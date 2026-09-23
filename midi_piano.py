#!/usr/bin/env python3
"""
Play a USB MIDI keyboard (e.g. an M-Audio Keystation / Oxygen / Axiom) as a piano.

What it does
------------
1. Finds MIDI input ports and picks your keyboard (auto-detects M-Audio by name).
2. Lists audio output devices and lets you choose one.
3. Turns incoming MIDI notes into sound in real time, either with a built-in
   physically-flavoured piano synth (no extra files needed) or with a SoundFont
   through FluidSynth if you pass --sf2.

Note: a USB MIDI keyboard is *not* an audio input device. It sends note
numbers, velocities and pedal messages; the sound is generated here.

Usage
-----
    python midi_piano.py --list                # show MIDI inputs + audio outputs
    python midi_piano.py                       # auto-pick keyboard + default output
    python midi_piano.py --midi 1 --output 3   # pick explicitly (index or name substring)
    python midi_piano.py --sf2 piano.sf2       # use a SoundFont instead of the built-in synth
    python midi_piano.py --demo                # play a test chord (no keyboard needed)

Requirements
------------
    pip install mido python-rtmidi sounddevice numpy
    pip install pyfluidsynth        # only if you want --sf2
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 44100
# Windows' default audio path (MME / HDMI) needs bigger chunks to avoid
# "output underflow" crackles; 1024 samples is ~23 ms, still fine to play on.
BLOCK_SIZE = 1024 if platform.system() == "Windows" else 256
MAX_VOICES = 32
MAX_PARTIALS = 12

# Product names that mean "this is probably the M-Audio keyboard".
MAUDIO_HINTS = (
    "m-audio", "m audio", "maudio", "keystation", "oxygen", "axiom",
    "hammer", "keyrig", "prokeys", "midisport", "code 25", "code 49", "code 61",
)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


# --------------------------------------------------------------------------- #
# Built-in piano synth
# --------------------------------------------------------------------------- #

class Voice:
    """One struck string, modelled as a stack of decaying inharmonic partials."""

    def __init__(self, note: int, velocity: int, sample_rate: int = SAMPLE_RATE):
        self.note = note
        self.sample_rate = sample_rate
        self.released = False
        self.finished = False
        self.age = 0.0                 # seconds since the key went down
        self.release_env = 1.0         # damper multiplier, 1.0 until key release

        vel = max(1, velocity) / 127.0
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)

        # Brighter tone the harder you hit the key.
        brightness = 0.50 + 0.40 * vel
        # Real strings are stiff, so partials sit slightly sharp of the harmonics.
        inharmonicity = 4.0e-4

        nyquist = 0.45 * sample_rate
        freqs, amps, decays = [], [], []

        # A0 rings for ~12 s, the top octave for barely more than a second.
        t60 = 12.0 * math.exp(-(note - 21) / 38.0)
        base_decay = 6.908 / max(0.25, t60)

        for k in range(1, MAX_PARTIALS + 1):
            f = f0 * k * math.sqrt(1.0 + inharmonicity * k * k)
            if f >= nyquist:
                break
            freqs.append(f)
            amps.append(brightness ** (k - 1) / k ** 1.4)
            decays.append(base_decay * (1.0 + 0.28 * (k - 1)))

        self.freqs = np.array(freqs, dtype=np.float64)
        self.amps = np.array(amps, dtype=np.float64)
        self.decays = np.array(decays, dtype=np.float64)
        self.phases = np.random.uniform(0.0, 2.0 * math.pi, size=self.freqs.shape)

        norm = float(np.sum(self.amps)) or 1.0
        self.gain = 0.55 * (vel ** 1.4) / norm

        self.attack_tau = 0.0015 + 0.006 * math.exp(-(note - 21) / 40.0)
        # Fast damper on high notes, slower on the long bass strings.
        self.release_tau = 0.06 + 0.10 * math.exp(-(note - 21) / 45.0)

    def note_off(self) -> None:
        self.released = True

    def render(self, frames: int, bend: float, out: np.ndarray) -> None:
        """Add `frames` samples of this voice into `out` (mono, float64)."""
        if self.finished or self.freqs.size == 0:
            return

        sr = self.sample_rate
        n = np.arange(1, frames + 1, dtype=np.float64)
        t = self.age + n / sr

        inc = (2.0 * math.pi * self.freqs * bend / sr)[:, None]
        angles = self.phases[:, None] + inc * n[None, :]
        partials = np.sin(angles)
        partials *= self.amps[:, None] * np.exp(-self.decays[:, None] * t[None, :])
        block = partials.sum(axis=0)

        # Soft hammer attack, so notes do not click on.
        block *= 1.0 - np.exp(-t / self.attack_tau)

        if self.released:
            damper = self.release_env * np.exp(-(n / sr) / self.release_tau)
            block *= damper
            self.release_env = float(damper[-1])

        out += block * self.gain

        self.phases = np.mod(angles[:, -1], 2.0 * math.pi)
        self.age = float(t[-1])

        peak_now = float(np.max(self.amps * np.exp(-self.decays * self.age))) * self.release_env
        if peak_now * self.gain < 1e-5:
            self.finished = True

    def loudness(self) -> float:
        if self.freqs.size == 0:
            return 0.0
        return float(np.sum(self.amps * np.exp(-self.decays * self.age))) * self.release_env


class PianoSynth:
    """Polyphonic voice pool plus sustain pedal and pitch bend handling."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, gain: float = 1.0,
                 max_voices: int = MAX_VOICES):
        self.sample_rate = sample_rate
        self.gain = gain
        self.max_voices = max_voices
        self.voices: list[Voice] = []
        self.sustain = False
        self.bend = 1.0
        self._held: set[int] = set()         # keys let go while the pedal is down
        self.lock = threading.Lock()

    # -- MIDI side ---------------------------------------------------------- #

    def note_on(self, note: int, velocity: int) -> None:
        if velocity == 0:
            self.note_off(note)
            return
        with self.lock:
            # Re-struck key: damp the old string first, like a real action.
            for v in self.voices:
                if v.note == note and not v.released:
                    v.note_off()
            if len(self.voices) >= self.max_voices:
                quietest = min(self.voices, key=Voice.loudness)
                self.voices.remove(quietest)
            self.voices.append(Voice(note, velocity, self.sample_rate))

    def note_off(self, note: int) -> None:
        with self.lock:
            if self.sustain:
                self._held.add(note)
                return
            for v in self.voices:
                if v.note == note and not v.released:
                    v.note_off()

    def control_change(self, control: int, value: int) -> None:
        if control == 64:                       # sustain pedal
            with self.lock:
                self.sustain = value >= 64
                if not self.sustain:
                    held, self._held = self._held, set()
                    for v in self.voices:
                        if v.note in held and not v.released:
                            v.note_off()
        elif control in (120, 123):              # all sound off / all notes off
            with self.lock:
                self._held = set()
                self.sustain = False
                for v in self.voices:
                    v.note_off()

    def pitch_bend(self, value: int, semitones: float = 2.0) -> None:
        self.bend = 2.0 ** ((value / 8192.0) * semitones / 12.0)

    # -- Audio side --------------------------------------------------------- #

    def render(self, frames: int) -> np.ndarray:
        buf = np.zeros(frames, dtype=np.float64)
        with self.lock:
            for v in self.voices:
                v.render(frames, self.bend, buf)
            self.voices = [v for v in self.voices if not v.finished]
        buf *= self.gain
        np.tanh(buf, out=buf)                    # gentle limiter instead of clipping
        return buf.astype(np.float32)

    def active_voices(self) -> int:
        return len(self.voices)


class SoundFontSynth:
    """Same interface as PianoSynth, but sample-based via FluidSynth."""

    def __init__(self, sf2_path: str, sample_rate: int = SAMPLE_RATE,
                 gain: float = 1.0, program: int = 0):
        import fluidsynth                        # pyfluidsynth

        self.sample_rate = sample_rate
        self.gain = gain
        self.fs = fluidsynth.Synth(samplerate=float(sample_rate))
        sfid = self.fs.sfload(sf2_path)
        if sfid == -1:
            raise RuntimeError(f"FluidSynth could not load SoundFont: {sf2_path}")
        self.fs.program_select(0, sfid, 0, program)
        self.lock = threading.Lock()

    def note_on(self, note: int, velocity: int) -> None:
        with self.lock:
            if velocity == 0:
                self.fs.noteoff(0, note)
            else:
                self.fs.noteon(0, note, velocity)

    def note_off(self, note: int) -> None:
        with self.lock:
            self.fs.noteoff(0, note)

    def control_change(self, control: int, value: int) -> None:
        with self.lock:
            self.fs.cc(0, control, value)

    def pitch_bend(self, value: int, semitones: float = 2.0) -> None:
        with self.lock:
            self.fs.pitch_bend(0, value)

    def render(self, frames: int) -> np.ndarray:
        with self.lock:
            samples = self.fs.get_samples(frames)
        stereo = np.asarray(samples, dtype=np.float32).reshape(-1, 2) / 32768.0
        return stereo.mean(axis=1) * self.gain

    def active_voices(self) -> int:
        return -1


# --------------------------------------------------------------------------- #
# Device discovery / selection
# --------------------------------------------------------------------------- #

def list_midi_inputs() -> list[str]:
    import mido
    return list(mido.get_input_names())


def pick_midi_input(names: list[str], requested: str | None) -> str:
    if not names:
        raise SystemExit(
            "No MIDI inputs found.\n"
            "  * Is the keyboard powered and plugged into USB?\n"
            "  * Linux: check `amidi -l` or `aconnect -i`.\n"
            "  * macOS: check Audio MIDI Setup > MIDI Studio.\n"
            "  * Windows: install the M-Audio USB driver if the class-compliant one is missing."
        )

    if requested is not None:
        if requested.isdigit() and int(requested) < len(names):
            return names[int(requested)]
        for name in names:
            if requested.lower() in name.lower():
                return name
        raise SystemExit(f"No MIDI input matches {requested!r}. Available: {names}")

    for name in names:
        if any(hint in name.lower() for hint in MAUDIO_HINTS):
            return name

    if len(names) == 1:
        return names[0]

    print("\nMIDI inputs:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    choice = input("Which one is the keyboard? [0]: ").strip() or "0"
    return names[int(choice)]


def pick_output_device(requested: str | None):
    import sounddevice as sd

    if requested is None:
        return None                              # system default
    if requested.isdigit():
        return int(requested)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and requested.lower() in dev["name"].lower():
            return i
    raise SystemExit(f"No audio output matches {requested!r}. Run --list to see the options.")


def print_devices() -> None:
    print("MIDI inputs:")
    try:
        names = list_midi_inputs()
    except Exception as exc:                     # pragma: no cover - environment dependent
        names = []
        print(f"  (could not query MIDI: {exc})")
    if not names:
        print("  (none found)")
    for i, name in enumerate(names):
        flag = "  <- looks like the M-Audio keyboard" if any(
            h in name.lower() for h in MAUDIO_HINTS) else ""
        print(f"  [{i}] {name}{flag}")

    print("\nAudio outputs:")
    import sounddevice as sd
    default_out = sd.default.device[1]
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] <= 0:
            continue
        flag = "  <- default" if i == default_out else ""
        print(f"  [{i}] {dev['name']}  ({dev['max_output_channels']} ch, "
              f"{int(dev['default_samplerate'])} Hz){flag}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def build_synth(args) -> object:
    if args.sf2:
        try:
            return SoundFontSynth(args.sf2, SAMPLE_RATE, args.gain, args.program)
        except ImportError:
            raise SystemExit("--sf2 needs pyfluidsynth: pip install pyfluidsynth")
    return PianoSynth(SAMPLE_RATE, args.gain)


def run(args) -> None:
    try:
        import mido
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency ({exc.name}). Install everything with:\n"
            "    pip install mido python-rtmidi sounddevice numpy"
        )

    synth = build_synth(args)
    device = pick_output_device(args.output)

    if args.demo:
        play_demo(synth, device, args.channels)
        return

    port_name = pick_midi_input(list_midi_inputs(), args.midi)

    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = synth.render(frames)
        outdata[:] = np.repeat(mono[:, None], outdata.shape[1], axis=1)

    def midi_callback(msg):
        if msg.type == "note_on":
            synth.note_on(msg.note, msg.velocity)
            if args.verbose and msg.velocity:
                print(f"  {note_name(msg.note):>4}  vel {msg.velocity:3d}")
        elif msg.type == "note_off":
            synth.note_off(msg.note)
        elif msg.type == "control_change":
            synth.control_change(msg.control, msg.value)
        elif msg.type == "pitchwheel":
            synth.pitch_bend(msg.pitch, args.bend_range)

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=args.blocksize,
        device=device,
        channels=args.channels,
        dtype="float32",
        latency="low",
        callback=audio_callback,
    )

    out_name = sd.query_devices(device if device is not None else sd.default.device[1])["name"]
    engine = "SoundFont" if args.sf2 else "built-in piano synth"

    with stream, mido.open_input(port_name, callback=midi_callback):
        print(f"MIDI in : {port_name}")
        print(f"Audio out: {out_name}")
        print(f"Engine   : {engine}  ({SAMPLE_RATE} Hz, {args.blocksize}-sample blocks)")
        print("Play. Ctrl-C to stop.\n")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopped.")


def play_demo(synth, device, channels: int) -> None:
    """Sanity check for the audio path: a C major chord, no keyboard needed."""
    import sounddevice as sd

    def audio_callback(outdata, frames, time_info, status):
        mono = synth.render(frames)
        outdata[:] = np.repeat(mono[:, None], outdata.shape[1], axis=1)

    with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=device,
                         channels=channels, dtype="float32", callback=audio_callback):
        print("Demo: C major chord...")
        for note in (60, 64, 67, 72):
            synth.note_on(note, 96)
            time.sleep(0.12)
        time.sleep(2.5)
        for note in (60, 64, 67, 72):
            synth.note_off(note)
        time.sleep(1.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list MIDI inputs and audio outputs, then exit")
    p.add_argument("--midi", help="MIDI input index or name substring (default: auto-detect M-Audio)")
    p.add_argument("--output", help="audio output index or name substring (default: system default)")
    p.add_argument("--sf2", help="path to a piano SoundFont (.sf2); needs pyfluidsynth")
    p.add_argument("--program", type=int, default=0, help="SoundFont program number (default 0)")
    p.add_argument("--gain", type=float, default=1.0, help="master gain (default 1.0)")
    p.add_argument("--channels", type=int, default=2, help="output channels (default 2)")
    p.add_argument("--blocksize", type=int, default=BLOCK_SIZE,
                   help=f"audio block size; raise it if you hear crackling (default {BLOCK_SIZE})")
    p.add_argument("--bend-range", type=float, default=2.0, help="pitch bend range in semitones")
    p.add_argument("--demo", action="store_true", help="play a test chord and exit")
    p.add_argument("--verbose", action="store_true", help="print each note as it is played")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list:
        print_devices()
        return
    run(args)


if __name__ == "__main__":
    main()
