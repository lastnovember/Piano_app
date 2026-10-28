#!/usr/bin/env python3
"""
Play a USB MIDI keyboard (e.g. an M-Audio Keystation / Oxygen / Axiom) as a piano.

What it does
------------
1. Finds MIDI input ports and picks your keyboard (auto-detects M-Audio by name).
2. Lists audio output devices and lets you choose one.
3. Turns incoming MIDI notes into sound in real time. With a piano SoundFont
   (.sf2) next to the script it plays real recorded piano samples; without one
   it falls back to a built-in synth.

Note: a USB MIDI keyboard is *not* an audio input device. It sends note
numbers, velocities and pedal messages; the sound is generated here.

Usage
-----
    python midi_piano.py --list                # show MIDI inputs + audio outputs
    python midi_piano.py                       # auto-pick keyboard + default output
    python midi_piano.py --midi 1 --output 3   # pick explicitly (index or name substring)
    python midi_piano.py --sf2 piano.sf2       # use a specific SoundFont (or just drop one next to the script)
    python midi_piano.py --presets             # show the instruments inside the SoundFont
    python midi_piano.py --demo                # play a test chord (no keyboard needed)

Requirements
------------
    pip install mido python-rtmidi sounddevice numpy
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
        return np.repeat(buf.astype(np.float32)[:, None], 2, axis=1)

    def active_voices(self) -> int:
        return len(self.voices)


# --------------------------------------------------------------------------- #
# SoundFont (.sf2) sampler: plays recorded piano samples, no FluidSynth needed
# --------------------------------------------------------------------------- #

# SoundFont generator ids we honour (the rest are ignored).
_G_START_OFF, _G_END_OFF, _G_STARTLOOP_OFF, _G_ENDLOOP_OFF = 0, 1, 2, 3
_G_START_COARSE, _G_END_COARSE, _G_STARTLOOP_COARSE, _G_ENDLOOP_COARSE = 4, 12, 45, 50
_G_PAN = 17
_G_DELAY, _G_ATTACK, _G_HOLD, _G_DECAY, _G_SUSTAIN, _G_RELEASE = 33, 34, 35, 36, 37, 38
_G_KEY_TO_HOLD, _G_KEY_TO_DECAY = 39, 40
_G_INSTRUMENT, _G_KEYRANGE, _G_VELRANGE = 41, 43, 44
_G_KEYNUM, _G_VELOCITY, _G_ATTENUATION = 46, 47, 48
_G_COARSE_TUNE, _G_FINE_TUNE, _G_SAMPLE_ID, _G_SAMPLE_MODES = 51, 52, 53, 54
_G_SCALE_TUNING, _G_ROOT_KEY = 56, 58

# Instrument-level defaults (SoundFont 2.04 spec, section 8.1.3).
_INST_DEFAULTS = {
    _G_DELAY: -12000, _G_ATTACK: -12000, _G_HOLD: -12000, _G_DECAY: -12000,
    _G_SUSTAIN: 0, _G_RELEASE: -12000, _G_KEY_TO_HOLD: 0, _G_KEY_TO_DECAY: 0,
    _G_PAN: 0, _G_ATTENUATION: 0, _G_COARSE_TUNE: 0, _G_FINE_TUNE: 0,
    _G_SCALE_TUNING: 100, _G_ROOT_KEY: -1, _G_SAMPLE_MODES: 0, _G_KEYNUM: -1,
    _G_VELOCITY: -1, _G_START_OFF: 0, _G_END_OFF: 0, _G_STARTLOOP_OFF: 0,
    _G_ENDLOOP_OFF: 0, _G_START_COARSE: 0, _G_END_COARSE: 0,
    _G_STARTLOOP_COARSE: 0, _G_ENDLOOP_COARSE: 0,
}
# Generators the preset layer may add on top of the instrument values.
_ADDITIVE = {_G_DELAY, _G_ATTACK, _G_HOLD, _G_DECAY, _G_SUSTAIN, _G_RELEASE,
             _G_KEY_TO_HOLD, _G_KEY_TO_DECAY, _G_PAN, _G_ATTENUATION,
             _G_COARSE_TUNE, _G_FINE_TUNE, _G_SCALE_TUNING}
_SILENCE_DB = -96.0


def _timecents(tc: float) -> float:
    """Timecents to seconds (2^(tc/1200)); -12000 is the spec's 'instant'."""
    return 0.0 if tc <= -12000 else 2.0 ** (tc / 1200.0)


class SoundFont:
    """Minimal .sf2 reader: enough of the spec to play sampled pianos."""

    def __init__(self, path: str):
        import struct

        self.path = path
        with open(path, "rb") as fh:
            data = fh.read()

        if data[:4] != b"RIFF" or data[8:12] != b"sfbk":
            raise ValueError(f"{path} is not a SoundFont (.sf2) file")

        chunks: dict[bytes, bytes] = {}

        def walk(start: int, end: int) -> None:
            pos = start
            while pos + 8 <= end:
                cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
                body = pos + 8
                if cid == b"LIST":
                    walk(body + 4, body + size)
                else:
                    chunks[cid] = data[body:body + size]
                pos = body + size + (size & 1)

        walk(12, len(data))
        for required in (b"smpl", b"phdr", b"pbag", b"pgen", b"inst", b"ibag", b"igen", b"shdr"):
            if required not in chunks:
                raise ValueError(f"{path}: missing {required.decode()} chunk")

        self.samples = np.frombuffer(chunks[b"smpl"], dtype="<i2")

        rec = lambda cid, dt: np.frombuffer(chunks[cid], dtype=np.dtype(dt))
        self.phdr = rec(b"phdr", [("name", "S20"), ("preset", "<u2"), ("bank", "<u2"),
                                  ("bag", "<u2"), ("lib", "<u4"), ("genre", "<u4"),
                                  ("morph", "<u4")])
        self.pbag = rec(b"pbag", [("gen", "<u2"), ("mod", "<u2")])
        self.pgen = rec(b"pgen", [("op", "<u2"), ("lo", "u1"), ("hi", "u1")])
        self.inst = rec(b"inst", [("name", "S20"), ("bag", "<u2")])
        self.ibag = rec(b"ibag", [("gen", "<u2"), ("mod", "<u2")])
        self.igen = rec(b"igen", [("op", "<u2"), ("lo", "u1"), ("hi", "u1")])
        self.shdr = rec(b"shdr", [("name", "S20"), ("start", "<u4"), ("end", "<u4"),
                                  ("startloop", "<u4"), ("endloop", "<u4"), ("rate", "<u4"),
                                  ("pitch", "u1"), ("correction", "i1"), ("link", "<u2"),
                                  ("type", "<u2")])

    # -- zone decoding ------------------------------------------------------ #

    @staticmethod
    def _zone_gens(gens, lo: int, hi: int) -> dict[int, object]:
        out: dict[int, object] = {}
        for g in gens[lo:hi]:
            op = int(g["op"])
            if op in (_G_KEYRANGE, _G_VELRANGE):
                out[op] = (int(g["lo"]), int(g["hi"]))
            else:
                raw = int(g["lo"]) | (int(g["hi"]) << 8)
                out[op] = raw - 65536 if raw >= 32768 else raw     # signed 16-bit
        return out

    def _zones(self, bags, gens, first_bag: int, last_bag: int, terminal_op: int):
        """Split an item's bags into (global_zone, [zones]); the first bag is
        global when it lacks the terminal generator (instrument / sampleID)."""
        zones = []
        for b in range(first_bag, last_bag):
            lo, hi = int(bags[b]["gen"]), int(bags[b + 1]["gen"])
            zones.append(self._zone_gens(gens, lo, hi))
        global_zone: dict[int, object] = {}
        if zones and terminal_op not in zones[0]:
            global_zone = zones.pop(0)
        return global_zone, [z for z in zones if terminal_op in z]

    def preset_names(self) -> list[tuple[int, int, str]]:
        return [(int(p["bank"]), int(p["preset"]), p["name"].split(b"\0")[0].decode(errors="replace"))
                for p in self.phdr[:-1]]

    def build_preset(self, bank: int = 0, program: int = 0) -> "Preset":
        index = None
        for i, p in enumerate(self.phdr[:-1]):
            if int(p["bank"]) == bank and int(p["preset"]) == program:
                index = i
                break
        if index is None:
            index = 0                            # fall back to the first preset
        header = self.phdr[index]
        name = header["name"].split(b"\0")[0].decode(errors="replace")

        p_global, p_zones = self._zones(self.pbag, self.pgen, int(header["bag"]),
                                        int(self.phdr[index + 1]["bag"]), _G_INSTRUMENT)
        regions: list[Region] = []
        for pz in p_zones:
            pz_full = {**p_global, **pz}
            inst_index = pz_full[_G_INSTRUMENT]
            i_global, i_zones = self._zones(self.ibag, self.igen,
                                            int(self.inst[inst_index]["bag"]),
                                            int(self.inst[inst_index + 1]["bag"]), _G_SAMPLE_ID)
            for iz in i_zones:
                gens = dict(_INST_DEFAULTS)
                gens.update(i_global)
                gens.update(iz)
                for op, val in pz_full.items():
                    if op in _ADDITIVE:
                        gens[op] = gens.get(op, 0) + val
                key_lo, key_hi = gens.get(_G_KEYRANGE, (0, 127))
                vel_lo, vel_hi = gens.get(_G_VELRANGE, (0, 127))
                pk_lo, pk_hi = pz_full.get(_G_KEYRANGE, (0, 127))
                pv_lo, pv_hi = pz_full.get(_G_VELRANGE, (0, 127))
                region = Region(self, gens, (max(key_lo, pk_lo), min(key_hi, pk_hi)),
                                (max(vel_lo, pv_lo), min(vel_hi, pv_hi)))
                if region.valid:
                    regions.append(region)
        return Preset(name, regions)


class Region:
    """One instrument zone, fully resolved: which sample, how to play it."""

    def __init__(self, font: SoundFont, g: dict, key_range, vel_range):
        self.key_lo, self.key_hi = key_range
        self.vel_lo, self.vel_hi = vel_range
        hdr = font.shdr[g[_G_SAMPLE_ID]]
        self.start = int(hdr["start"]) + g[_G_START_OFF] + 32768 * g[_G_START_COARSE]
        self.end = int(hdr["end"]) + g[_G_END_OFF] + 32768 * g[_G_END_COARSE]
        self.loop_start = int(hdr["startloop"]) + g[_G_STARTLOOP_OFF] + 32768 * g[_G_STARTLOOP_COARSE]
        self.loop_end = int(hdr["endloop"]) + g[_G_ENDLOOP_OFF] + 32768 * g[_G_ENDLOOP_COARSE]
        self.sample_rate = int(hdr["rate"]) or 44100
        self.root_key = g[_G_ROOT_KEY] if g[_G_ROOT_KEY] >= 0 else int(hdr["pitch"])
        if self.root_key > 127:
            self.root_key = 60
        self.tune_cents = (int(hdr["correction"]) + g[_G_FINE_TUNE] + 100 * g[_G_COARSE_TUNE])
        self.scale_tuning = g[_G_SCALE_TUNING]
        self.loop_mode = g[_G_SAMPLE_MODES] & 3
        self.pan = max(-500, min(500, g[_G_PAN])) / 1000.0 + 0.5      # 0 = left, 1 = right
        # Like FluidSynth, scale initialAttenuation by 0.4: most SoundFonts were
        # authored against hardware that behaved that way and sound too quiet otherwise.
        self.attenuation_db = 0.4 * max(0, g[_G_ATTENUATION]) / 10.0
        self.delay = g[_G_DELAY]
        self.attack = g[_G_ATTACK]
        self.hold = g[_G_HOLD]
        self.decay = g[_G_DECAY]
        self.sustain_db = -min(1000, max(0, g[_G_SUSTAIN])) / 10.0
        self.release = g[_G_RELEASE]
        self.key_to_hold = g[_G_KEY_TO_HOLD]
        self.key_to_decay = g[_G_KEY_TO_DECAY]
        self.fixed_key = g[_G_KEYNUM]
        self.fixed_vel = g[_G_VELOCITY]
        n = font.samples.size
        self.valid = (0 <= self.start < self.end <= n and self.key_lo <= self.key_hi
                      and self.vel_lo <= self.vel_hi)
        if self.loop_mode and not (self.start <= self.loop_start < self.loop_end <= self.end):
            self.loop_mode = 0

    def matches(self, key: int, vel: int) -> bool:
        return self.key_lo <= key <= self.key_hi and self.vel_lo <= vel <= self.vel_hi


class Preset:
    def __init__(self, name: str, regions: list[Region]):
        self.name = name
        self.regions = regions
        self.low = min((r.key_lo for r in regions), default=0)
        self.high = max((r.key_hi for r in regions), default=127)

    def regions_for(self, key: int, vel: int) -> list[Region]:
        return [r for r in self.regions if r.matches(key, vel)]


class SampleVoice:
    """One region playing back: pitch-shifted sample + SoundFont volume envelope."""

    def __init__(self, region: Region, samples: np.ndarray, key: int, velocity: int,
                 out_rate: int):
        self.region = region
        self.samples = samples
        self.note = key
        self.out_rate = out_rate
        self.released = False
        self.finished = False
        self.pos = float(region.start)
        self.age = 0.0
        self.release_t = 0.0
        self.release_level = 0.0

        key_used = region.fixed_key if region.fixed_key >= 0 else key
        vel_used = region.fixed_vel if region.fixed_vel >= 0 else velocity
        semis = (key_used - region.root_key) * region.scale_tuning / 100.0
        self.ratio = (2.0 ** (semis / 12.0 + region.tune_cents / 1200.0)
                      * region.sample_rate / out_rate)

        # Velocity to loudness: the spec's default concave curve is ~ (vel/127)^2.
        vel_gain = (max(1, vel_used) / 127.0) ** 2
        self.gain = 1.4 * vel_gain * 10.0 ** (-region.attenuation_db / 20.0)
        self.pan_l = math.cos(region.pan * math.pi / 2)
        self.pan_r = math.sin(region.pan * math.pi / 2)

        self.t_delay = _timecents(region.delay)
        self.t_attack = _timecents(region.attack)
        self.t_hold = _timecents(region.hold + (60 - key_used) * region.key_to_hold)
        self.t_decay = _timecents(region.decay + (60 - key_used) * region.key_to_decay)
        self.t_release = max(0.005, _timecents(region.release))
        self.sustain_db = region.sustain_db

    def note_off(self) -> None:
        if not self.released:
            self.released = True
            self.release_t = self.age
            self.release_level = self._env_level(np.array([self.age]))[0]

    def _env_level(self, t: np.ndarray) -> np.ndarray:
        """Delay / attack / hold / decay / sustain envelope, as linear gain."""
        env = np.ones_like(t)
        t0 = t - self.t_delay
        env[t0 < 0] = 0.0
        if self.t_attack > 0:
            att = t0 < self.t_attack
            env[att] = np.clip(t0[att] / self.t_attack, 0.0, 1.0)
        t1 = t0 - self.t_attack - self.t_hold
        if self.sustain_db < 0:
            if self.t_decay > 0:
                frac = np.clip(t1 / self.t_decay, 0.0, 1.0)
            else:
                frac = (t1 >= 0).astype(np.float64)
            env *= 10.0 ** (self.sustain_db * frac / 20.0)
        return env

    def render(self, frames: int, bend: float, out: np.ndarray) -> None:
        if self.finished:
            return
        r = self.region
        n = np.arange(frames, dtype=np.float64)
        pos = self.pos + n * (self.ratio * bend)

        looping = r.loop_mode == 1 or (r.loop_mode == 3 and not self.released)
        if looping:
            span = r.loop_end - r.loop_start
            over = pos >= r.loop_end
            pos[over] = r.loop_start + np.mod(pos[over] - r.loop_start, span)
            next_pos = pos[-1] + self.ratio * bend
        else:
            next_pos = pos[-1] + self.ratio * bend

        idx = np.floor(pos).astype(np.int64)
        frac = pos - idx
        valid = idx < r.end - 1
        idx = np.clip(idx, 0, r.end - 1)
        block = (self.samples[idx] * (1.0 - frac) + self.samples[idx + 1] * frac) / 32768.0
        block[~valid] = 0.0

        t = self.age + n / self.out_rate
        env = self._env_level(t)
        if self.released:
            # dB-linear fade from the level at key release down to silence.
            fade = 10.0 ** (_SILENCE_DB / 20.0 * (t - self.release_t) / self.t_release)
            env = np.minimum(env, self.release_level * fade)
        block *= env * self.gain

        out[:, 0] += block * self.pan_l
        out[:, 1] += block * self.pan_r

        self.pos = next_pos if looping else min(next_pos, float(r.end))
        self.age = t[-1] + 1.0 / self.out_rate

        if (not looping and self.pos >= r.end - 1) or (
                self.released and self.release_level * fade[-1] < 1e-5):
            self.finished = True

    def loudness(self) -> float:
        lvl = self._env_level(np.array([self.age]))[0]
        if self.released:
            lvl = min(lvl, self.release_level * 10.0 ** (
                _SILENCE_DB / 20.0 * (self.age - self.release_t) / self.t_release))
        return float(lvl * self.gain)


class SoundFontSynth:
    """Same interface as PianoSynth, playing a .sf2 preset's samples."""

    def __init__(self, sf2_path: str, sample_rate: int = SAMPLE_RATE,
                 gain: float = 1.0, program: int = 0, bank: int = 0,
                 max_voices: int = MAX_VOICES):
        self.font = SoundFont(sf2_path)
        self.preset = self.font.build_preset(bank, program)
        if not self.preset.regions:
            raise RuntimeError(f"Preset {bank}:{program} in {sf2_path} has no playable zones")
        self.samples = self.font.samples.astype(np.float32)
        self.sample_rate = sample_rate
        self.gain = gain
        self.max_voices = max_voices
        self.voices: list[SampleVoice] = []
        self.sustain = False
        self.bend = 1.0
        self.lock = threading.Lock()
        self._held: set[int] = set()

    @property
    def name(self) -> str:
        return self.preset.name

    def note_on(self, note: int, velocity: int) -> None:
        if velocity == 0:
            self.note_off(note)
            return
        regions = self.preset.regions_for(note, velocity)
        if not regions:
            return
        with self.lock:
            self._held.discard(note)
            for v in self.voices:
                if v.note == note and not v.released:
                    v.note_off()
            while len(self.voices) + len(regions) > self.max_voices and self.voices:
                self.voices.remove(min(self.voices, key=SampleVoice.loudness))
            for r in regions:
                self.voices.append(SampleVoice(r, self.samples, note, velocity, self.sample_rate))

    def note_off(self, note: int) -> None:
        with self.lock:
            if self.sustain:
                self._held.add(note)
                return
            for v in self.voices:
                if v.note == note and not v.released:
                    v.note_off()

    def control_change(self, control: int, value: int) -> None:
        if control == 64:
            with self.lock:
                self.sustain = value >= 64
                if not self.sustain:
                    held, self._held = self._held, set()
                    for v in self.voices:
                        if v.note in held and not v.released:
                            v.note_off()
        elif control in (120, 123):
            with self.lock:
                self._held = set()
                self.sustain = False
                for v in self.voices:
                    v.note_off()

    def pitch_bend(self, value: int, semitones: float = 2.0) -> None:
        self.bend = 2.0 ** ((value / 8192.0) * semitones / 12.0)

    def render(self, frames: int) -> np.ndarray:
        buf = np.zeros((frames, 2), dtype=np.float64)
        with self.lock:
            for v in self.voices:
                v.render(frames, self.bend, buf)
            self.voices = [v for v in self.voices if not v.finished]
        buf *= self.gain
        np.tanh(buf, out=buf)
        return buf.astype(np.float32)

    def active_voices(self) -> int:
        return len(self.voices)


def find_soundfont(explicit: str | None) -> str | None:
    """Resolve --sf2: an explicit file, an archive holding one, or any .sf2
    sitting next to this script (names containing 'piano' win)."""
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(
                f"SoundFont not found: {explicit}\n"
                "Check the file name and folder (look in your Downloads folder)."
            )
        if explicit.lower().endswith(".sf2"):
            return explicit
        return extract_soundfont(explicit, here)
    found = sorted(glob.glob(os.path.join(here, "*.sf2")),
                   key=lambda p: ("piano" not in os.path.basename(p).lower(), p))
    return found[0] if found else None


def extract_soundfont(archive: str, dest: str) -> str:
    """Pull the first .sf2 out of a .zip / .tar.* download into `dest`."""
    import os
    import tarfile
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".sf2")]
            if not members:
                raise SystemExit(f"No .sf2 inside {archive}")
            target = os.path.join(dest, os.path.basename(members[0]))
            with zf.open(members[0]) as src, open(target, "wb") as dst:
                dst.write(src.read())
            return target
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith(".sf2"):
                    target = os.path.join(dest, os.path.basename(m.name))
                    with tf.extractfile(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    return target
            raise SystemExit(f"No .sf2 inside {archive}")
    raise SystemExit(f"{archive} is not a .sf2, .zip or .tar archive")


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
    sf2 = find_soundfont(args.sf2)
    if sf2:
        synth = SoundFontSynth(sf2, SAMPLE_RATE, args.gain, args.program, args.bank)
        synth.label = f"SoundFont '{synth.name}' from {sf2}"
        return synth
    synth = PianoSynth(SAMPLE_RATE, args.gain)
    synth.label = "built-in synth (drop a piano .sf2 next to this script for real piano samples)"
    return synth


def fill_output(outdata: np.ndarray, block: np.ndarray) -> None:
    """Copy a stereo block into a device buffer with any channel count."""
    ch = outdata.shape[1]
    if ch == 2:
        outdata[:] = block
    elif ch == 1:
        outdata[:, 0] = block.mean(axis=1)
    else:
        outdata[:, :2] = block
        outdata[:, 2:] = 0.0


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
        fill_output(outdata, synth.render(frames))

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
    engine = synth.label

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
        fill_output(outdata, synth.render(frames))

    with sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=device,
                         channels=channels, dtype="float32", callback=audio_callback):
        print(f"Engine: {synth.label}")
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
    p.add_argument("--sf2", help="piano SoundFont: a .sf2 file, or a .zip/.tar download holding one "
                                 "(default: any .sf2 found next to this script)")
    p.add_argument("--program", type=int, default=0, help="SoundFont program number (default 0 = grand piano)")
    p.add_argument("--bank", type=int, default=0, help="SoundFont bank number (default 0)")
    p.add_argument("--presets", action="store_true", help="list the presets in the SoundFont, then exit")
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
    if args.presets:
        sf2 = find_soundfont(args.sf2)
        if not sf2:
            raise SystemExit("No SoundFont found. Pass --sf2 or put a .sf2 next to this script.")
        for bank, prog, name in SoundFont(sf2).preset_names():
            print(f"  bank {bank:3d}  program {prog:3d}  {name}")
        return
    run(args)


if __name__ == "__main__":
    main()
