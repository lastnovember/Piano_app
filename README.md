# Piano App

Plug in a USB MIDI keyboard (tested against M-Audio Keystation / Oxygen style
controllers), run one command, play piano.

The keyboard sends MIDI (which keys, how hard, pedal), not audio. This program
listens for those messages, generates the piano sound, and sends it to the audio
output you choose (speakers, headphones, an interface).

## Quick start

```bash
pip install -r requirements.txt
python midi_piano.py
```

That is it. With no arguments it:

1. finds your MIDI inputs and auto-selects the M-Audio keyboard,
2. uses your system's default audio output,
3. starts the built-in piano synth. Play, Ctrl-C to quit.

Try `python midi_piano.py --demo` first if you want to hear a test chord without
touching the keyboard.

## Choosing devices

```bash
python midi_piano.py --list              # show every MIDI input and audio output
python midi_piano.py --midi 1 --output 3 # pick by index...
python midi_piano.py --output "Headphones" # ...or by name
```

## Better piano sound (optional)

The built-in synth needs no extra files. For a sampled grand piano, install
FluidSynth and point the script at a SoundFont:

```bash
pip install pyfluidsynth       # also needs the fluidsynth library:
                               #   macOS: brew install fluid-synth
                               #   Ubuntu: sudo apt install fluidsynth
python midi_piano.py --sf2 /path/to/piano.sf2
```

Free piano SoundFonts: "Salamander Grand" (sf2 conversion), "FluidR3_GM.sf2"
(program 0 is an acoustic grand piano).

## Supported MIDI

- Note on/off with velocity (louder and brighter the harder you play)
- Sustain pedal (CC 64), all-notes-off (CC 120/123)
- Pitch bend (`--bend-range`, default 2 semitones)

## Troubleshooting

| Problem | Fix |
|---|---|
| "No MIDI inputs found" | Check the USB cable and that the keyboard is powered. On Windows install the M-Audio driver if it is not class compliant. |
| Crackling / dropouts | `--blocksize 512` or `1024`. Close other audio apps. |
| Wrong output device | `--list`, then `--output <index>`. |
| Too quiet / too loud | `--gain 1.5` / `--gain 0.5`. |
| Linux permission error on MIDI | Add your user to the `audio` group, or run with `sudo` once to test. |

## Options

```
--list          list MIDI inputs and audio outputs, then exit
--midi X        MIDI input index or name substring
--output X      audio output index or name substring
--sf2 FILE      SoundFont piano via FluidSynth
--program N     SoundFont program (default 0)
--gain G        master gain (default 1.0)
--blocksize N   audio block size (default 256)
--bend-range S  pitch bend range in semitones (default 2)
--demo          play a test chord and exit
--verbose       print each note as it is played
```
