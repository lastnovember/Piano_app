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
3. plays a piano SoundFont if one sits next to the script (see below), or the
   built-in synth otherwise. Play, Ctrl-C to quit.

Try `python midi_piano.py --demo` first if you want to hear a test chord without
touching the keyboard.

## Choosing devices

```bash
python midi_piano.py --list              # show every MIDI input and audio output
python midi_piano.py --midi 1 --output 3 # pick by index...
python midi_piano.py --output "Headphones" # ...or by name
```

## Real piano sound (recommended)

The built-in synth needs no extra files, but it sounds like a synth. For a real
piano, give the app a piano **SoundFont** (`.sf2`): a file of recorded piano
samples. The app plays SoundFonts itself; nothing else to install.

1. Download a free piano SoundFont. Good ones under a Creative Commons licence:
   - FreePats "YDP Grand Piano" (Yamaha Disklavier, ~36 MB) and "Upright Piano
     KW" from https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
     (pick the **SF2** download on that page)
   - Anything tagged "piano" and "sf2" on https://musical-artifacts.com
2. Put the `.sf2` file in this folder, next to `midi_piano.py`. If the download
   is a `.zip` or `.tar.bz2`, you can hand it straight to the app once and it
   will unpack the `.sf2` here for you:

   ```bash
   python midi_piano.py --sf2 "C:\Users\you\Downloads\YDP-GrandPiano-SF2-20160804.tar.bz2"
   ```

3. Run `python midi_piano.py` as usual. The startup line will say
   `Engine: SoundFont 'Grand Piano' from ...` instead of "built-in synth".

A SoundFont can hold many instruments. `python midi_piano.py --presets` lists
them; `--program N` (and `--bank N`) picks one. Program 0 is the grand piano in
General MIDI collections such as `FluidR3_GM.sf2` or `TimGM6mb.sf2`.

## Reverb and EQ

A light hall reverb is on by default. Adjust or add tone shaping with:

```bash
python midi_piano.py --reverb 0.5 --reverb-time 2.5   # bigger room (0 = dry)
python midi_piano.py --eq 500:3                       # +3 dB around 500 Hz (warmth / body)
python midi_piano.py --eq 500:3 --eq 3000:-2:2        # repeatable; FREQ:DB[:Q]
```

To keep your favourite settings without retyping them, create a file called
`settings.txt` next to `midi_piano.py` with the flags inside, one or more per
line:

```
--eq 500:3
--reverb 0.4
```

Anything typed on the command line still overrides what is in the file.

## Supported MIDI

- Note on/off with velocity (louder and brighter the harder you play)
- Sustain pedal (CC 64), all-notes-off (CC 120/123)
- Pitch bend (`--bend-range`, default 2 semitones)

## Troubleshooting

| Problem | Fix |
|---|---|
| "No MIDI inputs found" | Check the USB cable and that the keyboard is powered. On Windows install the M-Audio driver if it is not class compliant. |
| Crackling / dropouts | `--blocksize 512` or `1024`. Close other audio apps. |
| Delay between key and sound | On Windows the app already picks the low-delay WASAPI route when it exists; `--list` shows which route each output uses. For the least delay add `--exclusive` (other apps go silent while you play). Wired headphones or speakers in the PC's own jack beat monitor/HDMI audio, and Bluetooth always adds ~100 ms or more. |
| Wrong output device | `--list`, then `--output <index>`. |
| Too quiet / too loud | `--gain 1.5` / `--gain 0.5`. |
| Linux permission error on MIDI | Add your user to the `audio` group, or run with `sudo` once to test. |

## Options

```
--list          list MIDI inputs and audio outputs, then exit
--midi X        MIDI input index or name substring
--output X      audio output index or name substring
--sf2 FILE      piano SoundFont (.sf2, or a .zip/.tar download holding one);
                default: any .sf2 next to the script
--presets       list the instruments inside the SoundFont, then exit
--program N     SoundFont program (default 0 = grand piano)
--bank N        SoundFont bank (default 0)
--gain G        master gain (default 1.0)
--blocksize N   audio block size: smaller = less delay, bigger = fewer crackles
                (default 256, or 1024 on Windows' legacy MME route)
--exclusive     Windows: open the output exclusively (WASAPI) for the lowest delay
--bend-range S  pitch bend range in semitones (default 2)
--reverb A      reverb amount, 0 = dry (default 0.35)
--reverb-time S reverb tail in seconds (default 1.8)
--eq F:DB[:Q]   boost or cut a band, e.g. 500:3; repeatable
--demo          play a test chord and exit
--verbose       print each note as it is played
```
