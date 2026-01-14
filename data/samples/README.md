# Sample Audio Files

This directory contains sample Bengali speech audio for testing.

## Required Files

Download or record sample audio files:
```
samples/
├── dhoro_01.wav      # ধরো (grip)
├── chharo_01.wav     # ছাড়ো (release)
├── thamo_01.wav      # থামো (stop)
├── bame_01.wav       # বামে (left)
├── dane_01.wav       # ডানে (right)
└── noise_sample.wav  # Background noise for testing
```

## Audio Format

- Format: WAV (PCM)
- Sample Rate: 16000 Hz
- Channels: Mono
- Bit Depth: 16-bit

## Recording Tips

1. Use a good quality microphone
2. Record in a quiet environment
3. Speak clearly at normal pace
4. Include some samples with background noise

## Dataset Sources

For training, we used:
- [Mozilla Common Voice Bengali](https://commonvoice.mozilla.org/bn)
- Custom recorded robotic commands

## Quick Test

```bash
# Test with a sample file
python recognize.py --audio data/samples/dhoro_01.wav

# Test with microphone
python recognize.py --mic
```
