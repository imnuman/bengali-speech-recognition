#!/usr/bin/env python3
"""
Bengali Speech Recognition CLI
==============================
Command-line interface for Bengali voice command recognition.

Author: Al Numan
Project: BahuBol - ProjectX

Usage:
    python recognize.py --mic                    # Listen from microphone
    python recognize.py --audio test.wav         # Process audio file
    python recognize.py --mic --continuous       # Continuous recognition
    python recognize.py --test-mic               # Test microphone
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from bengali_speech import BengaliRecognizer, Command


def on_command(cmd: Command):
    """Callback for recognized commands."""
    print(f"\n{'='*50}")
    print(f"Bengali: {cmd.bengali}")
    print(f"English: {cmd.english}")
    print(f"Confidence: {cmd.confidence:.2%}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bengali Voice Command Recognition"
    )

    # Input source
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Listen from microphone"
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file"
    )

    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Test microphone"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="models/bengali_commands.pt",
        help="Model path"
    )

    # Recognition options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep listening after recognition"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop after timeout seconds"
    )

    # Commands
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List supported commands"
    )

    args = parser.parse_args()

    # List commands
    if args.list_commands:
        recognizer = BengaliRecognizer(args.model, device=args.device)
        print("\nSupported Commands:")
        print("-" * 50)
        for bengali, (phonetic, english) in recognizer.get_commands().items():
            print(f"  {bengali} ({phonetic}) -> {english}")
        print("-" * 50)
        return

    # Initialize recognizer
    recognizer = BengaliRecognizer(
        model=args.model,
        device=args.device,
        threshold=args.threshold
    )

    # Test microphone
    if args.test_mic:
        print("Testing microphone...")
        audio = recognizer.test_microphone(duration=3.0)
        print(f"Recorded {len(audio)} samples")
        print(f"Max amplitude: {audio.max():.4f}")
        print(f"RMS: {(audio ** 2).mean() ** 0.5:.4f}")
        return

    # Process audio file
    if args.audio:
        import soundfile as sf

        print(f"Processing: {args.audio}")
        audio, sr = sf.read(args.audio)

        if sr != recognizer.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=recognizer.sample_rate)

        command, latency = recognizer.recognize(audio)

        if command:
            print(f"\nRecognized: {command.bengali} ({command.english})")
            print(f"Confidence: {command.confidence:.2%}")
            print(f"Latency: {latency*1000:.0f}ms")
        else:
            print("\nNo command recognized")
        return

    # Listen from microphone
    if args.mic:
        print("Starting microphone recognition...")
        print("Speak Bengali commands. Press Ctrl+C to stop.")
        print()

        try:
            recognizer.listen(
                callback=on_command,
                continuous=args.continuous,
                timeout=args.timeout
            )
        except KeyboardInterrupt:
            print("\nStopped")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
