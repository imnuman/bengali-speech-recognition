"""
Bengali Speech Recognition
==========================
Real-time Bengali voice command recognition for robotic control.

Author: Al Numan
Project: BahuBol - ProjectX
"""

import numpy as np
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from dataclasses import dataclass
import threading
import queue
import time

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Recognized voice command."""
    bengali: str
    english: str
    confidence: float
    timestamp: float
    audio_duration: float


class BengaliRecognizer:
    """
    Bengali voice command recognizer.

    Optimized for real-time robotic control with low latency.

    Example:
        recognizer = BengaliRecognizer('models/bengali_commands.pt')

        def on_command(cmd):
            print(f"Recognized: {cmd.bengali} ({cmd.english})")

        recognizer.listen(callback=on_command)
    """

    # Default command vocabulary
    COMMANDS = {
        'ধরো': ('dhoro', 'grip'),
        'ছাড়ো': ('chharo', 'release'),
        'থামো': ('thamo', 'stop'),
        'বামে': ('bame', 'left'),
        'ডানে': ('dane', 'right'),
        'উপরে': ('upore', 'up'),
        'নিচে': ('niche', 'down'),
        'সামনে': ('samne', 'forward'),
        'পিছনে': ('pichone', 'backward'),
        'রাখো': ('rakho', 'place'),
        'হোম': ('home', 'home'),
        'রিসেট': ('reset', 'reset')
    }

    def __init__(
        self,
        model: str,
        device: str = 'cuda',
        threshold: float = 0.7,
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
        vad_threshold: float = 0.5
    ):
        """
        Initialize Bengali recognizer.

        Args:
            model: Path to model weights
            device: Device ('cuda' or 'cpu')
            threshold: Confidence threshold for accepting commands
            sample_rate: Audio sample rate (Hz)
            chunk_duration: Duration of audio chunks (seconds)
            vad_threshold: Voice activity detection threshold
        """
        self.model_path = Path(model)
        self.device = device
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.vad_threshold = vad_threshold

        self._model = None
        self._is_listening = False
        self._audio_queue = queue.Queue()
        self._callback = None

        # Load model
        self._load_model()

        logger.info(f"Bengali recognizer initialized on {device}")

    def _load_model(self):
        """Load speech recognition model."""
        try:
            import torch
            import torchaudio

            if not self.model_path.exists():
                logger.warning(f"Model not found: {self.model_path}")
                logger.info("Using fallback lightweight model")
                self._model = self._create_fallback_model()
            else:
                # Load pre-trained model
                self._model = torch.jit.load(
                    str(self.model_path),
                    map_location=self.device
                )
                self._model.set_mode('inference')

            logger.info("Model loaded successfully")

        except ImportError:
            logger.error("PyTorch not installed. pip install torch torchaudio")
            raise

    def _create_fallback_model(self):
        """Create a simple fallback model for testing."""
        import torch
        import torch.nn as nn

        class SimpleCommandModel(nn.Module):
            def __init__(self, num_commands):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(80, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                )
                self.fc = nn.Linear(256, num_commands)

            def forward(self, x):
                x = self.conv(x)
                x = x.squeeze(-1)
                return self.fc(x)

        model = SimpleCommandModel(len(self.COMMANDS))
        return model.to(self.device)

    def _extract_features(self, audio: np.ndarray):
        """Extract mel-spectrogram features from audio."""
        import torch
        import torchaudio.transforms as T

        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )

        mel = mel_transform(audio)
        mel = torch.log(mel + 1e-9)

        return mel.to(self.device)

    def recognize(self, audio: np.ndarray) -> Tuple[Optional[Command], float]:
        """
        Recognize command from audio.

        Args:
            audio: Audio waveform (numpy array)

        Returns:
            Tuple of (Command or None, processing_time)
        """
        import torch

        start_time = time.time()

        # Extract features
        features = self._extract_features(audio)

        # Run inference
        with torch.no_grad():
            logits = self._model(features)
            probs = torch.softmax(logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

        confidence = confidence.item()
        idx = idx.item()

        process_time = time.time() - start_time

        # Check threshold
        if confidence < self.threshold:
            return None, process_time

        # Get command
        commands = list(self.COMMANDS.keys())
        if idx < len(commands):
            bengali = commands[idx]
            phonetic, english = self.COMMANDS[bengali]

            return Command(
                bengali=bengali,
                english=english,
                confidence=confidence,
                timestamp=time.time(),
                audio_duration=len(audio) / self.sample_rate
            ), process_time

        return None, process_time

    def listen(
        self,
        callback: Callable[[Command], None],
        continuous: bool = True,
        timeout: Optional[float] = None
    ):
        """
        Start listening for voice commands.

        Args:
            callback: Function called with recognized command
            continuous: Keep listening after recognition
            timeout: Stop after timeout seconds (None = forever)
        """
        try:
            import pyaudio
        except ImportError:
            logger.error("PyAudio not installed. pip install pyaudio")
            raise

        self._callback = callback
        self._is_listening = True

        # Audio parameters
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk_samples
        )

        logger.info("Listening for Bengali commands...")
        start_time = time.time()

        try:
            while self._is_listening:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    break

                # Read audio
                data = stream.read(chunk_samples, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)

                # Voice activity detection
                energy = np.sqrt(np.mean(np.power(audio, 2)))
                if energy < self.vad_threshold * 0.01:
                    continue

                # Recognize
                command, latency = self.recognize(audio)

                if command:
                    logger.info(
                        f"Recognized: {command.bengali} "
                        f"(conf={command.confidence:.2f}, latency={latency*1000:.0f}ms)"
                    )
                    callback(command)

                    if not continuous:
                        break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self._is_listening = False

    def stop(self):
        """Stop listening."""
        self._is_listening = False

    def add_command(self, bengali: str, phonetic: str, english: str):
        """
        Add custom command to vocabulary.

        Args:
            bengali: Bengali text
            phonetic: Phonetic representation
            english: English translation/action
        """
        self.COMMANDS[bengali] = (phonetic, english)
        logger.info(f"Added command: {bengali} -> {english}")

    def get_commands(self) -> Dict[str, Tuple[str, str]]:
        """Get current command vocabulary."""
        return self.COMMANDS.copy()

    def test_microphone(self, duration: float = 3.0) -> np.ndarray:
        """
        Test microphone by recording audio.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio
        """
        import pyaudio

        p = pyaudio.PyAudio()
        samples = int(self.sample_rate * duration)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=samples
        )

        logger.info(f"Recording {duration} seconds...")
        data = stream.read(samples)
        audio = np.frombuffer(data, dtype=np.float32)

        stream.stop_stream()
        stream.close()
        p.terminate()

        rms = np.sqrt(np.mean(np.power(audio, 2)))
        logger.info(f"Recording complete. RMS: {rms:.4f}")
        return audio


class BengaliCommandNode:
    """
    ROS2 node for Bengali voice commands.

    Publishes recognized commands to /voice_commands topic.
    """

    def __init__(self, model_path: str = 'models/bengali_commands.pt'):
        """Initialize ROS2 node."""
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError:
            logger.error("ROS2 not installed")
            raise

        class _BengaliNode(Node):
            def __init__(self, recognizer):
                super().__init__('bengali_command_node')
                self.publisher = self.create_publisher(String, 'voice_commands', 10)
                self.recognizer = recognizer

            def publish_command(self, command: Command):
                msg = String()
                msg.data = f"{command.english}:{command.confidence:.2f}"
                self.publisher.publish(msg)
                self.get_logger().info(f"Published: {command.bengali}")

        self.recognizer = BengaliRecognizer(model_path)
        self._node = _BengaliNode(self.recognizer)

    def spin(self):
        """Start the node."""
        import rclpy

        # Start listening in background
        thread = threading.Thread(
            target=self.recognizer.listen,
            args=(self._node.publish_command,),
            daemon=True
        )
        thread.start()

        rclpy.spin(self._node)
