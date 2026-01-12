# Bengali Speech Recognition for Robotics

Real-time Bengali voice command recognition system for robotic control applications.

![Bengali Voice Control Demo](assets/demo.gif)

## Features

- **Bengali Language Support**: Native Bangla speech recognition
- **Low Latency**: <200ms response time for real-time control
- **Edge Deployment**: Optimized for NVIDIA Jetson and Raspberry Pi
- **Offline Mode**: Works without internet connection
- **Custom Commands**: Easy to add domain-specific vocabulary
- **ROS2 Integration**: Direct integration with robotic systems

## Supported Commands

| Bengali | English | Action |
|---------|---------|--------|
| ধরো | dhoro | Grip/Grasp |
| ছাড়ো | chharo | Release |
| থামো | thamo | Emergency Stop |
| বামে | bame | Move Left |
| ডানে | dane | Move Right |
| উপরে | upore | Move Up |
| নিচে | niche | Move Down |
| এটা তুলো | eta tulo | Pick this |
| রাখো | rakho | Place |
| শুরু করো | shuru koro | Start |

## Performance

| Platform | Latency | Accuracy | Model Size |
|----------|---------|----------|------------|
| Jetson Orin Nano | 120ms | 92% | 45MB |
| Jetson Nano 4GB | 180ms | 92% | 45MB |
| Raspberry Pi 4 | 250ms | 90% | 45MB |
| Desktop (GPU) | 50ms | 95% | 45MB |

## Installation

### Prerequisites

- Python 3.8+
- PyAudio (for microphone input)
- CUDA 11.4+ (for GPU acceleration)

### Quick Install

```bash
# Clone repository
git clone https://github.com/imnuman/bengali-speech-recognition.git
cd bengali-speech-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py
```

### Jetson Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pyaudio portaudio19-dev

# Install Python dependencies
pip install -r requirements-jetson.txt

# Download optimized models
python scripts/download_models.py --platform jetson
```

## Usage

### Basic Recognition

```python
from bengali_speech import BengaliRecognizer

# Initialize recognizer
recognizer = BengaliRecognizer(
    model='models/bengali_commands.pt',
    device='cuda'
)

# Start listening
recognizer.listen(callback=on_command)

def on_command(command, confidence):
    print(f"Command: {command}, Confidence: {confidence}")
```

### Command Line

```bash
# Test with microphone
python recognize.py --mic

# Process audio file
python recognize.py --audio test.wav

# Start continuous recognition
python recognize.py --mic --continuous
```

### ROS2 Integration

```python
from bengali_speech import BengaliCommandNode
import rclpy

rclpy.init()
node = BengaliCommandNode()
rclpy.spin(node)
```

```bash
# Launch ROS2 node
ros2 run bengali_speech command_node

# Subscribe to commands
ros2 topic echo /voice_commands
```

## Model Architecture

```
Bengali Speech Recognition Pipeline
├── Audio Input (16kHz, mono)
├── Feature Extraction
│   ├── Mel-Spectrogram (80 bins)
│   └── Delta features
├── Acoustic Model
│   ├── Conformer encoder
│   └── CTC decoder
├── Language Model
│   └── Bengali command vocabulary
└── Command Output
```

## Training Custom Commands

### Prepare Dataset

```
data/
├── audio/
│   ├── dhoro_001.wav
│   ├── dhoro_002.wav
│   └── ...
└── transcripts.csv
```

### Train Model

```bash
# Train with default config
python train.py --data configs/commands.yaml --epochs 50

# Train with augmentation
python train.py --data configs/commands.yaml --augment --epochs 100
```

## API Reference

### BengaliRecognizer

```python
class BengaliRecognizer:
    def __init__(
        self,
        model: str,              # Model path
        device: str = 'cuda',    # Device
        threshold: float = 0.7,  # Confidence threshold
        sample_rate: int = 16000 # Audio sample rate
    ):
        ...

    def recognize(self, audio: np.ndarray) -> Tuple[str, float]:
        """Recognize single audio clip."""
        ...

    def listen(
        self,
        callback: Callable,
        continuous: bool = True,
        timeout: float = None
    ):
        """Start continuous listening."""
        ...

    def add_command(self, bengali: str, phonetic: str):
        """Add custom command to vocabulary."""
        ...
```

## Project Structure

```
bengali-speech-recognition/
├── configs/
│   ├── commands.yaml      # Command definitions
│   └── model.yaml         # Model configuration
├── models/
│   └── bengali_commands.pt
├── scripts/
│   ├── download_models.py
│   └── record_samples.py
├── bengali_speech/
│   ├── __init__.py
│   ├── recognizer.py      # Main recognizer class
│   ├── audio.py           # Audio processing
│   ├── features.py        # Feature extraction
│   └── ros2_node.py       # ROS2 integration
├── recognize.py           # CLI tool
├── train.py              # Training script
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Areas where help is needed:

1. More Bengali voice samples for training
2. Additional command vocabulary
3. Dialect support (Sylheti, Chittagonian, etc.)
4. Integration with other robotic platforms

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Bengali Common Voice](https://commonvoice.mozilla.org/bn)
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

## Citation

```bibtex
@software{numan2024bengalispeech,
  author = {Al Numan},
  title = {Bengali Speech Recognition for Robotics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/imnuman/bengali-speech-recognition}
}
```

## Contact

- **Author**: Al Numan
- **Email**: admin@numanab.com
- **Project**: [ProjectX - BahuBol](https://projectxbd.com)

---

*Part of the BahuBol (বাহুবল) intelligent robotic arm project - Voice-controlled robotics for Bangladesh.*
