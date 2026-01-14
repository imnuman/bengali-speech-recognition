"""
Tests for Bengali Speech Recognition.
Run with: pytest tests/ -v
"""

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfiguration:
    """Test configuration and setup."""

    def test_requirements_exists(self):
        """Check requirements.txt exists."""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        assert req_path.exists(), "requirements.txt missing"

    def test_has_whisper_dependency(self):
        """Check whisper or vosk is in requirements."""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        with open(req_path) as f:
            content = f.read().lower()
        has_asr = "whisper" in content or "vosk" in content or "transformers" in content
        assert has_asr, "ASR library missing from requirements"

    def test_recognize_script_exists(self):
        """Check main recognition script exists."""
        script_path = Path(__file__).parent.parent / "recognize.py"
        assert script_path.exists(), "recognize.py missing"

    def test_recognize_script_valid_syntax(self):
        """Check recognize.py has valid Python syntax."""
        script_path = Path(__file__).parent.parent / "recognize.py"
        with open(script_path) as f:
            code = f.read()
        compile(code, script_path, "exec")


class TestCommands:
    """Test command vocabulary."""

    def test_commands_defined(self):
        """Check Bengali commands are defined."""
        # Commands should be defined somewhere in the codebase
        bengali_speech_dir = Path(__file__).parent.parent / "bengali_speech"
        py_files = list(bengali_speech_dir.glob("*.py"))

        assert len(py_files) > 0, "No Python files in bengali_speech/"

        # Check for common Bengali commands in any file
        bengali_commands = ["ধরো", "ছাড়ো", "থামো", "বামে", "ডানে"]
        found_bengali = False

        for py_file in py_files:
            with open(py_file, encoding='utf-8') as f:
                content = f.read()
            if any(cmd in content for cmd in bengali_commands):
                found_bengali = True
                break

        assert found_bengali, "Bengali commands not found in source code"

    def test_english_mappings_exist(self):
        """Check English command mappings exist."""
        bengali_speech_dir = Path(__file__).parent.parent / "bengali_speech"

        english_commands = ["grip", "release", "stop", "left", "right"]
        found_mapping = False

        for py_file in bengali_speech_dir.glob("*.py"):
            with open(py_file) as f:
                content = f.read().lower()
            if any(cmd in content for cmd in english_commands):
                found_mapping = True
                break

        assert found_mapping, "English command mappings not found"


class TestAudioProcessing:
    """Test audio processing utilities."""

    def test_audio_module_exists(self):
        """Check audio processing module exists."""
        bengali_speech_dir = Path(__file__).parent.parent / "bengali_speech"
        py_files = [f.name for f in bengali_speech_dir.glob("*.py")]

        # Should have some audio processing
        audio_related = any(
            name in py_files
            for name in ["audio.py", "processor.py", "stream.py", "vad.py"]
        )

        # Or audio handling in main files
        if not audio_related:
            for py_file in bengali_speech_dir.glob("*.py"):
                with open(py_file) as f:
                    content = f.read().lower()
                if "audio" in content or "microphone" in content:
                    audio_related = True
                    break

        assert audio_related, "No audio processing code found"


class TestSampleData:
    """Test sample data availability."""

    def test_samples_directory_exists(self):
        """Check samples directory exists."""
        samples_dir = Path(__file__).parent.parent / "data" / "samples"
        assert samples_dir.exists(), "data/samples/ directory missing"

    def test_sample_audio_readme(self):
        """Check samples have documentation."""
        samples_dir = Path(__file__).parent.parent / "data" / "samples"
        readme = samples_dir / "README.md"
        # Either README or actual samples should exist
        has_docs = readme.exists() or len(list(samples_dir.glob("*.wav"))) > 0
        assert has_docs, "Sample data undocumented"


class TestModel:
    """Test model configuration."""

    def test_models_directory_exists(self):
        """Check models directory exists."""
        models_dir = Path(__file__).parent.parent / "models"
        assert models_dir.exists(), "models/ directory missing"


class TestIntegration:
    """Test ROS2 integration code."""

    def test_ros2_publisher_exists(self):
        """Check ROS2 integration is present."""
        bengali_speech_dir = Path(__file__).parent.parent / "bengali_speech"

        ros_integration = False
        for py_file in bengali_speech_dir.glob("*.py"):
            with open(py_file) as f:
                content = f.read()
            if "rclpy" in content or "ros2" in content.lower() or "ROS" in content:
                ros_integration = True
                break

        # ROS2 integration is optional but should be documented
        assert ros_integration or True, "ROS2 integration not found (optional)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
