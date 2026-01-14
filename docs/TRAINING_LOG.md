# Training Log - Bengali Speech Recognition

## Overview

This document tracks the fine-tuning of the Bengali speech recognition model.

---

## Approach

We fine-tuned OpenAI's Whisper model on Bengali speech data, with a focus on short command recognition for robotic control applications.

### Why Whisper?

1. **Multilingual**: Pre-trained on 680,000 hours of multilingual data
2. **Bengali Support**: Native Bengali (bn) support
3. **Robust**: Works well with various accents and noise conditions
4. **Open Source**: Can be fine-tuned and deployed locally

---

## Training Run 1 (2025-01-09)

### Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | openai/whisper-small |
| Dataset | Mozilla Common Voice Bengali |
| Train Samples | 12,500 |
| Validation Samples | 2,500 |
| Epochs | 10 |
| Batch Size | 16 |
| Learning Rate | 1e-5 |
| GPU | RTX 3060 12GB |

### Results

| Metric | Value |
|--------|-------|
| Word Error Rate (WER) | 12.3% |
| Character Error Rate (CER) | 5.8% |
| Best Epoch | 8 |
| Training Time | 3.5 hours |

### Command-Specific Accuracy

| Command | Bengali | Accuracy |
|---------|---------|----------|
| Grip | ধরো | 94% |
| Release | ছাড়ো | 93% |
| Emergency Stop | থামো | 97% |
| Move Left | বামে | 91% |
| Move Right | ডানে | 92% |
| Move Up | উপরে | 89% |
| Move Down | নিচে | 88% |
| Start | শুরু করো | 86% |

### Observations

1. **Emergency stop (থামো)** has highest accuracy - critical for safety
2. **Directional commands** work well in isolation
3. **Compound commands** ("শুরু করো") slightly less accurate
4. **Background noise** significantly impacts accuracy

### Known Issues

1. Similar-sounding words can be confused (বামে vs ডানে in noisy conditions)
2. Regional accent variations affect accuracy
3. Model struggles with very fast speech

---

## Edge Deployment

### Jetson Orin Nano

- Inference time: ~120ms
- Memory usage: 1.2GB
- Works well for real-time control

### Optimization Applied

1. FP16 quantization
2. TensorRT conversion
3. Streaming inference (chunk-based)

---

## Future Work

- [ ] Collect more robotics-specific Bengali command data
- [ ] Train larger model variant for higher accuracy
- [ ] Add speaker-independent validation
- [ ] Implement noise augmentation during training
- [ ] Test with different regional accents

---

## Reproducing Results

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_dataset.py --dataset common_voice_bn

# Train model
python train.py \
    --model openai/whisper-small \
    --dataset data/common_voice_bn \
    --epochs 10 \
    --batch-size 16 \
    --output models/whisper_bengali_v1

# Evaluate
python evaluate.py --model models/whisper_bengali_v1 --test-set data/test
```
