# Place your TFLite model here

This directory should contain the Silent-Face-Anti-Spoofing TFLite model.

## Expected file:
- `silent_face_model.tflite`

## How to obtain the model:

### Option 1: Silent-Face-Anti-Spoofing (Recommended)
1. Visit: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
2. Download the pre-trained model
3. Convert to TFLite format if needed
4. Place the `.tflite` file in this directory

### Option 2: Use MobileFaceNet
1. Download MobileFaceNet model from TensorFlow Hub
2. Convert to TFLite format
3. Rename to `silent_face_model.tflite`

### Option 3: Train your own model
1. Use CASIA-FASD, OULU-NPU, or Replay-Attack datasets
2. Train a MobileNetV2-based classifier
3. Export to TFLite format

## Model Requirements:
- Input: 224x224x3 RGB image
- Output: 2 classes [spoof_probability, real_probability]
- Format: TensorFlow Lite (.tflite)
- Recommended size: < 6MB
