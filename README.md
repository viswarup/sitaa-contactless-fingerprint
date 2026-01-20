# Contactless Fingerprint Capture SDK

An Android application demonstrating contactless fingerprint capture using computer vision and machine learning.

## Features

- **MediaPipe Hand Detection** - Precise finger landmark detection (21 points)
- **Real-time Quality Assessment** - Blur detection, brightness analysis
- **4-Layer Liveness Detection**:
  - Texture analysis (Laplacian variance)
  - Color variance (HSV saturation)
  - Specular reflection (screen glare detection)
  - Motion analysis (micro-tremor detection)
- **Image Enhancement** - Convex hull segmentation + CLAHE
- **Finger Alignment Overlay** - Visual guide for positioning

## Requirements

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0)
- Physical Android device with camera

## Quick Start

```bash
# Clone the repository
git clone https://github.com/viswarup/sitaa-contactless-fingerprint.git

# Open in Android Studio
# Sync Gradle
# Connect device with USB debugging enabled
# Run the app
```

## Project Structure

```
app/src/main/java/com/grokking/contactlessfingerprint/
├── MainActivity.kt          # Entry point, instructions
├── CaptureActivity.kt       # Camera + detection + countdown
├── ResultActivity.kt        # Enhancement display + save
└── core/
    ├── FingerDetector.kt    # MediaPipe wrapper
    ├── QualityAnalyzer.kt   # Blur + brightness scoring
    ├── LivenessChecker.kt   # Texture + color + specular
    ├── MotionAnalyzer.kt    # Multi-frame motion detection
    └── ImageEnhancer.kt     # Convex hull + CLAHE pipeline
```

## Dependencies

- CameraX 1.3.1
- MediaPipe Tasks Vision 0.10.9
- OpenCV Android 4.x
- Kotlin Coroutines 1.7.3

## License

MIT License
