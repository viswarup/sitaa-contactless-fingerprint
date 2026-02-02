# Contactless Fingerprint Capture SDK

An Android application demonstrating contactless fingerprint capture using computer vision and deep learning. This project implements a robust pipeline for detection, quality assessment, liveness verification, and image enhancement.

## Project Tracks Status

This project fulfills the following tracks:
- **Track A (Implemented)**: Core Capture & Detection.
- **Track B (Implemented)**: Quality Assessment (Blur, Lighting, Distance).
- **Track D (Implemented)**: Liveness Detection (AI + Motion + Heuristics).
- **Track C (Research Phase)**: Advanced Matching & Enhancement.
    - *Status*: Literature review conducted and experimentation on **PolyU dataset** completed.
    - *Note*: Advanced enhancement (Gabor/Minutiae) was explored but excluded from the mobile demo to prioritize real-time performance.

## Features

### 1. Detection & Capture
- **MediaPipe Hand Detection**: Real-time 21-point 3D landmark detection to isolate valid hands.
- **ROI Extraction**: Automatically crops the finger region based on index finger landmarks (8, 12, 16, 20).
- **Auto-Flash**: Automatically triggers a white screen flash in Low Light or Backlit conditions.

### 2. Quality Assessment
- **Blur Detection**: Uses **Laplacian Variance** (Threshold > 15.0) to ensure ridge clarity.
- **Lighting Analysis**: Checks ROI brightness and global contrast to prevent dark or silhouetted captures.
- **Distance Check**: Validates finger width to ensure optimal DPI for processing.

### 3. Multi-Layer Liveness Detection (Anti-Spoofing)
A sophisticated defense system against prints, screens, and casts:
- **AI Deep Learning**: **MiniFASNet (ONNX)** models (2.7x and 4.0x scales) detect spoof artifacts like moiré patterns and screen pixels.
- **Motion Analysis**: **Optical Flow** verifies the natural micro-tremors (pulse/muscle tension) of a living hand vs. a static photo.
- **Specular Reflection**: detects unnatural screen glare/hotspots.
- **Texture/Color Analysis**: Checks for natural skin texture variance and color consistency.

### 4. Image Enhancement
- **Segmentation**: **Convex Hull** masking to remove background noise (wood grain, tables).
- **CLAHE**: **Contrast Limited Adaptive Histogram Equalization** to reveal ridges in shadowed areas without washing out details.
- **Grayscale Conversion**: Standard intensity mapping for AFIS compatibility.

---

## Technical Implementation

### Core Components
| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **FingerDetector** | `MediaPipe Hands` | Finds finger ROI robustly in complex backgrounds. |
| **OnnxLivenessDetector** | `Microsoft ONNX Runtime` | Runs MiniFASNet models for forensic spoof detection. |
| **MotionAnalyzer** | `OpenCV Optical Flow` | Distinguishes live movement from static photos. |
| **QualityAnalyzer** | `OpenCV Laplacian` | Ensures identifying features (ridges) are in focus. |
| **LightingAnalyzer** | `Histogram Analysis` | Triggers auto-flash to fix poor lighting. |
| **ImageEnhancer** | `Convex Hull` + `CLAHE` | Prepares the image for matching. |

### Excluded Advanced Features
The following features were explored but **excluded** from this demo due to complexity/performance trade-offs:

| Feature | Reason for Exclusion |
| :--- | :--- |
| **Gabor Filter Bank** | **Parameter Sensitivity**: Requires precise local frequency/orientation tuning per-block. untuned filters caused "Black Output" artifacts. |
| **Skeletonization** | **Noise Sensitivity**: Dependent on perfect Gabor output. Noise caused broken skeletons. |
| **Minutiae Extraction** | **High False Positives**: Without a clean skeleton, it detected thousands of noise artifacts. |
| **Neural Segmentation** | **Performance**: DeepLab/U-Net models were too heavy (>100ms) for real-time mobile preview. |

---

## Requirements

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0)
- Physical Android device with Camera (Emulator support is limited due to camera/gpu requirements)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/viswarup/sitaa-contactless-fingerprint.git

# Open in Android Studio
# Sync Gradle
# App ID: com.grokking.contactlessfingerprint
# Connect device with USB debugging enabled
# Run the app
```

## Project Structure

```
app/src/main/java/com/grokking/contactlessfingerprint/
├── MainActivity.kt          # Entry point
├── CaptureActivity.kt       # Main UI: Camera, Liveness, Auto-Flash
├── ResultActivity.kt        # Analysis Results, Enhancement Display
└── core/
    ├── FingerDetector.kt    # MediaPipe Implementation
    ├── OnnxLivenessDetector.kt # AI Model Inference (MiniFASNet)
    ├── MotionAnalyzer.kt    # Optical Flow Logic
    ├── QualityAnalyzer.kt   # Blur/Focus Check
    ├── LightingAnalyzer.kt  # Flash Trigger Logic
    ├── LivenessChecker.kt   # Heuristic Checks (Color/Texture)
    └── ImageEnhancer.kt     # Segmentation & CLAHE
```
## Key Learnings

Deep learning methods are more robust and accuracte, especifically for liveliness detection, segmentation and matching. Small deep learning models which can run on low end android devices should be preferred over traditional computer vision models.  

## Future Improvements

1.  **Fine-Tuning Liveness Models**: Current models are trained on Faces. Fine-tuning on a Finger-specific dataset (Live vs. Silicone/Latex/Play-Doh) would significantly improve accuracy against 3D spoofs.
2.  **MobileNetV2 Segmentation**: Replacing Convex Hull with a lightweight MobileNetV2 segmentation model would solve the "webbing" issue where background between spread fingers is captured.

## Dependencies

- `androidx.camera:camera-core:1.3.1`
- `com.google.mediapipe:tasks-vision:0.10.9`
- `org.opencv:opencv-android:4.x`
- `com.microsoft.onnxruntime:onnxruntime-android:1.16.0`
- `org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3`

## License

MIT License
