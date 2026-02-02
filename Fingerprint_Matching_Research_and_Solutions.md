# **Technical Feasibility and Solution Architecture for Contactless-to-Contact Fingerprint Interoperability in Mobile Biometrics**

## **1\. Executive Context and Problem Landscape**

### **1.1 The Paradigm Shift in Biometric Identity Management**

The global landscape of biometric identity management is currently undergoing a foundational transformation, migrating from dedicated, hardware-dependent acquisition modalities toward ubiquitous, software-defined solutions. For decades, the gold standard for fingerprint acquisition has been the optical or capacitive scanner, devices that rely on physical contact to capture the friction ridge patterns of the human finger. These systems, utilizing Frustrated Total Internal Reflection (FTIR), have underpinned the world's largest identity databases, including the Federal Bureau of Investigation’s Next Generation Identification (NGI) system and the Unique Identification Authority of India (UIDAI) Aadhaar ecosystem.1

However, the operational rigidities of contact-based systems have become increasingly apparent. The reliance on specialized hardware limits scalability, particularly in rural or resource-constrained environments where distributing certified scanners is logistically burdensome. Furthermore, the global COVID-19 pandemic catalyzed a permanent shift in user expectations regarding hygiene, rendering shared contact surfaces less desirable.3 Consequently, the industry is pivoting toward **contactless fingerprint acquisition**, leveraging the high-resolution RGB cameras embedded in commodity smartphones to capture "finger photos."

This transition, while operationally advantageous, introduces a profound technical schism known as the **interoperability gap**. The legacy databases (galleries) against which new users must be authenticated are populated almost exclusively with contact-based images. A smartphone camera captures a fundamentally different representation of the finger—a 3D, perspective-distorted, diffusely reflected image—compared to the 2D, flattened, frustration-contrast image of a contact scanner. The challenge, therefore, is not merely to capture a fingerprint, but to algorithmically translate the contactless modality into a representation that is mathematically indistinguishable from a contact scan by the legacy Automated Fingerprint Identification Systems (AFIS).4

### **1.2 The UIDAI/SITAA Challenge Framework**

The specific context for this research is the challenge posed by the UIDAI and the SITAA-Startup-Cohort1, which seeks to develop a solution for **contactless-to-contact (C2C) matching** that runs entirely on the mobile edge.1 The requirements are rigorous and delineate the boundaries of the necessary technical architecture:

1. **Legacy Interoperability:** The solution must produce output templates that are fully compatible with UIDAI’s existing AFIS. This implies that the system must function as a "virtual scanner," outputting standard ISO/IEC 19794-4 images despite the disparate input modality.1  
2. **Edge-Based Processing:** The entire pipeline—from capture to template generation—must execute on Android devices (version 9 and above) with a strict latency threshold of **≤ 2 seconds**. This constraint effectively disqualifies heavy server-side processing or massive deep learning models that cannot be quantized or optimized for mobile Neural Processing Units (NPUs).1  
3. **ISO Standardization:** The output must not only match visually but must adhere to the **ISO/IEC 19794-4** standard, which dictates a spatial resolution of 500 pixels per inch (PPI), specific grayscale depths, and Wavelet Scalar Quantization (WSQ) compression. This introduces a complex requirement for precise scale estimation, as mobile cameras do not have a fixed focal plane relative to the subject.7  
4. **Quality Assurance:** The system must incorporate mechanisms for "Readiness Scoring," quantifying metrics such as minutiae count and clarity (NFIQ) to flag acceptable versus rejected captures before they are transmitted for matching.1

The objective of this report is to deconstruct these requirements into a viable technical architecture, identifying the specific deep learning models and signal processing algorithms capable of bridging the C2C gap within the constraints of mobile edge computing.

## ---

**2\. The Physics and Mathematics of Interoperability**

To engineer a solution that allows a mobile photo to match a contact scan, one must first rigorously model the physical and geometric divergences between the two modalities. These differences are not merely cosmetic; they are structural discrepancies that prevent standard minutiae extractors from functioning.

### **2.1 Photometric Divergence: FTIR vs. Ambient Reflection**

The primary barrier to immediate interoperability lies in the physics of image formation.

**Frustrated Total Internal Reflection (FTIR):**

Legacy contact scanners operate on the principle of FTIR. A glass prism is illuminated from the side. When a finger presses against the platen, the ridges of the skin come into physical contact with the glass. At these points of contact, the refractive index boundary changes, causing the light to be absorbed or scattered rather than reflected. The valleys of the fingerprint, which do not touch the glass, essentially retain the air-glass interface, causing the light to be totally internally reflected into the sensor.

* *Result:* The resulting image is binary-like in its contrast. Ridges appear distinctively dark (or bright, depending on the sensor polarity), and valleys appear the opposite. The transition is sharp, and the image represents the **physical touch** of the skin, not its visual texture. The background is uniformly white or black, free of noise.2

**Diffuse Ambient Reflection (Contactless):**

In contrast, a smartphone camera captures light that is diffusely reflected from the skin's surface.

* *Mechanism:* The visibility of the friction ridges is determined by the interaction of ambient light (or the camera flash) with the 3D topography of the finger. Ridges are visible primarily due to shadows cast into the valleys (shading) or subtle differences in skin texture/color.  
* *Challenges:*  
  * **Low Contrast:** The luminance difference between a ridge and a valley in a photo is minimal compared to the binary contrast of FTIR.  
  * **Subsurface Scattering:** Human skin is translucent. Light penetrates the stratum corneum and scatters within the tissue. This "glow" can blur the high-frequency ridge details, especially under strong flash illumination, leading to a "washed out" appearance where ridges are indistinguishable.10  
  * **Noise:** The image includes complex background clutter, sensor noise (especially in low light), and variations in skin pigmentation which are irrelevant to the ridge pattern but can confuse feature extractors.5

**Implication for Modeling:** The solution cannot simply process the RGB image. It must essentially "hallucinate" or mathematically reconstruct the contact contrast. This suggests the need for **domain adaptation** models (like CycleGAN or specialized enhancers) that map the *Texture Domain* to the *Contact Domain*.12

### **2.2 Geometric Distortion and Projection Models**

The geometric mismatch is arguably the most significant cause of matching failure in C2C scenarios. A contact print is a 2D unrolling of a 3D surface; a contactless photo is a 2D projective transformation of that same surface.

**Perspective Distortion:**

A smartphone camera acts as a pinhole projector. Points on the finger that are closer to the lens (the bulb of the finger) appear spatially larger than points at the periphery.

* *Mathematical Model:* If we define the finger surface ![][image1] and the image plane ![][image2], the projection ![][image3] induces a non-uniform scaling factor ![][image4] across the image.  
* *Effect:* The ridge frequency (ridges per millimeter) appears to vary across the image. Standard AFIS algorithms, which assume a constant ridge frequency (approx. 0.5mm period), will fail to extract minutiae correctly in the periphery of a perspective-distorted image.13

**Cylindrical Deformation vs. Elastic Flattening:**

* **Contact (Elastic):** When a finger presses against a scanner, the skin undergoes non-linear elastic deformation. The 3D cylinder is physically flattened, introducing shear and tensile stress that locally stretches the ridge pattern.  
* **Contactless (Rigid):** The mobile photo captures the finger in its natural, semi-cylindrical state. The periphery of the finger curves away from the camera, resulting in foreshortening.  
* *The "Unrolling" Problem:* To make the contactless image match the contact gallery, the system must mathematically "unroll" this cylinder onto a flat plane. However, simply unrolling it geometrically (assuming a perfect cylinder) is insufficient because it does not account for the **elastic deformation** present in the contact gallery. A purely geometric unroll might produce a "correct" flat map, but the gallery image is "distorted" by pressure. Therefore, the effective solution must mimic the pressure distortion—a process termed **"virtual flattening"**.14

### **2.3 Scale Ambiguity and the 500 PPI Imperative**

ISO/IEC 19794-4 requires fingerprints to be stored at 500 PPI.7 In a contact scanner, the sensor size and optics are fixed; 500 PPI is a hardware constant. In mobile capture, the user moves the phone. A finger captured at 10cm is half the size (in pixels) of a finger captured at 5cm.

* *The Challenge:* The SDK must dynamically estimate the PPI of the captured image and rescale it. Without depth sensors (like LiDAR, which are rare on mid-range Androids), this must be done using **Shape-from-Texture** cues, specifically the ridge frequency itself.8

## ---

**3\. Computational Architectures for Segmentation and ROI Extraction**

The first stage of the operational pipeline is the isolation of the fingerprint from the chaotic background of a video stream. This is critical for mobile implementations where processing the full high-resolution frame would violate the 2-second latency constraint.

### **3.1 The Failure of Traditional Computer Vision**

Historical approaches utilized classical computer vision techniques:

* **Otsu’s Thresholding:** Converting to grayscale and finding a global threshold. This fails dynamically under variable lighting or when the background skin tone (palm) matches the finger.16  
* **Color Space Clustering (HSV/YCbCr):** Segmentation based on skin color. This is non-robust to diverse demographics (skin tones) and background clutter that mimics skin color (e.g., wood tables).1

Consequently, the industry has standardized on **Deep Semantic Segmentation**, specifically lightweight Convolutional Neural Networks (CNNs).

### **3.2 MobileUNetV3: The Optimal Balance**

For the specific constraints of Android deployment (low latency, high accuracy), **MobileUNetV3** emerges from the research as the most effective architecture.

**Architecture Analysis:**

* **Encoder (Backbone):** Instead of the heavy ResNet or VGG encoders used in server-side segmentation, this model uses **MobileNetV3-Small**.  
  * *Mechanism:* It employs **Depthwise Separable Convolutions**, which factorize standard convolutions into a depthwise layer (filtering inputs) and a pointwise layer (combining outputs). This reduces the computational cost (FLOPs) by a factor of 8-9x compared to standard convolutions.17  
  * *Squeeze-and-Excitation (SE):* MobileNetV3 introduces lightweight attention modules that reweight channel importance, improving the model's ability to distinguish "finger" from "background" without adding significant latency.  
* **Decoder:** A lightweight U-Net style decoder.  
  * *Skip Connections:* It retains the U-Net's signature skip connections, which concatenate high-resolution feature maps from the encoder with the upsampled features in the decoder. This is crucial for fingerprints because the **boundary precision** (the exact edge of the fingertip) must be preserved to avoid cutting off minutiae.18

**Performance Metrics:**

* **Accuracy:** MobileUNetV3 achieves Dice coefficients \> 0.96 and Intersection over Union (IoU) \> 0.90 on fingerprint segmentation tasks.19  
* **Latency:** On a standard Android device (Snapdragon 845 equivalent), inference for a 224x224 input is approximately **30-50ms**. This leaves ample room in the 2-second budget for subsequent processing.18

### **3.3 TipSegNet: Multi-Scale Robustness**

An alternative architecture identified in recent literature is **TipSegNet**.

* **Differentiation:** It utilizes a **Feature Pyramid Network (FPN)**.  
* **Mechanism:** FPNs extract features at multiple scales. This is particularly useful for the "Auto-Capture" phase where the user might be moving the phone closer or further away. TipSegNet can robustly detect fingertips at varying distances.20  
* **Trade-off:** While offering slightly higher accuracy (mIoU 0.987) 20, the backbone (often ResNeXt or heavier variants) is computationally more expensive than MobileNetV3. For the UIDAI requirement of "lightweight" models compatible with lower-end Androids, MobileUNetV3 is the pragmatic recommendation, while TipSegNet represents a high-end alternative.

### **3.4 Operational Strategy: The "Preview" Loop**

To minimize latency, the segmentation model should not run on the full 12MP image.

1. **Preview Stream:** Run the quantized (INT8) MobileUNetV3 on the viewfinder stream (e.g., 224x224 or 320x320 resolution).  
2. **Coordinate Mapping:** Once a high-confidence mask is detected, map the bounding box coordinates to the full-resolution sensor space.  
3. **Crop:** Perform the high-res capture and crop immediately. This ensures the subsequent costly steps (Unwarping) only process the pixels of interest.4

## ---

**4\. Geometric Rectification and Unwarping architectures**

Once the finger is segmented, the system faces the central challenge: transforming the curved, perspective-distorted 2D photo into a flat, 500 PPI representation that aligns with the contact gallery. This is the **C2C (Contactless to Contact)** transformation.

### **4.1 Theoretical Models for Unwarping**

**Thin Plate Spline (TPS) Models:**

TPS is a non-rigid interpolation method. It minimizes a "bending energy" function to map a set of source control points to target control points.

* *Limitations:* TPS requires reliable keypoints (minutiae) to be detected *before* unwarping to define the transformation. However, minutiae detection on the raw distorted photo is unreliable, creating a "chicken-and-egg" problem.13

**Parametric Cylinder Unrolling:**

This assumes the finger is a perfect cylinder.

* *Equation:* ![][image5], where ![][image6] is the horizontal distance from the center and ![][image7] is the estimated radius.  
* *Limitations:* Fingers are not perfect cylinders; they are tapered and irregular. This method fails to account for the "pitch" (finger pointing up/down) which causes trapezoidal distortion.8

### **4.2 The C2CL Framework: State-of-the-Art Architecture**

The most effective solution identified in the research is the **C2CL (Contact to Contactless)** framework proposed by Jain et al..4 It bypasses the need for manual geometric modeling by learning the transformation directly from data.

**Core Component: Spatial Transformer Network (STN)**

The STN is a differentiable module that can be inserted into a neural network to perform spatial manipulations.

* **Localization Network:** This sub-network takes the input feature map ![][image8] and regresses a set of transformation parameters ![][image9] (e.g., the 6 parameters of an affine transformation, or a grid of thin-plate spline control points).  
* **Grid Generator:** It generates a sampling grid ![][image10] based on the predicted parameters.  
* **Sampler:** It samples the input image at the grid points to produce the unwarped output ![][image11].  
  * *Equation:* ![][image12] (using bilinear interpolation).

**Why STN is Superior for Mobile:**

1. **Efficiency:** The sampling operation is a highly optimized matrix operation that can be parallelized on the GPU. It is significantly faster than iterative 3D reconstruction methods.  
2. **End-to-End Training:** The critical innovation in C2CL is how the STN is trained. It is not trained to just "flatten" the image visually. It is trained to **minimize the biometric feature distance** between the unwarped photo and the ground-truth contact scan.  
   * *Loss Function:* $L \= |

| D(I\_{unwarped}) \- D(I\_{contact}) ||^2$, where ![][image13] is a fixed-length fingerprint embedding (like DeepPrint). \* *Result:* The network learns to warp the photo specifically to maximize matching performance against the contact gallery, implicitly correcting for both perspective and elastic contact distortion.4

### **4.3 3D Reconstruction: Shape-from-Texture**

An alternative, albeit heavier, approach is **Monocular 3D Reconstruction**.

* **Method:** A Deep CNN predicts a surface normal map or a depth map from the single RGB image. The system then mathematically unrolls this 3D mesh.  
* **Performance:** This offers high accuracy for extreme poses (rotations \> 45 degrees).  
* **Mobile Viability:** Models estimating dense depth maps are computationally intensive (typically \>500ms). For the "lightweight" and "low latency" constraints of the SITAA challenge, the STN-based 2D warping (C2CL) is the superior architectural choice.14

### **4.4 Implementation on Android**

To implement the C2CL Unwarper on Android:

* **Model Format:** The STN must be exported to TensorFlow Lite.  
* **Custom Operators:** Standard TFLite sometimes lacks native support for the GridSample operator used in STNs.  
  * *Solution:* Implement a custom TFLite operator in C++ using the Android NDK, or use the tf.image. projective\_transform approximation if available in the GPU delegate.  
* **Optimization:** The Localization Network (which predicts the warp parameters) can be a very shallow CNN (e.g., 4 layers), as the global shape of the finger is a low-frequency feature. This keeps inference \< 100ms.4

## ---

**5\. Radiometric Enhancement and Domain Adaptation**

After geometric unwarping, the image is geometrically correct but radiometrically wrong. It still looks like a photo (gray skin, low contrast) rather than a scan (black/white ridges). The AFIS expects the latter.

### **5.1 Ridge Frequency Estimation and Scale Normalization**

Before contrast enhancement, the system **must** resolve the scale ambiguity to meet the ISO 19794-4 500 PPI requirement.

**Algorithm: Spectral Analysis (FFT)**

1. **Block Processing:** Divide the unwarped image into non-overlapping blocks (e.g., 32x32).  
2. **Fourier Transform:** Apply 2D FFT to each block.  
3. **Peak Detection:** Identify the peak energy frequency ![][image14]. This corresponds to the sine wave of the ridges.  
4. **Global Consensus:** Compute the median frequency across all blocks to reject noise.  
5. **Scaling:**  
   * Standard Ridge Frequency at 500 PPI ![][image15] ![][image16] to ![][image17] cycles per pixel (ridges are \~9-10 pixels apart).  
   * Calculate Resizing Factor ![][image18].  
   * Resize image by ![][image19].8

**Importance:** This step is non-negotiable. If a finger is captured too close, the ridges might be 20 pixels apart. The AFIS feature extractor (designed for 10-pixel separation) will fail to detect them or misclassify them as noise.

### **5.2 Contextual Gabor Filtering**

The gold standard for fingerprint enhancement is **Contextual Filtering**.

* **Concept:** Ridges flow locally in a specific orientation and frequency. A filter aligned with this flow will amplify the ridge and smooth out orthogonal noise.  
* **Gabor Filter Bank:** A set of filters ![][image20] are created for discrete orientations ![][image9] (e.g., 0°, 22.5°, 45°...) and the estimated frequency ![][image21].  
* **Application:** For each pixel, the system estimates the local orientation (using image gradients) and applies the corresponding Gabor filter.  
* **Pros:** Mathematically robust, restores broken ridges.  
* **Cons:** Computationally expensive (convolution with large kernels).  
* **Mobile Optimization:** Implementation should use separable filters or FFT-based convolution in the Android NDK (C++) layer to maintain speed.24

### **5.3 Generative Style Transfer (CycleGAN)**

A more modern approach uses **Generative Adversarial Networks (GANs)** for "Domain Adaptation."

* **Architecture:** **CycleGAN**. It consists of two mappings: ![][image22] (Photo to Scan) and ![][image23] (Scan to Photo). It ensures cycle consistency: ![][image24].  
* **Advantage:** It learns complex, non-linear mappings that Gabor filters miss, such as correcting lighting glare or specific sensor noise patterns.  
* **Risk:** GANs are prone to **hallucination**. In fingerprints, generating a "likely" ridge where none exists is a critical security failure (increasing False Accept Rate).  
* **Recommendation:** For the SITAA challenge, which prioritizes "Accuracy" and "AFIS Interoperability," standard Gabor filtering or **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is preferred over GANs due to predictability and lower computational overhead. If a GAN is used, it must be heavily constrained (e.g., using a "Pixel-to-Pixel" loss with strong regularization) and used only for texture enhancement, not structure generation.26

### **5.4 ISO 19794-4 Encoding (WSQ)**

The final step in the pipeline is encoding.

* **Format:** The raw pixel buffer (now 500 PPI, enhanced) must be compressed using **WSQ**.  
* **Library:** The **NIST Biometric Image Software (NBIS)** is the open-source reference implementation. The cwsq utility from NBIS can be compiled via the Android NDK.  
* **Quality Check:** Before encoding, the **NFIQ 2 (NIST Fingerprint Image Quality)** algorithm should be run. NFIQ 2 returns a score (0-100). The SDK should enforce a threshold (e.g., Score \> 40\) as per the "Readiness Scoring" requirement.28

## ---

**6\. Feature Extraction and Matching Paradigms**

While the primary output is the ISO image, the SDK must often perform local matching (1:1 verification) or quality pre-checks. Two distinct paradigms exist here.

### **6.1 Minutiae-Based Matching (The ISO Standard)**

This is the interoperability layer.

* **Model:** **VeriFinger** (or similar ISO 19794-2 extractors).  
* **Method:** Extracts ridge endings and bifurcations (x, y, theta).  
* **Pros:** Fully compatible with Aadhaar AFIS.  
* **Cons:** Extremely sensitive to image quality. If the unwarping (Stage 4\) is imperfect, the relative positions of minutiae shift, causing match failure.30

### **6.2 Deep Representation Matching (DeepPrint)**

This is the performance layer.

* **Model:** **DeepPrint**.4  
* **Architecture:** A deep CNN (e.g., Inception-v4 backbone) trained with a softmax classifier on millions of identities. It outputs a fixed-length vector (e.g., 192 floats).  
* **Method:** Matching is a simple Cosine Distance calculation between two vectors.  
* **Pros:**  
  * **Fast:** Matching takes microseconds.  
  * **Robust:** The deep network learns to ignore distortion and sensor noise, often succeeding where minutiae matchers fail on contactless images.  
  * **Guidance:** As mentioned in Section 4.2, DeepPrint is crucial for *training* the STN unwarper.  
* **Cons:** Not interoperable with legacy AFIS (you can't send a DeepPrint vector to Aadhaar).  
* **Integration Strategy:** Use DeepPrint for the **on-device 1:1 demo app** and for internal quality checks. Use the standard ISO image for the formal AFIS submission.

### **6.3 Fusion Strategies**

To maximize accuracy (TAR), the system can employ a **Two-Stage Search** or **Score Fusion**:

1. **Stage 1 (Fast):** Use DeepPrint to rapidly verify if the finger matches the local cache (1:1).  
2. **Stage 2 (Precise):** If DeepPrint passes, generate the ISO template and perform minutiae matching (or send to server). Research shows that fusing DeepPrint and Minutiae scores significantly boosts TAR, especially for "ugly" quality latent or contactless prints.21

## ---

**7\. Edge Implementation Strategy on Android**

The "SITAA-Startup-Cohort1" challenge requires this complex pipeline to run on Android 9+ with \< 2s latency. This requires aggressive optimization.

### **7.1 Android Neural Networks API (NNAPI) & TFLite**

Deep learning models (MobileUNet, STN) must be converted to **TensorFlow Lite (TFLite)**.

* **Quantization:** Convert weights from Float32 to **Int8**. This typically reduces model size by 4x and speeds up inference by 2-3x on mobile DSPs/NPUs with negligible accuracy loss (\<1%).34  
* **Delegates:**  
  * Use the **GPU Delegate** for the STN unwarping (grid sampling is parallelizable).  
  * Use the **NNAPI Delegate** for MobileUNet (segmentation), which allows the OS to schedule the workload on the DSP or NPU.35

### **7.2 Native C++ (NDK) for Signal Processing**

Do not implement FFT (Scaling) or Gabor Filtering in Java/Kotlin. The overhead of the Java Virtual Machine (JVM) and Garbage Collection (GC) is too high.

* **Strategy:** Use JNI (Java Native Interface) to call optimized C++ code.  
* **Libraries:** Use **OpenCV Android SDK** (C++) or **Eigen** for matrix operations (FFT, Filtering). The **NBIS** library (for WSQ/NFIQ) is already written in C and integrates directly.4

### **7.3 Pipeline Orchestration & Memory Management**

* **Zero-Copy:** Use ByteBuffer for passing image data between Java (Camera API), TFLite (Models), and C++ (OpenCV). Avoid creating new Bitmap objects at every stage to prevent GC pauses.  
* **Threading:**  
  * Thread 1 (UI): Camera Preview.  
  * Thread 2 (Worker): Deep Learning (TFLite interpreter).  
  * Thread 3 (Worker): Signal Processing (C++).  
  * This pipelining allows the Capture UI to remain responsive while the "heavy lifting" happens in the background.

## ---

**8\. Comparative Evaluation and Strategic Recommendations**

### **8.1 Comparative Analysis of Solution Architectures**

The following table synthesizes the available architectures against the specific SITAA/UIDAI constraints.

| Solution Component | Models Evaluated | Pros | Cons | Latency Impact | Recommendation |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Segmentation** | **MobileUNetV3** | Ultra-lightweight, high boundary precision (skip connections). | Less robust to extreme clutter than Mask R-CNN. | **Low** (\~30ms) | **Adopt** (Best balance). |
|  | TipSegNet | Multi-scale handling (FPN). Slightly higher mIoU. | Heavier backbone (ResNeXt). Slower inference. | Medium (\~100ms) | Backup (High-end only). |
|  | Otsu/HSV (Classical) | Zero latency. | Fails dynamically. Unreliable. | Negligible | Reject. |
| **Unwarping** | **C2CL (STN)** | End-to-end learnable. Optimizes for *matching* not just visuals. | Requires paired training data. "Black box" warping. | **Medium** (\~150ms) | **Adopt** (SOTA approach). |
|  | 3D Shape-from-X | Physically accurate 3D unrolling. | Computationally prohibitive for edge. Slow. | High (\>500ms) | Reject. |
|  | Parametric (Cylinder) | Fast, simple math. | Fails on tapered fingers or pitch rotation. | Low | Reject (Poor accuracy). |
| **Enhancement** | **Contextual Gabor** | Mathematically grounded. Restores ridge flow. | Computationally heavy (convolutions). | High (\~600ms) | **Adopt** (with C++ opt). |
|  | CycleGAN | Excellent visual style transfer. | Hallucination risk. Heavy model size. | Very High | Reject (Security risk). |
| **Feature Extraction** | **DeepPrint** | Fast matching. Robust to distortion. | Non-standard output. Not AFIS interoperable. | Low | **Hybrid Use** (Internal check). |
|  | **Minutiae (ISO)** | AFIS Interoperability. Standards compliant. | Sensitive to quality. Slow extraction. | Medium | **Adopt** (Required). |

### **8.2 Performance vs. Latency Trade-off**

| Step | Technique | Android Latency (Est.) |
| :---- | :---- | :---- |
| Capture | Camera2 API \+ YUV | 100 ms |
| Segmentation | MobileUNetV3 (Int8) | 50 ms |
| Unwarping | STN (GPU Delegate) | 150 ms |
| Scaling | FFT (Native C++) | 100 ms |
| Enhancement | Gabor / CLAHE (Native C++) | 600 ms |
| Encoding | WSQ (NBIS C++) | 100 ms |
| **Total** |  | **\~1100 ms (1.1s)** |

This projected latency of **1.1 seconds** sits comfortably within the **≤ 2 seconds** requirement, leaving a 900ms buffer for UI rendering, OS overhead, and the "Readiness Scoring" (NFIQ 2\) computation.4

### **8.3 Final Recommendations for Implementation**

1. **Architecture:** Adopt the **C2CL** framework logic. Train the Spatial Transformer Network (STN) offline using a large dataset of paired contact/contactless prints (e.g., PolyU, IIT Bombay databases) using the DeepPrint loss function. Deploy only the inference engine (TFLite) to the mobile.  
2. **Standards:** Prioritize **Ridge Frequency Estimation** (FFT) for scale normalization. Without this, the ISO 19794-4 output will be technically valid (file format) but biometrically useless (mismatched ridge density) to the Aadhaar AFIS.  
3. **Optimization:** Avoid Python/Java for pixel-level operations. Build a robust **JNI bridge** to a C++ library (OpenCV/NBIS) for the enhancement and encoding stages.  
4. **Quality Gate:** Implement a strict "Readiness Score" using **NFIQ 2 Mobile** or a lightweight blur detector (Laplacian variance) to reject poor frames *before* they enter the heavy processing pipeline. This improves the perceived user experience by providing instant feedback ("Hold Steady," "Move Closer").1

By strictly adhering to this architecture, the proposed solution will not only meet the functional requirements of the SITAA challenge but effectively bridge the gap between the mobile future of identity and the massive legacy infrastructure of national ID systems.

#### **Works cited**

1. fingerprint\_uid.pdf  
2. Techniques for contactless fingerprint recognition: A review | AIP, accessed January 26, 2026, [https://pubs.aip.org/aip/acp/article/3318/1/030030/3356802/Techniques-for-contactless-fingerprint-recognition](https://pubs.aip.org/aip/acp/article/3318/1/030030/3356802/Techniques-for-contactless-fingerprint-recognition)  
3. Touchless fingerprint recognition: A survey of recent developments and challenges, accessed January 26, 2026, [https://www.researchgate.net/publication/387832570\_Touchless\_fingerprint\_recognition\_A\_survey\_of\_recent\_developments\_and\_challenges](https://www.researchgate.net/publication/387832570_Touchless_fingerprint_recognition_A_survey_of_recent_developments_and_challenges)  
4. C2CL: Contact to Contactless Fingerprint Matching \- Biometrics Research Group, accessed January 26, 2026, [http://biometrics.cse.msu.edu/Publications/Fingerprint/C2CL\_\_Contact\_to\_Contactless\_Fingerprint\_Matcher\_\_TIFS\_final.pdf](http://biometrics.cse.msu.edu/Publications/Fingerprint/C2CL__Contact_to_Contactless_Fingerprint_Matcher__TIFS_final.pdf)  
5. Fingerprint Systems: Sensors, Image Acquisition, Interoperability and Challenges \- PMC, accessed January 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10384471/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10384471/)  
6. INVITATION FOR EXPRESSION OF INTEREST FOR Participation in Biometric Authentication Proof of Concept Studies \- uidai, accessed January 26, 2026, [https://uidai.gov.in/images/tenders/eoi\_for\_auth\_poc\_devices\_final.pdf](https://uidai.gov.in/images/tenders/eoi_for_auth_poc_devices_final.pdf)  
7. ISO-IEC-19794-4-Fingerprint-Image-Record-Standard \- BioEnable Aadhaar Solutions, accessed January 26, 2026, [https://aadhaar.bioenabletech.com/iso-iec-19794-4-fingerprint-image-record-standard](https://aadhaar.bioenabletech.com/iso-iec-19794-4-fingerprint-image-record-standard)  
8. Guidance for Evaluating Contactless Fingerprint Acquisition Devices \- NIST Technical Series Publications, accessed January 26, 2026, [https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.500-305.pdf](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.500-305.pdf)  
9. On the interoperability of capture devices in fingerprint presentation attacks detection \- CEUR-WS.org, accessed January 26, 2026, [https://ceur-ws.org/Vol-1816/paper-07.pdf](https://ceur-ws.org/Vol-1816/paper-07.pdf)  
10. Contactless Fingerprint Recognition Using Deep Learning—A Systematic Review \- MDPI, accessed January 26, 2026, [https://www.mdpi.com/2624-800X/2/3/36](https://www.mdpi.com/2624-800X/2/3/36)  
11. A Robust Contactless Fingerprint Enhancement Algorithm \- SciSpace, accessed January 26, 2026, [https://scispace.com/pdf/a-robust-contactless-fingerprint-enhancement-algorithm-4crr6y5u5i.pdf](https://scispace.com/pdf/a-robust-contactless-fingerprint-enhancement-algorithm-4crr6y5u5i.pdf)  
12. Finger-UNet: A U-Net based Multi-Task Architecture for Deep Fingerprint Enhancement \- arXiv, accessed January 26, 2026, [https://arxiv.org/html/2310.00629](https://arxiv.org/html/2310.00629)  
13. Deep Contactless Fingerprint Unwarping | Semantic Scholar, accessed January 26, 2026, [https://www.semanticscholar.org/paper/Deep-Contactless-Fingerprint-Unwarping-Dabouei-Soleymani/8c582981abc3e47dfd459a064b0c2ceea9d98839](https://www.semanticscholar.org/paper/Deep-Contactless-Fingerprint-Unwarping-Dabouei-Soleymani/8c582981abc3e47dfd459a064b0c2ceea9d98839)  
14. A new approach to unwrap a 3-D fingerprint to a 2-D rolled equivalent fingerprint, accessed January 26, 2026, [https://www.researchgate.net/publication/224083408\_A\_new\_approach\_to\_unwrap\_a\_3-D\_fingerprint\_to\_a\_2-D\_rolled\_equivalent\_fingerprint](https://www.researchgate.net/publication/224083408_A_new_approach_to_unwrap_a_3-D_fingerprint_to_a_2-D_rolled_equivalent_fingerprint)  
15. 3D to 2D fingerprints: Unrolling and distortion correction \- Semantic Scholar, accessed January 26, 2026, [https://www.semanticscholar.org/paper/3D-to-2D-fingerprints%3A-Unrolling-and-distortion-Zhao-Jain/e12333009b0763c0aa806fefcbbd85886454484e](https://www.semanticscholar.org/paper/3D-to-2D-fingerprints%3A-Unrolling-and-distortion-Zhao-Jain/e12333009b0763c0aa806fefcbbd85886454484e)  
16. TipSegNet: Fingertip Segmentation in Contactless Fingerprint Imaging \- PMC \- NIH, accessed January 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11946156/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946156/)  
17. Understanding and Implementing MobileNetV3 | by Rishabh Singh \- Medium, accessed January 26, 2026, [https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a](https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a)  
18. (PDF) MobileUNetV3—A Combined UNet and MobileNetV3 Architecture for Spinal Cord Gray Matter Segmentation \- ResearchGate, accessed January 26, 2026, [https://www.researchgate.net/publication/362390667\_MobileUNetV3-A\_Combined\_UNet\_and\_MobileNetV3\_Architecture\_for\_Spinal\_Cord\_Gray\_Matter\_Segmentation](https://www.researchgate.net/publication/362390667_MobileUNetV3-A_Combined_UNet_and_MobileNetV3_Architecture_for_Spinal_Cord_Gray_Matter_Segmentation)  
19. MobileUNetV3—A Combined UNet and MobileNetV3 Architecture for Spinal Cord Gray Matter Segmentation \- MDPI, accessed January 26, 2026, [https://www.mdpi.com/2079-9292/11/15/2388](https://www.mdpi.com/2079-9292/11/15/2388)  
20. TipSegNet: Fingertip Segmentation in Contactless Fingerprint Imaging \- MDPI, accessed January 26, 2026, [https://www.mdpi.com/1424-8220/25/6/1824](https://www.mdpi.com/1424-8220/25/6/1824)  
21. C2CL: Contact to Contactless Fingerprint Matching | Request PDF \- ResearchGate, accessed January 26, 2026, [https://www.researchgate.net/publication/356953456\_C2CL\_Contact\_to\_Contactless\_Fingerprint\_Matching](https://www.researchgate.net/publication/356953456_C2CL_Contact_to_Contactless_Fingerprint_Matching)  
22. Monocular 3D Fingerprint Reconstruction and Unwarping, accessed January 26, 2026, [https://ivg.au.tsinghua.edu.cn/\~jfeng/pubs/Cui\_PAMI23\_3DFinger.pdf](https://ivg.au.tsinghua.edu.cn/~jfeng/pubs/Cui_PAMI23_3DFinger.pdf)  
23. Guidance on Contactless Friction Ridge Image Compression, v1.0 \- GovInfo, accessed January 26, 2026, [https://www.govinfo.gov/content/pkg/GOVPUB-C13-98d2e58805dd8b30e1f8bed150272ab2/pdf/GOVPUB-C13-98d2e58805dd8b30e1f8bed150272ab2.pdf](https://www.govinfo.gov/content/pkg/GOVPUB-C13-98d2e58805dd8b30e1f8bed150272ab2/pdf/GOVPUB-C13-98d2e58805dd8b30e1f8bed150272ab2.pdf)  
24. Latent fingerprint enhancement for accurate minutiae detection \- arXiv, accessed January 26, 2026, [https://arxiv.org/html/2409.11802](https://arxiv.org/html/2409.11802)  
25. Fingerprint Image Enhancement: Algorithm and Performance Evaluation \- Biometrics Research Group \- Michigan State University, accessed January 26, 2026, [http://biometrics.cse.msu.edu/Publications/Fingerprint/MSU-CPS-97-35fenhance.pdf](http://biometrics.cse.msu.edu/Publications/Fingerprint/MSU-CPS-97-35fenhance.pdf)  
26. Synthetic Latent Fingerprint Generator \- YouTube, accessed January 26, 2026, [https://www.youtube.com/watch?v=9w5kD\_BfgQU](https://www.youtube.com/watch?v=9w5kD_BfgQU)  
27. Mobile Contactless Fingerprint Recognition: Implementation, Performance and Usability Aspects \- PMC \- PubMed Central, accessed January 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8839666/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8839666/)  
28. Compression Guidance for 1000 ppi Friction Ridge Imagery \- NIST Technical Series Publications, accessed January 26, 2026, [https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.SP.500-289.pdf](https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.SP.500-289.pdf)  
29. usnistgov/NFIQ2: Optical live-scan and ink fingerprint ... \- GitHub, accessed January 26, 2026, [https://github.com/usnistgov/NFIQ2](https://github.com/usnistgov/NFIQ2)  
30. Towards More Accurate Contactless Fingerprint Minutiae Extraction and Pose-Invariant Matching \- PolyU, accessed January 26, 2026, [https://www4.comp.polyu.edu.hk/\~csajaykr/myhome/papers/TIFS2020a.pdf](https://www4.comp.polyu.edu.hk/~csajaykr/myhome/papers/TIFS2020a.pdf)  
31. VeriFinger fingerprint recognition technology, algorithm and SDK for PC, smartphones and Web \- Neurotechnology, accessed January 26, 2026, [https://www.neurotechnology.com/verifinger.html](https://www.neurotechnology.com/verifinger.html)  
32. arxiv.org, accessed January 26, 2026, [https://arxiv.org/abs/2104.02811](https://arxiv.org/abs/2104.02811)  
33. \[PDF\] Fusion2Print: Deep Flash-Non-Flash Fusion for Contactless, accessed January 26, 2026, [https://www.semanticscholar.org/paper/Fusion2Print%3A-Deep-Flash-Non-Flash-Fusion-for-Sahoo-Namboodiri/40490fdbc2cebfdabd50085a0ac68aabd4b8cf5e](https://www.semanticscholar.org/paper/Fusion2Print%3A-Deep-Flash-Non-Flash-Fusion-for-Sahoo-Namboodiri/40490fdbc2cebfdabd50085a0ac68aabd4b8cf5e)  
34. Understanding Latency in Mobile & Android Systems — An End-to-End Perspective (Including Jetpack Compose) | by Subrajit Pandey | Dec, 2025 | Medium, accessed January 26, 2026, [https://medium.com/@subhrajeetpandey2001/understanding-latency-in-mobile-android-systems-an-end-to-end-perspective-including-jetpack-9712af44d8aa](https://medium.com/@subhrajeetpandey2001/understanding-latency-in-mobile-android-systems-an-end-to-end-perspective-including-jetpack-9712af44d8aa)  
35. Profiling MLC-LLM's OpenCL Backend on Android: Performance Insights \- Callstack, accessed January 26, 2026, [https://www.callstack.com/blog/profiling-mlc-llms-opencl-backend-on-android-performance-insights](https://www.callstack.com/blog/profiling-mlc-llms-opencl-backend-on-android-performance-insights)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAYCAYAAACr3+4VAAACRElEQVR4Xu2XO2gVQRSGD4qo+KoMgmAlFhY2ijYqCGIh+MAHNjYRQrDxAYogKonaBQs7RdCA2iXaCT5AEJIYiaKldoKVjaKo4Pv/mZncf493N7tLghr2g48998zcmzmzs7MTs4b/kg74GX6Ax13btOInnBnjX3CPtOWyAl6G2yR3TOJ/kbdwVoxZ6GFp+wN25MxcgQvhRgtfOgU/Sr+p5IS1Bnooxj3wbozp3tS5DcstLN9C+CPrfdJC/qRPTiEvJD4o8Wq4KsrCPT3wEdzq8hn6LRTUDubTsqjCFthtYbB0f7Y5l+cSd0lMbsTrtUw2C8c74JOJtCzakZfP4x0chhssLKWlcDFcoJ0KGJO4U2KSJmGf5GbDl/J51ArGnAq94BsqwLv+xSdr8FhiX+jVeNW7zgnVwr7Ce/I5Q5+1ik1eyvSYmG8+UZMRibXQixZWxX24S/LkDnxgYZL0GW9L2uXUV5ke+XD59PpkTbTQNI4f8XpU2iaFzdb6I2XgzK+DawpcOd67GH9H6Vq4xMLOWpvdPhG5buULPQN3wO0FbhrvXcwTidPS/RSvty2snspwAEd8MsIzY9lCubPyZT8ZPJM4Fcrj3VCMeaipzFN4yycjfC6qbEhlJ2UidDM5IPF5uAzOhw8lX4r0HPLLyqBVP/bNge99sgavJfYnMk4+2WkVN783cIaFAbJgvux57Zc+VeAS43uME8Wj2iILv1+GufC7ZXd9LlN9bc2z1g5Mb0rbX4PHv9PwLDxn0/z/xIaGhoYG8hvpSYTpUGomQQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAYCAYAAABeIWWlAAAB/klEQVR4Xu2XPUgcURDHR00hfhBtYmGlhaSy0k5stDMIFjY2J4KChQZSmIiViigiola2CqIilqLYiGAn+FWYKuEQwVjEL4TEb//De8vNDnK3+w7lhP3Bj52d2Tt23r739o4oIqPJhRfwCQ6p2rtnR8R3cFGcv3v4iRXYuNeev8g+/EfmAs9z3xWvxzcyI/8VdlNimi3Y+J7MzSdjBd7opMZr7K2ZFLGcXryuumAxPBJ5Dd9zvk5KPpC5aFcXQlANO2CnMAgTIp4TMXNsjzFYKAuWS/hRJzU9ZJpr0IUA/IS/YB2sgKXwE5kRD8KYiGdEzHgzqQRWygI4IPNQmCVZ0FyR25Tkz+TpZEhGRKyb+22PU74s0Txsg61kZsh/X1Xhst7ilBi5dBgWsWyuGdbAafIPACM3P3bZX07grbdtXUjBpk44IpuTN/wAB2GWqIfmB5kv+6ILSSiD7bAqhUHQT+4zmVdENtwQNSdc1lst/A4bUxiEURF705I3C4bff7xJOeOy3phVnXBkXMRyzZ3Z47XIhSKHTGN7uhCAW0pzPVjkS3xWxPWwycZ/RD4w/ALl5lp0ISCPZAYoHWRD6yJmTuyxHK7JQjIOyaw1fvR/yfx94N9xLpzCLTIbSBGFa5Z/V+ptXd8Hn3u1uL/0dsRgH+wns4UP+MsREREREZnDM2o4eAlU/8XcAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAoElEQVR4XmNgGAUTgPgjEP+H4u9A/A5NbBVcNQ4AU4gO5Bkg4rvQJZABSMFxdEEowGUwGEQwQCTd0SWAgJOBgOZrDLgl1zNA5ALQJWAAl8mODBDxiegSyACm+QMQvwfiH1D+ZSAWRlKHAWD+TUKXIAbcZMDuZKIALv8SBUAa76ALEgOqGSCa09El8IHJQPyZARKyoHT8FYj/oagYBUMdAABsmDE6TV027AAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAACd0lEQVR4Xu2XzatNURjGHx/5ivIVEsVAKQMGkgyIgaQkYnxTSCklnxncP8Cd3BFiZKKYKJKPDPEHmCgS9yZhQj7z7X3uWutY59nv3mffe0/3SOdXT+33ed+9ztprr7X2OkCXLpvVGGNWqVHFWjVacMB0Us0O8FuNMvYhFN/UhMNi00s1O8RS01s1y3iNeiPCmilqdpBPpm1qeqxG6PxuTWSsM31Vs8MsQ70XMwQLv6mZ8R3/xtpT2O9panrcQ/VoMDdVzYytpvOmmTHmbLhomtCoqMdc02XTck0YS9RA6NdpNT1mIxT3asKYgeqH51qYY5qEUHfLtMm0IcZ1WWC6YxqP4n0vHI+wvvbSYQNeIxvh++SqaUUWs+5Ndv0zy7Ui1fag+HuMr4hHzqFY6/LI9BSheKLk9kTf45TErNsbr4c7PflghHvBr8xPb3Rl5iVOoLxvDZ4jzHt2iMWXmrL+iHqsQb26VrCNg1l8NHoex1CeG4LfwMdZzGK9gacd9Txuo15dFTtQbOO94yXOojyHd6Yv4h1GuGFh5s2KngfX20C8Zg0HLME29NNyHMUlkON1mPE18RLc0D6rST6g2FCC/kPH4y6p0D+Dvx/dZ5LLORQ9/nYZu9B8X3rgsgM2c/1qLoqJsk3gLoqdY3xEPPIAIXc/xjwfMh5sVDTDAfihpsC3wjYo7tLalxzmOMNGDafWRzVHSFWHeWzMqRqs9KLaBhurWj91mGy6oGakD+E35sV4e4zHNSqa4WzZqeZo2GJ6ouYwqfrw86h3PV6vR3g4npA85pteqdkOeO7jf8iRMl0NgYeEG6b9mhDaOjWVHjXGGL7dLv8tfwCcY5nITsai/QAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKsAAAAYCAYAAACFtg3CAAAFkElEQVR4Xu2bd4gkRRSHnxEjKhgQRU5PUBQV9R8xIYg5gIgR5UQUVBBRVMSEomJOiILxTgUTiKAgmP4xY84Bwx1mD8w5+76tqu3an9U9O9NzN7tuf/DYrt+rqa7uqn5T9XrWrKOjo6Nj9CzttqWKHdOG/VVoyZoqDMjGKgyDF9z+UbFjWnCn2/YqtmRYc4F5tZmKbaFz96q4mPnc7XcLfcG+d/suK79fVZ32nGrDmRB7uj2sYkuWcntAxciXbn9ZNSbfuv2QlR+rqo6DvqKKbaDBJVUcARta6MuF6rDqhvwf+NPCteygjj5ZFPfjDrflVcw4ysJ5dxOdpST6T6LvWtAGZh+3P1QcETda/ZNIH/GtoY5pymwV+uR0t1dUHAK9HoD3rL5OXUBBW0fFQXjebXcVR0TdxULyLaGOGQr3YhsVW7Ke2yUqCpMZI4W165Mqlrja7eisfJHbmVm51PiooC+/qehsbcH3qDoG5FC3+93mu10pPsB/roXNC5zidsS4t+I4t4Vur9p//Ye4veb2iNuObs9FnQ3HCW7XuW0atdXdjrXQF/4CX7N3ue0Ry0qvcePr+mK3ZWOZvnLOJt5WoQDnfUJF5yQLvnNEh4OtR3/XsnCz4HirNi/wottl8XiqQH/p33minxz1M0QflLkW2mONBQwg5XzNzrlSlGBDsXk8PjL6ie6Ub45lPkv50lh+1sJkSXAONifAJOTbjPpMYtjAqiXQDW4fRZ1JjHZtLCfYWDUNPmtirm81C/U+dJtlYR7gq4NrbeJAC+3tLPoVUd9P9Jym/k5wrhTLW1j46uCY2d4vt9fYbW63WpgIt7jdZOHm90OaNEQg7M1YToM8LBgwvXGUnxbt8qjzEMFDVi1Bfo6+RJq8+8YyxxtV7jH0OqiTJmuuad9SRiSHqK9a4lO35bIy9e7Ljt/JfDl7ux2uovCWVWPEA0dblEuRVqnr7xhbZceaKskvZqpQGihAe13FAmk3mqLSZFnBwudIw+QQJUv9AfSmlBG7X+q8bGE5UAJ/abLOF41Jrv0gMKiWYImRIA1FPbIs0DTuv6hQoDRGy0TtGtEV6jSdfxxylnqSqQb9Y6milG5QiTRZ2a32gq9a6hJN58RjIlhO3WRN6+ez1CHk+UeMrEsOWmmyEr1yPol6ztyCVkKDVBOfqVCAtkobpXSNTeBfWcUSVOTrui3sFPuxybKuhT5eoA6b3I3oByKotkeZBzqnbrKuYr3vZ55em2UT9wsJyqXJmvYZiY+jnnNaQSvxo02uHmkwlohNHGahrV1ET0GiV/qzth+rWnDOtmq9uknmZz04lZhnoY+aX9026rUXOgC0xRpbNSLhLKs2UWxA685b16cP4t+STzXKpcn6hmilyMpGRrXEr1ZdH3XuyXwHWDndVddWzgIr12Oio7+rDqH02TGut+DkTcRL8Xj96OMJIiUylagbfH5ck/tYhxNV2kBbvNpNzLHqhcNOVqWKiJylPgHpJ3xXZRrpwfSDEnznZ74UPBKkkyizqclBI5LmlL4JoKStbUEn9XZQPJ4XfWQsiLQKkZH0Wi/yccg50YKeHjLK+kBwX0qfHSPtTjHCNhE2lc/O6o0abh5rxa/cvo5lcpY5T1noNxGCOm3hmyZfT+5lYZPAhCX3CvSDNRwTh4lNtFLYmKW9wN8WXismqH9M9GFEx5Qqe9ztC6vaXmBhPctGCo263A/gL2X0hTbxByu0y/pZoe18kpK3pvxMqiDwepV7UgebxTRG31jIhJC7z0nnJC/NCwCFvUHd+TtmAA9aeJDbUhvxhgjnICB0zGDaTjSWDf1shAdhOytneTpmGKyL71axD0jsL2p4K5aWQB0zHN4e8Tp4EHqlm9rC71u7/0DpmACbuUEYys/2GiD12NHR0dGxWPkXhGqrTp/rm1UAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAiklEQVR4XmNgGAUDASYCcSoSvwOIa5D4YCAOxJeg7Fwg/gXE/6H8s0DcA2WDAUwCBHigfH0gtoCyI5DkGYyQ2GUMqJo5kNgY4BMDqmK8AKRwMbogDAgwQBQoMyDcq4UkfxWJzTCTAaKAE4jPQdmKUDmQJ1dA2WDAyABRAMKuDBAbYPw6JHWjgHwAAGFEHDJYgssXAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAAy0lEQVR4Xu2SMQ8BURCER6XSqiVahd8gWr3/4g8olCqVH6KlUGkQnUonR0hEEGaz7708e+/UivuSSS4zs5fbzQElI+pMvZ1u1JF6RF7Dl4vwRcsM6jdtECOFuTVJB5qtbeDpQwtdG5AJNJsaP7BB+pOFonUCqUKbelJ74+eQQbnwklpRd+dV41IKv68cJmbr/J/skC4NoX7dBjGpfYUr1K/YIEYKC2ui+KWBAbTQswHyw+F5TF2oDHrlE/XyoaMFHThA//fad1zy53wAhPQ9J2j9tisAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAzElEQVR4Xu2QvQ5BQRCFp0Ci8AIalU7jGUThKRQqT6BRaJReQqdRUxFuVCqViodQ+Uk4Y3eTcbJLLXzJl5ucc2cyWZHfZQszuIQLuIYr03fgBs7hjLonJXiHVy48dXH9COapk5a4ckC5RfsoepaWBS48VTjlMKCDyc1gDGscBnTwxKEhubghruxxYUgO69NrmePCU4ETDgNnebMZtGGXw8CnxzpyYNHBPYcGvSyJDh849OjSMoeWocTP3sEmhzH64hbc/PcCiy9//PlWHt/LLnH14N1sAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAn0lEQVR4XmNgGHQgDYgL0AWRAQcQ/wdicSC2AeKfqNIIAFJkhcZnR+KDwRYg/o0mBlLoiiwgDBUMRBaEihUhC2yDCiIDFaiYO7IgSOAvEDNB+YxAPA0qDgcsUIF7QHwACYPEUBQmQQXkkQWhYteRBZZCBZHBIixiDNlYBEF8kE0oAGQlssI+IP6GxEcBX4E4BIiroGy8wBmIRdEFRzYAABkLJxqYpSHRAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAYCAYAAABTPxXiAAACZUlEQVR4Xu2WTYhPURjGHx/jc5SRWFAjC2NjYaPMbBSxsCBKkjQrJQt2E5nZSVko00zKx06sJSWhlNhY2FphahKTLEQoZt5n3nNnzv+Zc+79f4xJmV899b/P895z7z33vP9zgXn+Hw6q0SRb1Zgr3pjWqpmgw3TSdN20I/LXR79fmbZFxzO4bxpvQPVwyXRBTaEfPt6Y6YhpU/B+mtaFLIbHK8WbgiEHUU8H4StVL8VCVNf9gNd0amDshGd/xN9r+ibeJGvgbyKmuInX4pMRNRI8Ml1VM+Ijqh+S+aCacH+Dmg9MC8Q7Cy8+IP4S0xXxUvDcxWoGODHMN2ogsIYTrLA3nqt5Rg3jC9IztRq1zZZiM9LnkqVIL5MUuTGOIp/VwKK6ChNcRv7c9/BsnwYNkht/ikXwIr62MrgE29Q0niF/kVYmJ6ZyjD540X4NIoZM3fDm1SbjbFMpyh6Cm+Ju066gPfDlm4JjLFMz5ivyFyJsyBfhd5fpcZSRt6Z34hWUPcRp03lM1wzA94oUzFepGVN2IcKM/1LkhOlDlJGHyDdu1dik3posvDkW5PphOWoHuGe6Gx0TLrHcRW7Bs04NAvysYH5NAyE3/iQ34AW94hfcNI2abgexlrtozJbg5yhmWveAFVGWW0bkMBLjHzJ9h+8Nn4PYF78ws5h1x6JjzQvoc9fPMYzpGy72pDshY1+UwX58qWYj/IZvWOQcfHmk+GS6qOYswQdO/bXXDb+Jis+J3Fsg2juzRQ98hbQM9wDOdBVPTafUbBGuhNw32V+DX6vtajbJE9N2NeeK42o0Cb8Q5vmnmQCnAaC4tyZcSAAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAAo0lEQVR4XmNgGAUgwAjEH4D4PxJ+i6ICAv4yIORBbAwwnwEi6YAmjgxA8jhBAgNEQTWaOAxsBGJjdEFkoMwAMWAbugQQcAHxM3RBbABkwEd0QSD4hS6AC8ACCRkkA3ENmhhOgM0AdD5egG7ANSAWReITBLcYIAYwA/FOINZBlSYM5jFADPAD4ntockSBBAZMb5AEFBkgmtPRJUgBp9EFRsFgBwCn7iceXggXuAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAYCAYAAAB9ejRwAAABiklEQVR4Xu2WPS8FQRSGX5+JIBKFj0IhIho/QFQU1Er/QK9VIBESrRClwg8RtUpBhFLohEQignDOnd3N5t0zszNcNPdJ3mKfc2Zn9u7enQVa/C+jknaWJTolgyx/kwHJLUuDR0k/S4tdyZPkM8uL5IHcZdFtoz2xpPQWC7B4hb92I1llGWBbcsbSh056yjKjD66+QV7xLTaEjmljySzDNc5zoYT1S+5LnsnF8CHZYslcoDohYy1Kj2tPbnAoeWfJWBMyVo8eT5KLYRbVc1XQhhOWJRbhevhfqE7fP6mMoGZR+fM0R77MNVzPEPnQibWmz46P0FhcoaYB/glC43xjckJjG8VQQ/4nsLYQ9R0sIxhGeM5G0fd+2oSrT3MhQ2sTLCOYQWBRa3DFBfJLcNvMG3lGx66zjGAPxqIO4O53fus0+t7Q7UT3vmNJd9Ht5whuk01FL3aHZbPohXHFEXxnTBJ3khWWAfSROWfZbLqQduUpvT9iHO59V8c93AfhnzGF8OdIj2SMZYtUvgALq2kmc66QcgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAYCAYAAACWTY9zAAABqUlEQVR4Xu2VSyhGQRTHj8ijZOFRKCsrKQtS9koSWbKwsLVDNrJnY+mZItlYUErfgihlgbJTnitrLFBKnv+/OZO58z3qfql7y/erXzNzzkxz7r1z7xXJ8U+ogbOw0E9ESRv8hAPwy8tFCouZ1jZ2hTX4wajpk5jdpXrYBY/FFNYDOwMzIqIXjoopigef/eHAjIhhYRt+MA6wsG4/+MfMSchz3CohF2RJpYTcZ15CLsgS/lEW/GAmXiR1YfvwCo7AZbjq5KrFrFuB606c38FbeA6HYIGT4x785ZXAI52XES7Y8WK8i4S5PO0/idmoCb5pjLxqWw4/nLh/sRxXwA44Ae+C6WS4oN0PgloJFsB5LIrtGTyBl7BI889i/rMWt7AqHa85sYw0SvKVWRbhlDO289LNd+ODcM8Zz4h5K8sk/fofmOSjWYIPXs5i7xCZhOPa54c4X/tkV1t3w0cxj+xCx1zDc0nsvG1tAzDZr22xl7Mwx1/VDRxz4jy83OgUHjjxZngPr8V8griuRXNu0QkxRyAlfHs25fd8+NTBdz8YNaXwEG6JedNyxJJvTv5jS/T5p7EAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAmklEQVR4XmNgGAVDFGgCsSy6IBbAiMxRAOL/QPwPiL9B2WnICpAAJxCbIAuANKGD/UD8G4hl0MS/o/EZPNEFoEAYiP8yQFwCwzYoKogELOgCyKCTAWH6ajQ5dGCHzDkBxM1I/JkMEEOCkMRgoByI2ZAFepA5SOAyA8SQEiD2B+IrQHwDRQUDmklogA+I5wPxNSBOQZMbBUMDAABTahs6czwQ3gAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAXCAYAAAD3CERpAAABS0lEQVR4Xu2UvytFYRjHn8ispG6sZoOyG252i8WApISIuuou/oO7XFZlMBjM/gCDwSRllYGSFCWDJHy/3oPnfO85p/vGUcqnvp2ez/vjOe/b6Zj9E8eAirIZROZVxlKzuE2uVIA+5BR5RbZk7JNd5MnCJGYhPVzIvdRzFvboSOpZ5OVrOJuYpjNI1dVsxPVN5wjdprgUMU31BOMW1k+I/7jBXGKanki9bWH9mHi+3I803UD6xU3ZN066qDKDvE3o+ULq8ua/w8EllUI3sqMyoW7pBvvIo7gWOLisUjhAOlU6hpAj5BDpsTZPuqJSKNwgA86/UOnhhFWVjmFkTaWD6+9c3ZW4inMpei1MaOiA41qFwPXnrr61cNUt7CE3yKWFa+CTm/PXqDyoEKYtND5Lnrn/3naZREZVls2zit/gWEXZrFvBF1gWIyr+FG/ZnFP4B/8VjgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAXCAYAAACI2VaYAAABlUlEQVR4Xu2UTStFURSGl2SspG6ZGBghumSoDGQiP0EGZtyIYmhmYMDEgB9gYOA3GCgjEzE18pUQISUfYS1r72Od9+6zOQM5xVNvt/OsffZ9O19E//wuLSiKQgdnDOVPMUP5/uwMhWGN04rSsMR54lxxemGWsE666M1lPD2OcgfH25xX+tyrLT1OuOXMm+MHzoI5DpKn3CinH6Wjm7LLyTkyszQEXBV5yskVyiJWzt8lRNwwSkuecnsoDLFy4p9Rkvp9lJbvllvmNKE0fFUOn1VBvDx7mciCCsoAodti8eXacUDqr1GS+ui+MpxACdSTfiZi+HLyHUTE36Ak9bHn+GPBJEpgi1OLEvDlOnFA6h9RkvoDlBZZMIUSiF56hy9XxgFl3z5xqygtsmAapaGHM4sygC/XhQNmharL1ThXBz6hkXTBIg4M5ygyGCTdawgHDpnZl2WHwm8wbXAuOSecY/crJeRjidyjAC5Iz7V7iTu0i5hm0oKbnCPOaXqcnxHOAMqi8IKiSOyiKApznBLKotCH4k/zDr5AbJXXcQ99AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIQAAAAYCAYAAAA74FWfAAAEVklEQVR4Xu2ZaahOWxjHH9xMmT8Yo1MIiTKmKEXmsUgSQkqizCKJlC8+3HvNyZQxQ4nI/AEfTJkzhOTIdNW94Zoye/7WWud99mPv8+6D8+711v7Vv73Xf+13v89ee82bKCUlJSXFDxprw3fKsJpq0zOmsgZrMw9oxZqozRDasP7UZhI8Zn218pWPZArsLWubyvOdp9oIYSaZ9/A3efIerpMngYTQjUxs6HZxHBjM9p7/tRECnqu/PX5QeYmAQG5q0xOekL+VNRvjWN21qWhGHj4fAhquTU9AbGe1mSd80UYIq8mzCjGCfgxoEWuQ8nJNH8p0o+tZfVktA1f4z1VtCDqw+rFeUWbI6BW4IiFuU6ZCVGbdZ1UjM4FLEqwqUDER2yzWdDKFmC8sY9XXpgArDzwTnu+ePZ8SuCIhENBdVk3WPuu9t342tkRoM2sTayNrA2sda639TUlARYgTh4/EjRvXTdZmkiCgI6w1OsMDMNGNKtgq2vCI6mQaRjZqk3m+qjrjN4MGGlWOATCRxIU37PF4MDtxENMzbZIp8DgTttIgTsGeZJXTZgi56gGHsG5pMwxcJAPC+QqRzsaSEqqkIJ7F2mT+Yi3VZo6I8wLjXAN0+ZcWaPCxVpEIRu4/IL3Tnr8UfhKghSGeAuXDc/pH+NjtO0BmUrxf+IfJbLytJLPL2UTkVSDznMdYY1ltRd4pMkPpc5uW/1vcS2xPpuXHAfcpVF5F1i4ysa6y53NEfjvWG9YOCm6Jt2A9IPM+4f8h8ly8jViHWFtFXgBcOEylJ5H5tnFa+EkwiqILXvuYlO215+Upk++2uV0aR7mclvfBuZuX4ByVBWApiNUXuETZl+NhQ1wU+J+5ynNdu44NjCGzInG4RluL9Vn4unyQRkUqIFP59wRyLW5CI8G2MLwXyk8CtFAdH6hBP84f5HXYHZRzoeZk1vqao2T2NxzuHqhcOMc93rF6F10RHo/mtTYiQIXD/Vxlk2Bv6IxIu//FEfOTa/bowPONFGkZ51Cbnie8vAQPge5Ngy+CWOM7GlCwAApZXUR6O2u2SDvwm7r2vA5lWthBil4hZKsQo1k9tBkBeuKo+2FvyG1SlaVghQhD+mMo2CAwXGIU8HKLPBto3Qga4yiOlYLZ38GLwwtEy+pkPfmg7tztqSAtx1MHWrIbIrAP4z49T6DgHARfWQvIfGRz3fn9otwgn7QRAuJBt43/DKvwANegIgD0lJ2FL8E9gPQxjPSk4oeeGcLzGmz1opvG/CFq23cA6wJrofCWs86TWSFhwnRO5EV9PcQ8CYW2m0xBYSnrOEFmDoUJZT3h416XRVpTXB7AWI//am2PUSDvEes/MlvajoY2Dz2I7MUwGf6XdYfMbi4qCuYMQA6vV1gPRTovWMCaps1SAJNPR3EvJy7zyfRc2ehK0UMSQGNA5U7JIVhau+76Imu8yPtZ8KJ/FVQoLKVRudwqJyVHYE3eUZspKSnMN2oZGnQxRyVDAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAoElEQVR4XmNgGJSAEYhV0QWxgadA/B+KiQJXGEhQDFJ4DV0QFwApjkAXxAaiGDCd0ATE/mhiYHCTAaGYC4jvAzEfEH+Dq0ACIIW3gVgQiDdCxX5CxTEASHAnEM9El0AHMxgQJsyGslUQ0qgAPTJA7INQdj6SOBiAJKeh8VuQ2HDACRUQRRL7CMQbgLgHiA2RxMHAE10ACDyAmANdcBTAAACQdCSKrBERiwAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGMAAAAYCAYAAADu3kOXAAAEBUlEQVR4Xu2ZaahNURTHlyFEPiCzTBEphBSSJ3MoISlfZEhmkjkl5QPJUEI+4KHIlJIoQ5GklEJCpp4PokRkyMz6v713d9119z7Dfe/qvTq/Wt29/2uddfbZZ5+99zmXKCMj4/8wVQsZlUzRQlrasy1n28/WVehDRFlynG2oFmsB88hcZ3XRj22H0jBIDyotEUfZ/rI9ZZvA1p1tL9trtsHWpxnPdkmLNZxGZK6lNZlB9D3fXRQr2V6x7aLCfrrNNkNpkSDBT7Ym2sGsI+O/rx1UeOLaANosn3LUG4p6MSDHRPv7Q/lA4n76RfHB8Ov5bz3bXaXVdM6TGXQSXNtopaWhB8X3XwXbbi1qPpBJVE87FL6TQRukxRpMCzJtnqx0aCuUloZ95O8fiXtqgvQmE/BSOzz4Evk0yVy2rWwNbH0hmTUoLbiQcrb6Sm9KudygjSj7uECFbe5mtbFKT8JAMmvrJzI50M6oPPrcefwmE4CLSgsW7qjkmPrQec3IxL1g68y2xPqSspFtpv3V50P9gKrrGAl8uOa6tl6HzOCIOiaKBWSeKBz/3JaX5UXkgzhshLzENT6KTRQ+FrsK7FgciDsryo+FLw7s5ABupjwfOhT1XkJ7z3ZF1CUYGIhHnmvCqtIHDhy/WIseEDdLi46ohkxiG8lWxjacbRTlryuHKXysHB04BnGYDoC8SXFg5HaxZeTAe4/DjcikzCYT30np0B4pLQ2tyORIMrsgbosWHXC+1aJlEeWmBtgayj3e4JDV41hNyeKiGEYmhxwMH62WlD1UGI+FHFo7padhFRXmDYG47Vp0uI6OAv4nWmTWUvyx4DMli4viHhXmQP2c0qLYRoU5cEMxtVUFPFU6bwjELdWi4yGZgNALDxZb+KdrB+VGlY9vbEdsGTEnhW8a5W+HMRVtEHUfyOHWDuCmvv5CA9jZuGlNA59sb3NVl8xha6nFAMhRocUAiA19UqoEATA5BYGOwhfC52tLRsfrP24iyuXWh3PgSZG8IxOD6SzEA8o/lztGgpuapr0oY13U9KX4PBLE4QtFEhLlvEq5BuCxxS92S+C0C/KAuAFaZN5Q/k3Atx/Ub7kAAS4enRv3roO1DTnw1H21Zc0ZMv4Q6Hwc94fMd7cQ+JYUWkslmFGQr7F2eBhD/jZXG3iJuqnFIrmoBYHbiTlwUdhg+PiihSJJ8i6ENiTtYKwtmPZLStLGRIG9dwctWtxT5bij6ppnWigSfLkOgfNfJhMTNYgkUW2uNjazndBiSjCthcBFjLDlnRQ98q9TeDOSBmwWsAb5cAt/H/ubBLyIjtNiqbhBpnGlArutUxT/UbKnFooEm5coysj895MEDKRjWiw187WQUQk+jmZkZKTmHxa0CRJihaEdAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAoElEQVR4XmNgGHJAEoinAjEbugQ2YAHE/4A4Boj/o8lhBSBFPVCaaA2a6IK4QDgDkaYqA7EXEJ9ggGjwBWIPFBVowB+IixggikEeBrELUFTgACANa9AF8QGQBh90QVwgjYFID8MAzMNEA5DiN+iC+ABIQzO6IC7AxADRwIEugQ4cgdgPiJMYiHQ/SNEHIL4HxKVoclgBSMMuIL6OLjEEAABbmSIQUKWkPQAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF8AAAAYCAYAAACcESEhAAACa0lEQVR4Xu2Yy6tPURzFl0eIUCYGNwbKwMCICQNXGNwQUl4T/4CUMvBOMlUyuTGSRyaMiDJQJkYyMZBHKSNJ3gp5fpfv2f32Wff8fnef3z2n+xvsT63u7661H9993ucAmUwmM1AMmQ6YLpiWRP7q6HemYa6Y/ppemDaZlppGTa9Nq4qsDd7Bx441u8guif+j8CeTa6bvKNf1rNQCmBJl1NdyXIYNfprmaGAcgeePNWiYUKjCuvaoOQDch9d7XPzAedMtNZVfqF50DPPtajbMbfg8hyPvoWkk+n+QWIDuB8wa0yM1lY/wztM0EKomaJrpKC/mqulgJ24Fnukn1KxB1cbnTvki3hiWwzu+0qACnaAbK9SoyR/4XKdNlyVri9S1VcEa2f9c5CWN9xvecK4GfbIXPt5TDWqwAz7GuEdOgxxF/09yerby4BnvKvKfqlNmIsyAj7dNgxrwCavpulJ4b1qoZiKh3g+mRZJ1pdcit5rWm4ZNa00bkLhHJwDPwG+mO/C69pXjZFb2KT4OXkR9zsLr3alBL9jhrZoFXPhJdHbQIdPUUotm4djhQNBTuS5b+hQvdXdRH27D2rWmLJD5czVbQOsI96N54rfFA/glrx9StuMYnsA7zdSgYD8836VBD46pkQA39CzxeN/g3G2/2BE+KGxUM5HwJntTgxTCXtNLyuIoS4Xfg9j+jQZdWAdvf10D+P0lzM8FtkmdNSpn4P2HNUjlHjoL5V2ff08V2Y3QKJGXpmVqCtyxvLnxmw6fEHit3RzluwuPGdt8gr8MtsF8+DtFXfjN6TN8e4UauaZMJpPJZDKZTGay+AffIqo7ULxVqgAAAABJRU5ErkJggg==>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF4AAAAYCAYAAABz00ofAAACOklEQVR4Xu2XO2hUQRSGfxFFMBIhhWDjQmIhEZsEhDQGbMQipAmpYqeVhYUQULSIYm2VlJa+CrGMjRiIYIi2SSE2ImjhMwqiYjKHc0fO/snunZm9N6s4H/zszn/u3HnceQKZTCaT+V945LQeqCo47vQZze+VNPMDzc8caA53lVmnX2iu30sT/02xOya2iXade8/pJ5sdMgotb418T8PpE5t/GTegbbhL/iC07jvI38RO6AuWOFCwz+kJmxXQ6mP7+vwLcBt6oLM1iIvQzKfI31P89jrN2EBFrEDLnSK/7k7f63SFzUTeQOt7EjrCo+ru11zLJaeB4v8uaGXLGGKjhH5ouXYZ++LUZ9J1we1NpQF91/fit3R5sfB0OULpEM5A86xyoARb9rLTCROrExlYI2wm4tsQMjj/4NfTrRTDbmiecQ6U4E9Vr52uU6xuPqCa09J7aBtucaAd09BMY8Y75nTfpOtkP7T8FxyIZDhRXxHZYcRz6CyNHqxynOMMZ6HLzXZwHlp+7P7AyMBJkbR/HmnIMfJc8d9vsnJPCSL6S1XMW3Sv/EWnw2wGctVpzqQPQdvxzXgt8evyMw4kcpmNALr14eUwcJrNQCacFthERFtuQh+c5EACF6DveseBNhyE5nnMgW0gqIMIuc88Reu8D6Cxhxzw3IaubR+hO7Jcb+UcaqdOCq8QtjcchW5qtnyZolHHsQ6QDrzGZglyG/X1lXuP3Dcs0h7xJS7PSXrUPpDJZDKZTCaTyXTABq/7p6PRsZrCAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGwAAAAYCAYAAAAf1RgaAAAD4UlEQVR4Xu2Za+hNWRjGX+QSJndKM01TM0UKISG33BIzwpgU+aAouZWQywdFQqSUMi5lmmKED2rUhHwx4xKfhPgi/nwQIuM6jOv7tNZy1nnOXuusc2ZTZ/7nV09n7+d999p7rb32WmvvI1KnTp3a4kc2cqIPG4lMZcOjLxuNjQOqoWxm0Eo1U7VHNcHzm6m6evuOfqo/2EwEHWgvmx7v2aglToipQIoYNDyOjzFNzLH/qOaqvlEtsh5uVla5zVVv2KyQC6oZbFpwDY/YrDVCNwUcUr1mU8L5jgYxOcPIB20lfM63qv5sVkFW2Y7nqh/YrBVcT0evzOIL1SnyVqsukudzWkyZ7Tnggfgl8tpYPw8aVNvZtHwn+Z3ns7NMzMWPJx/zDminWucHxOQPIs+xS0wcw2EMDEtTyLuiOkZetXwv8ZuCWGs2I7RUDWQzg85s5M1jKa0YnqBv7TbmFPR8H873CQ11zCs2xBw3hk0PzJu7pfDk/qT6VcwokUXsOhDbzGaAG2Ly79nfq8XhIp6yUYZK61TSwD1pn8EJQnGsGhHbyIFEQuUCzDudVC3E5OFJHKUaYfezgD+YTQsWTFmdhlmqGkLeRDFlYwHls0a1hLwYFdfJzV9ZCrFWwnF3bEcOJBBaNYIjql7ePvLue9tYqGSB2Gw2LTslfD6ff9nw+FOK2+xycThKVXVaISY4yfN6qw57+wwe11BF3YVngflqrGqkmB40WswS24Ht0LEYon2QN8duB4cOMXmb2LS4upejCRsZYH6rlKrqhPGWLxrvSxgWQ/wipcc4YjdsgRQaCdoixV8zvrZ+OTDxp+QB5G1l07Jc0sv5SgrX/UDVpThcxDM2EkiuU6yBQ6yU8DEp5cVyQr7PcUnLA8hbzKblZ0krB6u+O1JYNWP1iRf7cx8zCjRV/c5mAkl1chPdeQ6UAUNbqPBZYmKTOWBBhRC/xgFLqFyM7bfsNnKwWnN0F9OJskAuLxgcmOBfsJnBEzYsmBtRPt5RR6o22P2UIRRUXKdtYhKncyCBUMMCvF8hPoADYlZliGHpmgViI9gU4++QwgvvTYqFKBdDG5QD3z9jrFJdV+3jQBmS64SlN+YuNOxD1d+ql2KGiFRQYOzz0UIxOZCbJ92TjIVHiAbVUTaVs2LKOGP3Xae4/TGjlHFS/oZ1YPMzUk2dqgZf0t2J8iR58k0Awy6/Jzm+lPzOUzN8qgq/U/Vgswpi14feHPvf7H/JetVBNnMAS2gM0f+Fk1L6bdTRTXWXzcbCX2JetPMGf33sZzMRfNr5jU2P2JPXKJjHRk7EPgLHmM+Gx3A26tSp05j5ADNUFrtUT9BtAAAAAElFTkSuQmCC>