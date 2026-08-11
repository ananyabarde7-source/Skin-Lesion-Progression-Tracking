# Multi-Modal Computer Vision Pipeline for Tele-Dermatology Skin Lesion Progression Tracking

This repository contains an end-to-end medical image processing and analytical decision support pipeline designed to track morphometric and colorimetric alterations in dermatological skin lesions over chronological intervals. The architecture integrates automated color-space segmentation, Deep Learning feature extraction (via MobileNetV2), and classical handcrafted visual metrics to profile lesion progression dynamics and evaluate clinical risk tiers.

## 🚀 Architectural Blueprint & Workflow

### 1. Robust Morphological Segmentation
To eliminate background artifacts, camera lens vignetting, and hair structures common in the HAM10000 dataset, the pipeline uses localized color-space processing:
* **Chrominance Isolation**: Input frames are transformed from BGR to the YCrCb color space. The **Cr (red-difference) channel** is isolated to optimize the contrast boundary between hemoglobin/melanin lesion density and surrounding healthy dermis.
* **Non-Parametric Thresholding**: Spatial noise is minimized using a Gaussian Blur ($5\times5$ kernel, $\sigma=0$), followed by **Otsu’s Automated Thresholding** to establish a global binarization mask.
* **Topological Filtering**: Morphological `CLOSE` operations close interior structural micro-holes. A **Connected Component Contour Filter** scans the canvas topologies, isolates the single largest geometric contour (the primary lesion), and wipes all stray peripheral artifacts completely black.

### 2. Feature Verification (MobileNetV2)
A Deep Convolutional Neural Network (CNN) leverages Transfer Learning for underlying semantic classification:
* **Feature Extractor**: A **MobileNetV2** base architecture pre-trained on ImageNet is frozen (`trainable=False`) to exploit generalized edge, texture, and shape descriptors.
* **Classification Head**: Global Average Pooling 2D downsamples spatial feature maps to a 1D vector, passing it through a dense bottleneck layer with a **Softmax activation function** to yield posterior probabilities across a binary diagnostic scale (`benign` vs. `malignant`).

### 3. Chronological Progression Metrics
Lesion alterations are quantified between sequential evaluations by computing differential spatial matrices against baseline cluster attributes:
* **Geometric Area Variation ($\Delta\text{Area}$)**: Measures the absolute percentage shift in non-zero mask pixels relative to standard class parameters:
  $$\Delta\text{Area} = \left| \frac{\text{Area}_{\text{Current}} - \mu_{\text{Baseline}}}{\mu_{\text{Baseline}} + 10^{-9}} \right| \times 100$$
* **Colorimetric Distance ($\Delta\text{Color}$)**: Computes the mean absolute color distance vector across the segmented three-channel pixel arrays against expected class configurations to track underlying vascularization or pigmentation shifts.

### 4. Deterministic Risk Level Allocation
Rather than misinterpreting tissue shrinkage as a baseline state, an **Absolute Variation Priority Logic Rule** analyzes the extracted properties to map progression variations into bounded severity categories (`LOW`, `MEDIUM`, or `HIGH` Risk).

---

## 📊 Analytical Performance & Pipeline Output

* **Deep Learning Classifier Accuracy**: **81.0%** on the validation split.

### Clinical Progression Visual Tracking
The side-by-side evaluation tool automates spatial tracking over sequential checks. It overlays deep classification probabilities, differential shift vectors ($\text{dArea}$, $\text{dColor}$), and assigned clinical risk tiers directly onto the output layout:

![Clinical Decision Support Visualizer](final_progression_tracking_output.png)

---

## 🛠️ Technology Stack & Core Dependencies
* **Core Runtime**: Python 3
* **Image Processing**: OpenCV (opencv-python)
* **Deep Learning Framework**: TensorFlow / Keras
* **Data Processing & Analytics**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
