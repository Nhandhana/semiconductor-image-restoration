# semiconductor-image-restoration
AI-based semiconductor image restoration using a Residual CNN for low-resolution degraded inspection images.
# Semiconductor Image Restoration using Deep Learning

## Overview

This project addresses the problem of restoring degraded semiconductor inspection images using deep learning.

In semiconductor manufacturing, inspection images can contain degradation, noise, and loss of fine details. These imperfections can make defect analysis difficult and may affect automated inspection systems.

Our approach uses a **deep residual convolutional neural network** to learn the difference between a degraded image and its corresponding high-quality ground-truth image.

The trained model reconstructs a cleaner and higher-quality image while preserving important semiconductor structures and details.

---

## Problem Statement

Develop an efficient deep-learning-based solution to restore degraded semiconductor inspection images and improve their visual quality for downstream inspection and defect analysis.

---

## Proposed Solution

We use a **Residual CNN-based image restoration model**.

The model learns a residual mapping:

**Restored Image = Degraded Image + Predicted Residual**

Instead of learning the complete image from scratch, the network learns the missing information that needs to be added to the degraded image.

### Pipeline

```text
Degraded Image
      ↓
Preprocessing
      ↓
Residual CNN
      ↓
Predicted Residual
      ↓
Residual Addition
      ↓
Restored Image
      ↓
Quality Evaluation
```

---

##  Model Architecture

The model consists of convolutional layers and residual blocks.

Each residual block contains:

* 3 × 3 convolution
* ReLU activation
* 3 × 3 convolution
* Residual/skip connection

Residual learning helps the network focus on recovering lost image information while preserving existing structures.

---

## Dataset

The dataset contains degraded semiconductor images and their corresponding ground-truth images.

The data is stored in NumPy (`.npy`) format.

```text
Dataset
├── NoisyLR
│   └── degraded images
│
└── GT
    └── ground-truth images
```

The ground-truth images are used during training to calculate the reconstruction loss.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* OpenCV / image processing tools
* Google Colab
* CUDA GPU acceleration
* GitHub

---

## Evaluation

The model is evaluated using standard image-restoration metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between the restored image and ground-truth image.

Lower MSE indicates better reconstruction.

### Peak Signal-to-Noise Ratio (PSNR)

Measures the reconstruction quality based on the ratio between signal strength and reconstruction error.

Higher PSNR indicates better image quality.

---

## Current Results

The trained model was evaluated on the test images.

Current evaluation:

* **Images evaluated:** 3200
* **Average MSE:** 0.0031068914
* **Average PSNR:** 25.08 dB

Additional evaluation during the restoration workflow produced:

* **MSE:** 0.009670979
* **PSNR:** 20.15 dB

The difference between these values comes from evaluating different subsets/stages of the restoration pipeline.

---

## Demonstration

A demonstration video is included to show the image restoration process.

The demo illustrates the transition from:

```text
Degraded Image → Deep Learning Restoration → Restored Image
```

The demonstration can be found in the `demo/` directory.

---

## Project Structure

```text
Semicon-Image-Restoration/
│
├── README.md
│
├── train_residual.py
├── inspect_data.py
├── inference.py
├── create_demo.py
├── model.py
│
├── demo/
│   └── semicon_restoration_demo.mp4
│
├── results/
│   ├── noisy_sample.png
│   ├── restored_sample.png
│   └── comparison.png
│
└── checkpoints/
    └── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy opencv-python matplotlib
```

### 2. Prepare the dataset

Place the dataset in the required directory structure:

```text
train/
├── NoisyLR/
└── GT/
```

### 3. Train the model

```bash
python train_residual.py
```

### 4. Run inference

```bash
python inference.py
```

### 5. Generate the demonstration

```bash
python create_demo.py
```

---

## Key Features

* Deep-learning-based image restoration
* Residual learning architecture
* Preserves important image structures
* Quantitative evaluation using MSE and PSNR
* Automated restoration pipeline
* Demonstration video for visual validation
* Designed for semiconductor inspection image restoration

---

## Uniqueness

The key idea is to use **residual learning specifically for semiconductor inspection image restoration**.

Instead of attempting to regenerate the entire image, the network focuses on learning the degradation-related information that must be corrected.

This makes the restoration process conceptually simple, computationally efficient, and suitable for further optimization toward practical semiconductor inspection systems.

---

## Future Improvements

Potential improvements include:

* Larger and more diverse training datasets
* Deeper residual architectures
* Attention mechanisms
* Multi-scale feature extraction
* SSIM-based evaluation
* GPU/edge inference optimization
* Real-time restoration
* Integration with semiconductor defect detection
* Deployment as an inspection pipeline

---

## 👥 Project

Developed as part of the **Semicon Hackathon 2026**.

The goal is to explore an efficient deep-learning solution for improving degraded semiconductor inspection imagery.
