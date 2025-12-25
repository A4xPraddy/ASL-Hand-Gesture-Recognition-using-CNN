# ASL Alphabet Recognition using Deep Learning

## Overview
This project implements a deep learning–based system to recognize **American Sign Language (ASL) alphabet gestures** from images and real-time webcam input. The model is trained from scratch using a Convolutional Neural Network (CNN) and can classify **29 classes** including A–Z, SPACE, DELETE, and NOTHING.

---

## Objective
To build an end-to-end ASL alphabet recognition system that:
- Learns visual hand gesture patterns
- Classifies ASL alphabets accurately
- Works on both static images and live webcam feed

---

## Dataset
- **Dataset Name:** ASL Alphabet Dataset
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Classes:** 29 (A–Z + SPACE + DELETE + NOTHING)
- **Total Images:** ~87,000

> Dataset is not included in this repository due to size constraints.

---

## Model Architecture
The CNN model is trained **from scratch** and consists of:
- Convolutional layers for feature extraction
- MaxPooling layers for spatial reduction
- Fully connected dense layers
- Softmax output layer for multi-class classification

**Input shape:** (128 × 128 × 3)  
**Output:** 29-class probability distribution

---

## Workflow
1. Download dataset from Kaggle
2. Preprocess images (resize, normalize)
3. Apply data augmentation
4. Train CNN model
5. Validate model performance
6. Save trained model
7. Perform real-time prediction using webcam

---

## How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
