# Alzheimer's Disease Classification using ConvNeXt

## Overview

This project implements a deep learning solution for classifying Alzheimer's disease in brain MRI images using the ConvNeXt architecture. The task involves binary classification to distinguish between patients with Alzheimer's Disease (AD) and Normal Cognitive (NC) patients using the ADNI brain dataset.

**Problem:** Classify Alzheimer's disease (normal and AD) of the ADNI brain data using ConvNeXt with a minimum accuracy of 0.8 on the test set.

## Project Description

### Problem Statement

Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early detection through medical imaging can significantly improve patient outcomes. This project develops a deep learning model to automatically classify brain MRI scans into two categories:
- **AD (Alzheimer's Disease)**: Patients diagnosed with Alzheimer's disease
- **NC (Normal Cognitive)**: Healthy control patients

### Algorithm

The solution uses **ConvNeXt** (pretrained via `timm`), a modern convolutional neural network architecture that adapts design principles from Transformers. Key features include:

1. **Pretrained Backbone**: Uses ImageNet-pretrained ConvNeXt from the `timm` library for transfer learning
2. **Custom Classification Head**: Adaptive pooling, dropout (0.5), and linear classifier for binary classification
3. **Grayscale Adaptation**: Converts single-channel MRI images to 3-channel format for compatibility with pretrained weights

The model architecture consists of:
- **Pretrained ConvNeXt Backbone**: Feature extraction using proven architecture
- **Adaptive Global Pooling**: Ensures fixed-size feature vectors
- **Dropout Layer**: Regularization to prevent overfitting (p=0.5)
- **Linear Classifier**: Final 2-class output layer

### How It Works

1. **Data Loading**: The dataset loads preprocessed brain MRI images from the ADNI dataset
2. **Preprocessing**: 
   - Training: Aggressive augmentation (rotation, translation, shear, color jitter, Gaussian blur)
   - Validation/Test: Resize to 224x224 and normalize only
3. **Training**: The model fine-tunes pretrained weights using:
   - AdamW optimizer with weight decay (1e-4)
   - OneCycleLR scheduler with warmup

## Dataset

The project uses the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** brain MRI dataset.

**Dataset Structure:**
```
ADNI/
└── AD_NC/
    ├── train/
    │   ├── AD/    (10,400 slices - Alzheimer's Disease)
    │   └── NC/    (11,120 slices - Normal Cognitive)
    └── test/
        ├── AD/    (4,460 slices)
        └── NC/    (4,540 slices)
```

- Training: 21,520 slices (48.3% AD, 51.7% NC)
- Test: 9,000 slices (49.6% AD, 50.4% NC)
- Test Patients: 450 unique patients
- Format: Grayscale JPEG images, 224x224 pixels

## Dependencies

Key dependencies:
- torch
- timm (for pretrained ConvNeXt model)
- torchvision
- numpy
- scikit-learn
- matplotlib
- Pillow
- tqdm
- pandas

## Usage

### Training

```bash
python train.py \
    --data_dir /home/groups/comp3710/ADNI \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.5 \
    --class_weight_nc 1.0 \
    --class_weight_ad 2.0 \
    --val_split 0.2 \
    --save_dir models
```

### Prediction

```bash
python improved_predict.py \
    --model_path checkpoints/best_model.pth \
    --data_dir /home/groups/comp3710/ADNI \
    --batch_size 64
```

## Performance

**Achieved Results (14 epochs):**
- Training Accuracy: 98.97%
- Validation Accuracy: 99.01%
- Best Validation Loss: 0.0229
- Training Time: ~20 minutes on A100 GPU

**Test Performance:**
- Patient-Level Accuracy: 78.67%
- Precision: 96.35%
- Recall: 59.19% (AD detection)
- F1-Score: 73.33%

**Confusion Matrix (Patient-Level):**
```
                Predicted
              NC      AD
True NC      222       5    (97.8% recall)
True AD       91     132    (59.2% recall)
```

## Performance Analysis
**Strengths:**
- Excellent NC detection: 97.8% recall (only 5 false positives)
- High precision: 96.35% (very few false alarms)
- Strong validation performance: 99% validation accuracy

**Challenges:**
- AD recall at 59.2% indicates conservative predictions
- Model bias toward NC class despite class weighting
- Gap between validation (99%) and test (78.67%) suggests domain shift or overfitting

## AI Usage

This project's documentation was developed with assistance from Claude (Anthropic).