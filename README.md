# 🌾 Rice Variety Classification with PyTorch

A deep learning project using **Transfer Learning (ResNet-18)** to classify various rice grain varieties with high accuracy. Built with PyTorch and Torchvision.

---

## 📌 Overview

This repository provides an end-to-end pipeline for:
- Loading and preprocessing the **Rice Image Dataset** via `torchvision.datasets.ImageFolder`.
- Fine-tuning a pre-trained **ResNet-18** model with PyTorch.
- Training with modern optimization techniques (**AdamW** and **CosineAnnealingLR**).
- Comprehensive evaluation including accuracy tracking, classification reports, and confusion matrix visualization.
- Single-image inference functionality ready for deployment.

---

## 📂 Dataset Structure

The code expects the dataset organized in standard classification directory format (e.g., [Rice Image Dataset from Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)):
```text
Rice_Image_Dataset/
├── Arborio/
├── Basmati/
├── Ipsala/
├── Jasmine/
└── Karacadag/
