# MED1C - Multi-code Extraction, Detection, Inference, and Classification

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Faster R-CNN](https://img.shields.io/badge/Model-Faster%20R--CNN-orange.svg)](https://arxiv.org/abs/1506.01497)
[![PyZBar](https://img.shields.io/badge/QR%20Decode-PyZBar-yellow.svg)](https://github.com/NaturalHistoryMuseum/pyzbar)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> A complete deep learning pipeline for medical QR code detection, localization, and content classification using Faster R-CNN with ResNet-50 backbone.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Dataset Format](#dataset-format)
- [Competition Submission](#competition-submission)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

MED1C is a comprehensive computer vision system designed for detecting and classifying QR codes in medical imaging contexts. The system implements a two-stage approach:

1. **Detection Stage**: Uses Faster R-CNN with ResNet-50 backbone to locate QR codes in images
2. **Decoding & Classification Stage**: Extracts QR content and classifies into medical categories (Batch Numbers, Manufacturer Codes, Regulatory/GTIN codes)

The project is optimized for both high-performance cloud training and local inference on limited hardware (6GB VRAM).

## Key Features

- **State-of-the-art Detection**: Faster R-CNN with ResNet-50 FPN for robust QR code localization
- **Medical QR Classification**: Specialized classification for medical industry QR codes
- **Memory Optimized**: Configurable for both 6GB VRAM local training and high-performance cloud training
- **Complete Pipeline**: End-to-end training, inference, and evaluation workflow
- **Competition Ready**: Generates submission files in required JSON format
- **Comprehensive Augmentation**: Advanced geometric and photometric augmentations for medical imaging
- **Visualization Tools**: Built-in prediction visualization and analysis tools
- **Robust QR Decoding**: Advanced QR code reading with multiple enhancement techniques

## Tech Stack

- **Python 3.12+**
- **PyTorch 2.1+**: Deep learning framework with CUDA support
- **TorchVision**: Computer vision models and utilities
- **OpenCV**: Image processing and computer vision
- **PyZBar**: QR code decoding library
- **Albumentations**: Advanced data augmentation
- **PIL/Pillow**: Image manipulation
- **NumPy**: Numerical computing
- **tqdm**: Progress bars and monitoring

## Environment Files

This project includes multiple environment setup options:

- **`environment.yml`**: Complete conda environment with CUDA PyTorch (required/recommended)
- **`requirements.txt`**: Standard pip requirements for manual installation (alternative)
- **`torch-req.txt`**: CPU-only PyTorch requirements for systems without CUDA

### Primary Setup (Required)
Use the provided `environment.yml` as specified:
```powershell
# Create environment from provided file
conda env create -f environment.yml
conda activate med1c-qr-detection
```

### Alternative Setup (Manual Installation)
For users who prefer pip or don't have conda:
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Project Structure

```
multiqr-hackathon/
├── main.py                     # Main pipeline entry point
├── config.py                   # Centralized configuration
├── train.py                    # Training pipeline
├── infer.py                    # Inference pipeline
├── evaluate.py                 # Evaluation metrics
├── format_submission.py        # Competition submission formatter
├── requirements.txt            # Python dependencies
├── torch-req.txt              # PyTorch CPU-only requirements
├── environment.yml            # Conda environment with CUDA PyTorch (recommended)
├── data/                      # Dataset directory
│   ├── annotations.json       # COCO-style annotations
│   ├── demo_images/           # Demo images for testing (included)
│   │   ├── img201.jpg         # Sample demo image 1
│   │   ├── img205.jpg         # Sample demo image 2
│   │   ├── img210.jpg         # Sample demo image 3
│   │   ├── img225.jpg         # Sample demo image 4
│   │   └── img240.jpg         # Sample demo image 5
│   ├── images/                # Image dataset (not included in repo)
│   │   ├── train/            # Training images
│   │   ├── val/              # Validation images
│   │   └── test/             # Test images
│   └── labels/               # YOLO format labels (generated)
│       ├── train/            # Training labels
│       └── val/              # Validation labels
├── src/                      # Source code modules
│   ├── models/               # Model definitions
│   │   ├── faster_rcnn_model.py  # Faster R-CNN implementation
│   │   └── __init__.py
│   ├── datasets/             # Dataset loaders
│   │   ├── qr_dataset.py     # QR dataset class
│   │   └── __init__.py
│   └── utils/                # Utility functions
│       ├── augmentation.py   # Data augmentation
│       ├── convert_annotations.py  # Annotation converter
│       ├── qr_processing.py  # QR decoding and classification
│       ├── target_matching.py     # Anchor matching utilities
│       └── __init__.py
├── runs/                     # Training outputs
│   └── MED1C_Detection/
│       ├── weights/          # Model checkpoints
│       └── logs/             # Training logs
└── outputs/                  # Inference outputs
    ├── submission_detection_1.json   # Stage 1 submission
    ├── submission_decoding_2.json    # Stage 2 submission
    ├── test_predictions.json         # Raw predictions
    ├── demo_predictions.json         # Demo predictions
    ├── visualizations/               # Prediction visualizations
    └── demo_visualizations/          # Demo visualizations
```

## Setup and Installation

### Option 1: Using Provided Environment (Required)

We provide a complete conda environment file with CUDA-enabled PyTorch as specified for this project.

```powershell
# Clone the repository
git clone https://github.com/TheRevanite/multiqr-hackathon.git
cd multiqr-hackathon

# Create environment from provided file
conda env create -f environment.yml

# Activate environment
conda activate med1c-qr-detection
```

### Option 2: Manual Installation (Alternative)

If you prefer to create your own environment or don't have conda:

#### 2.1. Create Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/TheRevanite/multiqr-hackathon.git
cd multiqr-hackathon

# Create virtual environment
python -m venv venv
venv\Scripts\activate
```

#### 2.2. Install Dependencies

**For CUDA-enabled GPU:**
```powershell
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**For CPU-only installation:**
```powershell
pip install -r torch-req.txt
pip install -r requirements.txt
```

### 3. Prepare Dataset
```powershell
# Convert annotations to YOLO format (if using full dataset)
python src/utils/convert_annotations.py
```

### 4. Verify Installation
```powershell
# Test with demo images (no training data required)
python main.py --mode full-demo
```

## Quick Start

### Demo Mode (Recommended for First Time Users)
```powershell
# Run inference on included demo images (no training data required)
python main.py --mode full-demo
```

### Single Command Training and Inference
```powershell
# Complete pipeline: train model and run inference
python main.py --mode train
python main.py --mode base-full
```

### Generate Competition Submissions
```powershell
# Create both detection and decoding submission files
python format_submission.py
```

## Detailed Usage

### Main Pipeline Commands

The `main.py` script provides a unified interface for all operations:

```powershell
# Validate environment and setup
python main.py --mode setup

# Train the model
python main.py --mode train

# Run inference and evaluation on test set
python main.py --mode base-full

# Run inference on custom directory
python main.py --mode full --test-dir ./custom_images

# Run demo with included sample images (no training data needed)
python main.py --mode full-demo

# Specify device
python main.py --mode train --device cuda
python main.py --mode train --device cpu
```

### Training Commands

The `train.py` script handles model training:

```powershell
# Basic training (recommended)
python main.py --mode train

# Direct training script
python train.py

# Training with specific device
python main.py --mode train --device cuda
python main.py --mode train --device cpu

# The training automatically:
# - Creates optimized dataloaders for 6GB VRAM
# - Implements early stopping
# - Saves best model checkpoints to runs/MED1C_Detection/weights/
# - Logs training progress to runs/MED1C_Detection/logs/
```

### Inference Commands

The `infer.py` script provides flexible inference options:

```powershell
# Run inference on test dataset
python infer.py --test

# Inference on single image
python infer.py --input path/to/image.jpg --output predictions.json

# Inference on directory
python infer.py --input path/to/images/ --output predictions.json

# Inference on demo images specifically
python infer.py --input data/demo_images/ --output submission.json

# Custom model and confidence threshold
python infer.py --model path/to/model.pth --confidence 0.7

# Generate visualizations
python infer.py --input image.jpg --visualize

# All options combined
python infer.py --model custom_model.pth --input ./test_images --output results.json --confidence 0.6 --visualize
```

### Evaluation Commands

The `evaluate.py` script calculates performance metrics:

```powershell
# Basic evaluation (uses default submission files)
python evaluate.py

# Custom evaluation files
python evaluate.py --detection submission_detection_1.json --decoding submission_decoding_2.json --ground-truth annotations.json
```

### Submission Formatting Commands

The `format_submission.py` script creates competition-ready submissions:

```powershell
# Generate both submission files from test predictions
python format_submission.py

# Custom input file
python format_submission.py --input custom_predictions.json

# Generate only detection submission
python format_submission.py --detection-only

# Generate only decoding submission  
python format_submission.py --decoding-only
```

### Dataset Conversion Commands

The `convert_annotations.py` script converts COCO to YOLO format:

```powershell
# Convert annotations (run once during setup)
python src/utils/convert_annotations.py

# The script automatically:
# - Reads annotations.json
# - Converts to YOLO format
# - Splits into train/val directories
# - Handles missing images gracefully
```

### Configuration Commands

The `config.py` file contains all settings. Key parameters can be modified:

```python
# Training Configuration
EPOCHS = 200                    # Number of training epochs
BATCH_SIZE = 1                  # Batch size (optimized for 6GB VRAM)
IMAGE_SIZE = 384                # Input image size
LEARNING_RATE = 2e-4            # Learning rate

# Model Configuration  
CONFIDENCE_THRESHOLD = 0.35     # Detection confidence threshold
NMS_THRESHOLD = 0.40           # Non-maximum suppression threshold

# Device Configuration
DEVICE = 'cuda'                # Training device (auto-detected)
MIXED_PRECISION = True         # Enable mixed precision training
```

### Demo Mode Commands

The demo mode allows you to test the system without requiring training data:

```powershell
# Run complete demo pipeline (requires trained model)
python main.py --mode full-demo

# The demo mode automatically:
# - Uses 5 included sample images from data/demo_images/
# - Runs inference with visualization
# - Generates submission files
# - Creates demo_visualizations/ folder with annotated results
# - No training data or annotations required
```

**Demo Mode Features:**
- Uses pre-selected QR code images for testing
- Generates visualizations with bounding boxes and confidence scores
- Creates submission files compatible with competition format
- Perfect for testing the complete pipeline functionality
- Requires only a trained model (from previous training or provided weights)

## Configuration

### Training Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `EPOCHS` | Training epochs | 200 | 50-500 |
| `BATCH_SIZE` | Batch size | 1 (6GB VRAM) | 1-8 |
| `IMAGE_SIZE` | Input image size | 384 | 256-640 |
| `LEARNING_RATE` | Learning rate | 2e-4 | 1e-5 to 1e-3 |

### Model Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `MODEL_BACKBONE` | CNN backbone | resnet50 | resnet50, resnet101 |
| `CONFIDENCE_THRESHOLD` | Detection threshold | 0.35 | 0.1-0.9 |
| `NMS_THRESHOLD` | NMS threshold | 0.40 | 0.1-0.9 |

### Hardware Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `DEVICE` | Computation device | auto | cuda, cpu, auto |
| `MIXED_PRECISION` | Enable FP16 training | True | True, False |
| `GRADIENT_ACCUMULATION_STEPS` | Gradient accumulation | 4 | 1-8 |

## Model Architecture

The system uses **Faster R-CNN** with the following specifications:

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **RPN**: Region Proposal Network for object localization
- **ROI Head**: Region of Interest head for classification and regression
- **Anchor Scales**: (32, 64, 128, 256, 512) pixels
- **Aspect Ratios**: (0.5, 1.0, 2.0) for each scale
- **Input Size**: 384x384 pixels (configurable)
- **Classes**: Background + QR_code (2 total)

### Training Strategy

1. **Transfer Learning**: Pre-trained ResNet-50 backbone
2. **Differential Learning Rates**: Lower LR for backbone, higher for heads
3. **Data Augmentation**: Geometric and photometric augmentations
4. **Early Stopping**: Prevents overfitting with patience-based stopping
5. **Mixed Precision**: Reduces memory usage and speeds up training

## Dataset Format

### Input Format (COCO-style)
```json
[
  {
    "image_id": "img001.jpg",
    "qrs": [
      {
        "bbox": [x_min, y_min, x_max, y_max],
        "value": "BATCH123",
        "type": "Batch_Number"
      }
    ]
  }
]
```

### YOLO Format (Generated)
```
# Each .txt file contains:
class_id x_center y_center width height
0 0.5 0.3 0.2 0.1
```

### Output Format
```json
[
  {
    "image_id": "img201",
    "qrs": [
      {
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.95,
        "value": "L123456",
        "type": "Batch_Number"
      }
    ]
  }
]
```

## Competition Submission


**Environment Setup for Organizers:**
```powershell
# Primary method: Use provided conda environment (required)
conda env create -f environment.yml
conda activate med1c-qr-detection

# Alternative: Manual pip installation (if conda unavailable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Stage 1: Detection (Mandatory)
```powershell
python main.py --mode base-full
# Generates: outputs/submission_detection_1.json
```

### Stage 2: Decoding & Classification (Bonus)
```powershell
# Automatically included in Stage 1 output
# Generates: outputs/submission_decoding_2.json
```

### Demo Mode for Organizers
```powershell
# Test the complete pipeline with included demo images
python main.py --mode full-demo

# This will generate:
# - outputs/demo_predictions.json (raw predictions)
# - outputs/demo_visualizations/ (annotated images)
# - outputs/submission_detection_1.json (competition format)
# - outputs/submission_decoding_2.json (competition format)
```

### Reproducibility Commands
```powershell
# Single command to reproduce results on test dataset
python main.py --mode base-full

# Single command to test functionality with demo images
python main.py --mode full-demo
```

## Performance

### Detection Metrics
- **mAP@IoU=0.5**: Achieved on validation set
- **mAP@IoU=0.75**: Achieved on validation set
- **Inference Speed**: ~2-3 FPS on GTX 1660 Ti (6GB)
- **Memory Usage**: <6GB VRAM during training

### Classification Accuracy
- **String Accuracy**: Percentage of correctly decoded QR values
- **Type Classification**: Accuracy of medical QR type classification
- **Categories**: Batch_Number, Manufacturer_Code, Regulator/GTIN

### Hardware Requirements
- **Minimum**: 6GB GPU VRAM, 8GB RAM
- **Recommended**: 8GB+ GPU VRAM, 16GB RAM
- **CPU Fallback**: Supported but significantly slower

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```powershell
# Reduce batch size in config.py
BATCH_SIZE = 1
IMAGE_SIZE = 256

# Enable gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8
```

#### Missing Dependencies
```powershell
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### PyZBar Installation Issues
```powershell
# Windows: Install Visual C++ Redistributable
# Then reinstall pyzbar
pip uninstall pyzbar
pip install pyzbar
```

#### Model Not Found
```powershell
# Ensure training completed successfully
python main.py --mode train

# Check weights directory
ls runs/MED1C_Detection/weights/
```

### Performance Optimization

#### For Limited Hardware
```python
# In config.py
BATCH_SIZE = 1
IMAGE_SIZE = 256
MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 8
```

#### For High-Performance Systems
```python
# In config.py  
BATCH_SIZE = 8
IMAGE_SIZE = 640
LEARNING_RATE = 1e-3
GRADIENT_ACCUMULATION_STEPS = 1
```

## Contributing

We welcome contributions to improve the MED1C system:

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update README for significant changes

### Areas for Contribution
- Additional QR code enhancement techniques
- Support for more medical QR formats
- Performance optimizations
- Mobile/edge device deployment
- Additional evaluation metrics

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

**Developed for medical QR code detection and classification in industrial applications.**