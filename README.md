A deep learning project that **reconstructs the physical configuration of a circular antenna array from its radiation pattern image**.
 
Given a polar plot of how an antenna radiates energy, the model predicts how many elements are placed on each of the 5 concentric rings.
 
---
 
## What is this project about?
 
A **Circular Antenna Array** consists of multiple rings, each ring holding a number of radiating elements. The arrangement of elements determines the antenna's *radiation pattern* — the shape of how the antenna sends/receives signals in different directions.
 
This project asks the reverse question:
 
> **"Given a radiation pattern image, can we figure out the antenna configuration that produced it?"**
 
We answer this with a CNN trained on synthetically generated data.
 
---
 
## Project Structure
 
```
antenna_project/
│
├── 1_generate_dataset.py   # Generate synthetic antenna images + labels
├── 2_split_dataset.py      # Split into train / test sets
├── 3_train.py              # Train the CNN model
├── 4_evaluate.py           # Evaluate and visualize results
├── cnn_model.py            # CNN architecture definition
│
├── data/
│   ├── train/              # Training images
│   ├── test/               # Test images
│   ├── train_labels.npy    # Labels for training set
│   └── test_labels.npy     # Labels for test set
│
├── dataset_polar/          # Raw generated images + labels.npy
├── cnn_model.pth           # Saved model weights (after training)
└── radiation_comparison.png
```
 
---
 
## How to Run (Step by Step)
 
### 1. Install dependencies
 
```bash
pip install torch torchvision numpy matplotlib pillow scipy
```
 
### 2. Generate the dataset
 
```bash
python 1_generate_dataset.py
```
 
Creates 5000 antenna radiation pattern images in `dataset_polar/`.
 
### 3. Split into train / test
 
```bash
python 2_split_dataset.py
```
 
Copies images into `data/train/` (70%) and `data/test/` (30%).
 
### 4. Train the model
 
```bash
python 3_train.py
```
 
Trains the CNN for 20 epochs. Saves the best model to `cnn_model.pth`.
 
### 5. Evaluate
 
```bash
python 4_evaluate.py
```
 
Reports MAE per ring, exact-match accuracy, and saves a radiation pattern comparison plot.
 
---
 
## Model Architecture
 
The model is a simple CNN for **regression**:
 
```
Input: grayscale image 256×256
 
Conv(1→16) → ReLU → MaxPool    # extracts low-level features
Conv(16→32) → ReLU → MaxPool   # extracts mid-level features
Conv(32→64) → ReLU → MaxPool   # extracts high-level features
 
Flatten
Linear(65536 → 128) → ReLU
Linear(128 → 5)                # 5 outputs = element count per ring
```
 
**Why regression?** The output is a count (0–10 elements per ring), not a category. We use **MSE loss** to minimize the distance between predicted and true counts.
 
---
 
## Dataset Details
 
| Property | Value |
|---|---|
| Total samples | 5000 |
| Train / Test split | 70% / 30% |
| Image size | 256 × 256 px (grayscale) |
| Label shape | (5,) integers in [0, 10] |
| Frequency | 2.45 GHz (WiFi band) |
| Rings | 5 |
| Max elements per ring | 10 |
 
Each image is a **polar plot** of the normalized array factor (in dB), simulated with a random steering angle.
 
---
 
## Metrics
 
- **MAE (Mean Absolute Error)**: average error in predicted element count per ring
- **Exact Match**: percentage of samples where all 5 rings are predicted correctly
- **Radiation pattern comparison**: visual overlap of true vs predicted antenna pattern
 
---
 
## Skills demonstrated
 
- Synthetic data generation for a physics-based problem
- CNN design for regression (not classification)
- PyTorch training loop with model checkpointing
- Signal processing (array factor, polar plots, dB scale)
- End-to-end ML pipeline: data → model → evaluation
 
---
 
## Possible improvements
 
- Add a validation set during training to monitor overfitting
- Use data augmentation (rotate/flip images)
- Try a deeper CNN or use transfer learning (ResNet)
- Frame it as classification (discretize element counts)
- Deploy as a simple web app using Streamlit
 
---
 
## Author
 
Built as a university project exploring the intersection of deep learning and antenna engineering.
