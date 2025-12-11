# Interpreting Breath Data for Health Screening

This repository contains a deep learning project for binary classification of breath sensor measurements (diabetic vs. healthy) using feedforward neural networks with an autoencoder bottleneck architecture. The work was developed and evaluated using the e-Nose Sensor Dataset for Predicting Human Diseases and executed primarily in a Kaggle environment.

> **Important**
> The Jupyter notebook contains detailed justifications, exploratory analysis, and interpretation alongside code. Please refer to the plot renderings in `results/plots/` or run them directly in Kaggle.

**Kaggle Notebook**: https://github.com/walmsley-lab/interpreting-breath-data-for-health-screening

## Problem Overview

The task is to classify breath samples captured by a six-sensor electronic nose array as either indicating diabetic or healthy status. This is a supervised tabular classification problem that requires extracting meaningful multivariate patterns from low-dimensional, static sensor measurements while preserving interpretability and handling potential class overlap.

Unlike image or sequence data, breath sensor readings lack spatial or temporal structure—sensor adjacency is arbitrary. This motivates the use of feedforward architectures rather than CNNs or RNNs, allowing inter-sensor relationships to be learned directly from the data without imposing inappropriate inductive biases.

## Repository Structure

```
interpreting-breath-data-for-health-screening/
├── notebook/                    # Kaggle-compatible experiment notebook
├── results/
│   ├── model_comparison.csv     # Aggregated evaluation results
│   ├── plots/                   # Saved EDA and performance figures
│   └── summary.md               # Short experiment summary
├── models/
│   └── enose_model.pth          # Trained model checkpoint
├── README.md
└── requirements.txt
```

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [e-Nose Sensor Dataset for Predicting Human Diseases](https://www.kaggle.com/datasets) (Kaggle) |
| **Samples** | ~1,000 |
| **Features** | 6 metal-oxide gas sensors |
| **Classes** | Normal, Diabetes (balanced) |
| **Sensors** | `TGS2610`, `TGS2611`, `TGS2600`, `TGS2602`, `TGS826`, `TGS2620` |

Each sensor exhibits cross-sensitivity to overlapping subsets of volatile organic compounds (VOCs), meaning informative signal arises from collective response patterns rather than individual measurements.

## Model Architecture

A combined **feedforward neural network with autoencoder bottleneck** (`ENoseNet`) was selected based on:

- Low intrinsic dimensionality (~3 principal components capture majority of variance)
- Static, unordered sensor measurements (no spatial/temporal structure)
- Need for interpretable latent representations

### Architecture Diagram

```
Input (6 sensors)
       │
       ▼
┌─────────────────┐
│  Encoder        │
│  6 → 32 → 16 → 8│
└────────┬────────┘
         │
    Bottleneck (8)
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│Decoder│ │Classifier │
│8→16→32│ │  8 → 1    │
│  → 6  │ │ (sigmoid) │
└───────┘ └───────────┘
```

### Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Architecture** | Feedforward + Autoencoder | No spatial/temporal structure in data |
| **Bottleneck** | 8 dimensions | Matches observed intrinsic dimensionality |
| **Regularization** | Dropout (0.3), L2 (0.01), BatchNorm | Prevent overfitting on modest sample size |
| **Loss Function** | $L = L_{\text{cls}} + \alpha \cdot L_{\text{recon}}$ | Joint classification and reconstruction |
| **Optimizer** | SGD with momentum (0.9) | More stable convergence than Adam |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Train/Val/Test Split** | 60% / 20% / 20% (stratified) |
| **Batch Size** | 32 |
| **Learning Rate** | 0.01 |
| **Weight Decay** | 0.01 |
| **Reconstruction Weight** ($\alpha$) | 0.3 |
| **Early Stopping Patience** | 15–20 epochs |
| **Cross-Validation** | 5-fold stratified |

## Key Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 1.000 |
| **AUC-ROC** | 1.000 |
| **Precision** | 1.000 |
| **Recall** | 1.000 |
| **F1 Score** | 1.000 |

### Cross-Validation Summary (5-Fold)

| Metric | Mean ± Std |
|--------|------------|
| **Accuracy** | $0.998 \pm 0.003$ |
| **AUC** | $1.000 \pm 0.001$ |

### Representation Quality

| Metric | Value |
|--------|-------|
| **Mean Reconstruction MSE** | ~0.02 |
| **Mean Reconstruction R²** | >0.90 |
| **Silhouette Score (Raw → Learned)** | Improved by ~0.3 |

> **Note**: Perfect classification performance warrants cautious interpretation. Results may reflect limited sample diversity rather than true population-level generalizability. External validation on independent cohorts is required.

## Results and Visualizations

All plots generated during analysis are saved under `results/plots/`. These include:

| Figure | Description |
|--------|-------------|
| **Sensor Distributions** | Raw sensor histograms by diagnosis class |
| **Effect Sizes** | Cohen's d for each sensor |
| **PCA/t-SNE Projections** | Dimensionality reduction of raw and learned spaces |
| **Training Convergence** | Loss and accuracy curves for SGD vs Adam |
| **ROC Curve** | Receiver operating characteristic with optimal threshold |
| **Confusion Matrix** | Test set classification results |
| **Reconstruction Analysis** | Per-sensor R² and error distributions |
| **Feature Interpretation** | Input–feature correlations and gradient-based importance |

## How to Run

The primary workflow is designed to run inside **Kaggle**.

1. Upload the repository notebook to Kaggle or open the existing project
2. Attach the [e-Nose Sensor Dataset](https://www.kaggle.com/datasets) 
3. Run all cells sequentially
4. Download outputs from `/kaggle/working/results`

### Local Execution

```bash
# Clone repository
git clone https://github.com/walmsley-lab/interpreting-breath-data-for-health-screening.git
cd interpreting-breath-data-for-health-screening

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebook/breath_data_analysis.ipynb
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- CUDA-compatible GPU (optional, CPU supported)

## Interpretability Highlights

The bottleneck architecture enables post-hoc interpretation of learned features:

| Analysis | Finding |
|----------|---------|
| **Dominant Sensors** | TGS2611 and TGS2610 show largest effect sizes and gradient importance |
| **Correlation Shifts** | Disease alters inter-sensor correlation structure |
| **Latent Features** | Subset of bottleneck dimensions show strong class-dependent activation |
| **Putative Compounds** | Acetone, methane (diabetes markers) align with sensor sensitivity profiles |

Interpretations are **hypothesis-generating** and do not constitute direct biomarker identification.

## Limitations

- Single dataset with binary labels and modest sample size (~1,000)
- Confounders (diet, medications, comorbidities) not controlled
- Sensor drift not explicitly modeled
- No external validation on independent cohorts
- Perfect performance may reflect limited population heterogeneity

## Future Directions

- External validation across diverse populations and acquisition settings
- Integration with complementary modalities (GC–MS, clinical data)
- Temporal modeling of sensor response dynamics
- Multi-class disease phenotyping
- Sensor drift calibration and transfer learning

## Notes

- All evaluation metrics computed on held-out test split
- Early stopping used to prevent overfitting
- Preprocessing performed independently within each CV fold to prevent leakage
- Model checkpoint includes scaler parameters for deployment consistency
