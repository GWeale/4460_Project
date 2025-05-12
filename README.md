# Spatial-BERT for CODEX Melanoma Data

This project implements a BERT-style Transformer model for analyzing spatial proteomics data from CODEX imaging of human melanoma samples. The model is designed to predict patient survival outcomes (high vs. low overall survival) based on the spatial organization of cells within the tumor microenvironment.

## Overview

The Spatial-BERT approach uses a windowed attention mechanism to learn from local cell neighborhoods. For each cell in the dataset, we:

1. Extract its k-nearest neighbors based on spatial coordinates
2. Represent each cell using protein marker expression values and cell type information
3. Apply positional encoding based on the relative spatial positions
4. Feed this neighborhood through a BERT-style transformer with a [CLS] token
5. Use the [CLS] token representation to make predictions at the patient level

The model architecture combines local spatial information with global patient features to make survival predictions.

## Requirements

All dependencies and exact versions are listed in `requirements.txt`, generated from the current environment using `pip freeze`. To install:

```bash
pip install -r requirements.txt
```

## Data Format

The code expects two main data files:

1. `metadata.csv`: Patient-level information with columns:
   - `donor`: Patient identifier (matching cell data)
   - `OS`: Overall survival in months
   - `Deceased`: Binary indicator of death
   - Demographics and clinical variables (Age, Gender, etc.)

2. `Melanoma_data.csv`: Cell-level data with columns:
   - Marker columns (protein expression levels)
   - `x`, `y`: Spatial coordinates
   - `donor`: Patient identifier (matching metadata)
   - `Cell_Type_Common`: Cell type classification
   - Other metadata (region, filename, etc.)

## Code Structure

The codebase is organized as follows:

- `data_prep.py`: Data loading, preparation, and preprocessing
- `model.py`: Model implementation, windowing, and embedding logic
- `train.py`: Training script with argument parsing and logging
- `evaluate.py`: Evaluation script for model testing

## Usage

### Data Preparation

The data preparation pipeline in `data_prep.py` handles:
- Loading metadata and cell data
- Creating a binary target variable for survival
- Merging patient-level labels with cell data
- Identifying feature columns and cell types
- Splitting data at the patient level
- Optional batch correction
- Feature normalization (using a StandardScaler)

**Note:** The fitted scaler is saved to `output/scaler.pkl` after training. This scaler is loaded in `evaluate.py` to ensure consistent normalization between training and evaluation.

### Training

To train the model with default parameters:

```bash
python train.py --metadata_path metadata.csv --cell_data_path Melanoma_data.csv --output_dir output
```

Additional important parameters:

```bash
# Data processing
--cell_type_col Cell_Type_Common  # Column to use for cell types
--os_threshold 24                 # Threshold for high/low survival (default: median)
--apply_batch_corr                # Apply batch correction

# Windowing
--k_neighbors 20                  # Number of neighbors per window
--windows_per_sample 500          # Number of windows per sample
--max_position 1000               # Maximum position value for positional encoding

# Model
--hidden_dim 256                  # Hidden dimension for model
--num_heads 8                     # Number of attention heads
--num_layers 6                    # Number of transformer layers
--dropout 0.1                     # Dropout rate
--use_global_features             # Use global features from metadata

# Training
--batch_size 32                   # Batch size
--epochs 50                       # Number of epochs
--lr 1e-4                         # Learning rate
--weight_decay 1e-5               # Weight decay for optimizer
--early_stopping 10               # Patience for early stopping
--model_name spatial_bert         # Model name for checkpoint files
--output_dir output               # Output directory
--device cuda                     # Device to use (cuda or cpu)
--seed 42                         # Random seed
```

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model_path output/spatial_bert_best.pt --output_dir evaluation
```

The evaluation script will:
- Load the specified model checkpoint
- Generate windows for the test data
- Calculate window-level and patient-level metrics
- Plot ROC curves and confusion matrices
- Save detailed results to the output directory

## Extending to Mouse Data

This codebase can be adapted for mouse data by adjusting the feature columns and input data paths. The core model and windowing logic should work with minimal changes.

## Acknowledgments

This project was developed for the analysis of CODEX melanoma spatial proteomics data.