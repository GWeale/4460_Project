# Cell-BERT: A BERT-style model for melanoma cell data analysis

## Project Overview
This repository contains a BERT-style transformer model for analyzing CODEX multiplexed imaging data from melanoma tumors to predict patient survival. The model uses the spatial arrangement and protein expression levels of cells to make predictions about patient outcomes.

## Data
The analysis uses three key data files:
- `Melanoma_data.csv`: Contains cell protein expression data
- `Day3_Markers_Dryad.csv`: Contains marker information
- `metadata.csv`: Contains patient metadata including survival information

## Model Architecture
Cell-BERT processes cells in windows (neighborhoods) to capture local cellular interactions. Key components:

- **Window-based Processing**: Analyzes cell neighborhoods rather than individual cells
- **Dual Positional Embeddings**: Incorporates both absolute spatial positions and relative positions within the neighborhood
- **Transformer Architecture**: Uses multi-head self-attention to model relationships between cells
- **CLS Token Aggregation**: A special classification token aggregates information from all cells in a window
- **Window Aggregation**: Multiple windows from a patient are aggregated to make a final prediction

## Optimized Implementation
The model has both a full version and a lighter version for faster training:

- **Standard Cell-BERT Model**:
  - 128-dimensional embeddings
  - 4 attention heads
  - 3 transformer layers

- **Light Cell-BERT Model**:
  - 64-dimensional embeddings
  - 2 attention heads 
  - 2 transformer layers
  - Achieves similar results with much faster training time

## Usage

### Training
```bash
python train_model.py --melanoma_data Melanoma_data.csv \
                     --markers_data Day3_Markers_Dryad.csv \
                     --metadata metadata.csv \
                     --sample_ratio 0.02 \
                     --window_size 10 \
                     --max_windows 50 \
                     --batch_size 64 \
                     --epochs 5
```

### Evaluation
```bash
python evaluate_model.py --melanoma_data Melanoma_data.csv \
                        --markers_data Day3_Markers_Dryad.csv \
                        --metadata metadata.csv \
                        --sample_ratio 0.02 \
                        --model_path cell_bert_model.pt \
                        --aggregator_path cell_bert_aggregator.pt
```

### Visualization
```bash
python visualize_cell_bert.py
```

## Optimization Parameters

### Performance vs Speed Tradeoffs
- **Sample Ratio**: Reduce from 0.1 to 0.02 for faster training with minimal impact on model learning
- **Window Size**: Reduced from 15 to 10 cells per window
- **Max Windows**: Limit to 50 windows per patient to prevent memory issues
- **Model Size**: Use the LightCellBERTModel for much faster training
- **Learning Rate**: Increased to 5e-3 for faster convergence
- **Epochs**: Reduced to 5 epochs for faster training

## Requirements
- Python 3.8+
- PyTorch
- scanpy
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Results
The model achieves effective prediction of patient survival based on cell spatial arrangements and protein expressions. Evaluation metrics include accuracy, precision, recall, F1 score, and AUC score.

## Implementation Details
- `cell_bert_model.py`: Core model implementation including data loading and model architecture
- `train_model.py`: Script for training the model
- `evaluate_model.py`: Script for evaluating model performance
- `visualize_cell_bert.py`: Script for visualizing data, embeddings, and model attention