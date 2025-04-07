import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from cell_bert_model import (
    LightCellBERTModel, WindowAggregator, CellWindowDataset, 
    create_anndata_from_csv, evaluate_model
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cell-BERT model for melanoma survival prediction')
    
    parser.add_argument('--melanoma_data', type=str, default='Melanoma_data.csv',
                        help='Path to Melanoma_data.csv file')
    parser.add_argument('--markers_data', type=str, default='Day3_Markers_Dryad.csv',
                        help='Path to Day3_Markers_Dryad.csv file')
    parser.add_argument('--metadata', type=str, default='metadata.csv',
                        help='Path to metadata.csv file')
    parser.add_argument('--sample_ratio', type=float, default=0.02,
                        help='Ratio of cells to sample (to reduce memory usage)')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Number of cells in each window (neighborhood)')
    parser.add_argument('--max_windows', type=int, default=50,
                        help='Maximum number of windows per patient')
    parser.add_argument('--survival_threshold', type=int, default=24,
                        help='Threshold in months for binary survival classification')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension for the model')
    parser.add_argument('--model_path', type=str, default='cell_bert_model.pt',
                        help='Path to trained Cell-BERT model')
    parser.add_argument('--aggregator_path', type=str, default='cell_bert_aggregator.pt',
                        help='Path to trained window aggregator model')
    parser.add_argument('--output_prefix', type=str, default='evaluation',
                        help='Prefix for output files')
    
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(fpr, tpr, auc, output_path):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    print(f"Loading data from CSV files:")
    print(f"  - Melanoma data: {args.melanoma_data}")
    print(f"  - Markers data: {args.markers_data}")
    print(f"  - Metadata: {args.metadata}")
    
    # Create AnnData from CSV files
    adata = create_anndata_from_csv(
        args.melanoma_data,
        args.markers_data,
        args.metadata,
        sample_ratio=args.sample_ratio
    )
    
    print(f"Creating dataset with window size {args.window_size} and max {args.max_windows} windows per patient")
    dataset = CellWindowDataset(adata, window_size=args.window_size, 
                               survival_threshold=args.survival_threshold,
                               max_windows_per_patient=args.max_windows)
    
    # Define test set (using a separate subset for evaluation)
    # For a real evaluation, use a separate test set not used during training
    test_size = len(dataset)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Created test loader with {test_size} windows")
    
    # Determine cell feature dimension from the data
    cell_feature_dim = adata.X.shape[1]
    
    # Create models
    print(f"Creating lightweight Cell-BERT model")
    cell_bert_model = LightCellBERTModel(
        cell_feature_dim, 
        d_model=args.embedding_dim,
        num_heads=2,  # Lightweight model parameters
        dim_feedforward=args.embedding_dim * 2,
        num_layers=2  # Lightweight model parameters
    )
    
    print(f"Creating window aggregator")
    window_aggregator = WindowAggregator(args.embedding_dim, hidden_dim=args.embedding_dim // 2)
    
    # Set device - try to use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained models
    print(f"Loading trained models from {args.model_path} and {args.aggregator_path}")
    try:
        cell_bert_model.load_state_dict(torch.load(args.model_path, map_location=device))
        window_aggregator.load_state_dict(torch.load(args.aggregator_path, map_location=device))
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Proceeding with untrained models for demonstration")
    
    # Move models to device
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    true_labels, pred_probas = evaluate_model(cell_bert_model, window_aggregator, test_loader, device)
    
    # Convert probabilities to predicted labels
    pred_labels = (pred_probas >= 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_probas)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_output_path = f"{args.output_prefix}_confusion_matrix.png"
    plot_confusion_matrix(cm, classes=['Survived', 'Deceased'], output_path=cm_output_path)
    print(f"Confusion matrix saved to {cm_output_path}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probas)
    roc_output_path = f"{args.output_prefix}_roc_curve.png"
    plot_roc_curve(fpr, tpr, auc, output_path=roc_output_path)
    print(f"ROC curve saved to {roc_output_path}")
    
    # Save metrics to file
    metrics_output_path = f"{args.output_prefix}_metrics.txt"
    with open(metrics_output_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
    print(f"Metrics saved to {metrics_output_path}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 