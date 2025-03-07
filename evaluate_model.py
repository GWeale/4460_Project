import argparse
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from cell_bert_model import CellBERTModel, WindowAggregator, CellWindowDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cell-BERT model for melanoma survival prediction')
    
    parser.add_argument('--data_path', type=str, default='protein_data.h5ad',
                        help='Path to AnnData (.h5ad) file with protein expression data')
    parser.add_argument('--bert_model_path', type=str, default='cell_bert_model.pt',
                        help='Path to trained Cell-BERT model')
    parser.add_argument('--aggregator_path', type=str, default='window_aggregator.pt',
                        help='Path to trained window aggregator model')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Number of cells in each window (neighborhood)')
    parser.add_argument('--survival_threshold', type=int, default=24,
                        help='Threshold in months for binary survival classification')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension for the model')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--output_prefix', type=str, default='results',
                        help='Prefix for output result files')
    
    return parser.parse_args()

def evaluate_model(cell_bert_model, window_aggregator, test_loader, device):
    cell_bert_model.eval()
    window_aggregator.eval()
    
    patient_cls_tokens = {}
    patient_true_labels = {}
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)
            labels = batch['label'].squeeze().cpu().numpy()
            patient_ids = batch['patient_id']
            
            # Forward pass for each window
            cls_outputs, _ = cell_bert_model(features, x_coords, y_coords)
            
            # Group by patient
            for i, patient_id in enumerate(patient_ids):
                if patient_id not in patient_cls_tokens:
                    patient_cls_tokens[patient_id] = []
                    patient_true_labels[patient_id] = labels[i]
                
                patient_cls_tokens[patient_id].append(cls_outputs[i])
    
    patient_ids = []
    true_labels = []
    pred_labels = []
    pred_probs = []
    
    for patient_id, cls_tokens in patient_cls_tokens.items():
        cls_tokens_tensor = torch.stack(cls_tokens)
        logits = window_aggregator(cls_tokens_tensor)
        
        # Get prediction
        probs = torch.softmax(logits, dim=0)
        pred = torch.argmax(logits).item()
        
        patient_ids.append(patient_id)
        true_labels.append(patient_true_labels[patient_id])
        pred_labels.append(pred)
        pred_probs.append(probs[1].item())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    auc = roc_auc_score(true_labels, pred_probs)
    cm = confusion_matrix(true_labels, pred_labels)
    
    report = classification_report(true_labels, pred_labels, 
                                   target_names=['Low Survival', 'High Survival'])
    
    return {
        'patient_ids': patient_ids,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'pred_probs': pred_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_results(results, output_prefix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Survival', 'High Survival'],
                yticklabels=['Low Survival', 'High Survival'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['true_labels'], results['pred_probs'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {results["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_roc_curve.png")
    except:
        print("Couldn't create ROC curve")
    
    with open(f"{output_prefix}_metrics.txt", 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data_path}")
    adata = sc.read(args.data_path)
    
    print(f"Creating dataset with window size {args.window_size}")
    dataset = CellWindowDataset(adata, window_size=args.window_size, 
                                survival_threshold=args.survival_threshold)
    
    dataset_size = len(dataset)
    test_size = int(args.test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    print(f"Dataset split: {train_size} training samples, {test_size} test samples")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    cell_feature_dim = adata.X.shape[1]
    
    print(f"Creating Cell-BERT model with embedding dimension {args.embedding_dim}")
    cell_bert_model = CellBERTModel(
        cell_feature_dim, 
        d_model=args.embedding_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.embedding_dim * 2,
        num_layers=args.num_layers
    )
    
    window_aggregator = WindowAggregator(args.embedding_dim, hidden_dim=args.embedding_dim // 2)
    
    print(f"Loading model weights from {args.bert_model_path} and {args.aggregator_path}")
    cell_bert_model.load_state_dict(torch.load(args.bert_model_path))
    window_aggregator.load_state_dict(torch.load(args.aggregator_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    print("Evaluating model on test set...")
    results = evaluate_model(cell_bert_model, window_aggregator, test_loader, device)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print(f"Saving results to {args.output_prefix}_*.png and {args.output_prefix}_metrics.txt")
    plot_results(results, args.output_prefix)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 