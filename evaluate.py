import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_prep import prepare_data
from model import SpatialBERTModel, WindowGenerator, WindowDataset, collate_windows
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Spatial-BERT model on melanoma data')
    
    # Data paths
    parser.add_argument('--metadata_path', type=str, default='metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--cell_data_path', type=str, default='Melanoma_data.csv', help='Path to cell data CSV')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

def load_model(model_path, device):
    """
    Load trained model from checkpoint
    
    Parameters:
    - model_path: Path to model checkpoint
    - device: Device to load model to
    
    Returns:
    - model: Loaded model
    - args: Arguments used for training
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get arguments used for training
    train_args = argparse.Namespace(**checkpoint['args'])
    
    # Get marker_dim from checkpoint or use default value
    marker_dim = None
    if 'marker_dim' in checkpoint['args']:
        marker_dim = checkpoint['args']['marker_dim']
    else:
        # Default to 58 markers (as seen in the training output)
        marker_dim = 58
        print(f"Warning: marker_dim not found in checkpoint, using default value: {marker_dim}")
    
    # Get cell_type parameters
    num_cell_types = checkpoint['args'].get('num_cell_types', None)
    if num_cell_types is None:
        # The model was trained with cell types, assume a default value
        num_cell_types = 20  # This is a reasonable default
        print(f"Warning: num_cell_types not found in checkpoint, using default value: {num_cell_types}")
    
    # Initialize model with the same parameters
    model = SpatialBERTModel(
        marker_dim=marker_dim,
        hidden_dim=train_args.hidden_dim,
        num_heads=train_args.num_heads,
        num_layers=train_args.num_layers,
        dropout=train_args.dropout,
        max_position=train_args.max_position,
        use_global_features=train_args.use_global_features,
        global_feature_dim=checkpoint['args'].get('global_feature_dim', 0),
        num_cell_types=num_cell_types,
        cell_type_map=None
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Training metrics: {checkpoint['metrics']}")
    
    return model, train_args

def prepare_global_features(metadata_df, use_global_features=False):
    """
    Prepare global features from metadata
    
    Parameters:
    - metadata_df: DataFrame with patient metadata
    - use_global_features: Whether to use global features
    
    Returns:
    - global_features_dict: Dict mapping donor to global features
    - global_feature_dim: Dimension of global features
    """
    if not use_global_features:
        return None, 0
    
    # Select features to use
    global_feature_cols = ['Age_baseline', 'Biopsy_time', 'Breslow_thickness', 'Ulceration', 'Gender']
    
    # Create a copy to avoid modifying the original
    metadata_copy = metadata_df.copy()
    
    # Encode categorical variables
    if 'Gender' in global_feature_cols:
        metadata_copy['Gender'] = metadata_copy['Gender'].map({'male': 0, 'female': 1})
    
    # Handle missing values
    for col in global_feature_cols:
        if col in metadata_copy.columns:
            # Fill missing values with median
            if metadata_copy[col].dtype in [np.float64, np.int64]:
                metadata_copy[col] = metadata_copy[col].fillna(metadata_copy[col].median())
            else:
                # For categorical, fill with mode
                metadata_copy[col] = metadata_copy[col].fillna(metadata_copy[col].mode()[0])
    
    # Normalize numerical features
    for col in global_feature_cols:
        if col in metadata_copy.columns and metadata_copy[col].dtype in [np.float64, np.int64]:
            # Skip binary columns
            if metadata_copy[col].nunique() > 2:
                metadata_copy[col] = (metadata_copy[col] - metadata_copy[col].mean()) / metadata_copy[col].std()
    
    # Create dict mapping donor to global features
    global_features_dict = {}
    for _, row in metadata_copy.iterrows():
        global_features_dict[row['donor']] = row[global_feature_cols].values.astype(np.float32)
    
    global_feature_dim = len(global_feature_cols)
    
    return global_features_dict, global_feature_dim

def evaluate_model(model, dataloader, device, global_features_dict=None):
    """
    Evaluate model on a dataset
    
    Parameters:
    - model: Model to evaluate
    - dataloader: DataLoader with evaluation data
    - device: Device to use
    - global_features_dict: Dict mapping donor to global features
    
    Returns:
    - results: Dict with evaluation results
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_group_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            marker_values = batch['marker_values'].to(device)
            rel_positions = batch['rel_positions'].to(device)
            labels = batch['label'].to(device) if 'label' in batch else None
            cell_types = batch['cell_types'].to(device) if 'cell_types' in batch else None
            
            # Prepare global features if used
            global_features = None
            if global_features_dict is not None and 'group_id' in batch:
                # Extract donor from group_id (first element of tuple)
                donors = [group_id[0] for group_id in batch['group_id']]
                global_features = torch.tensor(
                    np.array([global_features_dict[donor] for donor in donors]),
                    dtype=torch.float32,
                    device=device
                )
            
            # Forward pass
            outputs = model(
                marker_values=marker_values, 
                rel_positions=rel_positions,
                cell_types=cell_types,
                global_features=global_features
            )
            
            # Store outputs and targets
            all_outputs.extend(outputs.cpu().numpy())
            if labels is not None:
                all_targets.extend(labels.cpu().numpy())
            
            if 'group_id' in batch:
                all_group_ids.extend(batch['group_id'])
    
    # Convert outputs to numpy arrays
    all_outputs = np.array(all_outputs).squeeze()
    
    # If no targets, return only outputs
    if not all_targets:
        return {'outputs': all_outputs, 'group_ids': all_group_ids}
    
    # Convert targets to numpy arrays
    all_targets = np.array(all_targets).squeeze()
    
    # Calculate window-level metrics
    window_preds = (all_outputs > 0).astype(int)
    window_metrics = {
        'accuracy': accuracy_score(all_targets, window_preds),
        'f1': f1_score(all_targets, window_preds, zero_division=0),
        'precision': precision_score(all_targets, window_preds, zero_division=0),
        'recall': recall_score(all_targets, window_preds, zero_division=0)
    }
    
    # Calculate AUC if possible
    if len(np.unique(all_targets)) > 1:
        window_metrics['auc'] = roc_auc_score(all_targets, all_outputs)
    else:
        window_metrics['auc'] = float('nan')
    
    # If group_ids are available, calculate donor-level metrics
    donor_metrics = None
    donor_df = None
    
    if all_group_ids:
        # Create a DataFrame with outputs, targets, and group_ids
        df = pd.DataFrame({
            'output': all_outputs,
            'target': all_targets,
            'donor': [group_id[0] for group_id in all_group_ids]
        })
        
        # Aggregate by donor (taking mean of outputs and first target)
        donor_df = df.groupby('donor').agg({
            'output': 'mean',
            'target': 'first'
        })
        
        # Calculate donor-level metrics
        donor_outputs = donor_df['output'].values
        donor_targets = donor_df['target'].values
        
        donor_preds = (donor_outputs > 0).astype(int)
        donor_metrics = {
            'accuracy': accuracy_score(donor_targets, donor_preds),
            'f1': f1_score(donor_targets, donor_preds, zero_division=0),
            'precision': precision_score(donor_targets, donor_preds, zero_division=0),
            'recall': recall_score(donor_targets, donor_preds, zero_division=0)
        }
        
        if len(np.unique(donor_targets)) > 1:
            donor_outputs = np.nan_to_num(donor_outputs)
            donor_targets = np.nan_to_num(donor_targets)
            donor_metrics['auc'] = roc_auc_score(donor_targets, donor_outputs)
        else:
            donor_metrics['auc'] = float('nan')
    
    return {
        'outputs': all_outputs,
        'targets': all_targets,
        'group_ids': all_group_ids,
        'window_metrics': window_metrics,
        'donor_metrics': donor_metrics,
        'donor_df': donor_df
    }

def plot_roc_curve(targets, outputs, output_path):
    """
    Plot ROC curve
    
    Parameters:
    - targets: Target labels
    - outputs: Model outputs
    - output_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)

def plot_confusion_matrix(targets, preds, output_path):
    """
    Plot confusion matrix
    
    Parameters:
    - targets: Target labels
    - preds: Model predictions
    - output_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, train_args = load_model(args.model_path, args.device)
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data(
        metadata_path=args.metadata_path,
        cell_data_path=args.cell_data_path,
        os_threshold=train_args.os_threshold if hasattr(train_args, 'os_threshold') else None,
        apply_batch_corr=train_args.apply_batch_corr if hasattr(train_args, 'apply_batch_corr') else False
    )
    
    # Prepare global features if needed
    global_features_dict, global_feature_dim = prepare_global_features(
        data_dict['metadata_full'],
        use_global_features=train_args.use_global_features if hasattr(train_args, 'use_global_features') else False
    )
    
    # Initialize window generator
    print("Initializing window generator...")
    window_generator = WindowGenerator(
        k_neighbors=train_args.k_neighbors if hasattr(train_args, 'k_neighbors') else 20,
        max_position=train_args.max_position if hasattr(train_args, 'max_position') else 1000,
        cell_type_col=train_args.cell_type_col if hasattr(train_args, 'cell_type_col') else 'Cell_Type_Common',
        device=args.device
    )
    
    # Fit window generator to training data (needed for cell type mapping)
    window_generator.fit(data_dict['train_cells'])
    
    # Generate windows for test data
    print("Generating windows for test data...")
    test_windows = window_generator.generate_windows(
        data_dict['test_cells_normalized'],
        num_windows_per_sample=train_args.windows_per_sample if hasattr(train_args, 'windows_per_sample') else 500
    )
    
    # Add [CLS] tokens
    test_windows = window_generator.add_cls_tokens(test_windows)
    
    # Create dataset and data loader
    test_dataset = WindowDataset(test_windows)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_windows
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, args.device, global_features_dict)
    
    # Print metrics
    print("Test metrics:")
    print("Window-level metrics:")
    for metric, value in results['window_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    if results['donor_metrics'] is not None:
        print("Donor-level metrics:")
        for metric, value in results['donor_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    if results['donor_df'] is not None:
        results['donor_df'].to_csv(os.path.join(args.output_dir, 'donor_predictions.csv'))
    
    # Plot ROC curve
    if 'auc' in results['window_metrics'] and not np.isnan(results['window_metrics']['auc']):
        plot_roc_curve(
            results['targets'],
            results['outputs'],
            os.path.join(args.output_dir, 'roc_curve_window.png')
        )
    
    if results['donor_metrics'] is not None and 'auc' in results['donor_metrics'] and not np.isnan(results['donor_metrics']['auc']):
        donor_outputs = results['donor_df']['output'].values
        donor_targets = results['donor_df']['target'].values
        plot_roc_curve(
            donor_targets,
            donor_outputs,
            os.path.join(args.output_dir, 'roc_curve_donor.png')
        )
    
    # Plot confusion matrix
    window_preds = (results['outputs'] > 0).astype(int)
    plot_confusion_matrix(
        results['targets'],
        window_preds,
        os.path.join(args.output_dir, 'confusion_matrix_window.png')
    )
    
    if results['donor_metrics'] is not None:
        donor_outputs = results['donor_df']['output'].values
        donor_targets = results['donor_df']['target'].values
        donor_preds = (donor_outputs > 0).astype(int)
        plot_confusion_matrix(
            donor_targets,
            donor_preds,
            os.path.join(args.output_dir, 'confusion_matrix_donor.png')
        )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 