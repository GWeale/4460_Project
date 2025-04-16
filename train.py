import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_prep import prepare_data
from model import SpatialBERTModel, WindowGenerator, WindowDataset, collate_windows

def parse_args():
    parser = argparse.ArgumentParser(description='Train Spatial-BERT model on melanoma data')
    
    # Data paths
    parser.add_argument('--metadata_path', type=str, default='metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--cell_data_path', type=str, default='Melanoma_data.csv', help='Path to cell data CSV')
    
    # Data processing
    parser.add_argument('--cell_type_col', type=str, default='Cell_Type_Common', help='Column to use for cell types')
    parser.add_argument('--os_threshold', type=float, default=None, help='Threshold for high/low survival (default: median)')
    parser.add_argument('--apply_batch_corr', action='store_true', help='Apply batch correction')
    
    # Windowing
    parser.add_argument('--k_neighbors', type=int, default=20, help='Number of neighbors per window')
    parser.add_argument('--windows_per_sample', type=int, default=500, help='Number of windows per sample')
    parser.add_argument('--max_position', type=int, default=1000, help='Maximum position value for positional encoding')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_global_features', action='store_true', help='Use global features')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--model_name', type=str, default='spatial_bert', help='Model name')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def prepare_global_features(metadata_df, args):
    """
    Prepare global features from metadata
    
    Parameters:
    - metadata_df: DataFrame with patient metadata
    
    Returns:
    - global_features_dict: Dict mapping donor to global features
    - global_feature_dim: Dimension of global features
    """
    if not args.use_global_features:
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

def train_epoch(model, dataloader, optimizer, criterion, device, global_features_dict=None):
    """
    Train the model for one epoch
    
    Parameters:
    - model: Model to train
    - dataloader: DataLoader with training data
    - optimizer: Optimizer
    - criterion: Loss function
    - device: Device to use
    - global_features_dict: Dict mapping donor to global features
    
    Returns:
    - loss: Average loss for the epoch
    - outputs: List of model outputs
    - targets: List of target labels
    """
    model.train()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
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
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, np.array(all_outputs), np.array(all_targets)

def validate(model, dataloader, criterion, device, global_features_dict=None):
    """
    Validate the model
    
    Parameters:
    - model: Model to validate
    - dataloader: DataLoader with validation data
    - criterion: Loss function
    - device: Device to use
    - global_features_dict: Dict mapping donor to global features
    
    Returns:
    - loss: Average loss for the validation set
    - outputs: List of model outputs
    - targets: List of target labels
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    all_group_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
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
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)
            
            # Track loss and predictions
            total_loss += loss.item()
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
            
            if 'group_id' in batch:
                all_group_ids.extend(batch['group_id'])
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # If group_ids are available, we can aggregate predictions by sample
    if all_group_ids:
        # Convert outputs and targets to numpy arrays
        outputs_np = np.array(all_outputs).squeeze()
        targets_np = np.array(all_targets)
        
        # Create a DataFrame with outputs, targets, and group_ids
        df = pd.DataFrame({
            'output': outputs_np,
            'target': targets_np,
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
            'donor_accuracy': accuracy_score(donor_targets, donor_preds),
            'donor_f1': f1_score(donor_targets, donor_preds, zero_division=0),
            'donor_precision': precision_score(donor_targets, donor_preds, zero_division=0),
            'donor_recall': recall_score(donor_targets, donor_preds, zero_division=0)
        }
        
        if len(np.unique(donor_targets)) > 1:
            donor_metrics['donor_auc'] = roc_auc_score(donor_targets, donor_outputs)
        else:
            donor_metrics['donor_auc'] = float('nan')
        
        return avg_loss, np.array(all_outputs), np.array(all_targets), donor_metrics
    
    return avg_loss, np.array(all_outputs), np.array(all_targets), None

def calculate_metrics(outputs, targets):
    """
    Calculate metrics for model evaluation
    
    Parameters:
    - outputs: Model outputs
    - targets: Target labels
    
    Returns:
    - metrics: Dict with calculated metrics
    """
    # Convert outputs to predictions (0 or 1)
    preds = (np.array(outputs) > 0).astype(int).squeeze()
    targets = np.array(targets).squeeze()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'f1': f1_score(targets, preds, zero_division=0),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0)
    }
    
    # Calculate AUC if possible (requires both classes to be present)
    if len(np.unique(targets)) > 1:
        metrics['auc'] = roc_auc_score(targets, np.array(outputs).squeeze())
    else:
        metrics['auc'] = float('nan')
    
    return metrics

def save_model(model, optimizer, epoch, args, metrics, model_path):
    """
    Save model checkpoint
    
    Parameters:
    - model: Model to save
    - optimizer: Optimizer state
    - epoch: Current epoch
    - args: Command-line arguments
    - metrics: Validation metrics
    - model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'metrics': metrics
    }
    
    # Save checkpoint
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data(
        metadata_path=args.metadata_path,
        cell_data_path=args.cell_data_path,
        os_threshold=args.os_threshold,
        apply_batch_corr=args.apply_batch_corr
    )
    
    # Prepare global features if needed
    global_features_dict, global_feature_dim = prepare_global_features(data_dict['metadata_full'], args)
    
    # Initialize window generator
    print("Initializing window generator...")
    window_generator = WindowGenerator(
        k_neighbors=args.k_neighbors,
        max_position=args.max_position,
        cell_type_col=args.cell_type_col,
        device=args.device
    )
    
    # Fit window generator to training data
    window_generator.fit(data_dict['train_cells'])
    
    # Generate windows for training data
    print("Generating windows for training data...")
    train_windows = window_generator.generate_windows(
        data_dict['train_cells_normalized'],
        num_windows_per_sample=args.windows_per_sample
    )
    
    # Generate windows for validation data
    print("Generating windows for validation data...")
    val_windows = window_generator.generate_windows(
        data_dict['val_cells_normalized'],
        num_windows_per_sample=args.windows_per_sample
    )
    
    # Add [CLS] tokens
    train_windows = window_generator.add_cls_tokens(train_windows)
    val_windows = window_generator.add_cls_tokens(val_windows)
    
    # Create datasets and data loaders
    train_dataset = WindowDataset(train_windows)
    val_dataset = WindowDataset(val_windows)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_windows
    )
    
    # Initialize model
    print("Initializing model...")
    model = SpatialBERTModel(
        marker_dim=len(window_generator.marker_cols),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_cell_types=window_generator.num_cell_types + 1,  # +1 for [CLS] token
        cell_type_map=window_generator.cell_type_map,
        max_position=args.max_position,
        use_global_features=args.use_global_features,
        global_feature_dim=global_feature_dim
    )
    
    # Move model to device
    model = model.to(args.device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    best_val_auc = 0
    patience_counter = 0
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_aucs = []
    donor_aucs = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss, train_outputs, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, args.device, global_features_dict
        )
        
        # Validate
        val_loss, val_outputs, val_targets, donor_metrics = validate(
            model, val_loader, criterion, args.device, global_features_dict
        )
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_outputs, train_targets)
        val_metrics = calculate_metrics(val_outputs, val_targets)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_metrics['auc'])
        
        if donor_metrics is not None:
            donor_aucs.append(donor_metrics['donor_auc'])
        
        # Print metrics
        print(f"Train loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        if donor_metrics is not None:
            print(f"Donor AUC: {donor_metrics['donor_auc']:.4f}, "
                  f"Donor Accuracy: {donor_metrics['donor_accuracy']:.4f}")
        
        # Check if this is the best model so far
        current_val_metric = donor_metrics['donor_auc'] if donor_metrics is not None else val_metrics['auc']
        
        if current_val_metric > best_val_auc:
            best_val_auc = current_val_metric
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, f"{args.model_name}_best.pt")
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics={**val_metrics, **(donor_metrics or {})},
                model_path=best_model_path
            )
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= args.early_stopping:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save latest model
        latest_model_path = os.path.join(args.output_dir, f"{args.model_name}_latest.pt")
        save_model(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            metrics={**val_metrics, **(donor_metrics or {})},
            model_path=latest_model_path
        )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(val_aucs, label='Window-level AUC')
    if donor_aucs:
        plt.plot(donor_aucs, label='Donor-level AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC')
    plt.savefig(os.path.join(args.output_dir, 'auc_curve.png'))
    
    print("Training complete!")

if __name__ == "__main__":
    main() 