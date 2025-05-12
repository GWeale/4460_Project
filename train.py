import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
import scanpy as sc
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from data_prep import prepare_data, identify_feature_columns
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

def plot_attention_weights(attention_weights, window_coords, cell_types, inv_cell_type_map, output_path, title="Attention Weights"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Convert tensors to numpy and ensure correct shapes
    attention_weights = attention_weights.detach().cpu().numpy()
    window_coords = window_coords.detach().cpu().numpy()
    cell_types = cell_types.detach().cpu().numpy()
    
    # Debug prints
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Window coords shape: {window_coords.shape}")
    print(f"Cell types shape: {cell_types.shape}")
    
    num_heads = min(8, attention_weights.shape[0])  # Only plot first 8 heads
    unique_types = np.unique(cell_types[1:len(window_coords)])
    type_map = {t: i for i, t in enumerate(unique_types)}
    mapped_types = np.array([type_map[t] for t in cell_types[1:len(window_coords)]])
    
    print(f"Plotting {num_heads} attention heads for window...")
    
    for h in range(num_heads):
        # --- Spatial scatter with attention lines ---
        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(1, 20, figure=fig)
        ax = fig.add_subplot(gs[0, :18])  # Main plot takes up 18/20 of the width
        cbar_ax = fig.add_subplot(gs[0, 19])  # Colorbar takes up 1/20 of the width
        
        scatter = ax.scatter(window_coords[1:, 0], window_coords[1:, 1], c=mapped_types, cmap='tab20', alpha=0.6)
        ax.scatter(window_coords[0, 0], window_coords[0, 1], c='red', marker='*', s=200, label='CLS')
        
        head_weights = attention_weights[h]
        print(f"Head {h+1} weights shape: {head_weights.shape}")
        
        # Normalize attention weights to [0, 1] range for each row
        row_sums = head_weights.sum(axis=1, keepdims=True)
        normalized_weights = np.divide(head_weights, row_sums, out=np.zeros_like(head_weights), where=row_sums!=0)
        
        # Draw attention lines
        for i in range(1, len(window_coords)):
            for j in range(1, len(window_coords)):
                if normalized_weights[i, j] > 0.1:  # Only draw significant attention
                    ax.plot([window_coords[i, 0], window_coords[j, 0]],
                            [window_coords[i, 1], window_coords[j, 1]],
                            'k-', alpha=min(normalized_weights[i, j], 1.0), linewidth=1)
        
        # Draw CLS attention
        for i in range(1, len(window_coords)):
            if normalized_weights[0, i] > 0.1:
                ax.plot([window_coords[0, 0], window_coords[i, 0]],
                        [window_coords[0, 1], window_coords[i, 1]],
                        'r-', alpha=min(normalized_weights[0, i], 1.0), linewidth=1)
            if normalized_weights[i, 0] > 0.1:
                ax.plot([window_coords[i, 0], window_coords[0, 0]],
                        [window_coords[i, 1], window_coords[0, 1]],
                        'b-', alpha=min(normalized_weights[i, 0], 1.0), linewidth=1)
        
        ax.set_title(f'Head {h+1} (spatial)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_ticks(range(len(unique_types)))
        cbar.set_ticklabels([inv_cell_type_map[t] for t in unique_types])
        
        plt.savefig(output_path.replace('.png', f'_head{h+1}_spatial.png'), bbox_inches='tight')
        plt.close()
        
        # --- Separate attention heatmap for each head ---
        fig_hm, ax_hm = plt.subplots(figsize=(6, 6))
        sns.heatmap(normalized_weights, ax=ax_hm, cmap='viridis')
        ax_hm.set_title(f'Head {h+1} (attention)')
        ax_hm.set_xlabel('Token')
        ax_hm.set_ylabel('Token')
        plt.savefig(output_path.replace('.png', f'_head{h+1}_heatmap.png'), bbox_inches='tight')
        plt.close()
    
    print("Done plotting attention heads.")

def train_epoch(model, dataloader, optimizer, criterion, device, global_features_dict=None, val_loader=None, val_steps=70):
    """
    Train the model for one epoch
    
    Parameters:
    - model: Model to train
    - dataloader: DataLoader with training data
    - optimizer: Optimizer
    - criterion: Loss function
    - device: Device to use
    - global_features_dict: Dict mapping donor to global features
    - val_loader: Optional validation dataloader for step-wise validation
    - val_steps: Number of training steps between validations
    
    Returns:
    - step_train_losses: dict mapping step_num to train loss
    - step_val_losses: dict mapping step_num to val loss
    - train_outputs: List of model outputs
    - train_targets: List of target labels
    - attention_weights: Tuple of (attention_weights, coords, cell_types) for first window
    """
    model.train()
    total_loss = 0
    all_outputs = []
    all_targets = []
    all_attention_weights = []
    step_train_losses = {}  # step_num: loss
    step_val_losses = {}    # step_num: val_loss
    step_counter = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move batch to device
        marker_values = batch['marker_values'].to(device)
        rel_positions = batch['rel_positions'].to(device)
        labels = batch['label'].to(device) if 'label' in batch else None
        cell_types = batch['cell_types'].to(device) if 'cell_types' in batch else None
        
        # Prepare global features if used
        global_features = None
        if global_features_dict is not None and 'group_id' in batch:
            donors = [group_id[0] for group_id in batch['group_id']]
            global_features = torch.tensor(
                np.array([global_features_dict[donor] for donor in donors]),
                dtype=torch.float32,
                device=device
            )
        
        # Forward pass
        outputs, attention_weights = model(
            marker_values=marker_values, 
            rel_positions=rel_positions,
            cell_types=cell_types,
            global_features=global_features,
            return_attention=True
        )
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        step_train_losses[step_counter] = loss.item()
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())
        
        # Only store attention weights for first batch
        if batch_idx == 0:
            first_window_attn = attention_weights[0].detach().cpu()
            first_window_coords = rel_positions[0].detach().cpu()
            first_window_types = cell_types[0].detach().cpu()
            all_attention_weights = (first_window_attn, first_window_coords, first_window_types)
        
        step_counter += 1
        
        # Run validation every val_steps
        if val_loader is not None and step_counter % val_steps == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_marker_values = val_batch['marker_values'].to(device)
                    val_rel_positions = val_batch['rel_positions'].to(device)
                    val_labels = val_batch['label'].to(device) if 'label' in val_batch else None
                    val_cell_types = val_batch['cell_types'].to(device) if 'cell_types' in val_batch else None
                    
                    val_global_features = None
                    if global_features_dict is not None and 'group_id' in val_batch:
                        val_donors = [group_id[0] for group_id in val_batch['group_id']]
                        val_global_features = torch.tensor(
                            np.array([global_features_dict[donor] for donor in val_donors]),
                            dtype=torch.float32,
                            device=device
                        )
                    
                    val_outputs, _ = model(
                        marker_values=val_marker_values,
                        rel_positions=val_rel_positions,
                        cell_types=val_cell_types,
                        global_features=val_global_features
                    )
                    val_loss += criterion(val_outputs.squeeze(), val_labels).item()
            
            val_loss /= len(val_loader)
            step_val_losses[step_counter] = val_loss
            model.train()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, step_train_losses, step_val_losses, np.array(all_outputs), np.array(all_targets), all_attention_weights

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
    - donor_metrics: Dict with donor-level metrics
    - attention_weights: Tuple of (attention_weights, coords, cell_types) for first window
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    all_group_ids = []
    all_attention_weights = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            # Move batch to device
            marker_values = batch['marker_values'].to(device)
            rel_positions = batch['rel_positions'].to(device)
            labels = batch['label'].to(device) if 'label' in batch else None
            cell_types = batch['cell_types'].to(device) if 'cell_types' in batch else None
            
            # Prepare global features if used
            global_features = None
            if global_features_dict is not None and 'group_id' in batch:
                donors = [group_id[0] for group_id in batch['group_id']]
                global_features = torch.tensor(
                    np.array([global_features_dict[donor] for donor in donors]),
                    dtype=torch.float32,
                    device=device
                )
            
            # Forward pass
            outputs, attention_weights = model(
                marker_values=marker_values, 
                rel_positions=rel_positions,
                cell_types=cell_types,
                global_features=global_features,
                return_attention=True
            )
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)
            
            # Track loss and predictions
            total_loss += loss.item()
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
            
            if 'group_id' in batch:
                all_group_ids.extend(batch['group_id'])
            
            # Only store attention weights for first batch
            if batch_idx == 0:
                # Take first window from batch
                first_window_attn = attention_weights[0].detach().cpu()  # shape: (num_heads, window_size, window_size)
                first_window_coords = rel_positions[0].detach().cpu()  # shape: (window_size, 2)
                first_window_types = cell_types[0].detach().cpu()  # shape: (window_size,)
                all_attention_weights = (first_window_attn, first_window_coords, first_window_types)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Calculate donor-level metrics if group_ids are available
    donor_metrics = None
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
            'donor_auc': roc_auc_score(donor_targets, donor_outputs),
            'donor_accuracy': accuracy_score(donor_targets, donor_preds),
            'donor_outputs': donor_outputs,
            'donor_targets': donor_targets
        }
    
    return avg_loss, np.array(all_outputs), np.array(all_targets), donor_metrics, all_attention_weights

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

def plot_pca_visualization(data_dict, feature_columns, output_dir, n_samples=10000):
    """
    Perform PCA on marker data and create visualizations
    
    Parameters:
    - data_dict: Dict with train/val/test DataFrames
    - feature_columns: Dict with categorized column names
    - output_dir: Directory to save plots
    - n_samples: Number of samples to use for visualization (default: 10000)
    """
    # Subsample data from each split proportionally
    total_cells = len(data_dict['train_cells']) + len(data_dict['val_cells']) + len(data_dict['test_cells'])
    train_samples = int(n_samples * len(data_dict['train_cells']) / total_cells)
    val_samples = int(n_samples * len(data_dict['val_cells']) / total_cells)
    test_samples = n_samples - train_samples - val_samples
    
    # Sample from each split
    train_subset = data_dict['train_cells'].sample(n=train_samples, random_state=42)
    val_subset = data_dict['val_cells'].sample(n=val_samples, random_state=42)
    test_subset = data_dict['test_cells'].sample(n=test_samples, random_state=42)
    
    # Combine subsampled data for PCA
    all_markers = pd.concat([
        train_subset[feature_columns['markers']],
        val_subset[feature_columns['markers']],
        test_subset[feature_columns['markers']]
    ])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_markers)
    
    # Create DataFrame with PCA results and metadata
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    
    # Add split information
    pca_df['split'] = ['train'] * len(train_subset) + \
                      ['val'] * len(val_subset) + \
                      ['test'] * len(test_subset)
    
    # Add survival information
    pca_df['High_Survival'] = pd.concat([
        train_subset['High_Survival'],
        val_subset['High_Survival'],
        test_subset['High_Survival']
    ]).values
    
    # Plot 1: PCA by split
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='split', alpha=0.6)
    plt.title(f'PCA of Marker Expression by Data Split (n={n_samples:,} samples)')
    plt.savefig(os.path.join(output_dir, 'pca_by_split.png'))
    plt.close()
    
    # Plot 2: PCA by survival
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='High_Survival', alpha=0.6)
    plt.title(f'PCA of Marker Expression by Survival Status (n={n_samples:,} samples)')
    plt.savefig(os.path.join(output_dir, 'pca_by_survival.png'))
    plt.close()
    
    # Plot 3: Explained variance
    pca_10 = PCA(n_components=10)
    pca_10.fit(all_markers)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_10.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # Print explained variance
    print(f"First 2 PCs explain {pca.explained_variance_ratio_[:2].sum():.2%} of variance")
    print(f"Used {n_samples:,} samples for visualization ({train_samples:,} train, {val_samples:,} val, {test_samples:,} test)")
    
    return pca

def plot_donor_mean_pca(data_dict, feature_columns, output_dir, n_samples_per_donor=50, cells_per_sample=1000):
    """
    Plot PCA of mean expression per donor with subsampling
    
    Parameters:
    - data_dict: Dict with train/val/test DataFrames
    - feature_columns: Dict with categorized column names
    - output_dir: Directory to save plots
    - n_samples_per_donor: Number of subsamples per donor
    - cells_per_sample: Number of cells per subsample
    """
    # Get all donors
    all_donors = pd.concat([
        data_dict['train_cells']['donor'],
        data_dict['val_cells']['donor'],
        data_dict['test_cells']['donor']
    ]).unique()
    
    # Initialize lists for means and survival
    means = []
    survival = []
    donors = []
    
    # For each donor, create multiple subsamples
    for donor in tqdm(all_donors, desc="Processing donors"):
        # Get donor data
        donor_data = pd.concat([
            data_dict['train_cells'][data_dict['train_cells']['donor'] == donor],
            data_dict['val_cells'][data_dict['val_cells']['donor'] == donor],
            data_dict['test_cells'][data_dict['test_cells']['donor'] == donor]
        ])
        
        # Create multiple subsamples
        for _ in range(n_samples_per_donor):
            # Randomly sample cells
            if len(donor_data) > cells_per_sample:
                subsample = donor_data.sample(n=cells_per_sample, random_state=42)
            else:
                subsample = donor_data
                
            # Calculate mean expression
            mean_expr = subsample[feature_columns['markers']].mean()
            means.append(mean_expr)
            survival.append(subsample['High_Survival'].iloc[0])
            donors.append(donor)
    
    # Convert to numpy arrays
    means = np.array(means)
    survival = np.array(survival)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(means)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['High_Survival'] = survival
    pca_df['Donor'] = donors
    
    # Plot 1: PCA by survival
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='High_Survival', alpha=0.6)
    plt.title(f'PCA of Mean Expression per Donor\n({n_samples_per_donor} subsamples of {cells_per_sample} cells each)')
    plt.savefig(os.path.join(output_dir, 'pca_donor_mean_survival.png'))
    plt.close()
    
    # Plot 2: PCA by donor
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Donor', alpha=0.6)
    plt.title(f'PCA of Mean Expression per Donor\n({n_samples_per_donor} subsamples of {cells_per_sample} cells each)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_donor_mean_donor.png'))
    plt.close()
    
    # Print explained variance
    print(f"First 2 PCs explain {pca.explained_variance_ratio_[:2].sum():.2%} of variance")
    print(f"Created {len(means)} subsamples ({n_samples_per_donor} per donor)")

def plot_roc_curve(targets, outputs, output_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(targets, preds, output_path, title="Confusion Matrix", class_names=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds)
    plt.figure()
    if class_names is None:
        class_names = ['Short Survival', 'Long Survival']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(targets, outputs, output_path, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(targets, outputs)
    ap = average_precision_score(targets, outputs)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_output_distribution(outputs, targets, output_path, class_names=None):
    plt.figure()
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    plt.hist(np.array(outputs)[np.array(targets)==0], bins=50, alpha=0.5, label=class_names[0])
    plt.hist(np.array(outputs)[np.array(targets)==1], bins=50, alpha=0.5, label=class_names[1])
    plt.xlabel('Model Output (logit; >0 = Long Survival, <0 = Short Survival)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Model Outputs')
    plt.savefig(output_path)
    plt.close()

def plot_donor_confusion_matrices(train_outputs, train_targets, train_group_ids, test_outputs, test_targets, test_group_ids, output_path):
    """
    Plot donor-level confusion matrices for train and test as subplots.
    """
    def aggregate_by_donor(outputs, targets, group_ids):
        df = pd.DataFrame({
            'output': outputs,
            'target': targets,
            'donor': [gid[0] if isinstance(gid, (list, tuple, np.ndarray)) else gid for gid in group_ids]
        })
        donor_df = df.groupby('donor').agg({'output': 'mean', 'target': 'first'})
        donor_preds = (donor_df['output'].values > 0).astype(int)
        donor_targets = donor_df['target'].values.astype(int)
        return donor_preds, donor_targets

    train_preds, train_true = aggregate_by_donor(train_outputs, train_targets, train_group_ids)
    test_preds, test_true = aggregate_by_donor(test_outputs, test_targets, test_group_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, preds, true, title in zip(axes, [train_preds, test_preds], [train_true, test_true], ["Train", "Test"]):
        cm = confusion_matrix(true, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Donor-level Confusion Matrix ({title})')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spatial_windows_with_survival(data_dict, train_windows, feature_columns, output_dir, donor_col='donor', cell_type_col=None, n_windows=100, donor_id=None, window_logits=None, epoch=None):
    """
    Plot all cells in a tissue colored by cell type, overlaying windows from long/short survival donors.
    Windows are colored by predicted label (logit > 0: long, <= 0: short).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    if cell_type_col is None:
        cell_type_col = feature_columns['cell_types'][0]
    # Get specific donor or random one
    if donor_id is None:
        donor = data_dict['train_cells'][donor_col].sample(1).values[0]
    else:
        donor = donor_id
    print(f"Plotting spatial windows for donor: {donor}")
    donor_cells = data_dict['train_cells'][data_dict['train_cells'][donor_col] == donor]
    # Plot all cells colored by cell type
    plt.figure(figsize=(10, 10))
    cell_type_map = {v: k for k, v in enumerate(donor_cells[cell_type_col].unique())}
    plt.scatter(donor_cells['x'], donor_cells['y'], c=[cell_type_map[ct] for ct in donor_cells[cell_type_col]], cmap='tab20', alpha=0.3, label='Cells', s=0.1)
    # Filter windows to only include those from the current donor
    donor_window_indices = []
    for i, group_id in enumerate(train_windows['group_ids']):
        window_donor = group_id[0] if isinstance(group_id, (list, tuple, np.ndarray)) else group_id
        if window_donor == donor:
            donor_window_indices.append(i)
    if len(donor_window_indices) == 0:
        print(f"No windows found for donor {donor}!")
        return donor
    print(f"Found {len(donor_window_indices)} windows for donor {donor}")
    n_windows = min(n_windows, len(donor_window_indices))
    selected_indices = np.random.choice(donor_window_indices, size=n_windows, replace=False)
    # Overlay windows colored by predicted label
    for i, idx in enumerate(selected_indices):
        # Get center cell coordinate (first cell in window is the center)
        center_coord = train_windows['abs_coords'][idx][0].cpu().numpy()
        if window_logits is not None:
            pred_label = 1 if window_logits[idx] > 0 else 0
        else:
            pred_label = data_dict['train_cells'][data_dict['train_cells'][donor_col] == donor]['High_Survival'].iloc[0]
        color = 'red' if pred_label == 1 else 'blue'
        plt.scatter(center_coord[0], center_coord[1], c=color, alpha=0.5, s=200, marker='s')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Cells in Donor {donor} with Windows Overlayed by Predicted Survival')
    # Custom legend: only red and blue squares
    custom_legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=12, label='Predicted Long Survival'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=12, label='Predicted Short Survival')
    ]
    plt.legend(handles=custom_legend, loc='upper right')
    plt.tight_layout()
    if epoch is not None:
        save_path = os.path.join(output_dir, f'spatial_windows_survival_epoch_{epoch}.png')
    else:
        save_path = os.path.join(output_dir, 'spatial_windows_survival.png')
    plt.savefig(save_path)
    plt.close()
    return donor

def plot_cell_density_heatmap(cell_df, output_dir, donor_col='donor', grid_size=50, donor_id=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Use specific donor or random one
    if donor_id is None:
        donor = cell_df[donor_col].sample(1).values[0]
    else:
        donor = donor_id
    print(f"Plotting cell density heatmap for donor: {donor}")
    donor_cells = cell_df[cell_df[donor_col] == donor]
    x = donor_cells['x'].values
    y = donor_cells['y'].values
    # Create 2D histogram with smaller grid size for better resolution
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=grid_size)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Cell Count')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Cell Density Heatmap for Donor {donor}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_density_heatmap.png'))
    plt.close()

def plot_umap_cell_markers(data_dict, feature_columns, output_dir, n_cells=10000):
    """
    Plot UMAP of cell protein markers colored by cell type using scanpy.
    """
    # Subsample cells
    all_cells = pd.concat([
        data_dict['train_cells'],
        data_dict['val_cells'],
        data_dict['test_cells']
    ])
    subsample = all_cells.sample(n=min(n_cells, len(all_cells)), random_state=42)
    markers = subsample[feature_columns['markers']].values
    cell_types = subsample[feature_columns['cell_types'][0]].astype(str).values
    adata = sc.AnnData(markers)
    adata.obs['cell_type'] = cell_types
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='cell_type', show=False, save='_celltype.png', wspace=0.4)
    # Move the plot to the output_dir
    import shutil
    if os.path.exists('figures/umap_celltype.png'):
        shutil.move('figures/umap_celltype.png', os.path.join(output_dir, 'umap_celltype.png'))

def plot_example_windows(train_windows, cell_type_ids, inv_cell_type_map, colors, output_dir):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    # Get all unique cell types across all windows
    all_cell_types = np.unique(cell_type_ids.cpu().numpy())
    plt.figure(figsize=(15, 5))
    for i in range(3):  # Plot 3 example windows
        plt.subplot(1, 3, i+1)
        window_idx = np.random.randint(train_windows['abs_coords'].shape[0])
        window_coords = train_windows['abs_coords'][window_idx].cpu().numpy()
        window_cell_types = cell_type_ids[window_idx].cpu().numpy()
        scatter = plt.scatter(window_coords[:, 0], window_coords[:, 1], c=window_cell_types, cmap='tab20', alpha=0.7)
        plt.title(f'Window {i+1}')
        plt.xlabel('Absolute X')
        plt.ylabel('Absolute Y')
    # Global legend for all cell types
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=inv_cell_type_map[ct],
                          markerfacecolor=colors(ct), markersize=8) for ct in all_cell_types]
    plt.figlegend(handles=handles, loc='center right', title='Cell Type', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(output_dir, 'example_windows.png'))
    plt.close()

def plot_grid_cell_density_histogram(cell_df, output_dir, donor_col='donor', grid_size=50, donor_id=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Use specific donor or random one
    if donor_id is None:
        donor = cell_df[donor_col].sample(1).values[0]
    else:
        donor = donor_id
    donor_cells = cell_df[cell_df[donor_col] == donor]
    x = donor_cells['x'].values
    y = donor_cells['y'].values
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=grid_size)
    # Flatten and plot histogram
    plt.figure()
    plt.hist(heatmap.flatten(), bins=30, color='gray', edgecolor='black')
    plt.xlabel('Number of Cells per Grid Cell')
    plt.ylabel('Count')
    plt.title(f'Distribution of Cell Density per Grid Cell (Donor {donor})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grid_cell_density_hist.png'))
    plt.close()

def plot_cell_type_prevalence_by_window_survival(train_windows, cell_type_map, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    # Get cell types and window labels
    cell_types = train_windows['cell_types'].cpu().numpy()  # shape: (n_windows, window_size)
    labels = train_windows['labels'].cpu().numpy()  # shape: (n_windows,)
    # Invert cell_type_map for names
    inv_cell_type_map = {v: k for k, v in cell_type_map.items()}
    n_types = max(max(inv_cell_type_map.keys())+1, np.max(cell_types)+1)
    # Flatten all cell types for each window, grouped by label
    long_mask = labels == 1
    short_mask = labels == 0
    long_types = cell_types[long_mask].flatten()
    short_types = cell_types[short_mask].flatten()
    # Count prevalence
    long_counts = np.bincount(long_types, minlength=n_types)
    short_counts = np.bincount(short_types, minlength=n_types)
    # Normalize to fractions
    long_frac = long_counts / long_counts.sum() if long_counts.sum() > 0 else np.zeros(n_types)
    short_frac = short_counts / short_counts.sum() if short_counts.sum() > 0 else np.zeros(n_types)
    # Barplot
    x = np.arange(n_types)
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, long_frac, width, label='Long Survival Windows')
    plt.bar(x + width/2, short_frac, width, label='Short Survival Windows')
    xticklabels = [inv_cell_type_map.get(i, f'Unknown_{i}') for i in range(n_types)]
    plt.xticks(x, xticklabels, rotation=45, ha='right')
    plt.ylabel('Fraction of Cells')
    plt.title('Cell Type Prevalence in Long vs Short Survival Windows')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_type_prevalence_long_vs_short.png'))
    plt.close()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data(
        metadata_path=args.metadata_path,
        cell_data_path=args.cell_data_path,
        os_threshold=args.os_threshold,
        apply_batch_corr=args.apply_batch_corr
    )
    
    # Get feature columns
    feature_columns = identify_feature_columns(data_dict['train_cells'])
    
    # Perform PCA visualization before training
    print("Performing PCA visualization...")
    pca = plot_pca_visualization(data_dict, feature_columns, os.path.join(args.output_dir, 'figures'))
    
    # Perform donor mean PCA visualization
    print("Performing donor mean PCA visualization...")
    plot_donor_mean_pca(data_dict, feature_columns, os.path.join(args.output_dir, 'figures'))
    
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
    print(f"Number of training windows after filtering: {train_windows['marker_values'].shape[0]}")
    assert train_windows['marker_values'].shape[0] == train_windows['labels'].shape[0], "Mismatch between marker_values and labels in train_windows!"
    
    # Visualize some example windows
    print("Visualizing example windows...")
    print("Window structure:", {k: type(v) for k, v in train_windows.items()})
    print("Window shapes:", {k: v.shape if hasattr(v, 'shape') else len(v) for k, v in train_windows.items()})
    
    cell_type_ids = train_windows['cell_types']  # shape: (n_windows, window_size)
    cell_type_map = window_generator.cell_type_map
    inv_cell_type_map = {v: k for k, v in cell_type_map.items()}
    n_types = len(cell_type_map)
    colors = matplotlib.colormaps['tab20'].resampled(n_types)
    
    plot_example_windows(train_windows, cell_type_ids, inv_cell_type_map, colors, os.path.join(args.output_dir, 'figures'))
    
    # Add before the training loop:
    # Initialize model
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
    model = model.to(args.device)
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # Create datasets and data loaders
    train_dataset = WindowDataset(window_generator.add_cls_tokens(train_windows))
    val_windows = window_generator.generate_windows(
        data_dict['val_cells_normalized'],
        num_windows_per_sample=args.windows_per_sample
    )
    val_windows = window_generator.add_cls_tokens(val_windows)
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
    
    # After model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Model parameters: {num_params:,} ({param_size_mb:.2f} MB)")
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    best_val_auc = 0
    patience_counter = 0
    
    # Lists to store metrics
    all_train_losses = {}  # step_num: loss
    all_val_losses = {}    # step_num: val_loss
    val_aucs = []
    donor_aucs = []
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch with step-wise validation
        train_loss, step_train_losses, step_val_losses, train_outputs, train_targets, train_attention_weights = train_epoch(
            model, train_loader, optimizer, criterion, args.device, global_features_dict, val_loader, val_steps=70
        )
        
        # Store step-wise losses
        all_train_losses.update(step_train_losses)
        all_val_losses.update(step_val_losses)
        
        # Calculate metrics for the epoch
        train_metrics = calculate_metrics(train_outputs, train_targets)
        
        # Get final validation metrics for the epoch
        val_loss, val_outputs, val_targets, donor_metrics, val_attention_weights = validate(
            model, val_loader, criterion, args.device, global_features_dict
        )
        val_metrics = calculate_metrics(val_outputs, val_targets)
        
        # Store metrics
        val_aucs.append(val_metrics['auc'])
        if donor_metrics is not None:
            donor_aucs.append(donor_metrics['donor_auc'])
        
        # Print metrics
        print(f"Train loss: {train_loss:.4f}, Train AUC: {train_metrics['auc']:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f'Train accuracy: {train_metrics["accuracy"]}')
        print(f'Val accuracy: {val_metrics["accuracy"]}')
        print(f'Val f1: {val_metrics["f1"]}')
        print(f'Train f1: {train_metrics["f1"]}')
        print(f'Val precision: {val_metrics["precision"]}')
        print(f'Train precision: {train_metrics["precision"]}')
        print(f'Val recall: {val_metrics["recall"]}')
        print(f'Train recall: {train_metrics["recall"]}')
        
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

        # After each epoch, plot ROC for training and validation
        plot_roc_curve(train_targets, train_outputs, os.path.join(args.output_dir, 'figures', f'roc_curve_window_train_epoch_{epoch}.png'), title='ROC Curve (Window-level, Train)')
        plot_roc_curve(val_targets, val_outputs, os.path.join(args.output_dir, 'figures', f'roc_curve_window_val_epoch_{epoch}.png'), title='ROC Curve (Window-level, Val)')

        # Window-level
        window_preds = (val_outputs > 0).astype(int)
        plot_roc_curve(val_targets, val_outputs, os.path.join(args.output_dir, 'figures', 'roc_curve_window.png'), title='ROC Curve (Window-level)')
        plot_confusion_matrix(val_targets, window_preds, os.path.join(args.output_dir, 'figures', 'confusion_matrix_window.png'), title='Confusion Matrix (Window-level)', class_names=['Short Survival', 'Long Survival'])
        plot_pr_curve(val_targets, val_outputs, os.path.join(args.output_dir, 'figures', 'pr_curve_window.png'), title='PR Curve (Window-level)')
        plot_output_distribution(val_outputs, val_targets, os.path.join(args.output_dir, 'figures', 'output_dist_window.png'), class_names=['Short Survival', 'Long Survival'])

        # Donor-level (if donor_metrics is not None)
        if donor_metrics is not None and 'donor_auc' in donor_metrics:
            # You need donor-level outputs/targets
            donor_outputs = np.array([v for v in donor_metrics.get('donor_outputs', [])])
            donor_targets = np.array([v for v in donor_metrics.get('donor_targets', [])])
            if donor_outputs.size > 0 and donor_targets.size > 0:
                donor_preds = (donor_outputs > 0).astype(int)
                plot_confusion_matrix(donor_targets, donor_preds, os.path.join(args.output_dir, 'figures', 'confusion_matrix_donor.png'), title='Confusion Matrix (Donor-level)', class_names=['Short Survival', 'Long Survival'])
                plot_output_distribution(donor_outputs, donor_targets, os.path.join(args.output_dir, 'figures', 'output_dist_donor.png'), class_names=['Short Survival', 'Long Survival'])

        # After validation and after train_outputs are available, call:
        donor_id = plot_spatial_windows_with_survival(data_dict, train_windows, feature_columns, os.path.join(args.output_dir, 'figures'), window_logits=train_outputs, epoch=epoch)
        plot_cell_density_heatmap(data_dict['train_cells'], os.path.join(args.output_dir, 'figures'), donor_id=donor_id)
        plot_grid_cell_density_histogram(data_dict['train_cells'], os.path.join(args.output_dir, 'figures'), donor_id=donor_id)
        plot_cell_type_prevalence_by_window_survival(train_windows, cell_type_map, os.path.join(args.output_dir, 'figures'))

        # Training windows - only plot 1 window
        if train_attention_weights is not None:
            attn_weights, coords, types = train_attention_weights
            plot_attention_weights(
                attention_weights=attn_weights,
                window_coords=coords,
                cell_types=types,
                inv_cell_type_map=inv_cell_type_map,
                output_path=os.path.join(args.output_dir, 'figures', f'attention_weights_train_epoch_{epoch}_window_0.png'),
                title=f'Training Attention Weights (Epoch {epoch}, Window 0)'
            )

        # Validation windows - only plot 1 window
        if val_attention_weights is not None:
            attn_weights, coords, types = val_attention_weights
            plot_attention_weights(
                attention_weights=attn_weights,
                window_coords=coords,
                cell_types=types,
                inv_cell_type_map=inv_cell_type_map,
                output_path=os.path.join(args.output_dir, 'figures', f'attention_weights_val_epoch_{epoch}_window_0.png'),
                title=f'Validation Attention Weights (Epoch {epoch}, Window 0)'
            )

    # Plot step-wise training curves
    plt.figure(figsize=(15, 6))
    plt.plot(list(all_train_losses.keys()), list(all_train_losses.values()), label='Train Loss', alpha=0.6)
    plt.scatter(list(all_val_losses.keys()), list(all_val_losses.values()), label='Val Loss', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss (Step-wise)')
    plt.savefig(os.path.join(args.output_dir, 'figures', 'stepwise_loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(val_aucs, label='Window-level AUC')
    if donor_aucs:
        plt.plot(donor_aucs, label='Donor-level AUC')
    if len(val_aucs) == 1:
        plt.scatter([0], val_aucs, color='blue')
    if len(donor_aucs) == 1:
        plt.scatter([0], donor_aucs, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC')
    plt.savefig(os.path.join(args.output_dir, 'figures', 'auc_curve.png'))
    plt.close()
    
    # Aggregate donor-level predictions for train and test
    # For train
    train_loader_full = DataLoader(
        WindowDataset(window_generator.add_cls_tokens(window_generator.generate_windows(
            data_dict['train_cells_normalized'], num_windows_per_sample=args.windows_per_sample))),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)
    train_loss, train_outputs, train_targets, train_group_ids = 0, [], [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader_full:
            marker_values = batch['marker_values'].to(args.device)
            rel_positions = batch['rel_positions'].to(args.device)
            labels = batch['label'].to(args.device) if 'label' in batch else None
            cell_types = batch['cell_types'].to(args.device) if 'cell_types' in batch else None
            global_features = None
            if global_features_dict is not None and 'group_id' in batch:
                donors = [group_id[0] for group_id in batch['group_id']]
                global_features = torch.tensor(
                    np.array([global_features_dict[donor] for donor in donors]),
                    dtype=torch.float32, device=args.device)
            outputs, _ = model(marker_values=marker_values, rel_positions=rel_positions, cell_types=cell_types, global_features=global_features)
            train_outputs.extend(outputs.detach().cpu().numpy())
            if labels is not None:
                train_targets.extend(labels.detach().cpu().numpy())
            if 'group_id' in batch:
                train_group_ids.extend(batch['group_id'])
    # For test
    test_loader_full = DataLoader(
        WindowDataset(window_generator.add_cls_tokens(window_generator.generate_windows(
            data_dict['test_cells_normalized'], num_windows_per_sample=args.windows_per_sample))),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)
    test_outputs, test_targets, test_group_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader_full:
            marker_values = batch['marker_values'].to(args.device)
            rel_positions = batch['rel_positions'].to(args.device)
            labels = batch['label'].to(args.device) if 'label' in batch else None
            cell_types = batch['cell_types'].to(args.device) if 'cell_types' in batch else None
            global_features = None
            if global_features_dict is not None and 'group_id' in batch:
                donors = [group_id[0] for group_id in batch['group_id']]
                global_features = torch.tensor(
                    np.array([global_features_dict[donor] for donor in donors]),
                    dtype=torch.float32, device=args.device)
            outputs, _ = model(marker_values=marker_values, rel_positions=rel_positions, cell_types=cell_types, global_features=global_features)
            test_outputs.extend(outputs.detach().cpu().numpy())
            if labels is not None:
                test_targets.extend(labels.detach().cpu().numpy())
            if 'group_id' in batch:
                test_group_ids.extend(batch['group_id'])
    plot_donor_confusion_matrices(
        np.array(train_outputs).ravel(), np.array(train_targets).ravel(), train_group_ids,
        np.array(test_outputs).ravel(), np.array(test_targets).ravel(), test_group_ids,
        os.path.join(args.output_dir, 'figures', 'donor_confusion_matrices.png'))

    print("Training complete!")

if __name__ == "__main__":
    main() 