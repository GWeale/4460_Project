import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random

class PositionalEncoding2D(nn.Module):
    """
    Standard 2D positional encoding similar to ViT
    """
    def __init__(self, d_model, max_h=1000, max_w=1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encodings for height and width dimensions
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_w = torch.zeros(max_w, d_model // 2)
        
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def forward(self, x, y):
        """
        Args:
            x: x-coordinate (batch_size)
            y: y-coordinate (batch_size)
            
        Returns:
            Positional encoding (batch_size, d_model)
        """
        batch_size = x.size(0)
        
        # Ensure coordinates are within range
        x = torch.clamp(x, 0, self.pe_w.size(0) - 1).long()
        y = torch.clamp(y, 0, self.pe_h.size(0) - 1).long()
        
        # Get positional encoding for each coordinate
        pos_x = self.pe_w[x]  # (batch_size, d_model // 2)
        pos_y = self.pe_h[y]  # (batch_size, d_model // 2)
        
        # Concatenate x and y encodings
        pos = torch.cat([pos_x, pos_y], dim=1)  # (batch_size, d_model)
        
        return pos

class CellTypeEmbedding(nn.Module):
    """
    Cell type embedding based on cell features (protein expression)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, cell_features):
        """
        Args:
            cell_features: Cell features (batch_size, input_dim)
            
        Returns:
            Cell type embedding (batch_size, output_dim)
        """
        return self.embedding_network(cell_features)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Combine projections to reduce overhead
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None, return_attention=False):
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # Combined linear projection for Q, K, V
        qkv = self.qkv_linear(q).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def get_attention_map(self, x, mask=None):
        """Extract attention map without doing the full forward pass"""
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Get Q, K, V projections
        qkv = self.qkv_linear(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attention=False):
        # Multi-head attention with residual connection and layer normalization
        if return_attention:
            attn_output, attn_weights = self.attention(x, x, x, mask, return_attention=True)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x, attn_weights
        else:
            attn_output = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x

class LightCellBERTModel(nn.Module):
    def __init__(self, cell_feature_dim, d_model=64, num_heads=2, dim_feedforward=128, 
                 num_layers=2, dropout=0.1, max_h=1000, max_w=1000):
        super().__init__()
        
        # Embedding layers
        self.feature_embedding = nn.Linear(cell_feature_dim, d_model)
        self.pos_encoding = PositionalEncoding2D(d_model, max_h, max_w)
        self.cell_type_embedding = CellTypeEmbedding(cell_feature_dim, d_model//2, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # Binary classification (long vs short survival)
        )
        
    def forward(self, cell_features, x_coords, y_coords, mask=None, return_attention=False):
        """
        Args:
            cell_features: Cell protein expression (batch_size, num_cells, cell_feature_dim)
            x_coords: X coordinates (batch_size, num_cells)
            y_coords: Y coordinates (batch_size, num_cells)
            mask: Attention mask (batch_size, num_cells + 1, num_cells + 1)
            return_attention: Whether to return attention maps
            
        Returns:
            cls_output: CLS token output for classification (batch_size, d_model)
            logits: Classification logits (batch_size, 2) or attention maps if return_attention=True
        """
        batch_size, num_cells, _ = cell_features.size()
        
        # Feature embedding
        embeddings = self.feature_embedding(cell_features)  # (batch_size, num_cells, d_model)
        
        # Compute positional embeddings for each cell
        pos_embeddings = torch.zeros_like(embeddings)
        for i in range(batch_size):
            for j in range(num_cells):
                pos_embeddings[i, j] = self.pos_encoding(x_coords[i, j].unsqueeze(0), 
                                                        y_coords[i, j].unsqueeze(0))
        
        # Compute cell type embeddings for each cell
        cell_type_embeddings = self.cell_type_embedding(cell_features)  # (batch_size, num_cells, d_model)
        
        # Combine all embeddings
        embeddings = embeddings + pos_embeddings + cell_type_embeddings
        
        # Add CLS token
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, d_model)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch_size, num_cells + 1, d_model)
        
        # Update mask to include CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, mask.size(2)).to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            cls_mask = torch.ones(batch_size, mask.size(1), 1).to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=2)
        
        # Apply transformer blocks
        attention_maps = []
        for transformer_block in self.transformer_blocks:
            # Handle returning attention maps for visualization
            if return_attention and hasattr(transformer_block.attention, 'get_attention_map'):
                embeddings, attn_map = transformer_block.attention(
                    embeddings, embeddings, embeddings, mask, return_attention=True
                )
                attention_maps.append(attn_map)
            else:
                embeddings = transformer_block(embeddings, mask)
        
        # Layer normalization
        embeddings = self.norm(embeddings)
        
        # Extract CLS token representation
        cls_output = embeddings[:, 0]  # (batch_size, d_model)
        
        # Classification
        logits = self.mlp_head(cls_output)  # (batch_size, 2)
        
        if return_attention:
            # If attention maps couldn't be extracted directly, create a placeholder
            if not attention_maps:
                # Mock attention maps for compatibility
                attention_maps = [torch.ones(batch_size, 1, 1+num_cells, 1+num_cells)]
                print("Warning: Attention maps not available. Using placeholder.")
            return cls_output, attention_maps
        
        return cls_output, logits

# Backwards compatibility
CellBERTModel = LightCellBERTModel

class CellWindowDataset(Dataset):
    """
    Dataset for cell windows with nearest neighbors
    """
    def __init__(self, adata, window_size=10, survival_threshold=24, max_windows_per_patient=50):
        """
        Args:
            adata: AnnData object with protein expression data
            window_size: Number of cells in each window
            survival_threshold: Threshold for survival (in months) for binary classification
            max_windows_per_patient: Maximum number of windows to create per patient
        """
        self.adata = adata
        self.window_size = window_size
        self.max_windows_per_patient = max_windows_per_patient
        
        # Get cell coordinates
        self.x_coords = adata.obs['X'].values
        self.y_coords = adata.obs['Y'].values
        self.coords = np.column_stack([self.x_coords, self.y_coords])
        
        # Get protein expression data
        self.features = adata.X
        
        # Get patient IDs
        self.patient_ids = adata.obs['donor'].values
        self.unique_patients = np.unique(self.patient_ids)
        
        # Get survival data from metadata
        self.survival_data = {}
        metadata = adata.uns['metadata']
        for i, row in metadata.iterrows():
            patient_id = row['donor']
            os = row['OS']
            self.survival_data[patient_id] = 1 if os >= survival_threshold else 0
        
        # Create windows for each patient
        self.windows = []
        self.labels = []
        
        for patient_id in self.unique_patients:
            patient_mask = self.patient_ids == patient_id
            
            if patient_id not in self.survival_data:
                continue
                
            patient_features = self.features[patient_mask]
            patient_coords = self.coords[patient_mask]
            patient_label = self.survival_data[patient_id]
            
            # Find nearest neighbors for each cell
            nn_finder = NearestNeighbors(n_neighbors=window_size)
            nn_finder.fit(patient_coords)
            
            # If many cells, randomly sample them to limit windows per patient
            num_cells = len(patient_coords)
            if num_cells > self.max_windows_per_patient:
                # Random sample of indices
                sample_indices = random.sample(range(num_cells), self.max_windows_per_patient)
            else:
                sample_indices = range(num_cells)
            
            # Create windows using nearest neighbors
            for i in sample_indices:
                distances, indices = nn_finder.kneighbors([patient_coords[i]])
                window_indices = indices[0]
                window_features = patient_features[window_indices]
                window_coords = patient_coords[window_indices]
                
                self.windows.append({
                    'features': window_features,
                    'coords': window_coords,
                    'patient_id': patient_id
                })
                self.labels.append(patient_label)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        features = torch.FloatTensor(window['features'])
        coords = window['coords']
        x_coords = torch.FloatTensor(coords[:, 0])
        y_coords = torch.FloatTensor(coords[:, 1])
        label = torch.LongTensor([self.labels[idx]])
        
        return {
            'features': features,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'label': label,
            'patient_id': window['patient_id']
        }

class WindowAggregator(nn.Module):
    """
    Aggregates CLS tokens from multiple windows for final prediction
    """
    def __init__(self, d_model, hidden_dim=32):
        super().__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: CLS tokens from multiple windows (num_windows, d_model)
            
        Returns:
            logits: Classification logits (2)
        """
        # Simple mean pooling of CLS tokens
        mean_cls = torch.mean(cls_tokens, dim=0)
        
        # Classification
        logits = self.aggregator(mean_cls)
        
        return logits

def create_anndata_from_csv(melanoma_data_path, markers_data_path, metadata_path, sample_ratio=0.02):
    """
    Create AnnData object from CSV files
    
    Args:
        melanoma_data_path: Path to Melanoma_data.csv
        markers_data_path: Path to Day3_Markers_Dryad.csv
        metadata_path: Path to metadata.csv
        sample_ratio: Ratio of cells to sample (to reduce memory usage)
    
    Returns:
        AnnData object with protein expression data
    """
    print(f"Loading data from CSV files...")
    
    # Load metadata (this is small so we load it completely)
    metadata = pd.read_csv(metadata_path)
    
    # For large data files, we'll read in chunks and sample
    # First check the total number of rows to determine chunk size
    with pd.read_csv(melanoma_data_path, chunksize=1000) as reader:
        total_rows = sum(len(chunk) for chunk in reader)
    
    # Determine sample size and chunk size based on memory considerations
    sample_size = int(total_rows * sample_ratio)
    chunk_size = min(10000, max(1000, sample_size // 10))
    
    print(f"Sampling approximately {sample_size} rows from {total_rows} total rows")
    
    # Read melanoma data in chunks and sample rows
    sampled_rows = []
    
    # Read in chunks to save memory
    for chunk in pd.read_csv(melanoma_data_path, chunksize=chunk_size):
        # Sample from this chunk
        chunk_sample = chunk.sample(frac=sample_ratio)
        sampled_rows.append(chunk_sample)
    
    # Combine sampled chunks
    melanoma_data = pd.concat(sampled_rows, ignore_index=True)
    print(f"Sampled {len(melanoma_data)} rows from Melanoma_data.csv")
    
    # Print column names to debug
    print(f"Available columns: {melanoma_data.columns.tolist()}")
    
    # Load markers data (if it's manageable in size)
    try:
        # Just load a subset of the markers file directly
        markers_data = pd.read_csv(markers_data_path, nrows=10)
        print(f"Loaded first {len(markers_data)} marker profiles (only for column inspection)")
    except Exception as e:
        # If markers file causes issues, we'll skip it
        markers_data = None
        print(f"Skipped loading markers file due to error: {e}")
    
    # Identify protein expression columns and cell metadata columns
    # Try to find columns containing coordinates and patient IDs
    coordinate_cols = [col for col in melanoma_data.columns if 'coord' in col.lower() or 'position' in col.lower() 
                     or 'location' in col.lower() or col.lower() in ['x', 'y', 'xcoord', 'ycoord']]
    id_cols = [col for col in melanoma_data.columns if 'patient' in col.lower() or 'donor' in col.lower() 
              or 'id' in col.lower()]
    
    # Print discovered columns
    print(f"Discovered coordinate columns: {coordinate_cols}")
    print(f"Discovered ID columns: {id_cols}")
    
    # Assuming first 57 columns are protein markers (adjust based on actual data)
    # Let's try to identify them based on column types
    numeric_cols = melanoma_data.select_dtypes(include=np.number).columns.tolist()
    potential_marker_cols = [col for col in numeric_cols if col not in coordinate_cols]
    protein_cols = potential_marker_cols[:57]  # Take first 57 numeric columns
    metadata_cols = [col for col in melanoma_data.columns if col not in protein_cols]
    
    print(f"Using {len(protein_cols)} columns as protein markers")
    print(f"Using {len(metadata_cols)} columns as metadata")
    
    # Extract protein expression data and cell metadata
    protein_data = melanoma_data[protein_cols]
    cell_metadata = melanoma_data[metadata_cols]
    
    # Create AnnData object
    adata = AnnData(X=protein_data.values, obs=cell_metadata)
    
    # Make sure coordinates are present
    x_coord_col = None
    y_coord_col = None
    
    # Look for X coordinate columns
    for col in ['X', 'x', 'X_coord', 'x_coord', 'xcoord']:
        if col in adata.obs.columns:
            x_coord_col = col
            break
    
    # Look for Y coordinate columns
    for col in ['Y', 'y', 'Y_coord', 'y_coord', 'ycoord']:
        if col in adata.obs.columns:
            y_coord_col = col
            break
            
    # If not found, try to infer from coordinate_cols
    if x_coord_col is None and len(coordinate_cols) >= 1:
        x_coord_col = coordinate_cols[0]
    if y_coord_col is None and len(coordinate_cols) >= 2:
        y_coord_col = coordinate_cols[1]
        
    # If still not found, create dummy coordinates
    if x_coord_col is None:
        print("Warning: X coordinate not found in cell metadata. Using placeholder.")
        adata.obs['X'] = np.random.rand(len(adata))
    else:
        # Make sure we have a column named 'X' for consistency
        adata.obs['X'] = adata.obs[x_coord_col]
    
    if y_coord_col is None:
        print("Warning: Y coordinate not found in cell metadata. Using placeholder.")
        adata.obs['Y'] = np.random.rand(len(adata))
    else:
        # Make sure we have a column named 'Y' for consistency
        adata.obs['Y'] = adata.obs[y_coord_col]
    
    # Make sure donor ID is present
    donor_col = None
    for col in ['donor', 'Donor', 'patient_id', 'Patient_ID']:
        if col in adata.obs.columns:
            donor_col = col
            break
            
    # If not found, try to infer from id_cols
    if donor_col is None and id_cols:
        donor_col = id_cols[0]
        
    if donor_col is None:
        print("Warning: donor ID not found in cell metadata. Using placeholder.")
        adata.obs['donor'] = np.array(['patient1'] * len(adata))
    else:
        # Make sure we have a column named 'donor' for consistency
        adata.obs['donor'] = adata.obs[donor_col]
    
    # Store metadata in uns
    adata.uns['metadata'] = metadata
    
    # Normalize data - with careful handling of zeros and NaNs
    print("Normalizing data...")
    
    # Replace NaN values with zeros
    adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    # Add a small epsilon to zeros instead of replacing them
    epsilon = 1e-8
    adata.X[adata.X == 0] = epsilon
    
    # Use safe log transformation
    adata.X = np.log1p(np.abs(adata.X))
    
    # Final check for NaNs or infs
    if np.isnan(adata.X).any() or np.isinf(adata.X).any():
        print("Warning: NaN or Inf values found after normalization. Fixing...")
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Created AnnData object with shape {adata.shape}")
    print(f"Check for NaN in data: {np.isnan(adata.X).any()}")
    return adata

def quick_train_model(cell_bert_model, window_aggregator, train_loader, val_loader, 
                device, num_epochs=5, lr=5e-3):
    """
    Train the Cell-BERT model with faster settings
    """
    # Move models to device
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    # Loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    
    # Use lower learning rate for stability
    bert_optimizer = torch.optim.AdamW(cell_bert_model.parameters(), lr=lr, weight_decay=0.01)
    agg_optimizer = torch.optim.AdamW(window_aggregator.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler for faster convergence
    bert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(bert_optimizer, patience=1, factor=0.5)
    agg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(agg_optimizer, patience=1, factor=0.5)
    
    # Function to check for NaN values in model parameters
    def check_for_nan(model, model_name):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf values found in {model_name}.{name}")
                return True
        return False
    
    # Check models before training
    check_for_nan(cell_bert_model, "cell_bert_model")
    check_for_nan(window_aggregator, "window_aggregator")
    
    # Training loop
    for epoch in range(num_epochs):
        cell_bert_model.train()
        window_aggregator.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                features = batch['features'].to(device)
                x_coords = batch['x_coords'].to(device)
                y_coords = batch['y_coords'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                # Check for NaN in inputs
                if torch.isnan(features).any() or torch.isnan(x_coords).any() or torch.isnan(y_coords).any():
                    print(f"Warning: NaN values in input batch {batch_idx}. Skipping.")
                    continue
                
                # Forward pass
                cls_outputs, logits = cell_bert_model(features, x_coords, y_coords)
                
                # Check for NaN in outputs
                if torch.isnan(cls_outputs).any() or torch.isnan(logits).any():
                    print(f"Warning: NaN values in model output for batch {batch_idx}. Skipping.")
                    continue
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Check for NaN in loss
                if torch.isnan(loss).item():
                    print(f"Warning: NaN loss in batch {batch_idx}. Skipping.")
                    continue
                
                train_loss += loss.item()
                
                # Backward pass and optimization
                bert_optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN in gradients
                any_nan_grad = False
                for name, param in cell_bert_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Warning: NaN/Inf gradient in {name}. Skipping optimization.")
                            any_nan_grad = True
                            break
                
                if any_nan_grad:
                    continue
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(cell_bert_model.parameters(), 1.0)
                bert_optimizer.step()
                
                # Print diagnostics every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Simplified validation - only validate on a subset for speed
        cell_bert_model.eval()
        window_aggregator.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Only evaluate on a random subset of patients for speed
            patient_cls_tokens = {}
            patient_labels = {}
            
            # Limit validation to at most 10 batches
            for i, batch in enumerate(val_loader):
                if i >= 10:  # Only process first 10 batches for faster validation
                    break
                    
                try:
                    features = batch['features'].to(device)
                    x_coords = batch['x_coords'].to(device)
                    y_coords = batch['y_coords'].to(device)
                    labels = batch['label'].squeeze().to(device)
                    patient_ids = batch['patient_id']
                    
                    # Check for NaN values
                    if torch.isnan(features).any() or torch.isnan(x_coords).any() or torch.isnan(y_coords).any():
                        print(f"Warning: NaN values in validation batch {i}. Skipping.")
                        continue
                    
                    # Forward pass for each window
                    cls_outputs, _ = cell_bert_model(features, x_coords, y_coords)
                    
                    # Check for NaN in outputs
                    if torch.isnan(cls_outputs).any():
                        print(f"Warning: NaN values in validation outputs for batch {i}. Skipping.")
                        continue
                    
                    # Group by patient
                    for j, patient_id in enumerate(patient_ids):
                        if patient_id not in patient_cls_tokens:
                            patient_cls_tokens[patient_id] = []
                            patient_labels[patient_id] = labels[j].item()
                        
                        patient_cls_tokens[patient_id].append(cls_outputs[j])
                except Exception as e:
                    print(f"Error processing validation batch {i}: {e}")
                    continue
            
            # Aggregate windows for each patient
            for patient_id, cls_tokens in patient_cls_tokens.items():
                try:
                    cls_tokens_tensor = torch.stack(cls_tokens)
                    logits = window_aggregator(cls_tokens_tensor)
                    label = torch.tensor([patient_labels[patient_id]]).to(device)
                    
                    loss = criterion(logits.unsqueeze(0), label)
                    val_loss += loss.item()
                    
                    pred = torch.argmax(logits).item()
                    true_label = label.item()
                    
                    if pred == true_label:
                        correct += 1
                    total += 1
                except Exception as e:
                    print(f"Error aggregating windows for patient {patient_id}: {e}")
                    continue
        
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        # Update learning rate based on validation loss
        if not np.isnan(val_loss) and total > 0:
            bert_scheduler.step(val_loss)
            agg_scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/max(1, len(train_loader)):.4f}, "
              f"Val Loss: {val_loss/max(1, total):.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Check for NaN in model parameters after epoch
        if check_for_nan(cell_bert_model, "cell_bert_model") or check_for_nan(window_aggregator, "window_aggregator"):
            print("NaN values found in model parameters. Stopping training.")
            break
    
    return cell_bert_model, window_aggregator

# For backwards compatibility
train_model = quick_train_model

def evaluate_model(cell_bert_model, window_aggregator, test_loader, device):
    """
    Evaluate the Cell-BERT model on a test set
    
    Args:
        cell_bert_model: The cell BERT model
        window_aggregator: The window aggregator model
        test_loader: DataLoader for the test set
        device: Device to run the model on
        
    Returns:
        true_labels: List of true labels
        pred_probs: List of predicted probabilities
    """
    cell_bert_model.eval()
    window_aggregator.eval()
    
    # Group windows by patient
    patient_cls_tokens = {}
    patient_true_labels = {}
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                features = batch['features'].to(device)
                x_coords = batch['x_coords'].to(device)
                y_coords = batch['y_coords'].to(device)
                labels = batch['label'].squeeze().cpu().numpy()
                patient_ids = batch['patient_id']
                
                # Check for NaN values
                if torch.isnan(features).any() or torch.isnan(x_coords).any() or torch.isnan(y_coords).any():
                    print(f"Warning: NaN values in test batch. Skipping.")
                    continue
                
                # Forward pass for each window
                cls_outputs, _ = cell_bert_model(features, x_coords, y_coords)
                
                # Check for NaN outputs
                if torch.isnan(cls_outputs).any():
                    print(f"Warning: NaN values in model output. Skipping.")
                    continue
                
                # Group by patient
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_cls_tokens:
                        patient_cls_tokens[patient_id] = []
                        patient_true_labels[patient_id] = labels[i] if i < len(labels) else 0
                    
                    patient_cls_tokens[patient_id].append(cls_outputs[i])
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Aggregate windows for each patient and make predictions
    true_labels = []
    pred_probs = []
    
    for patient_id, cls_tokens in patient_cls_tokens.items():
        try:
            cls_tokens_tensor = torch.stack(cls_tokens)
            logits = window_aggregator(cls_tokens_tensor)
            
            # Get prediction probability
            probs = torch.softmax(logits, dim=0)
            prob_class_1 = probs[1].item()  # Probability of the positive class
            
            true_labels.append(patient_true_labels[patient_id])
            pred_probs.append(prob_class_1)
        except Exception as e:
            print(f"Error aggregating predictions for patient {patient_id}: {e}")
            continue
    
    return np.array(true_labels), np.array(pred_probs)

def main():
    # Create AnnData from CSV files with much smaller sample ratio
    adata = create_anndata_from_csv(
        'Melanoma_data.csv',
        'Day3_Markers_Dryad.csv',
        'metadata.csv',
        sample_ratio=0.02  # Reduced from 0.1 to 0.02
    )
    
    # Create datasets with smaller window size and limit windows per patient
    window_size = 10  # Reduced from 15 to 10
    survival_threshold = 24
    max_windows_per_patient = 50  # Limit windows per patient
    
    dataset = CellWindowDataset(adata, window_size=window_size, 
                               survival_threshold=survival_threshold,
                               max_windows_per_patient=max_windows_per_patient)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with larger batch size for faster training
    batch_size = 64  # Increased from 32 to 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2)  # Add num_workers for parallel loading
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2)
    
    # Create models - use smaller model with reduced parameters
    cell_feature_dim = adata.X.shape[1]
    d_model = 64  # Reduced from 128 to 64
    
    cell_bert_model = LightCellBERTModel(
        cell_feature_dim, 
        d_model=d_model,
        num_heads=2,  # Reduced from 4 to 2
        dim_feedforward=128,  # Reduced from 256 to 128
        num_layers=2,  # Reduced from 3 to 2
        dropout=0.1
    )
    
    window_aggregator = WindowAggregator(d_model, hidden_dim=32)
    
    # Train models with fewer epochs and higher learning rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_bert_model, window_aggregator = quick_train_model(
        cell_bert_model, window_aggregator, train_loader, val_loader, 
        device, num_epochs=5, lr=5e-3  # Reduced from 50 to 5 epochs, increased LR
    )
    
    # Save models
    torch.save(cell_bert_model.state_dict(), 'cell_bert_model.pt')
    torch.save(window_aggregator.state_dict(), 'window_aggregator.pt')

if __name__ == "__main__":
    main() 