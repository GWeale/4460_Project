import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm

class CellEmbedding(nn.Module):
    """
    Embedding module for cell data
    
    This module combines marker, spatial, and cell type embeddings
    """
    def __init__(self, marker_dim, hidden_dim, num_cell_types=None, cell_type_map=None, max_position=1000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.marker_dim = marker_dim
        self.cell_type_map = cell_type_map
        
        # Marker embedding (linear projection of marker values)
        self.marker_embedding = nn.Linear(marker_dim, hidden_dim)
        
        # Cell type embedding
        if num_cell_types is not None:
            self.cell_type_embedding = nn.Embedding(num_cell_types, hidden_dim)
        else:
            self.cell_type_embedding = None
            
        # Spatial positional encoding
        self.register_buffer(
            "position_table", 
            self._create_sinusoidal_embeddings(max_position, hidden_dim)
        )
        
        # [CLS] token embedding
        self.cls_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
    def _create_sinusoidal_embeddings(self, max_position, hidden_dim):
        """Create sinusoidal embeddings for spatial positions"""
        position = torch.arange(max_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        
        position_table = torch.zeros(max_position, hidden_dim)
        position_table[:, 0::2] = torch.sin(position * div_term)
        position_table[:, 1::2] = torch.cos(position * div_term)
        
        return position_table
    
    def get_positional_embedding(self, rel_pos):
        """Get positional embedding for relative positions"""
        # Clip positions to max_position
        rel_pos = torch.clamp(rel_pos, 0, self.position_table.shape[0] - 1)
        return self.position_table[rel_pos]
        
    def forward(self, marker_values, rel_positions, cell_types=None):
        """
        Generate embeddings for cells
        
        Parameters:
        - marker_values: tensor of marker values [batch_size, seq_len, marker_dim]
        - rel_positions: tensor of relative positions [batch_size, seq_len, 2]
        - cell_types: tensor of cell type indices [batch_size, seq_len]
        
        Returns:
        - combined embeddings [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = marker_values.shape[0], marker_values.shape[1]
        
        # Marker embedding
        marker_embed = self.marker_embedding(marker_values)
        
        # Positional embedding - calculate for x and y separately
        # Convert positions to integers for lookup
        rel_pos_x = rel_positions[:, :, 0].long()
        rel_pos_y = rel_positions[:, :, 1].long()
        
        pos_embed_x = self.get_positional_embedding(rel_pos_x)
        pos_embed_y = self.get_positional_embedding(rel_pos_y)
        pos_embed = pos_embed_x + pos_embed_y  # combine x and y embeddings
        
        # Combine embeddings
        combined_embed = marker_embed + pos_embed
        
        # Add cell type embedding if provided
        if cell_types is not None and self.cell_type_embedding is not None:
            cell_type_embed = self.cell_type_embedding(cell_types)
            combined_embed = combined_embed + cell_type_embed
            
        return combined_embed
    
    def get_cls_embedding(self, batch_size):
        """Return cls token embeddings for the batch"""
        return self.cls_embedding.expand(batch_size, 1, -1)


class SpatialBERTModel(nn.Module):
    """
    BERT-style transformer model for spatial transcriptomics
    """
    def __init__(
        self,
        marker_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        num_cell_types=None,
        cell_type_map=None,
        max_position=1000,
        use_global_features=False,
        global_feature_dim=0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_global_features = use_global_features
        
        # Cell embedding module
        self.cell_embedding = CellEmbedding(
            marker_dim=marker_dim,
            hidden_dim=hidden_dim,
            num_cell_types=num_cell_types,
            cell_type_map=cell_type_map,
            max_position=max_position
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Prediction head
        pred_input_dim = hidden_dim + global_feature_dim if use_global_features else hidden_dim
        self.prediction_head = nn.Sequential(
            nn.Linear(pred_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, marker_values, rel_positions, attention_mask=None, cell_types=None, global_features=None):
        """
        Forward pass through the model
        
        Parameters:
        - marker_values: tensor of marker values [batch_size, seq_len, marker_dim]
        - rel_positions: tensor of relative positions [batch_size, seq_len, 2]
        - attention_mask: attention mask for padding [batch_size, seq_len]
        - cell_types: tensor of cell type indices [batch_size, seq_len]
        - global_features: tensor of global features [batch_size, global_feature_dim]
        
        Returns:
        - logits for binary classification [batch_size, 1]
        """
        batch_size, seq_len = marker_values.shape[0], marker_values.shape[1]
        
        # Get embeddings for sequence cells (excluding CLS)
        cell_embeddings = self.cell_embedding(
            marker_values=marker_values[:, 1:],  # exclude CLS position
            rel_positions=rel_positions[:, 1:],
            cell_types=None if cell_types is None else cell_types[:, 1:]
        )
        
        # Get CLS token embedding
        cls_embedding = self.cell_embedding.get_cls_embedding(batch_size)
        
        # Combine CLS with cell embeddings
        combined_embeddings = torch.cat([cls_embedding, cell_embeddings], dim=1)
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Convert boolean mask to additive attention mask
            attn_mask = (1 - attention_mask.float()) * -10000.0
            transformer_output = self.transformer_encoder(combined_embeddings, src_key_padding_mask=attention_mask)
        else:
            transformer_output = self.transformer_encoder(combined_embeddings)
        
        # Extract CLS token
        cls_output = transformer_output[:, 0]
        
        # Combine with global features if provided
        if self.use_global_features and global_features is not None:
            combined_output = torch.cat([cls_output, global_features], dim=1)
        else:
            combined_output = cls_output
            
        # Get prediction
        logits = self.prediction_head(combined_output)
        
        return logits


class WindowGenerator:
    """
    Generates windows of cells for model input
    """
    def __init__(
        self,
        k_neighbors=20,
        max_position=1000,
        cell_type_col='Cell_Type_Common',
        coordinate_cols=['x', 'y'],
        marker_cols=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.k_neighbors = k_neighbors
        self.max_position = max_position
        self.cell_type_col = cell_type_col
        self.coordinate_cols = coordinate_cols
        self.marker_cols = marker_cols
        self.device = device
        
        # These will be set when fit is called
        self.cell_type_map = None
        self.num_cell_types = None
        
    def fit(self, cell_df):
        """
        Fit the window generator to the data
        
        Parameters:
        - cell_df: DataFrame with cell data
        
        Returns:
        - self
        """
        # Create mapping for cell types
        if self.cell_type_col in cell_df.columns:
            unique_cell_types = cell_df[self.cell_type_col].dropna().unique()
            self.cell_type_map = {cell_type: i for i, cell_type in enumerate(unique_cell_types)}
            self.num_cell_types = len(self.cell_type_map)
            print(f"Found {self.num_cell_types} unique cell types")
        
        # If marker_cols not provided, infer from data
        if self.marker_cols is None:
            # Exclude known non-marker columns
            excluded_cols = [
                'cellid', 'donor', 'filename', 'region', 'x', 'y', 'DAPI',
                'Cell_Type', 'Cell_Type_Common', 'Cell_Type_Sub', 'Overall_Cell_Type',
                'Neighborhood', 'High_Survival'
            ]
            self.marker_cols = [col for col in cell_df.columns if col not in excluded_cols]
            print(f"Inferred {len(self.marker_cols)} marker columns")
        
        return self
    
    def _find_neighbors(self, coordinates):
        """
        Find k nearest neighbors for each cell
        
        Parameters:
        - coordinates: numpy array of shape [n_cells, 2]
        
        Returns:
        - indices of neighbors for each cell [n_cells, k]
        """
        # Initialize nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=self.k_neighbors + 1)  # +1 because cell is its own neighbor
        nn_model.fit(coordinates)
        
        # Find neighbors
        distances, indices = nn_model.kneighbors(coordinates)
        
        # Exclude self (first column is the cell itself)
        return indices[:, 1:]
    
    def _generate_windows_for_sample(self, sample_df, center_indices=None, num_windows=None):
        """
        Generate windows for a sample (e.g., a single tissue section)
        
        Parameters:
        - sample_df: DataFrame with cells from a single sample
        - center_indices: Indices to use as window centers (if None, randomly sample)
        - num_windows: Number of windows to generate (if center_indices is None)
        
        Returns:
        - dict with window data
        """
        # Get coordinates
        coordinates = sample_df[self.coordinate_cols].values
        
        # Find neighbors for all cells
        neighbor_indices = self._find_neighbors(coordinates)
        
        # Select center cells
        if center_indices is None:
            if num_windows is None:
                num_windows = min(1000, len(sample_df))  # Default: use 1000 windows or all cells if fewer
            
            # Randomly sample center cells
            center_indices = np.random.choice(len(sample_df), size=num_windows, replace=False)
        
        # Get neighbors for center cells
        windows = neighbor_indices[center_indices]
        
        # Get coordinates for windows
        window_coordinates = np.array([coordinates[window] for window in windows])
        
        # Get center coordinates for each window
        center_coordinates = coordinates[center_indices].reshape(-1, 1, 2)
        
        # Calculate relative positions
        rel_positions = window_coordinates - center_coordinates
        
        # Scale and discretize positions
        # Map to range [0, max_position)
        max_dist = np.max(np.abs(rel_positions))
        scale_factor = (self.max_position // 2) / max_dist if max_dist > 0 else 1
        rel_positions = rel_positions * scale_factor
        rel_positions = rel_positions.astype(int) + (self.max_position // 2)
        
        # Clip to valid range
        rel_positions = np.clip(rel_positions, 0, self.max_position - 1)
        
        # Get marker values for windows
        marker_values = np.array([sample_df.iloc[window][self.marker_cols].values for window in windows])
        
        # Get cell types for windows if available
        if self.cell_type_col in sample_df.columns and self.cell_type_map is not None:
            cell_types = sample_df[self.cell_type_col].map(self.cell_type_map).fillna(0).astype(int)
            window_cell_types = np.array([cell_types.iloc[window].values for window in windows])
        else:
            window_cell_types = None
        
        # Get survival label for the sample
        if 'High_Survival' in sample_df.columns:
            # All cells in the sample should have the same label
            label = sample_df['High_Survival'].iloc[0]
        else:
            label = None
            
        return {
            'windows': windows,
            'marker_values': marker_values,
            'rel_positions': rel_positions,
            'cell_types': window_cell_types,
            'label': label,
            'center_indices': center_indices
        }
    
    def generate_windows(self, cell_df, group_cols=['donor', 'filename'], num_windows_per_sample=None):
        """
        Generate windows for all samples in the data
        
        Parameters:
        - cell_df: DataFrame with cell data
        - group_cols: Columns to use for grouping cells into samples
        - num_windows_per_sample: Number of windows to generate per sample
        
        Returns:
        - dict with window data for all samples
        """
        # Group data by sample
        grouped = cell_df.groupby(group_cols)
        
        # Initialize lists to store results
        all_marker_values = []
        all_rel_positions = []
        all_cell_types = []
        all_labels = []
        all_group_ids = []
        
        # Generate windows for each sample
        for group_id, sample_df in tqdm(grouped, desc="Generating windows"):
            window_data = self._generate_windows_for_sample(
                sample_df, 
                num_windows=num_windows_per_sample
            )
            
            # Store results
            all_marker_values.append(window_data['marker_values'])
            all_rel_positions.append(window_data['rel_positions'])
            
            if window_data['cell_types'] is not None:
                all_cell_types.append(window_data['cell_types'])
                
            if window_data['label'] is not None:
                all_labels.append(np.full(len(window_data['windows']), window_data['label']))
                
            all_group_ids.append([group_id] * len(window_data['windows']))
        
        # Concatenate results
        marker_values = np.concatenate(all_marker_values, axis=0)
        rel_positions = np.concatenate(all_rel_positions, axis=0)
        
        # Convert to tensors
        marker_values_tensor = torch.tensor(marker_values, dtype=torch.float32, device=self.device)
        rel_positions_tensor = torch.tensor(rel_positions, dtype=torch.long, device=self.device)
        
        result = {
            'marker_values': marker_values_tensor,
            'rel_positions': rel_positions_tensor,
            'group_ids': np.concatenate(all_group_ids, axis=0) if all_group_ids else None
        }
        
        if all_cell_types:
            result['cell_types'] = torch.tensor(
                np.concatenate(all_cell_types, axis=0), 
                dtype=torch.long, 
                device=self.device
            )
            
        if all_labels:
            result['labels'] = torch.tensor(
                np.concatenate(all_labels, axis=0),
                dtype=torch.float32,
                device=self.device
            )
            
        return result
    
    def add_cls_tokens(self, window_data):
        """
        Add [CLS] tokens to the window data
        
        Parameters:
        - window_data: dict with window data
        
        Returns:
        - dict with window data including [CLS] tokens
        """
        batch_size, seq_len = window_data['marker_values'].shape[:2]
        
        # Create [CLS] token for marker values (zeros)
        cls_markers = torch.zeros(
            (batch_size, 1, window_data['marker_values'].shape[2]), 
            device=window_data['marker_values'].device
        )
        
        # Add [CLS] to marker values
        window_data['marker_values'] = torch.cat([cls_markers, window_data['marker_values']], dim=1)
        
        # Create [CLS] token for relative positions (center position)
        cls_positions = torch.full(
            (batch_size, 1, 2), 
            self.max_position // 2,  # Center position
            device=window_data['rel_positions'].device
        )
        
        # Add [CLS] to relative positions
        window_data['rel_positions'] = torch.cat([cls_positions, window_data['rel_positions']], dim=1)
        
        # Add [CLS] to cell types if present
        if 'cell_types' in window_data and window_data['cell_types'] is not None:
            # Use a special index for [CLS] (e.g., num_cell_types)
            cls_cell_type = torch.full(
                (batch_size, 1), 
                self.num_cell_types,  # Special index for [CLS]
                device=window_data['cell_types'].device
            )
            
            # Add [CLS] to cell types
            window_data['cell_types'] = torch.cat([cls_cell_type, window_data['cell_types']], dim=1)
            
        return window_data


class WindowDataset(torch.utils.data.Dataset):
    """
    Dataset for windowed cell data
    """
    def __init__(self, window_data):
        """
        Initialize dataset with window data
        
        Parameters:
        - window_data: dict with window data
        """
        self.marker_values = window_data['marker_values']
        self.rel_positions = window_data['rel_positions']
        self.labels = window_data.get('labels')
        self.cell_types = window_data.get('cell_types')
        self.group_ids = window_data.get('group_ids')
        
    def __len__(self):
        return len(self.marker_values)
    
    def __getitem__(self, idx):
        item = {
            'marker_values': self.marker_values[idx],
            'rel_positions': self.rel_positions[idx]
        }
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
            
        if self.cell_types is not None:
            item['cell_types'] = self.cell_types[idx]
            
        if self.group_ids is not None:
            item['group_id'] = self.group_ids[idx]
            
        return item


def collate_windows(batch):
    """
    Collate function for batching windows
    
    Parameters:
    - batch: list of dicts with window data
    
    Returns:
    - dict with batched window data
    """
    # Get keys from first item
    keys = batch[0].keys()
    
    # Initialize dict for results
    result = {}
    
    # Batch each key
    for key in keys:
        if key == 'group_id':
            # Group IDs are not tensors
            result[key] = [item[key] for item in batch]
        else:
            # Stack tensors
            result[key] = torch.stack([item[key] for item in batch])
            
    return result 