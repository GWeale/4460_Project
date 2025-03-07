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

class PositionalEncoding2D(nn.Module):
    """
    Standard 2D positional encoding similar to ViT
    """
    def __init__(self, d_model, max_h=1000, max_w=1000):
        super().__init__()
        self.d_model = d_model
        
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
        
        x = torch.clamp(x, 0, self.pe_w.size(0) - 1).long()
        y = torch.clamp(y, 0, self.pe_h.size(0) - 1).long()
        
        pos_x = self.pe_w[x]  # (batch_size, d_model // 2)
        pos_y = self.pe_h[y]  # (batch_size, d_model // 2)
        
        pos = torch.cat([pos_x, pos_y], dim=1) 
        
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
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        return out

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
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CellBERTModel(nn.Module):
    def __init__(self, cell_feature_dim, d_model=128, num_heads=4, dim_feedforward=256, 
                 num_layers=3, dropout=0.1, max_h=1000, max_w=1000):
        super().__init__()
        
        self.feature_embedding = nn.Linear(cell_feature_dim, d_model)
        self.pos_encoding = PositionalEncoding2D(d_model, max_h, max_w)
        self.cell_type_embedding = CellTypeEmbedding(cell_feature_dim, d_model//2, d_model)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # Binary classification (long vs short survival)
        )
        
    def forward(self, cell_features, x_coords, y_coords, mask=None):
        """
        Args:
            cell_features: Cell protein expression (batch_size, num_cells, cell_feature_dim)
            x_coords: X coordinates (batch_size, num_cells)
            y_coords: Y coordinates (batch_size, num_cells)
            mask: Attention mask (batch_size, num_cells + 1, num_cells + 1)
            
        Returns:
            cls_output: CLS token output for classification (batch_size, d_model)
            logits: Classification logits (batch_size, 2)
        """
        batch_size, num_cells, _ = cell_features.size()
        
        embeddings = self.feature_embedding(cell_features)  # (batch_size, num_cells, d_model)
        
        pos_embeddings = torch.zeros_like(embeddings)
        for i in range(batch_size):
            for j in range(num_cells):
                pos_embeddings[i, j] = self.pos_encoding(x_coords[i, j].unsqueeze(0), 
                                                        y_coords[i, j].unsqueeze(0))
        
        cell_type_embeddings = self.cell_type_embedding(cell_features)  # (batch_size, num_cells, d_model)
        
        embeddings = embeddings + pos_embeddings + cell_type_embeddings
        
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, d_model)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch_size, num_cells + 1, d_model)
        
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, mask.size(2)).to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            cls_mask = torch.ones(batch_size, mask.size(1), 1).to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=2)
        
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings, mask)
        
        embeddings = self.norm(embeddings)
        
        cls_output = embeddings[:, 0]  # (batch_size, d_model)
        
        logits = self.mlp_head(cls_output)  # (batch_size, 2)
        
        return cls_output, logits

class CellWindowDataset(Dataset):
    """
    Dataset for cell windows with nearest neighbors
    """
    def __init__(self, adata, window_size=15, survival_threshold=24):
        """
        Args:
            adata: AnnData object with protein expression data
            window_size: Number of cells in each window
            survival_threshold: Threshold for survival (in months) for binary classification
        """
        self.adata = adata
        self.window_size = window_size
        
        self.x_coords = adata.obs['X'].values
        self.y_coords = adata.obs['Y'].values
        self.coords = np.column_stack([self.x_coords, self.y_coords])
        
        self.features = adata.X
        
        self.patient_ids = adata.obs['donor'].values
        self.unique_patients = np.unique(self.patient_ids)
        
        self.survival_data = {}
        metadata = adata.uns['metadata']
        for i, row in metadata.iterrows():
            patient_id = row['donor']
            os = row['OS']
            self.survival_data[patient_id] = 1 if os >= survival_threshold else 0
        
        self.windows = []
        self.labels = []
        
        for patient_id in self.unique_patients:
            patient_mask = self.patient_ids == patient_id
            
            if patient_id not in self.survival_data:
                continue
                
            patient_features = self.features[patient_mask]
            patient_coords = self.coords[patient_mask]
            patient_label = self.survival_data[patient_id]
            
            nn_finder = NearestNeighbors(n_neighbors=window_size)
            nn_finder.fit(patient_coords)
            distances, indices = nn_finder.kneighbors(patient_coords)
            
            for i in range(len(patient_coords)):
                window_indices = indices[i]
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
    def __init__(self, d_model, hidden_dim=64):
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

def train_model(cell_bert_model, window_aggregator, train_loader, val_loader, 
                device, num_epochs=50, lr=1e-4):
    """
    Train the Cell-BERT model
    """
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    criterion = nn.CrossEntropyLoss()
    bert_optimizer = torch.optim.Adam(cell_bert_model.parameters(), lr=lr)
    agg_optimizer = torch.optim.Adam(window_aggregator.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        cell_bert_model.train()
        window_aggregator.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = batch['features'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            cls_outputs, logits = cell_bert_model(features, x_coords, y_coords)
            
            loss = criterion(logits, labels)
            train_loss += loss.item()
            
            bert_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step()
        
        cell_bert_model.eval()
        window_aggregator.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Group windows by patient
            patient_cls_tokens = {}
            patient_labels = {}
            
            for batch in val_loader:
                features = batch['features'].to(device)
                x_coords = batch['x_coords'].to(device)
                y_coords = batch['y_coords'].to(device)
                labels = batch['label'].squeeze().to(device)
                patient_ids = batch['patient_id']
                
                cls_outputs, _ = cell_bert_model(features, x_coords, y_coords)
                
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_cls_tokens:
                        patient_cls_tokens[patient_id] = []
                        patient_labels[patient_id] = labels[i].item()
                    
                    patient_cls_tokens[patient_id].append(cls_outputs[i])
            
            for patient_id, cls_tokens in patient_cls_tokens.items():
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
        
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/total:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    return cell_bert_model, window_aggregator

def main():
    # Load data
    adata = sc.read('protein_data.h5ad')
    
    window_size = 15  # Window size (number of nearest neighbors)
    survival_threshold = 24  # Survival threshold in months
    
    dataset = CellWindowDataset(adata, window_size=window_size, 
                                survival_threshold=survival_threshold)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    cell_feature_dim = adata.X.shape[1]  # Number of protein markers
    d_model = 128  # Embedding dimension
    
    cell_bert_model = CellBERTModel(cell_feature_dim, d_model=d_model, 
                                   num_heads=4, dim_feedforward=256, 
                                   num_layers=3, dropout=0.1)
    
    window_aggregator = WindowAggregator(d_model, hidden_dim=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_bert_model, window_aggregator = train_model(
        cell_bert_model, window_aggregator, train_loader, val_loader, 
        device, num_epochs=50, lr=1e-4
    )
    
    torch.save(cell_bert_model.state_dict(), 'cell_bert_model.pt')
    torch.save(window_aggregator.state_dict(), 'window_aggregator.pt')

if __name__ == "__main__":
    main() 