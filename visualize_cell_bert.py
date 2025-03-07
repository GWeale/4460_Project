#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-BERT Model Visualization

This script visualizes the results of the Cell-BERT model for melanoma survival prediction.
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# Import our model
from cell_bert_model import CellBERTModel, WindowAggregator, CellWindowDataset


def load_data(data_path='protein_data.h5ad'):
    """Load and print basic information about the data"""
    print("1. Loading Data")
    adata = sc.read(data_path)
    print(f"Data shape: {adata.shape}")
    print(f"Number of patients: {len(adata.obs['donor'].unique())}")
    return adata


def visualize_patient_distributions(adata):
    """Visualize patient survival distribution"""
    print("\n2. Visualizing Patient Distributions")
    # Extract metadata
    metadata = adata.uns['metadata']
    
    # Plot survival distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(metadata['OS'], bins=15, kde=True)
    plt.axvline(x=24, color='red', linestyle='--', label='Survival Threshold (24 months)')
    plt.title('Overall Survival Distribution')
    plt.xlabel('Overall Survival (months)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('survival_distribution.png')
    plt.close()
    print("Saved 'survival_distribution.png'")


def visualize_spatial_distribution(adata):
    """Visualize spatial distribution of cells for a sample patient"""
    print("\n3. Visualizing Spatial Distribution of Cells")
    # Select a sample patient
    sample_patient = adata.obs['donor'].unique()[0]
    patient_mask = adata.obs['donor'] == sample_patient
    patient_data = adata[patient_mask]
    
    # Plot cell spatial distribution
    plt.figure(figsize=(10, 10))
    plt.scatter(patient_data.obs['X'], patient_data.obs['Y'], alpha=0.5, s=10)
    plt.title(f'Spatial Distribution of Cells for Patient {sample_patient}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().set_aspect('equal')
    plt.savefig('spatial_distribution.png')
    plt.close()
    print("Saved 'spatial_distribution.png'")
    
    return patient_data


def visualize_cell_neighborhoods(patient_data, window_size=15):
    """Visualize cell neighborhoods using nearest neighbors"""
    print("\n4. Visualizing Cell Neighborhoods")
    # Extract coordinates for the patient
    coords = np.column_stack([patient_data.obs['X'].values, patient_data.obs['Y'].values])
    
    # Find nearest neighbors
    nn_finder = NearestNeighbors(n_neighbors=window_size)
    nn_finder.fit(coords)
    distances, indices = nn_finder.kneighbors(coords)
    
    # Select a random cell
    np.random.seed(42)
    random_cell_idx = np.random.randint(0, len(coords))
    neighbor_indices = indices[random_cell_idx]
    
    # Plot the neighborhood
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.1, s=10, color='lightgray')
    plt.scatter(coords[neighbor_indices, 0], coords[neighbor_indices, 1], alpha=0.8, s=40, color='blue')
    plt.scatter(coords[random_cell_idx, 0], coords[random_cell_idx, 1], alpha=1.0, s=100, color='red')
    plt.title(f'Cell Neighborhood (Window Size = {window_size})')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().set_aspect('equal')
    plt.savefig('cell_neighborhood.png')
    plt.close()
    print("Saved 'cell_neighborhood.png'")


def create_and_load_model(adata, window_size=15, survival_threshold=24, batch_size=32):
    """Create and optionally load pre-trained Cell-BERT model"""
    print("\n5. Creating and Loading Model")
    # Create datasets
    dataset = CellWindowDataset(adata, window_size=window_size, 
                                survival_threshold=survival_threshold)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Total windows: {len(dataset)}")
    print(f"Training windows: {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create models
    cell_feature_dim = adata.X.shape[1]  # Number of protein markers
    d_model = 128  # Embedding dimension
    
    cell_bert_model = CellBERTModel(cell_feature_dim, d_model=d_model, 
                                   num_heads=4, dim_feedforward=256, 
                                   num_layers=3, dropout=0.1)
    
    window_aggregator = WindowAggregator(d_model, hidden_dim=64)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional: Load pre-trained models if available
    try:
        cell_bert_model.load_state_dict(torch.load('cell_bert_model.pt'))
        window_aggregator.load_state_dict(torch.load('window_aggregator.pt'))
        print("Loaded pre-trained models")
    except FileNotFoundError:
        print("No pre-trained models found. Will need to train from scratch.")
    
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    return cell_bert_model, window_aggregator, train_loader, val_loader, device


def extract_cell_embeddings(model, loader, device):
    """Extract CLS token embeddings from the model"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            features = batch['features'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)
            labels = batch['label'].squeeze().numpy()
            patient_ids = batch['patient_id']
            
            cls_output, _ = model(features, x_coords, y_coords)
            
            all_embeddings.append(cls_output.cpu().numpy())
            all_labels.extend(labels)
            all_patient_ids.extend(patient_ids)
    
    return np.vstack(all_embeddings), np.array(all_labels), np.array(all_patient_ids)


def visualize_embeddings(cell_bert_model, val_loader, device):
    """Visualize embeddings using PCA and t-SNE"""
    print("\n6. Extracting and Visualizing Cell Embeddings")
    
    # Extract embeddings from validation set
    embeddings, labels, patient_ids = extract_cell_embeddings(cell_bert_model, val_loader, device)
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Visualize embeddings using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], alpha=0.5, 
                    label=f"{'Long Survival' if label == 1 else 'Short Survival'}")
    
    plt.title('PCA of CLS Token Embeddings')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('pca_embeddings.png')
    plt.close()
    print("Saved 'pca_embeddings.png'")
    
    # Visualize embeddings using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], alpha=0.5, 
                    label=f"{'Long Survival' if label == 1 else 'Short Survival'}")
    
    plt.title('t-SNE of CLS Token Embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('tsne_embeddings.png')
    plt.close()
    print("Saved 'tsne_embeddings.png'")


def get_attention_weights(model, cell_features, x_coords, y_coords):
    """Get attention weights for a sample window"""
    model.eval()
    with torch.no_grad():
        # Get feature embedding
        batch_size, num_cells, _ = cell_features.size()
        embeddings = model.feature_embedding(cell_features)
        
        # Compute positional embeddings for each cell
        pos_embeddings = torch.zeros_like(embeddings)
        for i in range(batch_size):
            for j in range(num_cells):
                pos_embeddings[i, j] = model.pos_encoding(x_coords[i, j].unsqueeze(0), 
                                                         y_coords[i, j].unsqueeze(0))
        
        # Compute cell type embeddings
        cell_type_embeddings = model.cell_type_embedding(cell_features)
        
        # Combine embeddings
        embeddings = embeddings + pos_embeddings + cell_type_embeddings
        
        # Add CLS token
        cls_tokens = model.cls_token.repeat(batch_size, 1, 1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Get attention weights from the first layer
        transformer_block = model.transformer_blocks[0]
        q = transformer_block.attention.q_linear(embeddings)
        k = transformer_block.attention.k_linear(embeddings)
        
        # Reshape for multi-head attention
        head_dim = transformer_block.attention.head_dim
        num_heads = transformer_block.attention.num_heads
        
        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights


def visualize_attention_patterns(cell_bert_model, val_loader, device):
    """Visualize attention patterns in the model"""
    print("\n7. Visualizing Attention Patterns")
    
    # Get a sample batch
    sample_batch = next(iter(val_loader))
    sample_features = sample_batch['features'].to(device)
    sample_x_coords = sample_batch['x_coords'].to(device)
    sample_y_coords = sample_batch['y_coords'].to(device)
    
    # Get attention weights
    attention_weights = get_attention_weights(cell_bert_model, sample_features, sample_x_coords, sample_y_coords)
    
    # Visualize attention for the first sample in the batch and first attention head
    sample_idx = 0
    head_idx = 0
    
    plt.figure(figsize=(12, 10))
    plt.imshow(attention_weights[sample_idx, head_idx].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights (Head {head_idx})')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.savefig('attention_weights.png')
    plt.close()
    print("Saved 'attention_weights.png'")
    
    # Visualize CLS token attention to other tokens
    cls_attention = attention_weights[sample_idx, head_idx, 0, 1:].cpu().numpy()
    
    # Plot spatial attention
    plt.figure(figsize=(10, 10))
    
    # Get coordinates from the sample
    x_coords = sample_x_coords[sample_idx].cpu().numpy()
    y_coords = sample_y_coords[sample_idx].cpu().numpy()
    
    # Normalize attention weights for better visualization
    norm_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
    
    # Create scatter plot with size and color based on attention
    plt.scatter(x_coords, y_coords, s=norm_attention*500, c=norm_attention, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Normalized Attention from CLS Token')
    plt.title('Spatial Distribution of CLS Token Attention')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().set_aspect('equal')
    plt.savefig('spatial_attention.png')
    plt.close()
    print("Saved 'spatial_attention.png'")


def analyze_window_sizes(adata, window_sizes=[5, 10, 15, 20, 30]):
    """Analyze the impact of different window sizes on cell density"""
    results = []
    
    # Get all patient IDs
    patient_ids = adata.obs['donor'].unique()
    
    for patient_id in patient_ids[:5]:  # Analyze first 5 patients for speed
        patient_mask = adata.obs['donor'] == patient_id
        
        # Get coordinates
        x_coords = adata.obs['X'][patient_mask].values
        y_coords = adata.obs['Y'][patient_mask].values
        coords = np.column_stack([x_coords, y_coords])
        
        for window_size in window_sizes:
            # Create nearest neighbors finder
            nn_finder = NearestNeighbors(n_neighbors=window_size)
            nn_finder.fit(coords)
            
            # Random sample of 100 cells for speed
            sample_indices = np.random.choice(len(coords), min(100, len(coords)), replace=False)
            
            # Compute the average density of cells in the window
            total_area = 0
            for idx in sample_indices:
                center = coords[idx]
                distances, _ = nn_finder.kneighbors([center])
                max_distance = distances[0][-1]  # Furthest neighbor
                area = np.pi * (max_distance ** 2)  # Area of the circle
                total_area += area
            
            avg_area = total_area / len(sample_indices)
            avg_density = window_size / avg_area if avg_area > 0 else 0
            
            results.append({
                'patient_id': patient_id,
                'window_size': window_size,
                'avg_area': avg_area,
                'avg_density': avg_density
            })
    
    return pd.DataFrame(results)


def analyze_window_size_impact(adata):
    """Analyze and visualize the impact of different window sizes"""
    print("\n8. Analyzing Window Size Impact")
    
    # Analyze window sizes
    window_size_analysis = analyze_window_sizes(adata)
    
    # Plot window size analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='window_size', y='avg_density', data=window_size_analysis)
    plt.title('Cell Density for Different Window Sizes')
    plt.xlabel('Window Size (number of cells)')
    plt.ylabel('Cell Density (cells per unit area)')
    plt.savefig('window_size_analysis.png')
    plt.close()
    print("Saved 'window_size_analysis.png'")


def main():
    """Main function to run all visualization steps"""
    print("Cell-BERT Model Visualization")
    print("============================")
    
    # 1. Load data
    adata = load_data()
    
    # 2. Visualize patient distributions
    visualize_patient_distributions(adata)
    
    # 3. Visualize spatial distribution
    patient_data = visualize_spatial_distribution(adata)
    
    # 4. Visualize cell neighborhoods
    visualize_cell_neighborhoods(patient_data)
    
    # 5. Create and load model
    cell_bert_model, window_aggregator, train_loader, val_loader, device = create_and_load_model(adata)
    
    # 6. Extract and visualize cell embeddings
    visualize_embeddings(cell_bert_model, val_loader, device)
    
    # 7. Visualize attention patterns
    visualize_attention_patterns(cell_bert_model, val_loader, device)
    
    # 8. Analyze window size impact
    analyze_window_size_impact(adata)
    
    print("\nVisualization complete. All results saved as PNG files.")


if __name__ == "__main__":
    main() 