#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-BERT Model Visualization

This script visualizes the results of the Cell-BERT model for melanoma survival prediction.
"""

import os
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
from cell_bert_model import (
    LightCellBERTModel, WindowAggregator, CellWindowDataset, 
    create_anndata_from_csv
)


def load_data(melanoma_data='Melanoma_data.csv', 
              markers_data='Day3_Markers_Dryad.csv',
              metadata='metadata.csv',
              sample_ratio=0.02):
    """
    Load data from CSV files and create an AnnData object.
    
    Args:
        melanoma_data: Path to melanoma data CSV
        markers_data: Path to markers data CSV
        metadata: Path to metadata CSV
        sample_ratio: Ratio of cells to sample (to reduce memory usage)
    
    Returns:
        AnnData object containing the data
    """
    print(f"Loading data from CSV files:")
    print(f"  - Melanoma data: {melanoma_data}")
    print(f"  - Markers data: {markers_data}")
    print(f"  - Metadata: {metadata}")
    
    # Create AnnData from CSV files with a small sample ratio for visualization
    adata = create_anndata_from_csv(
        melanoma_data,
        markers_data,
        metadata,
        sample_ratio=sample_ratio
    )
    
    print(f"Data shape: {adata.X.shape} cells, {adata.X.shape[1]} features")
    print(f"Number of patients: {len(adata.obs['patient_id'].unique())}")
    
    return adata


def visualize_patient_distribution(adata, output_path="patient_survival_distribution.png"):
    """
    Visualize the distribution of patient survival times.
    
    Args:
        adata: AnnData object containing patient metadata
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique patients and their survival times
    patients = adata.obs[['patient_id', 'overall_survival']].drop_duplicates()
    
    # Create a histogram of survival times
    sns.histplot(patients['overall_survival'], bins=20, kde=True)
    plt.xlabel('Overall Survival Time (months)')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Patient Survival Times')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Patient survival distribution saved to {output_path}")
    
    # Print some basic statistics
    print(f"Survival time statistics:")
    print(f"  Mean: {patients['overall_survival'].mean():.1f} months")
    print(f"  Median: {patients['overall_survival'].median():.1f} months")
    print(f"  Min: {patients['overall_survival'].min():.1f} months")
    print(f"  Max: {patients['overall_survival'].max():.1f} months")


def visualize_spatial_distribution(adata, patient_id=None, output_path="spatial_distribution.png"):
    """
    Visualize the spatial distribution of cells for a patient.
    
    Args:
        adata: AnnData object containing cell data
        patient_id: ID of the patient to visualize (if None, selects the first patient)
        output_path: Path to save the visualization
    """
    if patient_id is None:
        # Get the first patient with a reasonable number of cells
        patients = adata.obs['patient_id'].value_counts()
        for pid, count in patients.items():
            if count > 100:
                patient_id = pid
                break
    
    # Filter data for the selected patient
    patient_data = adata[adata.obs['patient_id'] == patient_id]
    
    if len(patient_data) == 0:
        print(f"No data found for patient {patient_id}")
        return
    
    print(f"Visualizing spatial distribution for patient {patient_id} with {len(patient_data)} cells")
    
    plt.figure(figsize=(10, 8))
    
    # Get cell coordinates
    x_coords = patient_data.obs['x_coordinate'].values
    y_coords = patient_data.obs['y_coordinate'].values
    
    # Create a scatter plot of cell positions
    plt.scatter(x_coords, y_coords, s=10, alpha=0.7)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Spatial Distribution of Cells for Patient {patient_id}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Spatial distribution visualization saved to {output_path}")


def visualize_cell_neighborhood(adata, window_size=10, patient_id=None, 
                               output_path="cell_neighborhood.png"):
    """
    Visualize a typical cell neighborhood (window) used by the model.
    
    Args:
        adata: AnnData object containing cell data
        window_size: Number of cells in each neighborhood
        patient_id: ID of the patient to visualize (if None, selects the first patient)
        output_path: Path to save the visualization
    """
    if patient_id is None:
        # Get the first patient with a reasonable number of cells
        patients = adata.obs['patient_id'].value_counts()
        for pid, count in patients.items():
            if count > window_size * 2:
                patient_id = pid
                break
    
    # Filter data for the selected patient
    patient_data = adata[adata.obs['patient_id'] == patient_id]
    
    if len(patient_data) < window_size:
        print(f"Not enough cells for patient {patient_id} to visualize a neighborhood")
        return
    
    # Get coordinates for all cells
    all_coords = np.vstack([
        patient_data.obs['x_coordinate'].values,
        patient_data.obs['y_coordinate'].values
    ]).T
    
    # Select a random central cell
    center_idx = np.random.randint(0, len(patient_data))
    center_x, center_y = all_coords[center_idx]
    
    # Calculate distances from the central cell to all other cells
    distances = np.sqrt(np.sum((all_coords - np.array([center_x, center_y])) ** 2, axis=1))
    
    # Find the window_size nearest neighbors
    nearest_indices = np.argsort(distances)[:window_size]
    
    # Create a visualization of the neighborhood
    plt.figure(figsize=(8, 8))
    
    # Plot all cells in the patient
    plt.scatter(all_coords[:, 0], all_coords[:, 1], s=5, color='lightgray', alpha=0.4, label='Other cells')
    
    # Highlight the window cells
    window_coords = all_coords[nearest_indices]
    plt.scatter(window_coords[:, 0], window_coords[:, 1], s=30, color='blue', alpha=0.7, label='Window cells')
    
    # Highlight the central cell
    plt.scatter(center_x, center_y, s=80, color='red', marker='*', label='Central cell')
    
    # Draw lines from the central cell to each neighbor
    for i in range(1, len(nearest_indices)):
        x, y = window_coords[i]
        plt.plot([center_x, x], [center_y, y], 'k-', alpha=0.2)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Cell Neighborhood (Window Size = {window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Cell neighborhood visualization saved to {output_path}")


def create_or_load_model(adata, embedding_dim=64, model_path=None, aggregator_path=None):
    """
    Create a new model or load a pre-trained model.
    
    Args:
        adata: AnnData object to determine feature dimensions
        embedding_dim: Dimension of embeddings
        model_path: Path to pre-trained model (optional)
        aggregator_path: Path to pre-trained aggregator (optional)
        
    Returns:
        cell_bert_model: The BERT-style model
        window_aggregator: The window aggregation model
        device: The device to run the model on
    """
    # Determine cell feature dimension from the data
    cell_feature_dim = adata.X.shape[1]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create lightweight models
    print("Creating lightweight Cell-BERT model")
    cell_bert_model = LightCellBERTModel(
        cell_feature_dim, 
        d_model=embedding_dim,
        num_heads=2,
        dim_feedforward=embedding_dim * 2,
        num_layers=2
    )
    
    window_aggregator = WindowAggregator(embedding_dim, hidden_dim=embedding_dim // 2)
    
    # Try to load pre-trained models if specified
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading pre-trained model from {model_path}")
            cell_bert_model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model instead")
    else:
        print("Using untrained model (no pre-trained model specified or found)")
    
    if aggregator_path and os.path.exists(aggregator_path):
        try:
            print(f"Loading pre-trained aggregator from {aggregator_path}")
            window_aggregator.load_state_dict(torch.load(aggregator_path, map_location=device))
        except Exception as e:
            print(f"Error loading aggregator: {e}")
            print("Using untrained aggregator instead")
    else:
        print("Using untrained aggregator (no pre-trained aggregator specified or found)")
    
    cell_bert_model = cell_bert_model.to(device)
    window_aggregator = window_aggregator.to(device)
    
    return cell_bert_model, window_aggregator, device


def visualize_embeddings(adata, cell_bert_model, device, window_size=10, max_windows=50,
                         output_path_prefix="embedding"):
    """
    Extract CLS token embeddings for cells and visualize using PCA and t-SNE.
    
    Args:
        adata: AnnData object containing cell data
        cell_bert_model: The BERT-style model to extract embeddings
        device: Device to run the model on
        window_size: Number of cells in each window
        max_windows: Maximum number of windows per patient for memory efficiency
        output_path_prefix: Prefix for output files
    """
    print("Creating dataset for embedding visualization")
    dataset = CellWindowDataset(adata, window_size=window_size, 
                              survival_threshold=24, 
                              max_windows_per_patient=max_windows)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"Extracting embeddings from {len(dataset)} windows")
    
    # Extract CLS token embeddings
    cell_bert_model.eval()
    cls_embeddings = []
    window_labels = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)
            labels = batch['label'].cpu().numpy()
            p_ids = batch['patient_id']
            
            # Forward pass to get CLS embeddings
            cls_outputs, _ = cell_bert_model(features, x_coords, y_coords)
            
            cls_embeddings.append(cls_outputs.cpu().numpy())
            window_labels.append(labels)
            patient_ids.extend(p_ids)
    
    # Combine results
    cls_embeddings = np.vstack(cls_embeddings)
    window_labels = np.concatenate(window_labels)
    
    print(f"Generated {cls_embeddings.shape[0]} embeddings")
    
    # Create PCA visualization
    print("Applying PCA to reduce embedding dimensions")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(cls_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=window_labels, 
                         cmap='coolwarm', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Survival Category')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Cell-BERT CLS Token Embeddings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_pca.png")
    plt.close()
    
    print(f"PCA visualization saved to {output_path_prefix}_pca.png")
    
    # Create t-SNE visualization (can be slow for large datasets)
    if cls_embeddings.shape[0] > 5000:
        print("Too many points for t-SNE, sampling 5000 points")
        indices = np.random.choice(cls_embeddings.shape[0], 5000, replace=False)
        embeddings_sample = cls_embeddings[indices]
        labels_sample = window_labels[indices]
    else:
        embeddings_sample = cls_embeddings
        labels_sample = window_labels
    
    print("Applying t-SNE to reduce embedding dimensions (this may take a while)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_sample, 
                         cmap='coolwarm', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Survival Category')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of Cell-BERT CLS Token Embeddings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_tsne.png")
    plt.close()
    
    print(f"t-SNE visualization saved to {output_path_prefix}_tsne.png")


def visualize_attention(adata, cell_bert_model, device, window_size=10, 
                        output_path="attention_pattern.png"):
    """
    Visualize attention patterns from the transformer model.
    Shows how the CLS token attends to different cells in a window.
    
    Args:
        adata: AnnData object containing cell data
        cell_bert_model: The BERT-style model
        device: Device to run the model on
        window_size: Number of cells in each window
        output_path: Path to save the visualization
    """
    print("Creating dataset for attention visualization")
    dataset = CellWindowDataset(adata, window_size=window_size, 
                              survival_threshold=24,
                              max_windows_per_patient=10)  # Just need a few windows
    
    if len(dataset) == 0:
        print("No data available for attention visualization")
        return
    
    # Get a single batch
    sample_idx = np.random.randint(0, len(dataset))
    sample = dataset[sample_idx]
    
    batch = {
        'features': sample['features'].unsqueeze(0).to(device),
        'x_coords': sample['x_coords'].unsqueeze(0).to(device),
        'y_coords': sample['y_coords'].unsqueeze(0).to(device),
        'patient_id': [sample['patient_id']]
    }
    
    # Get attention maps from the model
    cell_bert_model.eval()
    with torch.no_grad():
        _, attn_maps = cell_bert_model(
            batch['features'], 
            batch['x_coords'], 
            batch['y_coords'],
            return_attention=True
        )
    
    # Take the last layer's attention map
    last_layer_attn = attn_maps[-1][0]  # Shape: [num_heads, seq_len, seq_len]
    
    # Visualize how the CLS token (index 0) attends to other tokens
    cls_attention = last_layer_attn[:, 0, 1:].cpu().numpy()  # Shape: [num_heads, window_size]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cls_attention, cmap='viridis', 
               xticklabels=[f'Cell {i+1}' for i in range(window_size)],
               yticklabels=[f'Head {i+1}' for i in range(cls_attention.shape[0])])
    plt.xlabel('Cell Position in Window')
    plt.ylabel('Attention Head')
    plt.title('Attention from CLS Token to Cells')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Attention visualization saved to {output_path}")
    
    # Also visualize average attention across heads
    plt.figure(figsize=(10, 4))
    avg_attention = cls_attention.mean(axis=0)
    plt.bar(range(1, window_size + 1), avg_attention)
    plt.xlabel('Cell Position in Window')
    plt.ylabel('Average Attention Score')
    plt.title('Average Attention from CLS Token to Cells (Across All Heads)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_avg.png'))
    plt.close()
    
    print(f"Average attention visualization saved to {output_path.replace('.png', '_avg.png')}")


def analyze_window_sizes(adata, window_sizes=[5, 10, 15, 20, 30], 
                        output_path="window_size_analysis.png"):
    """
    Analyze the impact of different window sizes on cell density.
    
    Args:
        adata: AnnData object containing cell data
        window_sizes: List of window sizes to analyze
        output_path: Path to save the visualization
    """
    print("Analyzing impact of different window sizes")
    
    # Get all patients with sufficient number of cells
    patients = adata.obs['patient_id'].value_counts()
    viable_patients = patients[patients > max(window_sizes)].index.tolist()
    
    if len(viable_patients) == 0:
        print("No patients with sufficient cells for window size analysis")
        return
    
    # Sample some patients for analysis
    if len(viable_patients) > 10:
        selected_patients = np.random.choice(viable_patients, 10, replace=False)
    else:
        selected_patients = viable_patients
    
    # Analyze window densities for different window sizes
    results = []
    
    for patient_id in selected_patients:
        patient_data = adata[adata.obs['patient_id'] == patient_id]
        
        # Get coordinates for all cells
        all_coords = np.vstack([
            patient_data.obs['x_coordinate'].values,
            patient_data.obs['y_coordinate'].values
        ]).T
        
        for window_size in window_sizes:
            # Sample some central points
            if len(patient_data) > 20:
                center_indices = np.random.choice(len(patient_data), 20, replace=False)
            else:
                center_indices = range(len(patient_data))
            
            # Calculate average window density
            densities = []
            
            for center_idx in center_indices:
                center_coords = all_coords[center_idx]
                
                # Calculate distances from center to all other cells
                distances = np.sqrt(np.sum((all_coords - center_coords) ** 2, axis=1))
                
                # Get the window_size nearest neighbors
                nearest_distances = np.sort(distances)[:window_size]
                
                # Calculate average distance (a proxy for density)
                avg_distance = np.mean(nearest_distances)
                densities.append(avg_distance)
            
            # Store results
            results.append({
                'patient_id': patient_id,
                'window_size': window_size,
                'avg_density': np.mean(densities)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='window_size', y='avg_density', data=results_df)
    plt.xlabel('Window Size (Number of Cells)')
    plt.ylabel('Average Distance Between Cells')
    plt.title('Impact of Window Size on Cell Neighborhood Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Window size analysis saved to {output_path}")


def main():
    """Run all visualization steps."""
    # Load data
    adata = load_data()
    
    # Visualize patient distribution
    visualize_patient_distribution(adata)
    
    # Visualize spatial distribution of cells
    visualize_spatial_distribution(adata)
    
    # Visualize cell neighborhood
    visualize_cell_neighborhood(adata, window_size=10)
    
    # Create or load model
    cell_bert_model, window_aggregator, device = create_or_load_model(
        adata,
        embedding_dim=64,
        model_path='cell_bert_model.pt',
        aggregator_path='cell_bert_aggregator.pt'
    )
    
    # Visualize embeddings
    visualize_embeddings(adata, cell_bert_model, device, window_size=10, max_windows=50)
    
    # Visualize attention pattern
    visualize_attention(adata, cell_bert_model, device, window_size=10)
    
    # Analyze impact of different window sizes
    analyze_window_sizes(adata)
    
    print("Visualization complete! All plots saved to current directory.")


if __name__ == "__main__":
    main() 