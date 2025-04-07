import argparse
import torch
import scanpy as sc
from torch.utils.data import DataLoader
from cell_bert_model import LightCellBERTModel, WindowAggregator, CellWindowDataset, quick_train_model, create_anndata_from_csv

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cell-BERT model for melanoma survival prediction')
    
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
                        help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension for the model')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--output_prefix', type=str, default='cell_bert',
                        help='Prefix for output model files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from CSV files:")
    print(f"  - Melanoma data: {args.melanoma_data}")
    print(f"  - Markers data: {args.markers_data}")
    print(f"  - Metadata: {args.metadata}")
    
    # Create AnnData from CSV files with reduced sampling ratio
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
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training windows, {val_size} validation windows")
    
    # Create data loaders with parallel loading
    print(f"Creating data loaders with batch size {args.batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    # Determine cell feature dimension from the data
    cell_feature_dim = adata.X.shape[1]
    
    print(f"Creating lightweight Cell-BERT model:")
    print(f"  - Feature dimension: {cell_feature_dim}")
    print(f"  - Embedding dimension: {args.embedding_dim}")
    print(f"  - Attention heads: {args.num_heads}")
    print(f"  - Transformer layers: {args.num_layers}")
    
    # Create lightweight model with reduced parameters
    cell_bert_model = LightCellBERTModel(
        cell_feature_dim, 
        d_model=args.embedding_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.embedding_dim * 2,
        num_layers=args.num_layers
    )
    
    print("Creating window aggregator model")
    window_aggregator = WindowAggregator(args.embedding_dim, hidden_dim=args.embedding_dim // 2)
    
    # Set device - try to use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Train model with accelerated settings
    print(f"Training for {args.epochs} epochs with learning rate {args.lr}")
    cell_bert_model, window_aggregator = quick_train_model(
        cell_bert_model, window_aggregator, 
        train_loader, val_loader, 
        device, 
        num_epochs=args.epochs, 
        lr=args.lr
    )
    
    # Save models
    bert_model_path = f"{args.output_prefix}_model.pt"
    aggregator_path = f"{args.output_prefix}_aggregator.pt"
    
    print(f"Saving models to {bert_model_path} and {aggregator_path}")
    torch.save(cell_bert_model.state_dict(), bert_model_path)
    torch.save(window_aggregator.state_dict(), aggregator_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 