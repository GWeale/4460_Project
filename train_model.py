import argparse
import torch
import scanpy as sc
from torch.utils.data import DataLoader
from cell_bert_model import CellBERTModel, WindowAggregator, CellWindowDataset, train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cell-BERT model for melanoma survival prediction')
    
    parser.add_argument('--data_path', type=str, default='protein_data.h5ad',
                        help='Path to AnnData (.h5ad) file with protein expression data')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Number of cells in each window (neighborhood)')
    parser.add_argument('--survival_threshold', type=int, default=24,
                        help='Threshold in months for binary survival classification')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension for the model')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--output_prefix', type=str, default='cell_bert',
                        help='Prefix for output model files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data_path}")
    adata = sc.read(args.data_path)
    
    print(f"Creating dataset with window size {args.window_size}")
    dataset = CellWindowDataset(adata, window_size=args.window_size, 
                                survival_threshold=args.survival_threshold)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Creating data loaders with batch size {args.batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    cell_feature_dim = adata.X.shape[1]
    
    print(f"Creating CellBERT model with embedding dimension {args.embedding_dim}")
    cell_bert_model = CellBERTModel(
        cell_feature_dim, 
        d_model=args.embedding_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.embedding_dim * 2,
        num_layers=args.num_layers
    )
    
    print("Creating window aggregator model")
    window_aggregator = WindowAggregator(args.embedding_dim, hidden_dim=args.embedding_dim // 2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Training for {args.epochs} epochs with learning rate {args.lr}")
    cell_bert_model, window_aggregator = train_model(
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