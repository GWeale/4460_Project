import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Metadata (Human)
def load_metadata(file_path='metadata.csv'):
    """Load patient metadata and return as DataFrame"""
    metadata_df = pd.read_csv(file_path)
    print(f"Loaded metadata with {metadata_df.shape[0]} patients and {metadata_df.shape[1]} columns")
    return metadata_df

# 2. Define Target Variable
def define_target_variable(metadata_df, os_threshold=None):
    """
    Create binary target variable based on OS (Overall Survival)
    
    Parameters:
    - metadata_df: DataFrame with patient metadata
    - os_threshold: Threshold in months to define high/low survival (default: median OS)
    
    Returns:
    - metadata_df with new High_Survival column
    """
    # Handle missing OS values if any
    if metadata_df['OS'].isna().any():
        print(f"Warning: {metadata_df['OS'].isna().sum()} patients have missing OS values")
        
    # If no threshold provided, use median OS
    if os_threshold is None:
        os_threshold = metadata_df['OS'].median()
        print(f"Using median OS as threshold: {os_threshold} months")
    
    # Create binary target
    metadata_df['High_Survival'] = (metadata_df['OS'] > os_threshold).astype(int)
    print(f"Created binary target: {metadata_df['High_Survival'].sum()} high survival, "
          f"{metadata_df.shape[0] - metadata_df['High_Survival'].sum()} low survival")
    
    return metadata_df

# 3. Load Cell Data (Human)
def load_cell_data(file_path='Melanoma_data.csv'):
    """Load single-cell data and return as DataFrame"""
    cell_df = pd.read_csv(file_path, index_col=0)
    print(f"Loaded cell data with {cell_df.shape[0]} cells and {cell_df.shape[1]} columns")
    return cell_df

# 4. Merge Target Variable
def merge_target_with_cell_data(cell_df, metadata_df):
    """
    Merge the binary target variable from metadata onto the cell data
    
    Parameters:
    - cell_df: DataFrame with single-cell data
    - metadata_df: DataFrame with patient metadata including High_Survival column
    
    Returns:
    - cell_df with High_Survival column added
    """
    # Create a lookup dictionary from donor to High_Survival
    donor_to_survival = dict(zip(metadata_df['donor'], metadata_df['High_Survival']))
    
    # Add High_Survival to cell data
    cell_df['High_Survival'] = cell_df['donor'].map(donor_to_survival)
    
    # Check if any cells have missing target
    missing_target = cell_df['High_Survival'].isna().sum()
    if missing_target > 0:
        print(f"Warning: {missing_target} cells ({missing_target/cell_df.shape[0]:.1%}) have missing survival data")
    
    return cell_df

# 5. Identify Feature Columns
def identify_feature_columns(cell_df):
    """
    Identify and categorize columns in the cell data
    
    Parameters:
    - cell_df: DataFrame with single-cell data
    
    Returns:
    - dict with categorized column names
    """
    # From the data sample, we can identify these categories
    # This may need adjustment based on the full dataset
    
    # Identify marker columns (biological markers like CCR7 to pERK1/2)
    # Exclude DAPI as it's a nuclear stain, not a biological marker
    marker_columns = [col for col in cell_df.columns if col not in [
        'cellid', 'donor', 'filename', 'region', 'x', 'y', 
        'Cell_Type', 'Cell_Type_Common', 'Cell_Type_Sub', 'Overall_Cell_Type',
        'Neighborhood', 'DAPI', 'High_Survival'
    ]]
    
    # Coordinate columns
    coordinate_columns = ['x', 'y']
    
    # Cell type columns (categorical)
    cell_type_columns = ['Cell_Type', 'Cell_Type_Common', 'Cell_Type_Sub', 'Overall_Cell_Type']
    
    # Group identifiers
    group_columns = ['donor', 'filename', 'region']
    
    feature_columns = {
        'markers': marker_columns,
        'coordinates': coordinate_columns,
        'cell_types': cell_type_columns,
        'groups': group_columns
    }
    
    # Print summary
    print(f"Identified {len(marker_columns)} marker columns")
    print(f"Identified {len(coordinate_columns)} coordinate columns")
    print(f"Identified {len(cell_type_columns)} cell type columns")
    print(f"Identified {len(group_columns)} group identifier columns")
    
    return feature_columns

# 6. Data Splitting (Patient-Level)
def split_data(cell_df, metadata_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the data at the patient (donor) level
    
    Parameters:
    - cell_df: DataFrame with single-cell data
    - metadata_df: DataFrame with patient metadata
    - test_size: Proportion of donors to use for testing
    - val_size: Proportion of donors to use for validation (from training set)
    - random_state: Random seed for reproducibility
    
    Returns:
    - dict with train/val/test DataFrames for both cells and metadata
    """
    # Get unique donors
    unique_donors = metadata_df['donor'].unique()
    
    # First split: training+validation vs test
    train_val_donors, test_donors = train_test_split(
        unique_donors, test_size=test_size, random_state=random_state
    )
    
    # Second split: training vs validation
    train_donors, val_donors = train_test_split(
        train_val_donors, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Filter cell data based on donor splits
    train_cells = cell_df[cell_df['donor'].isin(train_donors)]
    val_cells = cell_df[cell_df['donor'].isin(val_donors)]
    test_cells = cell_df[cell_df['donor'].isin(test_donors)]
    
    # Filter metadata based on donor splits
    train_metadata = metadata_df[metadata_df['donor'].isin(train_donors)]
    val_metadata = metadata_df[metadata_df['donor'].isin(val_donors)]
    test_metadata = metadata_df[metadata_df['donor'].isin(test_donors)]
    
    # Print summary
    print(f"Data split complete:")
    print(f"  Training: {len(train_donors)} donors, {train_cells.shape[0]} cells")
    print(f"  Validation: {len(val_donors)} donors, {val_cells.shape[0]} cells")
    print(f"  Test: {len(test_donors)} donors, {test_cells.shape[0]} cells")
    
    return {
        'train_cells': train_cells,
        'val_cells': val_cells,
        'test_cells': test_cells,
        'train_metadata': train_metadata,
        'val_metadata': val_metadata,
        'test_metadata': test_metadata,
        'train_donors': train_donors,
        'val_donors': val_donors,
        'test_donors': test_donors
    }

# 7. Batch Correction (Optional)
def apply_batch_correction(data_dict, feature_columns, batch_key='donor'):
    """
    Apply batch correction to marker columns
    
    This is a placeholder for batch correction. In a real implementation,
    you might want to use scanpy.pp.combat or other batch correction methods.
    
    Parameters:
    - data_dict: Dict with train/val/test DataFrames
    - feature_columns: Dict with categorized column names
    - batch_key: Column to use as batch identifier (e.g., 'donor')
    
    Returns:
    - data_dict with batch-corrected DataFrames
    """
    print("Batch correction would be applied here")
    print("Recommended: Use scanpy.pp.combat or similar methods")
    print(f"Using '{batch_key}' as batch key")
    
    # This is where you would implement batch correction
    # For example:
    # import scanpy as sc
    # adata = sc.AnnData(X=train_cells[feature_columns['markers']])
    # adata.obs[batch_key] = train_cells[batch_key]
    # sc.pp.combat(adata, key=batch_key)
    
    return data_dict

# 8. Final Normalization
def normalize_features(data_dict, feature_columns):
    """
    Apply normalization to marker columns if needed
    
    Parameters:
    - data_dict: Dict with train/val/test DataFrames
    - feature_columns: Dict with categorized column names
    
    Returns:
    - data_dict with normalized DataFrames
    - scaler: Fitted StandardScaler for later use
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler on training data
    scaler.fit(data_dict['train_cells'][feature_columns['markers']])
    
    # Transform all datasets
    for split in ['train', 'val', 'test']:
        # Create a copy to avoid modifying the original
        data_dict[f'{split}_cells_normalized'] = data_dict[f'{split}_cells'].copy()
        
        # Apply scaling to marker columns
        data_dict[f'{split}_cells_normalized'][feature_columns['markers']] = scaler.transform(
            data_dict[f'{split}_cells'][feature_columns['markers']]
        )
    
    print("Applied normalization to marker columns")
    
    return data_dict, scaler

# Main execution function
def prepare_data(metadata_path='metadata.csv', cell_data_path='Melanoma_data.csv', 
                os_threshold=None, apply_batch_corr=False):
    """
    Execute the full data preparation pipeline
    
    Parameters:
    - metadata_path: Path to metadata CSV
    - cell_data_path: Path to cell data CSV
    - os_threshold: Threshold for high/low survival (default: median)
    - apply_batch_corr: Whether to apply batch correction
    
    Returns:
    - processed_data: Dict with processed DataFrames and metadata
    """
    # 1. Load metadata
    metadata_df = load_metadata(metadata_path)
    
    # 2. Define target variable
    metadata_df = define_target_variable(metadata_df, os_threshold)
    
    # 3. Load cell data
    cell_df = load_cell_data(cell_data_path)
    
    # 4. Merge target with cell data
    cell_df = merge_target_with_cell_data(cell_df, metadata_df)
    
    # 5. Identify feature columns
    feature_columns = identify_feature_columns(cell_df)
    
    # 6. Split data
    data_dict = split_data(cell_df, metadata_df)
    
    # 7. Apply batch correction if requested
    if apply_batch_corr:
        data_dict = apply_batch_correction(data_dict, feature_columns)
    
    # 8. Normalize features
    data_dict, scaler = normalize_features(data_dict, feature_columns)
    
    # Add extra metadata to the return dict
    data_dict['feature_columns'] = feature_columns
    data_dict['scaler'] = scaler
    data_dict['metadata_full'] = metadata_df
    data_dict['cell_data_full'] = cell_df
    
    return data_dict

if __name__ == "__main__":
    # Example usage
    processed_data = prepare_data()
    
    # You can access the processed data like:
    # train_cells = processed_data['train_cells']
    # marker_columns = processed_data['feature_columns']['markers']
    
    print("Data preparation complete!") 