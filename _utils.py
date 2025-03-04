import pandas as pd
import scanpy as sc
from anndata import AnnData
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_anndata():
    df = pd.read_csv('23_10_11_Melanoma_Marker_Cell_Neighborhood.csv', delimiter=',', header=0,index_col=0)  
    metadata = pd.read_csv('metadata.csv', delimiter=',', header=0)
    protein_data = df.iloc[:,:57]
    obs = df.iloc[:,57:]
    adata = AnnData(protein_data, obs=obs)
    adata.uns['metadata'] = metadata
    # for each patient,  normalize the protein data and log1p transform
    for donor in adata.obs['donor'].unique():
        mask = adata.obs['donor'] == donor
        sc.pp.normalize_total(adata[mask])
        sc.pp.log1p(adata[mask])
    adata.write('protein_data.h5ad')
    overall_survival = metadata['OS']
    os_binary = overall_survival<np.mean(overall_survival)
    adata.obs['os_binary'] = os_binary
    return adata