import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt

def create_anndata():
    df = pd.read_csv('23_10_11_Melanoma_Marker_Cell_Neighborhood.csv', delimiter=',', header=0,index_col=0)  
    metadata = pd.read_csv('metadata.csv', delimiter=',', header=0)
    protein_data = df.iloc[:,:56]
    obs = df.iloc[:,56:]
    adata = AnnData(protein_data, obs=obs)
    adata.uns['metadata'] = metadata
    adata.write('protein_data.h5ad')
    return adata