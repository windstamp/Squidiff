import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(self, adata, control_adata=None,use_drug_structure=False,comb_num=1):
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
                
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            return {'feature':self.features[idx], 'drug_dose':self.encode_drug_doses[idx], 'group': self.encoded_obs_tensor[idx],'control_feature':self.control_features[idx]}
        else:
            return {'feature':self.features[idx], 'group': self.encoded_obs_tensor[idx]}
            
    

def create_mock_adata(num_cells=1000, num_genes=100, num_groups=3, use_drug_structure=False):
    """
    Create mock AnnData object for testing.
    
    Parameters:
    -----------
    num_cells : int
        Number of cells (samples)
    num_genes : int
        Number of genes (features)
    num_groups : int
        Number of different groups/cell types
    use_drug_structure : bool
        Whether to include drug structure information
    
    Returns:
    --------
    adata : AnnData
        Mock AnnData object
    """
    import anndata
    
    np.random.seed(42)
    
    X = np.random.randn(num_cells, num_genes).astype(np.float32)
    X = np.maximum(X, 0)
    
    adata = anndata.AnnData(X)
    
    adata.obs['Group'] = np.random.randint(1, num_groups + 1, size=num_cells)
    
    if use_drug_structure:
        mock_smiles = [
            'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
            'CC(=O)Oc1ccccc1C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ]
        adata.obs['SMILES'] = np.random.choice(mock_smiles, size=num_cells)
        adata.obs['dose'] = np.random.uniform(0.1, 10.0, size=num_cells)
    
    return adata


def prepared_data(data_dir=None, control_data_dir=None, batch_size=64, 
                  use_drug_structure=False, comb_num=1, use_mock_data=False,
                  mock_num_cells=1000, mock_num_genes=100):
    """
    Prepare data loader for training.
    
    Parameters:
    -----------
    data_dir : str
        Path to h5ad file
    control_data_dir : str
        Path to control h5ad file (for drug structure mode)
    batch_size : int
        Batch size for DataLoader
    use_drug_structure : bool
        Whether to use drug structure information
    comb_num : int
        Number of drug combinations
    use_mock_data : bool
        If True, use mock data instead of loading from file
    mock_num_cells : int
        Number of cells in mock data
    mock_num_genes : int
        Number of genes in mock data
    
    Returns:
    --------
    dataloader : DataLoader
        PyTorch DataLoader object
    """
    
    if use_mock_data:
        print(f"Using mock data: {mock_num_cells} cells, {mock_num_genes} genes")
        train_adata = create_mock_adata(
            num_cells=mock_num_cells,
            num_genes=mock_num_genes,
            use_drug_structure=use_drug_structure
        )
        if use_drug_structure:
            control_adata = create_mock_adata(
                num_cells=mock_num_cells,
                num_genes=mock_num_genes,
                use_drug_structure=False
            )
        else:
            control_adata = None
    else:
        train_adata = sc.read_h5ad(data_dir)
        if use_drug_structure:
            control_adata = sc.read_h5ad(control_data_dir)
        else:
            control_adata = None
    
    _data_dataset = AnnDataDataset(train_adata, control_adata, use_drug_structure, comb_num)

    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader