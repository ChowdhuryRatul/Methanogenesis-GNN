# Date: 07.11.2024
# Author Noor

import rdkit
from rdkit import Chem
import pickle
import numpy as np

# Database of molecules in SDF format containing all entries.
# Extracts the SMILES string and generic names for all the entries.

'''
This labels each SMILES with generic names for latter identification.
'''

def extract_mapped_data_from_sdf(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    mapped_data = []
    current_smiles = None
    current_generic_name = None
    
    for i, line in enumerate(lines):
        if line.strip() == '> <SMILES>':
            current_smiles = lines[i + 1].strip()
        elif line.strip() == '> <GENERIC_NAME>':
            current_generic_name = lines[i + 1].strip()
        elif line.strip() == '$$$$':
            if current_smiles and current_generic_name:
                m = Chem.MolFromSmiles(current_smiles)
                if m is not None:
                    rdkit_smiles = Chem.MolToSmiles(m, kekuleSmiles=True)
                else:
                    rdkit_smiles = None
                
                mapped_data.append((current_smiles, rdkit_smiles, current_generic_name))
            
            current_smiles = None
            current_generic_name = None
    
    return mapped_data


'''
Only for preprocessing data from MCR inhibitor molecules from txt file
'''

def extract_mapped_data_from_txt(file_path):
    compounds = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            category, generic_name, iupac_name, smiles = line.split('|')
            compounds.append((category, generic_name, iupac_name, smiles))

    mapped_data = []
    for category, generic_name, iupac_name, original_smiles in compounds:
        # Convert to RDKit format
        mol = Chem.MolFromSmiles(original_smiles)
        if mol is not None:
            rdkit_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        else:
            rdkit_smiles = None
        
        mapped_data.append((original_smiles, rdkit_smiles, generic_name))
    
    return mapped_data


'''
Loading embedding from pickle file
'''
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    embeddings = np.array([item['embedding'] for item in data])
    original_smiles = [item['original_smiles'] for item in data]
    rdkit_smiles = [item['rdkit_smiles'] for item in data]
    names = [item['generic_name'] for item in data]
    
    return embeddings, original_smiles, rdkit_smiles, names

# Load sample data
# sample_embeddings, sample_smiles, sample_names = load_data(DATABASE_PATHS['sample'])
# print(f"Shape of sample embeddings: {sample_embeddings.shape}")



