# Labelled Mapped SMILES
# Date: 07.11.2024
# Author Arun, Noor

import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle
from smiles import extract_mapped_data_from_sdf, extract_mapped_data_from_txt

print('Starting methanogenesis_gnn4_embd.py')

def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    encoding[choices.index(value)] = 1
    return encoding

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    Chem.Kekulize(mol)
    adj = Chem.GetAdjacencyMatrix(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(one_hot_encoding(atom.GetAtomicNum(), [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42, 46, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 60, 65, 73, 74, 75, 79, 80, 81, 82, 83, 92]))
    # total different number of elements in the MCDB and BMDB is 58
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(np.where(adj), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(58, 64)
        self.conv2 = GCNConv(64, 128)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def atom_molecule_agg(gnn_output):
    return torch.mean(gnn_output, dim=0)

def process_database(database_path, database_name):
    print(f'Processing {database_name}')
    if database == 'sample':
        mapped_data = extract_mapped_data_from_txt(database_path)
    else:
        mapped_data = extract_mapped_data_from_sdf(database_path)
        
    print(f'Finished extracting mapped data for {database_name}')

    print('Initiating GNN model')
    model = GNN()

    # Filter out None entries and process the mapped data
    processed_data = []
    for original_smiles, rdkit_smiles, generic_name in mapped_data:
        if rdkit_smiles is not None and rdkit_smiles.lower() != 'none':
            processed_data.append((original_smiles, rdkit_smiles, generic_name))

    all_molecule_data = []
    for original_smiles, rdkit_smiles, generic_name in processed_data:
        graph = smiles_to_graph(rdkit_smiles)
        # debug
        # print(f"Processing: {generic_name} - {rdkit_smiles}")
        if graph is not None:
            success = False
            while not success:
                try:
                    # Forward pass through the GNN
                    output = model(graph)
                    # debug
                    # print('Output from GNN:', output)
                    # print(output.shape)

                    # Atom to molecule
                    molecule_embedding = atom_molecule_agg(output)
                    # print('GNN output -> atom to molecule (tensor):', molecule_embedding)

                    # Convert PyTorch tensor to NumPy array
                    output_np = molecule_embedding.detach().numpy()
                    success = True
                    # print('Output (numpy) from GNN:', output_np)
                    # print(output_np.shape)

                    all_molecule_data.append({
                        'generic_name': generic_name,
                        'original_smiles': original_smiles,
                        'rdkit_smiles': rdkit_smiles,
                        'embedding': output_np
                    })
                except Exception as e:
                    print(f"Error processing {generic_name}: {str(e)}")
        else:
            print(f'Graph is None for {generic_name}')

    print(f'Processed all molecules for {database_name}')
    print(f'Number of processed molecules: {len(all_molecule_data)}')

    return all_molecule_data

# Define database paths 
# Update path to required database
# 'sample' represent the selected MCR molecules
DATABASE_PATHS = {
    'mcdb': 'metabolites/milk_metabolites_structures.sdf',
    'bmdb': 'metabolites/bovine_metabolites_structures.sdf',
    'sample': 'metabolites/randy_metabolites.txt',
}

# Apply GNN to desired database. 
database = 'mcdb'
path_to_database = DATABASE_PATHS.get(database)
if path_to_database is None:
    raise ValueError(f"Unknown database: {database}")

molecule_data = process_database(path_to_database, database)
print(f"Successfully processed {database} database. First entry:")
print(molecule_data[0])

# contains embeddings, original_smiles, rdkit_smiles, names
with open(f"{database}__molecule_data.pkl", 'wb') as f:
        pickle.dump(molecule_data, f)