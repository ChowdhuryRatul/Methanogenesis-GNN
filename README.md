# Methanogenesis GNN

## Overview
This project focuses on analyzing and visualizing the complex molecular structures present in milk and bovine metabolites. We use Graph Neural Networks (GNNs) to generate molecular embeddings from SMILES string and t-SNE for dimensionality reduction and visualization.

## Databases Used
We utilize two metabolite databases:
The Milk Composition Database (MCDB) - 2,360 entries
The Bovine Metabolome Database (BMDB) - 51,682 entries
Sample - 16 selected MCR molecules

Note: The MCDB database is included in the metabolites folder of this repository. Due to its large size, the BMDB is not incorporated directly into this repository.
The databses are publicly available here: https://metabolomicscentre.ca/software-databases/databases/. Bovine metabolome database can also be accessed from: https://iastate.box.com/s/8xz7ipb6rxqdfgfpcurxyq0ynd5l9g29

## Workflow
These can be combined for smaller database.
1. Embedding Generation (`methanogenesis_gnn_embd.py`)
* Network architecture: 58 input nodes, 64 hidden layers, 128 output nodes
* Generates embeddings (N, 128) where N, number of atoms in each molecule
* Averages tensors across the atomic dimension to produce a 128-dimensional vector for each molecule

2. Dimensionality Reduction and Visualization (`methanogenesis_gnn_tsne.py`)
* Applies (t-SNE) to reduce embeddings to 2D
* Perplexity: set to min(30, total number of molecules in the database)
* Visualization along with experimentally validated MCR molecules

Note: Embeddings from `methanogenesis_gnn_embd.py` are loaded in `methanogenesis_gnn_tsne.py` before performing t-SNE.

Note: Both scripts can be mergred while working on small databse. BMDB being too large, t-SNE and plotting is time and memory intensive. Better run as separate scripts. 
