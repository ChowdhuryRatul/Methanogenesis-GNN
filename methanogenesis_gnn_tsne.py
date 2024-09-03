# t-SNE and plotting
# Date: 07.11.2024
# Author Arun, Noor

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy.spatial.distance import cdist
from smiles import load_data

# path to embeding pkl files to be loaded to perform t-SNE
# 'sample' represent the selected MCR molecules
DATABASE_PATHS_2 = {
    'mcdb': 'metabolites/mcdb_molecule_data.pkl',
    'bmdb': 'metabolites/bmdb_molecule_data.pkl',
    'sample': 'metabolites/sample_molecule_data.pkl',
}

# Address the database to work on
database = 'mcdb'
path_to_database = DATABASE_PATHS.get(database)

if path_to_database is None:
    raise ValueError(f"Unknown database: {database}")

print('starting methanogenesis_gnn_tsne.py')
print(f"working on {database}")

# Load sample data from MCR embeddings
sample_embeddings, sample_original_smiles, sample_rdkit_smiles, sample_names = load_data(DATABASE_PATHS_2['sample'])
print(f"Shape of sample embeddings: {sample_embeddings.shape}")

# Load database
database = 'mcdb'  # Change this to 'BMDB' or 'test' as needed
db_embeddings, db_original_smiles, db_rdkit_smiles, db_names = load_data(DATABASE_PATHS_2[database])
print(f"Shape of {database} embeddings: {db_embeddings.shape}")

# Combine the embeddings
all_embeddings = np.vstack((db_embeddings, sample_embeddings))
all_smiles = db_rdkit_smiles + sample_rdkit_smiles
all_names = db_names + sample_names

# Create labels for coloring
labels = np.array([database] * len(db_embeddings) + ['Sample'] * len(sample_embeddings))

# Perform t-SNE
print('Initiating t-SNE')
perplexity = min(30, len(all_embeddings) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
tsne_result = tsne.fit_transform(all_embeddings)
print('completed t-SNE')

# Dumping tsne results for quick plotting later
with open(f"{database}_molecule_data_tsne.pkl", 'wb') as f:
        pickle.dump(tsne_result, f)


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle(f"t-SNE visualization of GNN node embeddings for {database} and sample molecules", fontsize=16)

# Plot 1: Entire t-SNE result
print('Plotting full 2D t-SNE')
colors = {'Sample': 'red', database: 'black'}
for label in np.unique(labels):
    mask = labels == label
    ax1.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                c=colors[label], label=label, alpha=0.6, s=20)

ax1.set_xlabel('t-SNE component 1')
ax1.set_ylabel('t-SNE component 2')
ax1.legend()

# Add rectangle to indicate the filtered region
# It varies depending on the database and the focus on the cluster
# mcdb
rect = Rectangle((55, -10), 15, 20, fill=False, edgecolor='red', linewidth=1)
# bmdb
# rect = Rectangle((-100, 0), 20, 10, fill=False, edgecolor='red', linewidth=1)
ax1.add_patch(rect)

# Plot 2: Filtered t-SNE result
print('Plotting filtered 2D t-SNE')
# mcdb
mask_tsne = (tsne_result[:, 1] >= -10) & (tsne_result[:, 1] <= 10) & (tsne_result[:, 0] >= 55) & (tsne_result[:, 0] <= 70)
# bmdb
# mask_tsne = (tsne_result[:, 1] >= 0) & (tsne_result[:, 1] <= 10) & (tsne_result[:, 0] >= -100) & (tsne_result[:, 0] <= -80)
filtered_tsne = tsne_result[mask_tsne]
filtered_labels = labels[mask_tsne]

for label in np.unique(filtered_labels):
    mask = filtered_labels == label
    ax2.scatter(filtered_tsne[mask, 0], filtered_tsne[mask, 1], 
                c=colors[label], label=label, alpha=0.6, s=50)

ax2.set_xlabel('t-SNE component 1')
ax2.set_ylabel('t-SNE component 2')
# mcdb
ax2.set_xlim(55, 70)
ax2.set_ylim(-10, 10)
# bmdb
# ax2.set_xlim(-100, -80)
# ax2.set_ylim(0, 10)
ax2.legend()

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the overall title
plt.savefig(f"tsne_cluster_{database}_full_and_filtered.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Visualization saved as 'tsne_cluster_{database}_full_and_filtered.png'")

# Print some statistics
total_points = len(tsne_result)
filtered_points = len(filtered_tsne)
print(f"Total points: {total_points}")
print(f"Filtered points: {filtered_points}")
print(f"Percentage of points in filtered plot: {filtered_points/total_points*100:.2f}%")