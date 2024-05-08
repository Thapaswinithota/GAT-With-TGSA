from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data

def get_atom_features(atom):
    # Example: encoding atomic number
    return np.array([atom.GetAtomicNum()])

def get_bond_features(bond):
    # Example: encoding bond type with one-hot encoding
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return np.array([1, 0, 0, 0])
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return np.array([0, 1, 0, 0])
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return np.array([0, 0, 1, 0])
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return np.array([0, 0, 0, 1])
    return np.array([0, 0, 0, 0])  # Default case for unknown bond type

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)

    # Convert list of numpy arrays into a single numpy array before conversion to tensor
    atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    bond_features = []
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[start, end], [end, start]])
        bond_feat = get_bond_features(bond)
        bond_features.extend([bond_feat, bond_feat])  # Add features for both directions

    # Convert lists to numpy arrays before creating tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(bond_features), dtype=torch.float)  # Convert to numpy array first

    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

# Example usage
smiles = 'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
graph = smiles_to_graph(smiles)
print("Node features (x):", graph.x)
print("Edge indices (edge_index):", graph.edge_index)
print("Edge attributes (edge_attr):", graph.edge_attr)

