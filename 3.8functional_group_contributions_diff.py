import torch
from model import CHIGraphModel
from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.featurization.convert_mol_to_data import convert_mol_to_data
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
from chienn import collate_with_circle_index
import pandas as pd
import scipy.sparse as sp
import os
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import FragmentCatalog

MODEL = 'Explation'
column = 'IC'

def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash

def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash
    sorted_all_hit_fg_at = sorted(all_hit_fg_at, key=lambda fg: len(fg), reverse=True)
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)
    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash

def calculate_atom_contributions(edge_index, attributions):
    atom_contributions = np.zeros(torch.max(edge_index).item() + 1)
    for i in range(edge_index.shape[1]):
        row_idx = edge_index[0, i].item()
        col_idx = edge_index[1, i].item()
        contribution = attributions[i]
        atom_contributions[row_idx] += contribution
        atom_contributions[col_idx] += contribution
    return atom_contributions

def return_mask_tensor(edge_index):
    masktensor = []
    for j in range(edge_index.shape[1]):
        smask_list = []
        for k in range(edge_index.shape[1]):
            if j == k:
                smask_list.append(0)
            else:
                smask_list.append(1)
        mask = torch.tensor(smask_list)
        masktensor.append(mask)
    return masktensor
def get_functional_groups(smiles):
    """
    Get the atom indices and names of the functional groups in a given molecule.
    
    Args:
    smiles (str): The SMILES string of the molecule.
    
    Returns:
    list of tuple: Each tuple contains the atom indices and the name of a functional group.
    """
    # 39 function group config
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                           'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                           '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                           '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                           '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                           '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                           '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                           '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                           '[C;D4]([C;D1])([C;D1])-[C;D1]',
                           '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                           '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]

    hit_fg_at, hit_fg_name = return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)
    
    functional_groups = []
    for a, hit_fg in enumerate(hit_fg_at):
        for b, hit_fg_b in enumerate(hit_fg):
            functional_groups.append((hit_fg_b, hit_fg_name[a]))
    
    return functional_groups
def get_functional_groups_and_chirality(smiles):
    """
    Get the atom indices and names of the functional groups in a given molecule, 
    and identify if they contain or are connected to chiral atoms.
    
    Args:
    smiles (str): The SMILES string of the molecule.
    
    Returns:
    list of tuple: Each tuple contains the atom indices, the name of a functional group,
                   and a boolean indicating if it contains/connected to chiral atoms.
    """
    mol = Chem.MolFromSmiles(smiles)

    # 识别所有手性原子及其相连的原子
    chiral_atoms = set()
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            chiral_atoms.add(atom.GetIdx())
            for neighbor in atom.GetNeighbors():
                chiral_atoms.add(neighbor.GetIdx())

    # 39 function group config
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                           'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                           '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                           '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                           '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                           '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                           '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                           '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                           '[C;D4]([C;D1])([C;D1])-[C;D1]',
                           '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                           '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]

    hit_fg_at, hit_fg_name = return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)

    functional_groups = []
    for a, hit_fg in enumerate(hit_fg_at):
        for b, hit_fg_b in enumerate(hit_fg):
            contains_chiral = any(atom_idx in chiral_atoms for atom_idx in hit_fg_b)
            functional_groups.append((hit_fg_b, fg_name_list[a], contains_chiral))
    
    return functional_groups

# Functions remain the same, only the main execution code is changed

device = torch.device("cuda:1" if torch.cuda.is_available() else torch.device("cpu"))

print('使用的设备：', device)

if device.type == 'cuda':
    print('GPU编号：', torch.cuda.current_device())
    print('GPU名称：', torch.cuda.get_device_name(torch.cuda.current_device()))

nn_params = {
    'num_tasks': 3,
    'num_layers': 3,
    'emb_dim': 128,
    'drop_ratio': 0,
    'graph_pooling': 'sum',
    'descriptor_dim': 1
}
model = CHIGraphModel(**nn_params).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(num_params)

model.load_state_dict(torch.load(f'saves/model_{column}CHI/model_save_1500.pth'))

if MODEL == 'Explation':
    csv_file_path = f'alphacsv/{column}_charity_0823.csv'
    df = pd.read_csv(csv_file_path)

    data_graph = []
    edge_indexs = []
    smiles_list = []
    speeds = []
    
    indices = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol_index = int(row['index'])  # Assuming this is the index for enantiomer pairs
        speed = row['Speed']
        
        
        mol = smiles_to_3d_mol(smiles)
        if mol is None:
            print(f"Failed to convert SMILES to 3D molecule: {smiles}")
            continue
        data = mol_to_data(mol)
        if data is None:
            print(f"Failed to convert molecule to graph data: {smiles}")
            continue
        edge_index = data.edge_index
        data = to_edge_graph(data)
        
        data_graph.append(data)
        edge_indexs.append(edge_index)
        smiles_list.append(smiles)
        speeds.append(speed)
        
        indices.append(mol_index)

    results_df = pd.DataFrame(columns=['Index', 'Functional_Group', 'Contribution'])

    enantiomer_pairs = {}
    for i in range(len(data_graph)):
        index = indices[i]
        if index not in enantiomer_pairs:
            enantiomer_pairs[index] = []
        enantiomer_pairs[index].append(i)

    for index, pair_indices in enantiomer_pairs.items():
        if len(pair_indices) == 2:
            i1, i2 = pair_indices

            # Calculate atom contributions for the first enantiomer
            edge_index1 = edge_indexs[i1]
            data1 = data_graph[i1]
            y1 = torch.Tensor([float(speeds[i1])])
            
            data1 = collate_with_circle_index([data1], k_neighbors=3)
            model.eval()
            mol_pred1, h_graph1 = model(data1.to(device))
            masktensor1 = return_mask_tensor(edge_index1)
            attributions1 = []
            for mask in masktensor1:
                data1 = data_graph[i1]
                y1 = torch.Tensor([float(speeds[i1])])
                
                data1 = collate_with_circle_index([data1], k_neighbors=3)
                model.eval()
                sub_pred1, h_graph1 = model(data1.to(device),  mask=mask.to(device))
                attribution1 = (mol_pred1[0][1].detach().cpu().data.numpy() / speeds[i1] - sub_pred1[0][1].detach().cpu().data.numpy() / speeds[i1]) / (mol_pred1[0][1].detach().cpu().data.numpy() / speeds[i1])
                attributions1.append(attribution1)

            atom_contributions1 = calculate_atom_contributions(edge_index1, attributions1)

            # Calculate atom contributions for the second enantiomer
            edge_index2 = edge_indexs[i2]
            data2 = data_graph[i2]
            y2 = torch.Tensor([float(speeds[i2])])
            
            data2 = collate_with_circle_index([data2], k_neighbors=3)
            model.eval()
            mol_pred2, h_graph2 = model(data2.to(device))
            masktensor2 = return_mask_tensor(edge_index2)
            attributions2 = []
            for mask in masktensor2:
                data2 = data_graph[i2]
                y2 = torch.Tensor([float(speeds[i2])])
                
                data2 = collate_with_circle_index([data2], k_neighbors=3)
                model.eval()
                sub_pred2, h_graph2 = model(data2.to(device), mask=mask.to(device))
                attribution2 = (mol_pred2[0][1].detach().cpu().data.numpy() / speeds[i2] - sub_pred2[0][1].detach().cpu().data.numpy() / speeds[i2]) / (mol_pred2[0][1].detach().cpu().data.numpy() / speeds[i2])
                attributions2.append(attribution2)

            atom_contributions2 = calculate_atom_contributions(edge_index2, attributions2)

            # Get functional groups and their contributions for both enantiomers
            functional_groups1 = get_functional_groups(smiles_list[i1])
            functional_groups2 = get_functional_groups(smiles_list[i2])

            # Compare contributions of corresponding functional groups and calculate difference
            for fg_atoms1, fg_name1 in functional_groups1:
                fg_contribution1 = sum(atom_contributions1[atom] for atom in fg_atoms1)
                
                for fg_atoms2, fg_name2  in functional_groups2:
                    # 确保每个官能团对的位置都被记录下来
                    if set(fg_atoms1) == set(fg_atoms2):
                        fg_contribution2 = sum(atom_contributions2[atom] for atom in fg_atoms2)
                        contribution_diff = abs(fg_contribution1 - fg_contribution2)
                        
                        results_df = pd.concat([results_df, pd.DataFrame({
                            'Index': [index],
                            'Functional_Group': [fg_name1],
                            'Contribution': [contribution_diff]
                        })], ignore_index=True)

# 将结果转换为 DataFrame 并导出到CSV
if not os.path.exists('./functional_group'):
    os.makedirs('./functional_group')
results_df.to_csv(f'functional_group/enantiomer_fg_contribution_diff_{column}.csv', index=False)
print("CSV文件已保存。")