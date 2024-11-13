import torch
from model import CHIGraphModel
from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.featurization.convert_mol_to_data import convert_mol_to_data
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
from chienn import collate_with_circle_index
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
import os
from PIL import Image
import io

MODEL = 'Explation'
column = 'IC'

def Convert_adjacency_matrix(edge_index, attributions, plot_name, plot=False):
    size = edge_index.shape[1]
    atom_num = torch.max(edge_index) + 1

    edge_w = [attributions[i] for i in range(size)]

    adj_matrix = sp.lil_matrix((atom_num, atom_num))
    for i in range(size):
        row_idx = edge_index[0, i].item()
        col_idx = edge_index[1, i].item()
        adj_matrix[row_idx, col_idx] = edge_w[i]

    if plot:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_xticks(np.arange(0, atom_num, 1))
        ax.set_yticks(np.arange(0, atom_num, 1))

        ax.set_xticklabels(np.arange(0, atom_num, 1), fontsize=24)
        ax.set_yticklabels(np.arange(0, atom_num, 1), fontsize=24)

        ax.set_xticks(np.arange(-.5, atom_num - 1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, atom_num - 1, 1), minor=True)

        #ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.1)
        midpoint = 0
        #norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=midpoint)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=min(attributions), vmax=max(attributions), vcenter=midpoint)
        cax = ax.matshow(adj_matrix.toarray(), cmap='coolwarm', norm=norm)
        cbar=plt.colorbar(cax)
        cbar.ax.yaxis.set_tick_params(labelsize=24)
        #plt.title('Adjacency Matrix with Attributions', pad=20, fontsize=18)
        plt.xlabel('Atoms', fontsize=24)
        plt.ylabel('Atoms', fontsize=24)
        plt.tight_layout()
        plt.savefig(f'fig_save/{plot_name}_adjacency.png', bbox_inches='tight', pad_inches=0.1)
        # 保存为SVG格式
        plt.savefig(f'fig_save/{plot_name}_adjacency.svg', bbox_inches='tight')
        # 保存为PDF格式
        plt.savefig(f'fig_save/{plot_name}_adjacency.pdf', bbox_inches='tight')
        plt.show()

    return adj_matrix.toarray()

def calculate_atom_contributions(edge_index, attributions):
    atom_contributions = np.zeros(torch.max(edge_index).item() + 1)
    for i in range(edge_index.shape[1]):
        row_idx = edge_index[0, i].item()
        col_idx = edge_index[1, i].item()
        contribution = attributions[i]
        atom_contributions[row_idx] += contribution
        atom_contributions[col_idx] += contribution
    return atom_contributions

def atom_attribution_visualize(smiles, atom_attribution, save_path='./image', cmap_name='coolwarm', show_values=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)  # 不显示氢原子
    cmap = plt.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    #norm = matplotlib.colors.Normalize(vmin=min(atom_attribution), vmax=max(atom_attribution))
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    atom_radii = {}

    for i in range(mol.GetNumAtoms()):
        highlight_atom_colors[i] = [plt_colors.to_rgba(float(atom_attribution[i]))]
        atom_radii[i] = 0.6

    rdDepictor.Compute2DCoords(mol)

    # 绘制分子图，标记原子贡献
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
    drawer.DrawMoleculeWithHighlights(mol, '', highlight_atom_colors, {}, atom_radii, {})
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{smiles}_atom.png'), 'wb') as f:
        f.write(png)

    # 从文件读取图像以进行显示
    image = Image.open(io.BytesIO(png))

    # 在图中添加 colorbar 和原子贡献值
    fig, ax = plt.subplots()
    ax.imshow(image, aspect='equal')
    ax.axis('off')

    if show_values:
        # 在图中添加原子贡献值
        for i in range(mol.GetNumAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            ax.text(pos.x, pos.y, f'{atom_attribution[i]:.2f}', fontsize=8, color='black')

    # 添加 colorbar
    midpoint = 0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=midpoint)
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=min(atom_attribution), vmax=max(atom_attribution), vcenter=midpoint)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar=plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.yaxis.set_tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{smiles}_with_values.png'), bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(save_path, f'{smiles}_with_values.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'{smiles}_with_values.pdf'), bbox_inches='tight')
    plt.show()

    return png



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

model.load_state_dict(
    torch.load(f'saves/model_{column}CHI/model_save_1500.pth'))

if MODEL == 'Explation':
    draw_picture = True
    smiles_list = ['CCOC(=O)[C@@]1(CC(=C(NC(=O)c2ccccc2)C(=O)O1)c3ccccc3)c4ccccc4Br', 'CCOC(=O)[C@]1(CC(=C(NC(=O)c2ccccc2)C(=O)O1)c3ccccc3)c4ccccc4Br']
    #smiles_list = ['[H][C@@]1(C2=CC([N+](=O)[O-])=CC=C2)C(C(=O)OCCOC)=C(C)NC(C)=C1C(=O)OC(C)C', '[H][C@]1(C2=CC([N+](=O)[O-])=CC=C2)C(C(=O)OCCOC)=C(C)NC(C)=C1C(=O)OC(C)C']
    #smiles_list = ['N[C@H](C(O)=O)C', 'N[C@@H](C(O)=O)C']
    #smiles_list = ['COc1ccc2cc(ccc2c1)[C@H](C)C(O)=O', 'COc1ccc2cc(ccc2c1)[C@@H](C)C(O)=O']
    #smiles_list = ['Clc1cccc(c1)[C@H]2CCc3ccccc3N2', 'Clc1cccc(c1)[C@@H]2CCc3ccccc3N2']
    y_pred = []

    speed = [1.0, 1.0]

    data_graph = []
    edge_indexs = []
    dataset = []
    for smile in smiles_list:
        mol = smiles_to_3d_mol(smile)
        if draw_picture:
            index = 0
            smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
            plt.imshow(smiles_pic)
            plt.axis('off')
            if not os.path.exists('./fig_save'):
                os.makedirs('./fig_save')
            plt.savefig(f'fig_save/molecular_{index}.png')
            index += 1
            plt.clf()
        data = mol_to_data(mol)
        edge_index = data.edge_index
        data = to_edge_graph(data)
        data_graph.append(data)
        edge_indexs.append(edge_index)

    results_df = pd.DataFrame(columns=['Unnamed: 0', 'index', 'SMILES', 'label', 'masktensor', 'sub_pred_mean10per', 'sub_pred_mean', 'sub_pred_mean90per', 'mol_pred_mean10per', 'mol_pred_mean', 'mol_pred_mean90per', 'attribution'])
    for i in range(len(data_graph)):
        edge_index = edge_indexs[i]
        data = data_graph[i]
        y = torch.Tensor([float(speed[i])])
        data = collate_with_circle_index([data], k_neighbors=3)
        model.eval()
        mol_pred, h_graph = model(data.to(device))
        y_pred.append(mol_pred.detach().cpu().data.numpy() / speed[i])
        masktensor = return_mask_tensor(edge_index)
        attributions = []
        for mask in masktensor:
            data = data_graph[i]
            y = torch.Tensor([float(speed[i])])
            data = collate_with_circle_index([data], k_neighbors=3)
            model.eval()
            sub_pred, h_graph = model(data.to(device), mask=mask.to(device))
            attribution = (mol_pred[0][1].detach().cpu().data.numpy() / speed[i] - sub_pred[0][1].detach().cpu().data.numpy() / speed[i])/(mol_pred[0][1].detach().cpu().data.numpy() / speed[i])
            #attribution = mol_pred[0][1].detach().cpu().data.numpy() / speed[i] - sub_pred[0][1].detach().cpu().data.numpy() / speed[i]
            attributions.append(attribution)
            results_df = pd.concat([results_df, pd.DataFrame({
                'Unnamed: 0': [i],
                'index': [i],
                'SMILES': [smiles_list[i]],
                'label': [y.item()],
                'masktensor': [mask.tolist()],
                'sub_pred_mean10per': [sub_pred[0][0].detach().cpu().data.numpy() / speed[i]],
                'sub_pred_mean': [sub_pred[0][1].detach().cpu().data.numpy() / speed[i]],
                'sub_pred_mean90per': [sub_pred[0][2].detach().cpu().data.numpy() / speed[i]],
                'mol_pred_mean10per': [mol_pred[0][0].detach().cpu().data.numpy() / speed[i]],
                'mol_pred_mean': [mol_pred[0][1].detach().cpu().data.numpy() / speed[i]],
                'mol_pred_mean90per': [mol_pred[0][2].detach().cpu().data.numpy() / speed[i]],
                'attribution': [attribution]
            })], ignore_index=True)

        adjacency = Convert_adjacency_matrix(edge_index, attributions, smiles_list[i], plot=True)
        atom_contributions = calculate_atom_contributions(edge_index, attributions)
        print(f'Atom contributions for {smiles_list[i]}: {atom_contributions}')
        # 保存邻接矩阵和原子贡献
        np.save(f'fig_save/{smiles_list[i]}.npy', {'adjacency': adjacency, 'atom_contributions': atom_contributions})

        plt.figure(figsize=(12, 4))
        # cmap = plt.get_cmap('coolwarm')
        # norm = matplotlib.colors.Normalize(vmin=min(atom_contributions), vmax=max(atom_contributions))
        # colors = [cmap(norm(contribution)) for contribution in atom_contributions]
        
        # plt.bar(range(len(atom_contributions)), atom_contributions, color=colors)
        plt.bar(range(len(atom_contributions)), atom_contributions, color='#ADD8E6')
        #plt.xticks(np.arange(0, len(atom_contributions)+1, 5))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.xlabel('Atom Index')
        # plt.ylabel('Contribution')
        #plt.title(f'Atom Contributions for {smiles_list[i]}')
        plt.savefig(f'fig_save/atom_contributions_{smiles_list[i]}.png')
        plt.savefig(f'fig_save/atom_contributions_{smiles_list[i]}.svg', bbox_inches='tight')
        plt.savefig(f'fig_save/atom_contributions_{smiles_list[i]}.pdf', bbox_inches='tight')
        plt.show()

        # 可视化分子图上的原子贡献
        atom_attribution_visualize(smiles_list[i], atom_contributions, save_path='./fig_save', show_values=False)

    results_df.to_csv('Enantiomer_Visualization.csv', index=False)
    print(y_pred)

       
