import os
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
from model import CHIGraphModel
from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.featurization.convert_mol_to_data import convert_mol_to_data
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from torch_geometric.data import Batch
from chienn import collate_with_circle_index
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import matplotlib
import scipy.sparse as sp

# 模型参数
nn_params = {
    'num_tasks': 3,
    'num_layers': 3,
    'emb_dim': 128,
    'drop_ratio': 0,
    'graph_pooling': 'sum',
    'descriptor_dim': 1
}

# 选择数据集和模型
column = 'IC'  # 你可以在这里改变数据集的选择

columns_data = {
    'ADH': 'ADH_charity_0823',
    'ODH': 'ODH_charity_0823',
    'IC': 'IC_charity_0823',
    'IA': 'IA_charity_0823',
    'OJH': 'OJH_charity_0823',
    'ASH': 'ASH_charity_0823',
    'IC3': 'IC3_charity_0823',
    'IE': 'IE_charity_0823',
    'ID': 'ID_charity_0823',
    'OD3': 'OD3_charity_0823',
    'IB': 'IB_charity_0823',
    'AD': 'AD_charity_0823',
    'AD3': 'AD_charity_0823',
    'IF': 'IF_charity_0823',
    'OD': 'OD_charity_0823',
    'AS': 'AS_charity_0823',
    'OJ3': 'OJ3_charity_0823',
    'IG': 'IG_charity_0823',
    'AZ': 'AZ_charity_0823',
    'IAH': 'IAH_charity_0823',
    'OJ': 'OJ_charity_0823',
    'ICH': 'ICH_charity_0823',
    'OZ3': 'OZ3_charity_0823',
    'IF3': 'IF3_charity_0823',
    'IAU': 'IAU_charity_0823'
}

model_file = f'saves/model_{column}CHI/model_save_1500.pth'
csv_file_path = f'columncsv/{columns_data[column]}.csv'
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('使用的设备：', device)
if device.type == 'cuda':
    print('GPU编号：', torch.cuda.current_device())
    print('GPU名称：', torch.cuda.get_device_name(torch.cuda.current_device()))
# 加载模型
model = CHIGraphModel(**nn_params).to(device)
model.load_state_dict(torch.load(model_file))
model.eval()

def visualize_and_save(csv_file_path, save_dir):
    # 创建保存图片的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    pred10_list = []
    pred_list = []
    pred90_list = []
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol_index = int(row['Unnamed: 0'])
        speed = row['Speed']
        
        # 生成分子图像
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)  # 不显示氢原子
        mol_img = Draw.MolToImage(mol, size=(200, 200))
        
        # 保存分子图像
        img_path = os.path.join(save_dir, f"{mol_index}.png")
        mol_img.save(img_path)
        
        # 显示图片（可选）
        plt.imshow(mol_img)
        plt.axis('off')
        plt.show()
        
        # 分子图像并进行可视化
        # 分子图像并进行可视化
        mol = smiles_to_3d_mol(smiles)
        if mol is None:
            print(f"Failed to convert SMILES to 3D molecule: {smiles}")
            pred10_list.append(None)
            pred_list.append(None)
            pred90_list.append(None)
            continue
        #gdata = convert_mol_to_data(mol)
        gdata = mol_to_data(mol)
        if gdata is None:
            print(f"Failed to convert molecule to graph data: {smiles}")
            pred10_list.append(None)
            pred_list.append(None)
            pred90_list.append(None)
            continue
        edge_index = gdata.edge_index
        edata = to_edge_graph(gdata)
        
        y = torch.Tensor([speed])
        data = collate_with_circle_index([edata], k_neighbors=3)
        mol_pred, h_graph = model(data.to(device))
        # 保存pred10, pred, pred90
        pred10_list.append(mol_pred[0][0].item())
        pred_list.append(mol_pred[0][1].item())
        pred90_list.append(mol_pred[0][2].item())
        masktensor = return_mask_tensor(edge_index)
        attributions = []
        for mask in masktensor:
            data = collate_with_circle_index([edata], k_neighbors=3)
            sub_pred, _ = model(data.to(device), mask=mask.to(device))
            attribution = (mol_pred[0][1].detach().cpu().data.numpy() / speed - sub_pred[0][1].detach().cpu().data.numpy() / speed)/(mol_pred[0][1].detach().cpu().data.numpy() / speed)
            #attribution = mol_pred[0][1].detach().cpu().data.numpy() / speed - sub_pred[0][1].detach().cpu().data.numpy() / speed
            attributions.append(attribution)
        
        adjacency = Convert_adjacency_matrix(edge_index, attributions, f"{save_dir}/{mol_index}", plot=True)
        atom_contributions = calculate_atom_contributions(edge_index, attributions)
        atom_attribution_visualize(smiles, atom_contributions, save_path=save_dir, img_name=f"{mol_index}", show_values=False)
    # 将新的列添加到DataFrame
    df['pred10'] = pred10_list
    df['pred'] = pred_list
    df['pred90'] = pred90_list

    # 保存更新后的CSV
    df.to_csv(csv_file_path, index=False)
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
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.set_xticks(np.arange(0, atom_num, 1))
        ax.set_yticks(np.arange(0, atom_num, 1))

        ax.set_xticklabels(np.arange(0, atom_num, 1), fontsize=10)
        ax.set_yticklabels(np.arange(0, atom_num, 1), fontsize=10)

        ax.set_xticks(np.arange(-.5, atom_num - 1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, atom_num - 1, 1), minor=True)

        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        midpoint = 0
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=midpoint)
        #norm = matplotlib.colors.TwoSlopeNorm(vmin=min(attributions), vmax=max(attributions), vcenter=midpoint)
        cax = ax.matshow(adj_matrix.toarray(), cmap='coolwarm', norm=norm)
        plt.colorbar(cax)
        plt.title('Adjacency Matrix with Attributions', pad=20, fontsize=12)
        plt.xlabel('Atoms', fontsize=12)
        plt.ylabel('Atoms', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{plot_name}_adjacency.png', bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close()  # Close the figure to free up memory

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

def atom_attribution_visualize(smiles, atom_attribution, save_path='./image', img_name='molecule', cmap_name='coolwarm', show_values=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)  # 不显示氢原子
    cmap = plt.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=min(atom_attribution), vmax=max(atom_attribution))
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
    with open(os.path.join(save_path, f'{img_name}_atom.png'), 'wb') as f:
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
    # vmin, vmax = min(atom_attribution), max(atom_attribution)
    # if vmin < midpoint < vmax:
    #     norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=midpoint)
    # else:
    #     norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{img_name}_with_values.png'), bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(save_path, f'{img_name}_with_values.pdf'), bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(save_path, f'{img_name}_with_values.svg'), bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)  # Close the figure to free up memory

    return png

def return_mask_tensor(edge_index):
    masktensor = []
    for j in range(edge_index.shape[1]):
        smask_list = []
        for k in range(edge_index.shape[1]):
            smask_list.append(0 if j == k else 1)
        mask = torch.tensor(smask_list)
        masktensor.append(mask)
    return masktensor

# 保存图像的目录
save_folder = os.path.join('column_picture', column)
visualize_and_save(csv_file_path, save_folder)
