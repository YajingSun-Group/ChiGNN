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

# 色谱柱数据
columns_data = {

    'IC': 'IC_charity_0823',
    'IA': 'IA_charity_0823',
    }
    

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('使用的设备：', device)
if device.type == 'cuda':
    print('GPU编号：', torch.cuda.current_device())
    print('GPU名称：', torch.cuda.get_device_name(torch.cuda.current_device()))

# 加载所有模型
models = {}
for column in columns_data:
    model_file = f'saves/model_{column}CHI/model_save_1100.pth'
    model = CHIGraphModel(**nn_params).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    models[column] = model

# 给定的一对对映体分子及其对应的speed和eluent
enantiomers_data = [
    {'smiles': 'CN(C)C(=O)[C@H]([C@@H](C(=O)N(C)C)O)O', 'speed': 1.0, 'eluent': 0.5},  # 示例数据
    {'smiles': 'CN(C)C(=O)[C@@H]([C@H](C(=O)N(C)C)O)O', 'speed': 1.0, 'eluent': 0.5}
]

def predict_retention_times(enantiomers_data):
    predictions = {column: [] for column in columns_data}
    
    for enantiomer_data in enantiomers_data:
        smiles = enantiomer_data['smiles']
        speed = enantiomer_data['speed']
        
        for column, model in models.items():
            mol = smiles_to_3d_mol(smiles)
            if mol is None:
                print(f"Failed to convert SMILES to 3D molecule: {smiles}")
                continue
            gdata = mol_to_data(mol)
            if gdata is None:
                print(f"Failed to convert molecule to graph data: {smiles}")
                continue
            edata = to_edge_graph(gdata)
            data = collate_with_circle_index([edata], k_neighbors=3)
            with torch.no_grad():
                mol_pred, h_graph = model(data.to(device))
            
            predictions[column].append(mol_pred[0].cpu().numpy())
    
    return predictions

# 预测保留时间
predictions = predict_retention_times(enantiomers_data)

# 转换为DataFrame
data = []
for column, times in predictions.items():
    for i, (RT10, RT, RT90) in enumerate(times):
        enantiomer = f'Enantiomer{i+1}'
        data.append([column, enantiomer, RT10, RT, RT90])

df = pd.DataFrame(data, columns=['Column', 'Enantiomer', 'RT10', 'RT', 'RT90'])

# 绘制条形图带误差条和折线
fig, ax = plt.subplots(figsize=(14, 8))

width = 0.35  # 条形图的宽度
x = np.arange(len(columns_data))  # 色谱柱的位置

# 记录每个色谱柱下的保留时间，以绘制折线
lines = {enantiomer: [] for enantiomer in [f'Enantiomer{i+1}' for i in range(len(enantiomers_data))]}

for i, enantiomer in enumerate([f'Enantiomer{i+1}' for i in range(len(enantiomers_data))]):
    subset = df[df['Enantiomer'] == enantiomer]
    bar_positions = x + i * width  # 条形的位置
    center_positions = bar_positions 
    ax.bar(
        bar_positions, subset['RT'], width,
        yerr=[abs(subset['RT'] - subset['RT10']), abs(subset['RT90'] - subset['RT'])],
        label=enantiomer, capsize=5
    )
    lines[enantiomer] = (center_positions, subset['RT'].values)

# 绘制折线
for enantiomer, (positions, values) in lines.items():
    ax.plot(positions, values, marker='o', linestyle='-', label=f'{enantiomer} Line')
# 设置标签和标题的字体大小
ax.set_xlabel('Chromatographic Columns', fontsize=16)
ax.set_ylabel('Predicted Retention Time', fontsize=16)
ax.set_title('Predicted Retention Time for Enantiomers across Different Chromatographic Columns', fontsize=20)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(list(columns_data.keys()), fontsize=14)
ax.legend(title='Enantiomer', fontsize=14)

plt.grid(True)

# 保存图像
if not os.path.exists('./column_csp'):
    os.makedirs('./column_csp')
output_file = "column_csp/predicted_retention_times"
plt.savefig(f"{output_file}.png", format='png')
plt.savefig(f"{output_file}.svg", format='svg')
plt.savefig(f"{output_file}.pdf", format='pdf')

plt.show()

print(f"图像已保存为 {output_file}")
df.to_csv("column_csp/predicted_retention_times.csv")
