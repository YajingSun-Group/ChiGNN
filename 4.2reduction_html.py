import pickle
import numpy as np
import torch
from model import CHIGraphModel
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from chienn import collate_with_circle_index
from faerun import Faerun
import matplotlib.pyplot as plt

column = 'ADH'  # 目标列
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
    'AD3': 'AD3_charity_0823',
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

# 加载数据集
with open(f'alphacsv/{columns_data[column]}_graph_data.pkl', 'rb') as f:
    data_graphs, data_graph_informations, props, big_index = pickle.load(f) 

# 加载 CSV 文件中的 alpha 列
HPLC = pd.read_csv(f'alphacsv/{columns_data[column]}.csv')
alpha_values = HPLC['beta'].values
index_values = HPLC['Unnamed: 0'].values
smiles_values = HPLC['SMILES'].values

# 创建索引到 alpha 和 SMILES 的映射
index_to_alpha = {idx: alpha for idx, alpha in zip(index_values, alpha_values)}
index_to_smiles = {idx: smiles for idx, smiles in zip(index_values, smiles_values)}

# 提取有效的索引
valid_indices = [info.data_index.item() for info in data_graph_informations]
valid_y_value = [info.y.item() for info in data_graph_informations]
# 对应的 alpha 标签和 SMILES
valid_alpha_values = [index_to_alpha[idx] for idx in valid_indices]
valid_smiles_values = [index_to_smiles[idx] for idx in valid_indices]

total_num = len(data_graphs)
print('data num:', total_num, len(data_graph_informations), len(props))

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

model.eval()

# 提取 h_graph 特征
h_graphs = []
labels = valid_alpha_values  # 将标签换成 alpha

with torch.no_grad():
    for i in range(total_num):
        data = data_graphs[i]
        data = collate_with_circle_index([data], k_neighbors=3)
        mol_pred, h_graph = model(data.to(device))
        h_graphs.append(h_graph.cpu().numpy())

h_graphs = np.concatenate(h_graphs, axis=0)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
h_graphs_2d_tsne = tsne.fit_transform(h_graphs)

# 使用 PCA 进行降维
#pca = PCA(n_components=2)
#h_graphs_2d_pca = pca.fit_transform(h_graphs)

# 创建 hover_labels
hover_labels = []
for idx, smile, y in zip(valid_indices, valid_smiles_values, valid_y_value):
    label = f"{smile}__alpha: {index_to_alpha[idx]:.2f}_idx: {idx}_y: {y:.2f}"
    hover_labels.append(label)
# 绘制 t-SNE 和 PCA 结果
faerun = Faerun(view='front', clear_color='#111265',coords=False)

faerun.add_scatter(
    "t_sne_h_graph",
    {"x": h_graphs_2d_tsne[:, 0], "y": h_graphs_2d_tsne[:, 1], "c": labels, "labels": hover_labels},
    point_scale=5.0,  # 调整点的大小
    colormap=plt.get_cmap("tab20"),  # 使用更鲜艳的颜色
    has_legend=True
)

# faerun.add_scatter(
#     "pca_h_graph",
#     {"x": h_graphs_2d_pca[:, 0], "y": h_graphs_2d_pca[:, 1], "c": valid_alpha_values, "labels": hover_labels},
#     point_scale=1.0,
#     colormap=plt.get_cmap("viridis"),
#     has_legend=True,
# )

faerun.plot("h_graph_dimensionality_reduction", template="smiles")


