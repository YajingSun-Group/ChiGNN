import pickle
import numpy as np
import torch
from model import CHIGraphModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from chienn import collate_with_circle_index
import pandas as pd
from faerun import Faerun

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float64))  # 转换为 double 类型
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")
    plt.contour(xx, yy, Z, linewidths=1, colors='k')
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

column = 'ADH'
goal = '隐藏'
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

with open(f'singlecsv/{columns_data[column]}_graph_data.pkl', 'rb') as f:
    data_graphs, data_graph_informations, props, big_index = pickle.load(f)

HPLC = pd.read_csv(f'singlecsv/{columns_data[column]}.csv')
alpha_values = HPLC['alpha'].values
index_values = HPLC['Unnamed: 0'].values
smiles_values = HPLC['SMILES'].values

index_to_alpha = {idx: alpha for idx, alpha in zip(index_values, alpha_values)}
index_to_smiles = {idx: smiles for idx, smiles in zip(index_values, smiles_values)}

valid_indices = [info.data_index.item() for info in data_graph_informations]

valid_alpha_values = [index_to_alpha[idx] for idx in valid_indices]
valid_smiles_values = [index_to_smiles[idx] for idx in valid_indices]
valid_y_value = [info.y.item() for info in data_graph_informations]
total_num = len(data_graphs)
print('data num:', total_num, len(data_graph_informations), len(props))
if goal == '隐藏':
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
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
    h_graphs = []
    with torch.no_grad():
        for i in range(total_num):
            data = data_graphs[i]
            data = collate_with_circle_index([data], k_neighbors=3)
            mol_pred, h_graph = model(data.to(device))
            h_graphs.append(h_graph.cpu().numpy())
        h_graphs = np.concatenate(h_graphs, axis=0).astype(np.float64)  # 转换为 double 类型    
elif goal == '图结构':             
    h_graphs = []
    for i in range(total_num):
        data = torch.sum(data_graphs[i].x,dim=0).reshape(1,-1)
        h_graphs.append(data.cpu().numpy())
    h_graphs = np.concatenate(h_graphs, axis=0).astype(np.float64)  # 转换为 double 类型

tsne = TSNE(n_components=2, random_state=0)
h_graphs_2d_tsne = tsne.fit_transform(h_graphs)

kmeans_tsne = KMeans(n_clusters=40, random_state=0)
clusters_tsne = kmeans_tsne.fit_predict(h_graphs_2d_tsne.astype(np.float64))

hover_labels = []
for idx, smile, y in zip(valid_indices, valid_smiles_values, valid_y_value):
    label = f"{smile}__alpha: {index_to_alpha[idx]:.2f}_idx: {idx}_y: {y:.2f}"
    hover_labels.append(label)

# 使用 Faerun 绘制 t-SNE 结果
faerun = Faerun(view='front',clear_color='#111265', coords=False)

faerun.add_scatter(
    "t_sne_h_graph",
    {"x": h_graphs_2d_tsne[:, 0], "y": h_graphs_2d_tsne[:, 1], "c": clusters_tsne, "labels": hover_labels},
    point_scale=5.0,  # 调整点的大小
    colormap=plt.get_cmap("tab20"),  # 使用更鲜艳的颜色
    has_legend=True
)

faerun.plot("h_graph_聚类", template="smiles")
