import pickle
import numpy as np
import torch
from model import CHIGraphModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加 3D 绘图工具
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from chienn import collate_with_circle_index
import pandas as pd
import mplcursors  # 添加 mplcursors 库以实现交互功能
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from matplotlib.colors import ListedColormap, BoundaryNorm
from rdkit import Chem
from rdkit.Chem import AllChem

def plot_tsne_only(h_graphs_2d_tsne, labels, valid_indices, valid_smiles_values):
    cmap = ListedColormap(['#440154', '#21918c', '#fde725'])
    norm = BoundaryNorm([1, 1.2, 2, 10], cmap.N)
    plt.figure(figsize=(8, 6))
    scatter_tsne = plt.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=labels, cmap=cmap,norm=norm, alpha=0.5,s=100,marker='o')
    cbar_tsne = plt.colorbar(scatter_tsne,  boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
    cbar_tsne.set_ticklabels(['<1.2', '1.2-2', '>2'], fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
    crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))
    plt.axis('off')  # 去掉坐标轴
    cbar_tsne.remove()  # 去掉colorbar
    plt.tight_layout()
    plt.savefig('tsne_only.png')
    plt.savefig('tsne_only.svg')
    plt.savefig('tsne_only.pdf')
    plt.show()



def plot_tsne_only_coolwarm(h_graphs_2d_tsne, labels, valid_indices, valid_smiles_values):
    plt.figure(figsize=(8, 6))
    scatter_tsne = plt.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=labels, cmap='coolwarm', alpha=0.5,s=100,marker='o')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
    crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))
    plt.axis('off')  # 去掉坐标轴
    plt.tight_layout()
    plt.savefig('tsne_only.png')
    plt.savefig('tsne_only.svg')
    plt.savefig('tsne_only.pdf')
    plt.show()    
def plot_dimensionality_reduction(h_graphs_2d_tsne, h_graphs_2d_pca, h_graphs_2d_umap, labels, valid_indices, valid_smiles_values, plot_type='2D'):
    cmap = ListedColormap(['#440154', '#21918c', '#fde725'])
    norm = BoundaryNorm([1, 1.2, 2, 10], cmap.N)

    if plot_type == '2D':
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))

        scatter_tsne = ax[0].scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_tsne = plt.colorbar(scatter_tsne, ax=ax[0], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_tsne.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax[0].set_title('t-SNE of h_graph')

        scatter_pca = ax[1].scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_pca = plt.colorbar(scatter_pca, ax=ax[1], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_pca.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax[1].set_title('PCA of h_graph')

        scatter_umap = ax[2].scatter(h_graphs_2d_umap[:, 0], h_graphs_2d_umap[:, 1], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_umap = plt.colorbar(scatter_umap, ax=ax[2], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_umap.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax[2].set_title('UMAP of h_graph')

        # 添加交互标签
        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_umap = mplcursors.cursor(scatter_umap, hover=True)
        crs_umap.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    elif plot_type == '3D':
        fig = plt.figure(figsize=(22, 7))

        ax_tsne = fig.add_subplot(131, projection='3d')
        scatter_tsne = ax_tsne.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], h_graphs_2d_tsne[:, 2], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_tsne = plt.colorbar(scatter_tsne, ax=ax[0], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_tsne.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax_tsne.set_title('t-SNE of h_graph')

        ax_pca = fig.add_subplot(132, projection='3d')
        scatter_pca = ax_pca.scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], h_graphs_2d_pca[:, 2], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_pca = plt.colorbar(scatter_pca, ax=ax[1], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_pca.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax_pca.set_title('PCA of h_graph')

        ax_umap = fig.add_subplot(133, projection='3d')
        scatter_umap = ax_umap.scatter(h_graphs_2d_umap[:, 0], h_graphs_2d_umap[:, 1], h_graphs_2d_umap[:, 2], c=labels, cmap=cmap, norm=norm, s=5)
        cbar_umap = plt.colorbar(scatter_umap, ax=ax[2], boundaries=[1, 1.2, 2, 10], ticks=[1.1, 1.5, 2.5])
        cbar_umap.set_ticklabels(['<1.2', '1.2-2', '>2'])
        ax_umap.set_title('UMAP of h_graph')

        # 添加交互标签
        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_umap = mplcursors.cursor(scatter_umap, hover=True)
        crs_umap.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    plt.tight_layout()
    plt.savefig('h_graph_dimensionality_reduction.png')
    plt.savefig('h_graph_dimensionality_reduction.svg')
    plt.savefig('h_graph_dimensionality_reduction.pdf')
    plt.show()
def plot_RT_dimensionality_reduction(h_graphs_2d_tsne, h_graphs_2d_pca, h_graphs_2d_umap, labels, valid_indices, valid_smiles_values, plot_type='2D'):
    if plot_type == '2D':
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))
        scatter_tsne = ax[0].scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar(scatter_tsne, ax=ax[0])
        ax[0].set_title('t-SNE of h_graph')

        scatter_pca = ax[1].scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar(scatter_pca, ax=ax[1])
        ax[1].set_title('PCA of h_graph')

        scatter_umap = ax[2].scatter(h_graphs_2d_umap[:, 0], h_graphs_2d_umap[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar(scatter_umap, ax=ax[2])
        ax[2].set_title('UMAP of h_graph')
        # 添加交互标签
        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_umap = mplcursors.cursor(scatter_umap, hover=True)
        crs_umap.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    elif plot_type == '3D':
        fig = plt.figure(figsize=(22, 7))

        ax_tsne = fig.add_subplot(131, projection='3d')
        scatter_tsne = ax_tsne.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], h_graphs_2d_tsne[:, 2], c=labels, cmap='viridis', s=5)
        ax_tsne.set_title('t-SNE of h_graph')
        fig.colorbar(scatter_tsne, ax=ax_tsne)

        ax_pca = fig.add_subplot(132, projection='3d')
        scatter_pca = ax_pca.scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], h_graphs_2d_pca[:, 2], c=labels, cmap='viridis', s=5)
        ax_pca.set_title('PCA of h_graph')
        fig.colorbar(scatter_pca, ax=ax_pca)

        ax_umap = fig.add_subplot(133, projection='3d')
        scatter_umap = ax_umap.scatter(h_graphs_2d_umap[:, 0], h_graphs_2d_umap[:, 1], h_graphs_2d_umap[:, 2], c=labels, cmap='viridis', s=5)
        ax_umap.set_title('UMAP of h_graph')
        fig.colorbar(scatter_umap, ax=ax_umap)

        # 添加交互标签
        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_umap = mplcursors.cursor(scatter_umap, hover=True)
        crs_umap.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    plt.tight_layout()
    plt.savefig('h_graph_dimensionality_reduction.png')
    plt.savefig('h_graph_dimensionality_reduction.svg')
    plt.savefig('h_graph_dimensionality_reduction.pdf')
    plt.show()    
def plot_density(result, title):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=result[:, 0], y=result[:, 1], cmap='Blues', fill=True, bw_adjust=0.5)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()

column = 'IC'  # 目标列
goal = 'hidden'
only_tsne = False  # 仅绘制t-SNE图
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

# 加载数据集
with open(f'alphacsv/{columns_data[column]}_graph_data.pkl', 'rb') as f:
    data_graphs, data_graph_informations, props, big_index = pickle.load(f) 

# 加载 CSV 文件中的 alpha 列
HPLC = pd.read_csv(f'alphacsv/{columns_data[column]}.csv')
alpha_values = HPLC['beta'].values
index_values = HPLC['Unnamed: 0'].values
smiles_values = HPLC['SMILES'].values
enantiomers_index_values = HPLC['index'].values

# 创建索引到 alpha 和 SMILES 的映射
index_to_alpha = {idx: alpha for idx, alpha in zip(index_values, alpha_values)}
index_to_smiles = {idx: smiles for idx, smiles in zip(index_values, smiles_values)}
index_to_enantiomers = {idx: enantiomers for idx, enantiomers in zip(index_values, enantiomers_index_values)}

# 创建 enantiomer_to_index 的映射
enantiomer_to_index = {}
for idx, enantiomer in zip(index_values, enantiomers_index_values):
    if enantiomer not in enantiomer_to_index:
        enantiomer_to_index[enantiomer] = []
    enantiomer_to_index[enantiomer].append(idx)

# 提取有效的索引
valid_indices = [info.data_index.item() for info in data_graph_informations]
valid_y_value = [info.y.item() for info in data_graph_informations]

# 对应的 alpha 标签和 SMILES
valid_alpha_values = [index_to_alpha[idx] for idx in valid_indices]
valid_smiles_values = [index_to_smiles[idx] for idx in valid_indices]
valid_enantiomers_values = [index_to_enantiomers[idx] for idx in valid_indices]
labels = valid_alpha_values

total_num = len(data_graphs)
print('data num:', total_num, len(data_graph_informations), len(props))

if goal == 'hidden':
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
    h_graphs = np.concatenate(h_graphs, axis=0)        
elif goal == 'graph':             
    h_graphs = []
    for i in range(total_num):
        data = torch.sum(data_graphs[i].x,dim=0).reshape(1,-1)
        h_graphs.append(data.cpu().numpy())
    h_graphs = np.concatenate(h_graphs, axis=0)
elif goal == 'Morgan':             
    h_graphs = []
    for i in range(total_num):
        mol = Chem.MolFromSmiles(valid_smiles_values[i])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        h_graphs.append(np.array(fp))
    h_graphs = np.array(h_graphs)    

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
h_graphs_2d_tsne = tsne.fit_transform(h_graphs)
# 如果仅绘制t-SNE图
if only_tsne:
    plot_tsne_only(h_graphs_2d_tsne, labels, valid_indices, valid_smiles_values)
else:
    # 使用 PCA 进行降维
    pca = PCA(n_components=2)
    h_graphs_2d_pca = pca.fit_transform(h_graphs)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_components=2)
    h_graphs_2d_umap = umap_model.fit_transform(h_graphs)

    # 调用绘图函数，选择 '2D' 或 '3D'
    plot_dimensionality_reduction(h_graphs_2d_tsne, h_graphs_2d_pca, h_graphs_2d_umap, labels, valid_indices, valid_smiles_values, plot_type='2D')

    # 绘制密度图
    plot_density(h_graphs_2d_tsne, 't-SNE Density Plot')
    plot_density(h_graphs_2d_pca, 'PCA Density Plot')
    plot_density(h_graphs_2d_umap, 'UMAP Density Plot')

    # 计算聚类指标
    label = labels
    sil_score = silhouette_score(h_graphs_2d_umap, label)
    print(f'Silhouette Score (UMAP): {sil_score}')

    db_index = davies_bouldin_score(h_graphs_2d_umap, label)
    print(f'Davies-Bouldin Index (UMAP): {db_index}')

    sil_score = silhouette_score(h_graphs_2d_tsne, label)
    print(f'Silhouette Score (t-SNE): {sil_score}')

    db_index = davies_bouldin_score(h_graphs_2d_tsne, label)
    print(f'Davies-Bouldin Index (t-SNE): {db_index}')

    sil_score = silhouette_score(h_graphs_2d_pca, label)
    print(f'Silhouette Score (PCA): {sil_score}')

    db_index = davies_bouldin_score(h_graphs_2d_pca, label)
    print(f'Davies-Bouldin Index (PCA): {db_index}')

    # 计算分离度量
    distances = []
    for i, idx in enumerate(valid_indices):
        enantiomer_idx = index_to_enantiomers[idx]
        if enantiomer_idx in enantiomer_to_index:
            enantiomer_pair = enantiomer_to_index[enantiomer_idx]
            if len(enantiomer_pair) == 2 and enantiomer_pair[0] in valid_indices and enantiomer_pair[1] in valid_indices:
                j = valid_indices.index(enantiomer_pair[1] if enantiomer_pair[0] == idx else enantiomer_pair[0])
                dist = euclidean(h_graphs_2d_umap[i], h_graphs_2d_umap[j])
                distances.append(dist)

    average_distance = np.mean(distances)
    print(f'Average Separation Distance: {average_distance}')
