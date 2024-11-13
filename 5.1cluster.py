import pickle
import numpy as np
import torch
from model import CHIGraphModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from chienn import collate_with_circle_index
import pandas as pd
import mplcursors
from sklearn.metrics import silhouette_score
import umap
import os

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float64))  # 转换为 double 类型
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_dimensionality_reduction(h_graphs_2d_tsne, h_graphs_2d_pca, h_graphs_2d_umap, labels, valid_indices, valid_smiles_values, plot_type='2D', cluster=None, clusterer_tsne=None, clusterer_pca=None, clusterer_umap=None):
    if plot_type == '2D':
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))

        plt.sca(ax[0])
        plot_decision_boundaries(clusterer_tsne, h_graphs_2d_tsne)
        ax[0].set_title('t-SNE of h_graph')

        scatter_tsne = ax[0].scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=cluster[0], cmap='viridis', s=5, alpha=0)

        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        plt.sca(ax[1])
        plot_decision_boundaries(clusterer_pca, h_graphs_2d_pca)
        ax[1].set_title('PCA of h_graph')

        scatter_pca = ax[1].scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], c=cluster[1], cmap='viridis', s=5, alpha=0)

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))
        plt.sca(ax[2])
        plot_decision_boundaries(clusterer_umap, h_graphs_2d_umap)
        ax[2].set_title('UMAP of h_graph')

        scatter_umap = ax[2].scatter(h_graphs_2d_umap[:, 0], h_graphs_2d_umap[:, 1], c=cluster[2], cmap='viridis', s=5, alpha=0)

        crs_umap = mplcursors.cursor(scatter_umap, hover=True)
        crs_umap.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))
    elif plot_type == '3D':
        fig = plt.figure(figsize=(15, 7))

        ax_tsne = fig.add_subplot(121, projection='3d')
        scatter_tsne = ax_tsne.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], h_graphs_2d_tsne[:, 2], c=cluster, cmap='viridis', s=5)
        ax_tsne.set_title('t-SNE of h_graph')
        fig.colorbar(scatter_tsne, ax=ax_tsne)

        ax_pca = fig.add_subplot(122, projection='3d')
        scatter_pca = ax_pca.scatter(h_graphs_2d_pca[:, 0], h_graphs_2d_pca[:, 1], h_graphs_2d_pca[:, 2], c=cluster, cmap='viridis', s=5)
        ax_pca.set_title('PCA of h_graph')
        fig.colorbar(scatter_pca, ax=ax_pca)

        crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
        crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

        crs_pca = mplcursors.cursor(scatter_pca, hover=True)
        crs_pca.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    plt.tight_layout()
    plt.savefig('h_graph_dimensionality_reduction.png')
    plt.savefig('h_graph_dimensionality_reduction.svg')
    plt.savefig('h_graph_dimensionality_reduction.pdf')
    plt.show()

def plot_tsne_only(h_graphs_2d_tsne, valid_indices, valid_smiles_values, cluster_tsne, clusterer_tsne):
    plt.figure(figsize=(7, 5))
    plot_decision_boundaries(clusterer_tsne, h_graphs_2d_tsne)
    scatter_tsne = plt.scatter(h_graphs_2d_tsne[:, 0], h_graphs_2d_tsne[:, 1], c=cluster_tsne, cmap='viridis', s=5, alpha=0)
    #plt.title('t-SNE of h_graph', fontsize=18)
    
    plt.xlabel('x', fontsize=16)  # 设置横轴标题及其字体大小
    plt.ylabel('y', fontsize=16)  # 设置纵轴标题及其字体大小
    
    plt.xticks(fontsize=14)  # 设置横轴数字的字体大小
    plt.yticks(fontsize=14)  # 设置纵轴数字的字体大小
    crs_tsne = mplcursors.cursor(scatter_tsne, hover=True)
    crs_tsne.connect("add", lambda sel: sel.annotation.set_text(f"Index: {valid_indices[sel.index]}\nSMILES: {valid_smiles_values[sel.index]}"))

    plt.tight_layout()
    plt.savefig('tsne_only.png')
    plt.savefig('tsne_only.svg')
    plt.savefig('tsne_only.pdf')
    plt.show()
column = 'IC'
goal = 'hidden'
kiskonwn = True
only_tsne = True  # 如果为True，则只绘制t-SNE图

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
    model.load_state_dict(torch.load(f'saves/model_{column}CHI/model_save_1500.pth'))
    model.eval()
    h_graphs = []
    with torch.no_grad():
        for i in range(total_num):
            data = data_graphs[i]
            data = collate_with_circle_index([data], k_neighbors=3)
            prop = props[i]
            mol_pred, h_graph = model(data.to(device), prop.to(device))
            h_graphs.append(h_graph.cpu().numpy())
    h_graphs = np.concatenate(h_graphs, axis=0).astype(np.float64)  # 转换为 double 类型    
elif goal == 'graph':             
    h_graphs = []
    for i in range(total_num):
        data = torch.sum(data_graphs[i].x, dim=0).reshape(1, -1)
        h_graphs.append(data.cpu().numpy())
    h_graphs = np.concatenate(h_graphs, axis=0).astype(np.float64)  # 转换为 double 类型

tsne = TSNE(n_components=2, random_state=0)
h_graphs_2d_tsne = tsne.fit_transform(h_graphs)

if only_tsne:
    k = 38 
    kmeans_tsne = KMeans(n_clusters=k, random_state=0)
    clusters_tsne = kmeans_tsne.fit_predict(h_graphs_2d_tsne.astype(np.float64))  # 转换为 double 类型
    plot_tsne_only(h_graphs_2d_tsne, valid_indices, valid_smiles_values, clusters_tsne, kmeans_tsne)
else:
    if kiskonwn is True:
        k = 38
    else:    
        # 确定最佳 k 值
        kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(h_graphs_2d_tsne)
                        for k in range(1, 100)]
        silhouette_scores = [silhouette_score(h_graphs_2d_tsne, model.labels_)
                            for model in kmeans_per_k[1:]]

        plt.figure(figsize=(6, 6), dpi=400)
        plt.plot(range(2, 100), silhouette_scores, "bo-", c='y')
        plt.xlabel("Number of clusters", fontsize=5)
        plt.ylabel("Silhouette score", fontsize=5)
        plt.tick_params(labelsize=5)
        plt.legend(["Silhouette score"], fontsize=5)
        plt.tight_layout()
        ax = plt.gca()
        ax.set_facecolor('black')
        plt.savefig('bestk.png')

        # 使用最佳 k 值进行聚类
        best_k = np.argmax(silhouette_scores) + 2  # 因为 range(2, 100) 对应于索引 [0, 98] 
        print(best_k)
        k = best_k  

    kmeans_tsne = KMeans(n_clusters=k, random_state=0)
    clusters_tsne = kmeans_tsne.fit_predict(h_graphs_2d_tsne.astype(np.float64))  # 转换为 double 类型

    pca = PCA(n_components=2)
    h_graphs_2d_pca = pca.fit_transform(h_graphs)
    kmeans_pca = KMeans(n_clusters=k, random_state=0)
    clusters_pca = kmeans_pca.fit_predict(h_graphs_2d_pca.astype(np.float64))  # 转换为 double 类型

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_components=2)
    h_graphs_2d_umap = umap_model.fit_transform(h_graphs)
    kmeans_umap = KMeans(n_clusters=k, random_state=0)
    clusters_umap = kmeans_umap.fit_predict(h_graphs_2d_umap.astype(np.float64))  # 转换为 double 类型

    plot_dimensionality_reduction(h_graphs_2d_tsne, h_graphs_2d_pca, h_graphs_2d_umap, labels, valid_indices, valid_smiles_values, plot_type='2D', cluster=[clusters_tsne, clusters_pca, clusters_umap], clusterer_tsne=kmeans_tsne, clusterer_pca=kmeans_pca, clusterer_umap=kmeans_umap)
