import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.data import Batch
from chienn.data.edge_graph.collate_circle_index import collate_circle_index
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from chienn.model.chienn_layer import ChiENNLayer


class ChiENNModel(nn.Module):
    """
    A simplified version of the ChiENN model used in the experimental part of the paper. To make this implementation
    concise and clear, we excluded computation of RWSE positional encodings. To reproduce the results from the paper,
    use the `experiments` module. Note that this model behaves like GPSModel from experiments/graphgps/network.gps_model.py
    with `local_gnn_type` set to `ChiENN' and `global_model_type` set to `None` (except for the positional encodings).
    Therefore, we wrapped `ChiENNLayer` with `GPSLayer`.
    """

    def __init__(
            self,
            k_neighbors: int = 3,
            in_node_dim: int = 93,
            hidden_dim: int = 128,
            n_layers: int = 3,
            dropout: float = 0.0,
    ):
        """

        Args:
            k_neighbors: number of k consecutive neighbors used to create a chiral-sensitive message. It's `k` from
                the eq. (4) in the paper.
            in_node_dim: number of input node features. Default (93) differs from the value used in the `experiments`
                module (118) as here we explicitly excluded chiral tags, while in the `experiments` we masked them.
            out_dim: output dimension.
            n_layers: number of ChiENN layers.
            dropout: dropout probability.
        """
        super().__init__()
        self.embedding_layer = nn.Linear(in_node_dim, hidden_dim)
        self.gps_layers = nn.ModuleList([
            GPSLayer(
                gnn_layer=ChiENNLayer(
                    hidden_dim=hidden_dim,
                    k_neighbors_embeddings_names=['linear'] * k_neighbors
                ),
                hidden_dim=hidden_dim,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        

    def forward(self, batch: Batch) -> Tensor:
        """
        Run ChiENN model.

        Args:
            batch: a batch representing `batch_size` graphs. Contains the following attributes:
                - x: (num_nodes, hidden_dim) node features
                - batch (num_nodes,): batch indices of the nodes.
                - edge_index (2, num_edges): edge indices
                - circle_index (num_nodes, circle_size): indices of neighbors forming an order around a node.
                - parallel_node_index (num_nodes,): indices of parallel nodes.

        Returns:
            Output of the shape (batch_size, out_dim).
        """
        batch.x = self.embedding_layer(batch.x)
        for gps_layers in self.gps_layers:
            batch.x = gps_layers(batch)   
        x=batch.x
        return x


class GPSLayer(nn.Module):
   
    def __init__(
            self,
            gnn_layer: nn.Module,
            hidden_dim: int = 128,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, batch: Batch) -> Tensor:
        residual_x = batch.x
        x = self.gnn_layer(batch) + residual_x
        x = self.norm_1(x)
        x = self.mlp(x) + x
        return self.norm_2(x)
class CHIGraphModel(nn.Module):

    def __init__(self, num_tasks=3, num_layers=3, emb_dim=128, drop_ratio=0,  graph_pooling="attention",
                 descriptor_dim=1781):
        
        super(CHIGraphModel, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.descriptor_dim=descriptor_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.chienn=ChiENNModel(k_neighbors=3)
        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))

        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, self.num_tasks),
        )

        self.descriptor = nn.Sequential(nn.Linear(self.descriptor_dim, self.emb_dim),
                                           nn.Sigmoid(),
                                           nn.Linear(self.emb_dim, self.emb_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, data_graphs, mask=None):
        h_graph=self.chienn(data_graphs)
        if mask is not None:
            h_graph = h_graph * mask.unsqueeze(-1)  # Apply mask
        h_graph = self.pool(h_graph, data_graphs.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output,h_graph
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=1e8),h_graph
def q_loss(q,y_true,y_pred):
    e = (y_true-y_pred)
    return torch.mean(torch.maximum(q*e, (q-1)*e))
def eval(model, device, loader_data_graphs,loader_data_graph_informations, data_graphs,batch_size):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_10 = []
    y_pred_90 = []

    num_data = len(data_graphs)

    with torch.no_grad():
        for step, batch in enumerate(zip(loader_data_graphs,loader_data_graph_informations)):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, num_data)
            batch_data_graphs=batch[0]
            batch_data_graph_informations=batch[1]
            
        
            # 添加circle_index到batch_data_list，假设是相关函数
            batch_data_graphs.circle_index = collate_circle_index(data_graphs[start_idx:end_idx], 3)
            batch_data_graphs = batch_data_graphs.to(device)
            batch_data_graph_informations=batch_data_graph_informations.to(device)
            
            # 模型预测
            pred=model(batch_data_graphs)[0]
            # 收集结果
            y_true.append(batch_data_graph_informations.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:,1].detach().cpu())
            y_pred_10.append(pred[:, 0].detach().cpu())
            y_pred_90.append(pred[:, 2].detach().cpu())

    # 合并所有批次的结果
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred_10 = torch.cat(y_pred_10, dim=0)
    y_pred_90 = torch.cat(y_pred_90, dim=0)

    # 可视化代码已注释，根据需要取消注释来使用
    # plt.plot(y_pred.cpu().data.numpy(),c='blue')
    # plt.plot(y_pred_10.cpu().data.numpy(),c='yellow')
    # plt.plot(y_pred_90.cpu().data.numpy(),c='black')
    # plt.plot(y_true.cpu().data.numpy(),c='red')
    # plt.show()

    return torch.mean((y_true - y_pred) ** 2).data.numpy(),1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())

# 注意，此处假设数据集已经是Data对象的列表形式，每个列表项能被Batch.from_data_list正确处理。




def train(model, device, loader_data_graphs,loader_data_graph_informations, optimizer, criterion_fn, data_graphs,batch_size):
    model.train()
    loss_accum = 0
    num_data =len(data_graphs)
    num_steps = (num_data + batch_size - 1) // batch_size  # 计算必要的步骤数，确保包括所有数据
    for step, batch in enumerate(zip(loader_data_graphs,loader_data_graph_informations)):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, num_data)
        batch_data_graphs=batch[0]
        batch_data_graph_informations=batch[1]
        
        # 添加circle_index到batch_data_list，假设是相关函数
        batch_data_graphs.circle_index = collate_circle_index(data_graphs[start_idx:end_idx], 3)
        batch_data_graphs = batch_data_graphs.to(device)
        batch_data_graph_informations=batch_data_graph_informations.to(device)
        # 模型预测
        pred = model(batch_data_graphs)[0]
        true = batch_data_graph_informations.y
        # 优化器清零
        optimizer.zero_grad()

        # 计算损失
        loss = q_loss(0.1, true, pred[:, 0]) + torch.mean((true - pred[:, 1]) ** 2) + q_loss(0.9, true, pred[:, 2]) + \
               torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + torch.mean(torch.relu(pred[:, 1] - pred[:, 2])) + torch.mean(torch.relu(2 - pred))
        #loss = criterion_fn(pred, true)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / num_steps

# 注意：collate_circle_index需要你自己定义或调整以适应你的数据结构和需求
#定义FLAG函数
def flag(model, data_loader, optimizer, M, alpha, device):
    model.train()
    for data_graphs, data_graph_informations in data_loader:
        data_graphs = data_graphs.to(device)
        data_graph_informations = data_graph_informations.to(device)
        origin=data_graphs.x.clone()
        optimizer.zero_grad()
        true = data_graph_informations.y
        # 初始化扰动
        perturb = torch.FloatTensor(*data_graphs.x.shape).uniform_(-alpha, alpha).to(device)
        perturb.requires_grad_()
        data_graphs.x  =origin + perturb
        out = model(data_graphs)[0]
        loss = (q_loss(0.1, true, out[:, 0]) + torch.mean((true - out[:, 1]) ** 2) + q_loss(0.9, true, out[:, 2]) + \
               torch.mean(torch.relu(out[:, 0] - out[:, 1])) + torch.mean(torch.relu(out[:, 1] - out[:, 2])) + torch.mean(torch.relu(2 - out))) / M
        for _ in range(M):
            loss.backward()
            # 更新扰动
            with torch.no_grad():
                perturb_data = perturb + alpha * torch.sign(perturb.grad)
                perturb.data = perturb_data.data
                perturb.grad.zero_()
            data_graphs.x  =origin + perturb
            out = model(data_graphs)[0]
            loss = (q_loss(0.1, true, out[:, 0]) + torch.mean((true - out[:, 1]) ** 2) + q_loss(0.9, true, out[:, 2]) + \
               torch.mean(torch.relu(out[:, 0] - out[:, 1])) + torch.mean(torch.relu(out[:, 1] - out[:, 2])) + torch.mean(torch.relu(2 - out))) / M
        data_graphs.x  =origin
        loss.backward()
        optimizer.step()
# FLAG整合训练
def train_with_flag(model, device, loader_data_graphs, loader_data_graph_informations, optimizer, criterion_fn, data_graphs, batch_size, M, alpha):
    model.train()
    loss_accum = 0
    num_data = len(data_graphs)
    num_steps = (num_data + batch_size - 1) // batch_size
    for step, batch in enumerate(zip(loader_data_graphs, loader_data_graph_informations)):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, num_data)
        batch_data_graphs = batch[0]
        batch_data_graph_informations = batch[1]
        
        batch_data_graphs.circle_index = collate_circle_index(data_graphs[start_idx:end_idx], 3)
        # FLAG扰动
        flag(model, [(batch_data_graphs, batch_data_graph_informations)], optimizer, M, alpha, device)
        pred = model(batch_data_graphs)[0]
        true = batch_data_graph_informations.y
        optimizer.zero_grad()
        loss = q_loss(0.1, true, pred[:, 0]) + torch.mean((true - pred[:, 1]) ** 2) + q_loss(0.9, true, pred[:, 2]) + \
               torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + torch.mean(torch.relu(pred[:, 1] - pred[:, 2])) + torch.mean(torch.relu(2 - pred))
        #loss = criterion_fn(pred, true)
         # 反向传播和优化
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / num_steps
def test(model, device, loader_data_graphs,loader_data_graph_informations, data_graphs,batch_size):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_10 = []
    y_pred_90 = []

    num_data =len(data_graphs)
    with torch.no_grad():
        for step, batch in enumerate(zip(loader_data_graphs,loader_data_graph_informations)):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, num_data)
            batch_data_graphs=batch[0]
            batch_data_graph_informations=batch[1]
        
            # 添加circle_index到batch_data_list，假设是相关函数
            batch_data_graphs.circle_index = collate_circle_index(data_graphs[start_idx:end_idx], 3)
            batch_data_graphs = batch_data_graphs.to(device)
            batch_data_graph_informations=batch_data_graph_informations.to(device)
            
            # 模型预测
            pred=model(batch_data_graphs)[0]
            # 收集结果
            y_true.append(batch_data_graph_informations.y.detach().cpu().reshape(-1))
            y_pred.append(pred[:, 1].detach().cpu())
            y_pred_10.append(pred[:, 0].detach().cpu())
            y_pred_90.append(pred[:, 2].detach().cpu())

    # 合并所有批次的结果
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_pred_10 = torch.cat(y_pred_10, dim=0)
    y_pred_90 = torch.cat(y_pred_90, dim=0)

    # 计算R^2和MAE
    R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    test_mse = torch.mean((y_true - y_pred) ** 2)


    print(R_square)
    return y_pred, y_true, R_square, test_mse, y_pred_10, y_pred_90

            