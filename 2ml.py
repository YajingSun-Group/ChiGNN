import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CHIGraphModel,eval,train,test,train_with_flag
from torch_geometric.data import Batch
from chienn import collate_with_circle_index
from torch_geometric.loader import DataLoader
import pandas as pd
import warnings
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings('ignore')
MODEL ='Test'   #['Train','Test','T']
column = 'IC'  # 目标列：['ADH','ODH','IC','IA','OJH','ASH','IC3','IE','ID','OD3', 'IB','AD','AD3','IF','OD','AS','OJ3','IG','AZ','IAH','OJ','ICH','OZ3','IF3','IAU']
test_mode='fixed' #fixed or random
sample = 'random_sampling'  # random_sampling or stratified_sampling
augmentation=False
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
    'IAU': 'IAU_charity_0823',
    'All':'All_charity_0823'
}

with open(f'columncsv/{columns_data[column]}_graph_data.pkl', 'rb') as f:
    data_graphs,data_graph_informations,big_index = pickle.load(f) 
total_num = len(data_graphs)
print('data num:',total_num,len(data_graph_informations))
def parse_args():
    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='ChiGNN',
                        help='task name')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_root', type=str, default="dataset",
                        help='dataset root')
    args = parser.parse_args()

    return args
def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)
    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)
train_ratio = 0.90
validate_ratio = 0.05
test_ratio = 0.05
 
# 根据分层抽样或随机抽样划分数据集
if sample == 'random_sampling':
    #given random seed
    if column=='ODH':
        np.random.seed(388)
    elif column=='ADH':
        np.random.seed(505)
    elif column=='IC':
        #np.random.seed(152)
        np.random.seed(526)
    elif column=='IA':
        np.random.seed(388)
    elif column=='All':
        np.random.seed(388)    
    else:
        np.random.seed(526) 
    # 自动数据加载和划分
    data_array = np.arange(0, total_num, 1)
    np.random.shuffle(data_array)

    train_num = int(len(data_array) * train_ratio)
    test_num = int(len(data_array) * test_ratio)
    val_num = int(len(data_array) * validate_ratio)

    train_index = data_array[0:train_num]
    valid_index = data_array[train_num:train_num + val_num]
    if test_mode == 'fixed':
        test_index = data_array[total_num - test_num:]
    elif test_mode == 'random':
        test_index = data_array[train_num + val_num:train_num + val_num + test_num]

elif sample =='stratified_sampling':
    # 使用分层抽样划分数据集
    from sklearn.model_selection import train_test_split
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    tsne = TSNE(n_components=2, random_state=0)
    h_graphs_2d_tsne = tsne.fit_transform(np.concatenate([graph.x.cpu().numpy().sum(axis=0).reshape(1, -1) for graph in data_graphs]))
    kmeans = KMeans(n_clusters=30, random_state=0)
    clusters = kmeans.fit_predict(h_graphs_2d_tsne)

    stratified_indices = []
    for label in set(clusters):
        indices = np.where(clusters == label)[0]
        stratified_indices.append(indices)
    train_indices, valid_indices, test_indices = [], [], []
    for indices in stratified_indices:
        train_idx, temp_idx = train_test_split(indices, test_size=0.1, random_state=42)
        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=0)
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)
        test_indices.extend(test_idx)

    train_index = np.array(train_indices)
    valid_index = np.array(valid_indices)
    test_index = np.array(test_indices)

torch.random.manual_seed(525)

train_data_graphs = []
valid_data_graphs = []
test_data_graphs = []
train_data_graph_informations = []
valid_data_graph_informations = []
test_data_graph_informations = []

for i in test_index:
    test_data_graphs.append(data_graphs[i])
    test_data_graph_informations.append(data_graph_informations[i])
for i in valid_index:
    valid_data_graphs.append(data_graphs[i])
    valid_data_graph_informations.append(data_graph_informations[i])
for i in train_index:
    train_data_graphs.append(data_graphs[i])
    train_data_graph_informations.append(data_graph_informations[i])
# 检查当前是否有可用的GPU，如果有则使用GPU，否则使用CPU
args = parse_args()
prepartion(args)
device = args.device

print('使用的设备：', device)

# 获取GPU的编号，如果使用CPU则返回None
if device.type == 'cuda':
    print('GPU编号：', torch.cuda.current_device())
    print('GPU名称：', torch.cuda.get_device_name(torch.cuda.current_device())) 
criterion_fn = torch.nn.MSELoss()
nn_params = {
    'num_tasks': 3,
    'num_layers': args.num_layers,
    'emb_dim': args.emb_dim,
    'drop_ratio': args.drop_ratio,
    'graph_pooling': args.graph_pooling,
    'descriptor_dim': 1
}
model = CHIGraphModel(**nn_params).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(num_params)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
writer = SummaryWriter(log_dir=args.save_dir)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5) 
print('===========Data Prepared================')
train_loader_data_graphs = DataLoader(train_data_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
train_loader_data_graph_informations = DataLoader(train_data_graph_informations, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
valid_loader_data_graphs = DataLoader(valid_data_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
valid_loader_data_graph_informations = DataLoader(valid_data_graph_informations, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader_data_graphs = DataLoader(test_data_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader_data_graph_informations = DataLoader(test_data_graph_informations, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
if MODEL == 'Train':
    if augmentation is False:
        try:
            os.makedirs(f'saves/model_{column}CHI')
        except OSError:
            pass
        # 创建一个空的 DataFrame 用来保存结果
        results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Valid MSE', 'Valid R2'])   
        for epoch in tqdm(range(args.epochs)):
            train_loss = train(model, device, train_loader_data_graphs,train_loader_data_graph_informations, optimizer, criterion_fn,train_data_graphs,batch_size=args.batch_size)
            if (epoch + 1) % 100 == 0:
                valid_mse,valid_r2 = eval(model, device, valid_loader_data_graphs,valid_loader_data_graph_informations,valid_data_graphs,batch_size=args.batch_size)
                print(train_loss,valid_mse,valid_r2)
                # 将结果添加到 DataFrame 中
                results_df = results_df.append({'Epoch': epoch + 1, 'Train Loss': train_loss, 'Valid MSE': valid_mse, 'Valid R2': valid_r2}, ignore_index=True)
                #torch.save(model.state_dict(), f'saves/model_{column}/model_save_{epoch + 1}.pth')
                torch.save(model.state_dict(), f'saves/model_{column}CHI/model_save_{epoch + 1}.pth')
        # 保存 DataFrame 到 CSV 文件
        #results_df.to_csv(f'saves/model_{column}/record.csv', index=False) 
        results_df.to_csv(f'saves/model_{column}CHI/record.csv', index=False)        
    elif augmentation is True:
        try:
            os.makedirs(f'saves/model_{column}CHI_aug')
        except OSError:
            pass
        # 创建一个空的 DataFrame 用来保存结果
        results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Valid MSE', 'Valid R2'])    
        for epoch in tqdm(range(args.epochs)):
            train_loss = train_with_flag(model, device, train_loader_data_graphs,train_loader_data_graph_informations, optimizer, criterion_fn,train_data_graphs,batch_size=args.batch_size, M=3, alpha=0.01)
            if (epoch + 1) % 100 == 0:
                valid_mse,valid_r2 = eval(model, device, valid_loader_data_graphs,valid_loader_data_graph_informations,valid_data_graphs,batch_size=args.batch_size)
                print(train_loss,valid_mse,valid_r2)
                # 将结果添加到 DataFrame 中
                results_df = results_df.append({'Epoch': epoch + 1, 'Train Loss': train_loss, 'Valid MSE': valid_mse, 'Valid R2': valid_r2}, ignore_index=True)
                #torch.save(model.state_dict(), f'saves/model_{column}/model_save_{epoch + 1}.pth')
                torch.save(model.state_dict(), f'saves/model_{column}CHI_aug/model_save_{epoch + 1}.pth')
        # 保存 DataFrame 到 CSV 文件
        results_df.to_csv(f'saves/model_{column}CHI_aug/record.csv', index=False)    
elif MODEL == 'Test':
    model.load_state_dict(
            torch.load(f'saves/model_{column}CHI/model_save_1500.pth'))
    y_pred, y_true, R_square, test_mse,y_pred_10,y_pred_90 = test(model, device, test_loader_data_graphs,test_loader_data_graph_informations,test_data_graphs,batch_size=args.batch_size)
    y_pred=y_pred.cpu().data.numpy()
    y_true = y_true.cpu().data.numpy()
    y_pred_10=y_pred_10.cpu().data.numpy()
    y_pred_90=y_pred_90.cpu().data.numpy()
    # 计算指标
    relative_error = np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())

    print('relative_error', relative_error)
    print('MAE', mae)
    print('RMSE', rmse)
    print('R_square', r_square)
    # 绘制散点图并美化
    plt.figure(1, figsize=(2.5, 2.5), dpi=300)  # 增大图表尺寸
    plt.style.use('ggplot')
    plt.scatter(y_true, y_pred, c='#8983BF', s=15, alpha=0.4, edgecolors='w', linewidth=0.5)

    # 绘制 y=x 参考线
    plt.plot(np.arange(0, 60), np.arange(0, 60), linewidth=1.5, linestyle='--', color='black')
    # # 设置刻度
    plt.yticks(np.arange(0, 66, 10), np.arange(0, 66, 10), fontproperties='Times New Roman', size=8)
    plt.xticks(np.arange(0, 66, 10), np.arange(0, 66, 10), fontproperties='Times New Roman', size=8)
    # # 设置标签
    plt.xlabel('Experiment', fontproperties='Times New Roman', size=10)
    plt.ylabel('Predicted', fontproperties='Times New Roman', size=10)

    # # 添加 R²、RMSE 和 MAE 注释
    metrics_text = f'R²: {r_square:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', fontproperties='Times New Roman',bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.8))

    # # 保存图表
    plt.savefig('testrg.png', bbox_inches='tight')
    # plt.figure(1,figsize=(2.5,2.5),dpi=300)
    # plt.style.use('ggplot')
    # plt.scatter(y_true, y_pred, c='#8983BF',s=15,alpha=0.4)
    # plt.plot(np.arange(0, 60), np.arange(0, 60),linewidth=1.5,linestyle='--',color='black')
    # plt.yticks(np.arange(0,66,10),np.arange(0,66,10),fontproperties='Arial', size=8)
    # plt.xticks(np.arange(0,66,10),np.arange(0,66,10),fontproperties='Arial', size=8)
    # plt.xlabel('Observed data', fontproperties='Arial', size=8)
    # plt.ylabel('Predicted data', fontproperties='Arial', size=8)
    # plt.savefig('testrg.png') 
else: 
    # 手动提取数据
    model.load_state_dict(
            torch.load(f'saves/model_{column}CHI/model_save_1500.pth'))
    results_df = pd.DataFrame(columns=['i', 'data_index','real', 'pred', 'ae'])
    model.eval()
    with torch.no_grad():
        for i in test_index:
            data_graph=data_graphs[i]
            graph_information=data_graph_informations[i]
            data_graph = collate_with_circle_index([data_graph], k_neighbors=3)
            data_graph = data_graph.to(device)
            graph_information=graph_information.to(device)
        
            pred = model(data_graph)[0] 
            new_row = {'i': i,
                       'data_index':graph_information.data_index.detach().cpu().data.numpy(),
                        'real': graph_information.y[0].detach().cpu().data.numpy(),
                        'pred': pred[0][1].detach().cpu().data.numpy(),
                        'ae': abs(graph_information.y[0].detach().cpu().data.numpy()-pred[0][1].detach().cpu().data.numpy())
                        }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        new_row = {'i': 0,
                   'data_index':graph_information.data_index.detach().cpu().data.numpy(),
                        'real':0,
                        'pred':0,
                        'ae':0
                        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        for i in valid_index:
            data_graph=data_graphs[i]
            graph_information=data_graph_informations[i]
            data_graph = collate_with_circle_index([data_graph], k_neighbors=3)
            data_graph = data_graph.to(device)
            graph_information=graph_information.to(device)
        
            pred = model(data_graph)[0] 
            new_row = {'i': i,
                       'data_index':graph_information.data_index.detach().cpu().data.numpy(),
                        'real': graph_information.y[0].detach().cpu().data.numpy(),
                        'pred': pred[0][1].detach().cpu().data.numpy(),
                        'ae': abs(graph_information.y[0].detach().cpu().data.numpy()-pred[0][1].detach().cpu().data.numpy())
                        }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)    
    results_df.to_csv('valtest.csv', index=False)    