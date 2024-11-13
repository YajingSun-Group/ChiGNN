import pickle
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from torch_geometric.data import Data
from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.featurization.convert_mol_to_data import convert_mol_to_data
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from rdkit import Chem
from chienn import smiles_to_data_with_circle_index
column = 'IA'  # 目标列：['ADH','ODH','IC','IA','OJH','ASH','IC3','IE','ID','OD3', 'IB','AD','AD3','IF','OD','AS','OJ3','IG','AZ','IAH','OJ','ICH','OZ3','IF3','IAU']

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
def Construct_dataset(data_index, T1, speed, All_smiles):
    big_index = []
    data_graphs=[]
    data_graph_informations=[]
    for i in tqdm(range(len(All_smiles))):
        y = torch.Tensor([float(T1[i]) * float(speed[i])])
        data_index_int=torch.from_numpy(np.array(data_index[i])).to(torch.int64)

        if y[0]>60:
            big_index.append(i)
            continue
        smile=All_smiles[i]
        try:
            mol=smiles_to_3d_mol(smile)
            data = mol_to_data(mol)
    
            edge_index=data.edge_index
            data = to_edge_graph(data)
            data.pos = None
        except:
            print(i,smile)
            continue 
              
        data_graphs.append(data)   
        data_graph_information = Data(edge_index=edge_index,y=y,data_index=data_index_int)
        data_graph_informations.append(data_graph_information)
    return data_graphs,data_graph_informations,big_index
if column in columns_data:
    HPLC = pd.read_csv(f'columncsv/{columns_data[column]}.csv')
    all_smile = HPLC['SMILES'].values
    T1 = HPLC['RT'].values
    Speed = HPLC['Speed'].values
    index = HPLC['Unnamed: 0'].values
    data_graphs,data_graph_informations,big_index = Construct_dataset(index, T1, Speed, all_smile)
    with open(f'columncsv/{columns_data[column]}_graph_data.pkl', 'wb') as f:
       pickle.dump((data_graphs,data_graph_informations,big_index), f)
    with open(f'columncsv/{columns_data[column]}_graph_data.pkl', 'rb') as f:
       data_graphs,data_graph_informations,big_index = pickle.load(f) 
    print(len(data_graphs),data_graphs[0],len(data_graph_informations),big_index)             