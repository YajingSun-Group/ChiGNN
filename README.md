# ChiGNN: Interpretable Algorithm Framework of Molecular Chiral Knowledge-Embedding and Stereo-Sensitive Properties Prediction  
This document describes the codes used in the experiments and how to replicate them.

<img src="\20241111-Figure.png" alt="20241111-Figure" style="zoom: 25%;" />

## Environment
The following packages are required.
- python=3.10
- numpy=1.25.0
- pandas=1.5.3
- tqdm=4.65.0
- rdkit=2023.3.3
- PyTorch=2.0.1  
- faerun=0.4.7
- matplotlib=3.9.0
- scikit-learn =1.2.2
- scipy=1.10.1
- seaborn =0.12.2
- torch_geometric=2.5.3


You can use the following command to create a Conda code environment. For the PyTorch package, install the appropriate version according to your GPU environment.
```sh
conda create -n ChiGNN python==3.10
conda activate ChiGNN
conda install faerun matplotlib numpy pandas rdkit scikit-learn scipy seaborn torch torch_geometric tqdm 
```





## Production and preservation of graph datasets
First of all, according to the definition of Trinity Graph, smiles is transformed intoTrinity Graph and saved in pkl format to provide dataset for subsequent training.
```sh
python 1make_dataset.py
```

## Training
In the training step, modify the MODEL of 2ml.py to Train, and modify the column parameter to the data set to be trained ( such as ' IC ' ). At the same time, other training conditions can also be modified. If not modified, the training model will be configured by default, as shown in the following commands.
```sh
python 2ml.py
```

## Test
In the test step, modify the MODEL of 2ml.py to Test, and modify the column parameter to the data set to be tested ( such as ' IC ' ), as shown in the following command.
```sh
python 2ml.py
```

## Enantiomer visualization
After the model is trained, a pair of enantiomers can be visualized and their retention time under different chromatographic column conditions can be predicted according to the Trinity Masking and Contribution Decomposition ( TMCD ) technique, as shown in the following commands.
```sh
python 3.1enantiomer_visualization.py
```


## Column picture
Similarly, the visual images, adjacency matrices and other data of the chiral enantiomer molecules of the same column can be saved, as shown in the following commands.
```sh
python 3.3column_picture.py
```

## Column CSPs
Similarly, when we trained the model of multiple CSPs, the retention time of the same pair of enantiomers under different chromatographic column conditions can be predicted, as shown in the following commands.
```sh
python 3.4column_csp.py
```

## Functional group contributions
Using TMCD technology, the contributions of all functional groups under different data sets can be obtained, and their counts and average contributions can be counted.
```sh
python 3.5functional_group_contributions.py
python a_fg_analyze.py
```

## Functional group contributions difference
Similarly, the difference of contribution of the same functional group under different chiral conditions can be obtained, and the count and average contribution can be obtained by statistics.
```sh
python 3.8functional_group_contributions_diff.py
python a_fg_analyze.py
```


## Dimensionality reduction and clustering
Using the trained model, the hidden vector of the chiral molecule and the hidden space of the data set can be obtained. The picture and html files can be obtained by using dimensionality reduction and clustering techniques respectively, as shown in the following commands.
```sh
python 4.1reduction.py
python 4.2reduction_html.py
python 5.1cluster.py
python 5.2cluster_html.py
```


## Drawing
The drawing program involved in this work is written in drawing.ipynb. It can be drawn and modified as needed.

