import collections
import os
import json
import time
import pandas as pd
import torch
import pickle
import random
import numpy as np
import os.path as osp
from pathlib import Path
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, vstack, save_npz
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Dataset
from Configures import data_args
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
random.seed(time.time())


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graphs.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def get_id_2_gene(species_data_path, species, tissue, filetype):
    data_path = species_data_path
    data_files = os.path.join(data_path, f'{species}_{tissue}{data_args.amount}_data.{filetype}')

    if filetype == 'csv':
        data = pd.read_csv(data_files, dtype='str', header=0).values[:, 0]
    else:
        data = pd.read_csv(data_files, compression='gzip', header=0).values[:, 0]  # data是['A1BG'  ...  'ZUP1']这样的基因名称

    gene = set(data)
    id2gene = list(gene)
    id2gene.sort()

    return id2gene


def get_id_2_label_and_label_statistics(species_data_path, species, tissue, id=0):
    data_path = species_data_path
    cell_file = os.path.join(data_path, f'{species}_{tissue}{data_args.amount}_celltype.csv')
    cell_types = set()
    cell_type_list = list()

    df = pd.read_csv(cell_file, dtype=str, header=0)
    df['Cell_type'] = df['Cell_type'].map(str.strip)  # 删除df中Cell_type列中的每个字符串的前导和尾随空格
    cell_types = set(df.values[:, 2]) | cell_types
    cell_type_list.extend(df.values[:, 2].tolist())

    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_label.txt'
    with open(gene_path, 'w', encoding='utf-8') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in id2label:
            f.write(label + '\r\n')


def label_classification(all_labels, num_labels):
    label_classes = [[] for _ in range(num_labels)]
    for idx, label in enumerate(all_labels):
        label_classes[int(label)].append(idx)
    return label_classes


def compute_similarity(vectors):
    batch_size = len(vectors)
    vector_size = vectors[0].size(0)
    # 归一化向量
    batch_vectors = F.normalize(vectors)
    # 计算余弦相似度矩阵
    similarities = torch.mm(batch_vectors, batch_vectors.t())

    return similarities


def make_single_graph(data, pearson_threshold, label):  # pearson
    # data : (num_cells x genes)
    data = data.T
    adj = torch.corrcoef(data)
    # adj = compute_similarity(data)
    torch.diagonal(adj, 0).zero_()
    split = adj > pearson_threshold
    adj[~split] = 0
    edges = torch.nonzero(adj)
    mask = edges[:, 0] < edges[:, 1]
    edges = edges[mask]
    # label_dict = dict(collections.Counter([labels[i] for i in idxes]))
    # y = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)[0][0]

    return Data(x=data.to(torch.float), edge_index=edges.long().T, y=label )



def make_graph(data, label_classes, n, pearson_threshold):
    graphs = []
    groups0 = []
    for label in range(len(label_classes)):
        label_this=label_classes[label]
        data1 = data[label_this]
        num_cells = len(data1)

        cells = [False for _ in range(num_cells)]
        groups = []

        sim_matrix = cosine_similarity(data1)
        for i in range(num_cells):
            if cells[i]:
                continue
            sim = [(idx, j) for idx, j in enumerate(sim_matrix[i])]
            sim = sorted(sim, key=lambda x: x[1], reverse=True)
            cnt = 0
            group = []
            for j, _ in sim:
                if not cells[j]:
                    group.append(label_this[j])
                    cells[j] = True
                    cnt += 1
                    if cnt == data_args.num_cells:
                        break
            if cnt < data_args.num_cells:
                break
            else:
                groups.extend(group)
        groups0.append(groups)

    for i in range(len(groups0)):
        L = len(groups0[i])
        for j in range(0, L, n):
            slices = groups0[i][j:j + n]
            if (len(slices) != n):
                continue
            graphs.append(make_single_graph(data[slices], pearson_threshold, i))

    return graphs


def get_dataset(dataset_dir, mode, task=None):
    return SingleCellDataset(dataset_dir, mode)


class SingleCellDataset(InMemoryDataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None):
        self.root = root
        self.mode = mode  # train / test
        super(SingleCellDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices['x']) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'SingleCellDataset', 'raw')

    @property
    def raw_file_names(self):
        return ['']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'SingleCellDataset', f'{self.mode}_pretrained', data_args.species, 'graphs')

    @property
    def processed_file_names(self):
        return [f'{data_args.species}_{data_args.tissue}{data_args.amount}_{data_args.num_cells}.pt']

    def process(self):
        species = data_args.species
        tissue = data_args.tissue

        species_data_path = Path(f'./datasets/SingleCellDataset/{self.mode}/{species}')
        graph_path = Path(f'./datasets/SingleCellDataset/{self.mode}_pretrained/{species}/graphs')

        if not species_data_path.exists():
            raise NotImplementedError
        if not graph_path.exists():
            graph_path.mkdir(parents=True)

        id2gene = get_id_2_gene(species_data_path, species, tissue, filetype=data_args.filetype)

        id2label, _ = get_id_2_label_and_label_statistics(species_data_path, species, tissue)
        id2label = sorted(id2label)

        gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
        num_genes = len(id2gene)
        self.num_genes = num_genes

        num_labels = len(id2label)
        label2id = {label: idx for idx, label in enumerate(id2label)}

        print(f"The builing graph contains {self.num_genes} genes with {num_labels} labels supported.")

        all_labels = []
        matrices = []
        num_cells = 0

        data_path = species_data_path
        data_file = os.path.join(data_path, f'{data_args.species}_{tissue}{data_args.amount}_data.csv')
        type_file = os.path.join(data_path, f'{data_args.species}_{tissue}{data_args.amount}_celltype.csv')

        cell2type = pd.read_csv(type_file, index_col=0)
        cell2type.columns = ['cell', 'type']
        cell2type['type'] = cell2type['type'].map(str.strip)
        cell2type['id'] = cell2type['type'].map(label2id)

        filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
        cell2type = cell2type.iloc[filter_cell]

        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        all_labels += cell2type['id'].tolist()

        df = pd.read_csv(data_file, index_col=0)  # (gene, cell)

        df = df.transpose()  # (cell, gene)
        df = df.iloc[filter_cell]

        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(
            f'{data_args.species}_{tissue}{data_args.amount}_data.{data_args.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        arr = df.to_numpy()
        # 对细胞归一化
        cell_feat = torch.from_numpy(arr)  # cells x genes
        # cell_feat = cell_feat / (torch.sum(cell_feat, dim=0, keepdims=True) + 1e-6)
        label_classes = label_classification(all_labels, num_labels)

        graphs = make_graph(cell_feat, label_classes, data_args.num_cells, data_args.pearson_threshold)
        # graphs = make_graph(cell_feat, label_classes, data_args.pearson_threshold)
        random.shuffle(graphs)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=42):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """
    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))
        # idx_list = [i for i in range(len(dataset))]
        # train = Subset(dataset, idx_list)
        # eval = Subset(dataset, idx_list)
        # test = Subset(dataset, idx_list)

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader
