import argparse
import collections
import os.path
import random
import time
from pathlib import Path
from scipy.sparse import csr_matrix, vstack, save_npz
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader as pygDataloader
from torch.utils.data import random_split, Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from Configures import data_args, train_args, model_args, group_args
from models import GnnNets
from models.MLP import MLPNet
from models.GRAPHTRANSFORMER import GraphTransformer
from sklearn.neighbors import kneighbors_graph
from models.Transformer import TransformerEncoder
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

random.seed(999)


def predict_GC(test_dataloader, gnnNets, labels):
    predictions_list = []
    labels_list = []

    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, _, _, a, _, _ = gnnNets(batch)

            # record
            _, preds = torch.max(logits, -1)
            predictions_list.append(preds)
            labels_list.append(batch.y)


    all_predictions = torch.cat(predictions_list).cpu().numpy()
    all_labels = torch.cat(labels_list).cpu().numpy()

    accuracy = accuracy_score(all_labels, all_predictions)

    # 计算精确率、召回率和 F1 分数
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return accuracy, precision, recall, f1,list(all_predictions),list(all_labels)


def get_id_2_gene(species_data_path, species, tissue, amount, filetype='csv'):
    data_path = species_data_path
    data_files = os.path.join(data_path, f'{species}_{tissue}{amount}_data.{filetype}')

    if filetype == 'csv':
        data = pd.read_csv(data_files, dtype='str', header=0).values[:, 0]
    else:
        data = pd.read_csv(data_files, compression='gzip', header=0).values[:, 0]  # data是['A1BG'  ...  'ZUP1']这样的基因名称

    gene = set(data)

    id2gene = sorted(list(gene))
    return id2gene


def get_id_2_label_and_label_statistics(species_data_path, species, tissue, amount):
    data_path = species_data_path
    cell_file = os.path.join(data_path, f'{species}_{tissue}{amount}_celltype.csv')
    print(os.path.join(data_path, f'{species}_{tissue}{amount}_celltype.csv'))
    cell_types = set()
    cell_type_list = list()

    df = pd.read_csv(cell_file, dtype=str, header=0)
    df['Cell_type'] = df['Cell_type'].map(str.strip)  # 删除df中Cell_type列中的每个字符串的前导和尾随空格
    cell_types = set(df.values[:, 2]) | cell_types
    cell_type_list.extend(df.values[:, 2].tolist())

    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def make_group_graph(data):
    adj = torch.corrcoef(data)
    # adj = compute_similarity(data)
    torch.diagonal(adj, 0).zero_()
    split = adj > group_args.pearson_threshold
    adj[~split] = 0
    edge_index = torch.nonzero(adj)

    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask].T
    return edge_index


def make_single_graph(data, pearson_threshold, label):  # pearson
    # data : (num_cells x genes)
    data = data.T
    adj = torch.corrcoef(data)
    torch.diagonal(adj, 0).zero_()
    split = adj > pearson_threshold
    adj[~split] = 0
    edge_index = torch.nonzero(adj)
    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask]

    return Data(x=data.to(torch.float), edge_index=edge_index.long().T, y=label)


def make_graph(data, label_classes, real_labels):
    graphs = []
    groups0 = []
    for label in range(len(label_classes)):
        label_this = label_classes[label]
        if len(label_this) == 0:
            continue
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

    n = data_args.num_cells
    cnt = 0
    for i in range(len(groups0)):
        L = len(groups0[i])
        for j in range(0, L, n):
            slices = groups0[i][j:j + n]
            if (len(slices) != n):
                continue

            real_y = [real_labels[i] for i in slices]
            real_y = collections.Counter(real_y)
            real_y = real_y.most_common(1)[0][0]
            if real_y == i:
                cnt += 1
            graphs.append(make_single_graph(data[slices], data_args.pearson_threshold, real_y))

    return graphs, cnt


def label_classification(all_labels, num_labels):
    label_classes = [[] for _ in range(num_labels)]
    for idx, label in enumerate(all_labels):
        label_classes[int(label)].append(idx)
    return label_classes


class GroupDataset(Dataset):
    def __init__(self, adata, label, training=True):
        super(GroupDataset, self).__init__()
        self.training = training
        self.node_features = torch.from_numpy(adata.values).to(torch.float32)
        if self.training:
            self.label = torch.tensor(label)

        self.input_dim = self.node_features.shape[1]

    def __len__(self):
        return self.node_features.shape[0]  # 返回节点数量

    def __getitem__(self, idx):
        if self.training:
            return self.node_features[idx], self.label[idx]
        return self.node_features[idx]


class PredictDataset(InMemoryDataset):
    def __init__(self, root,id1,id2, num_labels, transform=None, pre_transform=None):
        self.root = root
        self.amount = id2
        self.preamount = id1
        self.num_labels = num_labels
        super(PredictDataset, self).__init__(root, transform, pre_transform)
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
        return os.path.join(self.root, 'SingleCellDataset', 'test_pretrained', data_args.species, 'graphs')

    @property
    def processed_file_names(self):
        return [f'{data_args.species}_{data_args.tissue}{self.amount}_{data_args.num_cells}.pt']

    def process(self):
        species = data_args.species
        tissue = data_args.tissue

        species_data_path_train = Path(f'./datasets/SingleCellDataset/train/{species}')
        species_data_path_test = Path(f'./datasets/SingleCellDataset/test/{species}')
        graph_path = Path(f'./datasets/SingleCellDataset/test_pretrained/{species}/graphs')

        if not species_data_path_train.exists() or not species_data_path_test.exists():
            raise NotImplementedError
        if not graph_path.exists():
            graph_path.mkdir(parents=True)

        id2gene = get_id_2_gene(species_data_path_train, species, tissue, self.preamount, filetype=data_args.filetype)
        gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
        self.num_genes = len(id2gene)

        id2label, _ = get_id_2_label_and_label_statistics(species_data_path_train, species, tissue, self.preamount)
        id2label = sorted(id2label)
        label2id = {label: idx for idx, label in enumerate(id2label)}

        num_genes = len(id2gene)
        num_labels = len(id2label)

        print(f"The builing graph contains {num_genes} genes with {num_labels} labels supported.")

        data_path = species_data_path_test
        data_file = os.path.join(data_path, f'{species}_{tissue}{self.amount}_data.csv')

        adata = pd.read_csv(data_file, index_col=0)
        adata = adata.transpose()
        data = adata.values

        group_model_ori = torch.load(f'./checkpoint/{species}_{tissue}_{self.preamount}_{self.amount}/group_latest.pt')

        group_model = GraphTransformer(data.shape[1], num_labels, model_args).cuda()
        group_model.load_state_dict(group_model_ori['net'])

        group_model.eval()
        fake_labels = []

        with torch.no_grad():
            x = torch.from_numpy(data).to(torch.float32)
            edge_index = make_group_graph(x)
            x, edge_index = x.to(group_args.device), edge_index.to(group_args.device)
            logits, _ = group_model(x, edge_index)

            _, prediction = torch.max(logits, -1)
            fake_labels.extend(prediction.detach().cpu().numpy())

        # record correct rate
        real_labels = \
            pd.read_csv(os.path.join(species_data_path_test, f'{species}_{tissue}{self.amount}_celltype.csv'))[
                'Cell_type'].tolist()
        real_labels = [label2id[i] for i in real_labels]

        assert len(fake_labels) == len(real_labels)

        cnt = 0
        for i, j in zip(real_labels, fake_labels):
            if i == j:
                cnt += 1
        correct_rate1 = cnt / len(real_labels)
        print(f'correct_rate1: {correct_rate1}')

        print(
            f'{data_args.species}_{tissue}{data_args.amount}_data.{data_args.filetype} -> Nonzero Ratio: {data.astype(bool).sum().sum() / data.size * 100:.2f}%')

        cell_feat = torch.from_numpy(data)  # cells x genes
        label_classes = label_classification(fake_labels, num_labels)
        graphs, graph_cnt = make_graph(cell_feat, label_classes, real_labels)

        correct_rate2 = graph_cnt / len(graphs)
        print(f'correct_rate2: {correct_rate2}')
        random.shuffle(graphs)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='straightly given the file name')
    parser.add_argument('--id1', type=str, default=None)
    parser.add_argument('--id2', type=str, default=None)
    args = parser.parse_args()
    if args.file is not None:
        file = args.file.split('_')
        data_args.species = file[0]
        data_args.amount = ''.join(list(filter(str.isdigit, args.file)))
        data_args.tissue = '_'.join(file[1:-1]).strip(data_args.amount)

    # attention the multi-task here
    print('start loading data====================')
    celltype_data = pd.read_csv(
        f'./datasets/SingleCellDataset/test/{data_args.species}/{data_args.species}_{data_args.tissue}{data_args.amount}_celltype.csv')
    labels = celltype_data['Cell_type'].tolist()
    id2celltypes = sorted(list(set(labels)))
    celltypes2id = {type: i for i, type in enumerate(id2celltypes)}
    labels = list(map(lambda x: celltypes2id[x], labels))

    dataset = PredictDataset('./datasets', args.id1, args.id2, len(id2celltypes))
    input_dim = dataset.num_node_features
    output_dim = len(id2celltypes)
    num_gene = dataset.num_genes

    dataloader = pygDataloader(dataset, batch_size=train_args.batch_size, shuffle=True)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index / 2 :.4f}")

    gnnNets = GnnNets(input_dim, output_dim, num_gene, model_args).cuda()
    checkpoint = torch.load(f'./checkpoint/human_{data_args.tissue}_{args.id1}_{args.id2}/gcn_latest.pt')
    gnnNets.update_state_dict(checkpoint['net'])

    acc, pre, rec, f1,prediction_list,all_label = predict_GC(dataloader, gnnNets, labels)

    with open(f'./datasets/figs/test.csv', 'a') as f:
        f.write(str(acc) + ',' + str(pre) + ',' + str(rec) + ',' + str(f1) + '\n')


