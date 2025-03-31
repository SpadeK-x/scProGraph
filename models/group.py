import os
import argparse
import random
import time
import pandas as pd
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import scanpy as sc
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset
from torch_geometric.nn import MessagePassing
from Configures import data_args, group_args
from sklearn.neighbors import kneighbors_graph
from my_mcts import mcts
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from models.MLP import MLPNet
from models.GRAPHTRANSFORMER import GraphTransformer
from models.Transformer import TransformerEncoder


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


def make_graph(data):
    adj = torch.corrcoef(data)
    torch.diagonal(adj, 0).zero_()
    split = adj > group_args.pearson_threshold
    adj[~split] = 0
    edge_index = torch.nonzero(adj)

    mask = edge_index[:, 0] < edge_index[:, 1]
    edge_index = edge_index[mask].T
    return edge_index


def save_best(ckpt_dir, gnnNets, is_best=None):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict()
    }
    pt_name = f"group_latest.pt"
    best_pt_name = f'group_best.pt'

    ckpt_path = os.path.join(ckpt_dir, pt_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pt_name))
    gnnNets.to(group_args.device)


def train_GC(id1,id2):
    print('start loading data====================')
    data_path = f'./datasets/SingleCellDataset/train/{data_args.species}/{data_args.species}_{data_args.tissue}{data_args.amount}_data.csv'
    label_path = f'./datasets/SingleCellDataset/train/{data_args.species}/{data_args.species}_{data_args.tissue}{data_args.amount}_celltype.csv'

    adata = pd.read_csv(data_path, index_col=0)
    adata = adata.transpose()
    label = pd.read_csv(label_path, index_col=0)['Cell_type'].tolist()
    label2id = {l: id for id, l in enumerate(sorted(list(set(label))))}

    label = [label2id[i] for i in label]

    data = adata.values
    input_dim = data.shape[1]
    output_dim = len(label2id)
    model = GraphTransformer(input_dim, output_dim, group_args)
    model.cuda()

    ckpt_dir = f"./checkpoint/{data_args.species}_{data_args.tissue}_{id1}_{id2}/"
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=group_args.group_learning_rate)

    best_acc = 0.0

    data_size = data.shape[0]
    print(f'The total num of dataset is {data_size}')

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    train_loss = []
    for epoch in range(1, group_args.max_epochs + 1):
        loss_list, acc = [], []
        model.train()

        x = torch.from_numpy(data).to(torch.float32)
        Y = torch.tensor(label)
        edge_index = make_graph(x)
        x,edge_index,Y = x.to(group_args.device),edge_index.to(group_args.device), Y.to(group_args.device)
        # x, Y = x.to(group_args.device), Y.to(group_args.device)
        # edge_index = torch.from_numpy(np.vstack((edge_index.row, edge_index.col))).to(group_args.device)
        logits, _ = model(x,edge_index)

        loss = criterion(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## record
        _, prediction = torch.max(logits, -1)
        loss_list.append(loss.item())

        acc.extend(prediction.eq(Y).cpu().numpy())

        acc = np.array(acc).mean()
        best_acc = max(best_acc, acc)
        print(
            f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Acc: {acc:.3f}")
        train_loss.append(np.average(loss_list))

        if epoch % group_args.save_epoch == 0:
            save_best(ckpt_dir, model)
    print(f"The best validation accuracy is {best_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model for grouping')
    parser.add_argument('--file', type=str, default=None, help='straightly given the file name')
    parser.add_argument('--id1', type=str, default=None)
    parser.add_argument('--id2', type=str, default=None)
    args = parser.parse_args()

    if args.file is not None:
        file = args.file.split('_')
        data_args.species = file[0]
        data_args.amount = ''.join(list(filter(str.isdigit, args.file)))
        data_args.tissue = '_'.join(file[1:-1]).strip(data_args.amount)

    train_GC(args.id1,args.id2)
