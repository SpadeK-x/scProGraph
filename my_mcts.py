import os
import math

import networkx
import torch
import torch.nn as nn
import networkx as nx
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mul
import tqdm
from torch.utils.data import Dataset, DataLoader
from Configures import mcts_args
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from functools import partial
from collections import Counter
from multiprocessing import Pool
from torch_geometric.utils import to_networkx, to_undirected


def get_edges(points, edges, num_nodes):
    valid_edges = edges[(torch.tensor([u.item() in points for u in edges[:, 0]]) &
                         torch.tensor([v.item() in points for v in edges[:, 1]]))]

    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)
    adjacency_matrix[valid_edges[0], valid_edges[1]] = 1
    adjacency_matrix[valid_edges[1], valid_edges[0]] = 1

    return adjacency_matrix,None
    return adjacency_matrix, edge_index


class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def mcts_rollout(tree_node, state_map, data, graph, score_func):
    cur_graph_coalition = tree_node.coalition
    # 递归结束条件，满足最小值就退出递归
    if len(cur_graph_coalition) <= mcts_args.min_atoms:
        return tree_node.P
    # 如果这个节点没有子节点，那么扩展此节点
    if len(tree_node.children) == 0:
        # 获取当前节点图的子图并按度排序
        node_degree_list = list(graph.subgraph(cur_graph_coalition).degree)
        node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=mcts_args.high2low)
        all_nodes = [x[0] for x in node_degree_list]
        # 取出要扩展的节点
        if len(all_nodes) < mcts_args.expand_atoms:
            expand_nodes = all_nodes
        else:
            expand_nodes = all_nodes[:mcts_args.expand_atoms]

        for each_node in expand_nodes:
            subgraph_coalition = [node for node in all_nodes if node != each_node]

            # 将所有的节点构成的图分成几个连接区域，再取连接区域中最大的
            subgraphs = [graph.subgraph(c)
                         for c in nx.connected_components(graph.subgraph(subgraph_coalition))]
            main_sub = subgraphs[0]
            for sub in subgraphs:
                if sub.number_of_nodes() > main_sub.number_of_nodes():
                    main_sub = sub

            new_graph_coalition = sorted(list(main_sub.nodes()))

            # 检查state map然后合并相同的子图
            Find_same = False
            for old_graph_node in state_map.values():
                if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                    new_node = old_graph_node
                    Find_same = True

            # 若无相同子图，则创建一个新节点
            if Find_same == False:
                new_node = MCTSNode(new_graph_coalition, data=data, ori_graph=graph)
                state_map[str(new_graph_coalition)] = new_node

            # 寻找是否有相同的子节点
            Find_same_child = False
            for cur_child in tree_node.children:
                if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                    Find_same_child = True

            # 如果没有相同子节点，将新节点变为子节点
            if Find_same_child == False:
                tree_node.children.append(new_node)
        # 运算子节点分数
        scores = compute_scores(score_func, tree_node.children)
        for child, score in zip(tree_node.children, scores):
            child.P = score
    # 选择下一个节点
    sum_count = sum([c.N for c in tree_node.children])
    selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))

    # 递归运行下一轮
    v = mcts_rollout(selected_node, state_map, data, graph, score_func)
    selected_node.W += v
    selected_node.N += 1
    return v


def mcts(data, gnnNet, prototype,num_genes=None):
    if num_genes is not None:
        mcts_args.min_atoms=num_genes-150
        mcts_args.max_atoms=num_genes
    data = Data(x=data.x, edge_index=data.edge_index)
    graph = to_networkx(data, to_undirected=True)
    data = Batch.from_data_list([data])
    num_nodes = graph.number_of_nodes()
    root_coalition = sorted([i for i in range(num_nodes)])
    root = MCTSNode(root_coalition, data=data, ori_graph=graph)
    state_map = {str(root.coalition): root}
    score_func = partial(gnn_prot_score, data=data, gnnNet=gnnNet, prototype=prototype)

    for rollout_id in range(mcts_args.rollout):
        mcts_rollout(root, state_map, data, graph, score_func)

    # 将所有节点按分数排序
    explanations = [node for _, node in state_map.items()]
    explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
    explanations = sorted(explanations, key=lambda x: len(x.coalition))

    # 寻找分数最高的符合条件的节点
    result_node = explanations[0]
    for result_idx in range(len(explanations)):
        x = explanations[result_idx]
        if len(x.coalition) <= mcts_args.max_atoms and x.P > result_node.P:
            result_node = x

    # adj, ret_edge_index = get_edges(result_node.coalition, data.edge_index, data.num_nodes)

    # 计算投影的原型向量，返回数据
    mask = torch.zeros(data.num_nodes).type(torch.float32)
    mask[result_node.coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)

    row, col = data.edge_index
    edge_mask = (mask[row] == 1) & (mask[col] == 1)
    ret_edge_index = data.edge_index[:, edge_mask]

    adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.int32)
    adj[ret_edge_index[0], ret_edge_index[1]] = 1
    adj[ret_edge_index[1], ret_edge_index[0]] = 1

    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data_ = Batch.from_data_list([mask_data])
    _, _, _, emb, _, _ = gnnNet(mask_data_, protgnn_plus=False)

    return result_node.coalition, result_node.P, emb, adj


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition)
        else:
            score = child.P
        results.append(score)
    return results


def gnn_prot_score(coalition, data, gnnNet, prototype):
    """ the similarity value of subgraph with selected nodes """
    epsilon = 1e-4
    mask = torch.zeros(data.num_nodes).type(torch.float32)
    mask[coalition] = 1.0
    ret_x = data.x * mask.unsqueeze(1)
    ret_edge_index = data.edge_index
    # row, col = data.edge_index
    # edge_mask = (mask[row] == 1) & (mask[col] == 1)
    # ret_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    _, _, _, emb, _, _ = gnnNet(mask_data, protgnn_plus=False)
    distance = torch.norm(emb - prototype) ** 2
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity.item()


if __name__ == '__main__':
    from models import GnnNets, GnnNets_NC
    from load_dataset import get_dataset, get_dataloader
    from Configures import data_args, train_args, model_args
    import numpy as np

    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, 'train', task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)

    data_indices = dataloader['train'].dataset.indices
    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    prototype_shape = (output_dim * model_args.num_prototypes_per_class, 128)
    prototype_vectors = nn.Parameter(torch.rand(prototype_shape),
                                     requires_grad=False).to(model_args.device)
    checkpoint = torch.load('./checkpoint/human_Adipose/gcn_best.pth')
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    batch_indices = np.random.choice(data_indices, 64, replace=False)

    func = partial(mcts, gnnNet=gnnNets, prototype=prototype_vectors[0])
    pool = Pool(4)
    results = pool.map(mcts, (dataset, gnnNets, prototype_vectors[0]))
