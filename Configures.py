import os
from typing import List

os.environ['OMP_NUM_THREADS'] = '1'


class DataParser():
    def __init__(self):
        super().__init__()
        self.species = 'human'
        self.tissue = 'GSM'
        self.amount = 1
        self.dataset_dir = './datasets'

        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1,
                                       0.1]  # the ratio of training, validation and testing set for random split
        self.seed = 42
        self.filetype = 'csv'
        self.exclude_rate = 0.1
        self.threshold = 0
        self.pearson_threshold = 0.9
        self.num_cells = 5


class GATParser():  # hyper-parameter for gat model
    def __init__(self):
        super().__init__()
        self.gat_dropout = 0.5  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3


class ModelParser():
    def __init__(self):
        super().__init__()
        self.device: int = 0
        self.model_name: str = 'gcn'
        self.checkpoint: str = './checkpoint'
        self.concate: bool = False  # whether to concate the gnn features before mlp
        self.trans_dim: List[int] = [128]  # the hidden units for each gnn layer
        self.latent_dim: List[int] = [256,64]  # the hidden units for each gnn layer
        self.mlp_latent_dim: List[int] = [128]  # the hidden units for MLP
        self.readout: 'str' = 'sum'  # the graphs pooling method
        self.mlp_hidden: List[int] = []  # the hidden units for mlp classifier
        self.gnn_dropout: float = 0.0  # the dropout after gnn layers
        self.dropout: float = 0.5  # the dropout after mlp layers
        self.adj_normlize: bool = True  # the edge_weight normalization for gcn conv
        self.emb_normlize: bool = False  # the l2 normalization after gnn layer
        self.enable_prot = True  # whether to enable prototype training
        self.num_prototypes_per_class = 3  # the num_prototypes_per_class
        self.nheads = 4  # transformer
        self.num_layers = 1  # transformer
        self.graph_transformer_emb = [64]  # transformer
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 3  # multi-head
        self.gat_hidden = 5  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 2

    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass


class MCTSParser(DataParser, ModelParser):
    rollout: int = 1  # the rollout number
    high2low: bool = False  # expand children with different node degree ranking method
    c_puct: float = 5  # the exploration hyper-parameter
    min_atoms: int = 20
    max_atoms: int = 150
    expand_atoms: int = 10  # # of atoms to expand children

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_best.pth")


class RewardParser():
    def __init__(self):
        super().__init__()
        self.reward_method: str = 'mc_l_shapley'  # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
        self.local_raduis: int = 4  # (n-1) hops neighbors for l_shapley
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100  # sample time for monte carlo approximation


class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.batch_size = 64
        self.weight_decay = 0
        self.max_epochs = 200
        self.save_epoch = 10
        self.early_stopping = 100
        self.last_layer_optimizer_lr = 1e-4  # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                                    'add_on_layers': 3e-3,
                                    'prototype_vectors': 3e-3}  # the learning rates of the joint training optimizer
        self.warm_epochs = 10  # the number of warm epochs
        self.proj_epochs = 10  # the epoch to start mcts


class GroupParser():
    def __init__(self):
        super().__init__()
        self.device = 0
        self.model_name = 'mlp'
        self.save_epoch = 10
        self.batch_size = 128
        self.max_epochs = 100
        self.group_learning_rate = 0.001
        self.mlp_latent_dim = [4]
        self.graph_transformer_emb = [64]
        self.pearson_threshold = 0.8
        self.nheads = 4
        self.num_layers = 1
        self.dropout = 0.6
        self.data_split_ratio: List = [0.9, 0.1, 0]


data_args = DataParser()
model_args = ModelParser()
mcts_args = MCTSParser()
reward_args = RewardParser()
train_args = TrainParser()
group_args = GroupParser()

import random
import numpy as np

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
