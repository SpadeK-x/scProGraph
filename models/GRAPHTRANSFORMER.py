import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add sequence dimension
        attn_output, _ = self.attention(x, x, x)
        x = self.linear(x.squeeze(0))  # Remove sequence dimension
        return x


class GraphTransformer(nn.Module):
    def __init__(self, node_features, output_dim,model_args):
        super(GraphTransformer, self).__init__()
        embed_dim=model_args.graph_transformer_emb[0]
        self.gnn = GCNConv(node_features, embed_dim)
        self.transformer = TransformerLayer(embed_dim, model_args.nheads)
        self.dropout = nn.Dropout(p=model_args.dropout)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gnn(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.transformer(x))
        x = self.fc(x)

        logits = x
        # probs = F.softmax(logits)
        return logits, None

class GraphTransformer2(nn.Module):
    def __init__(self, input_dim, output_dim, num_gene, model_args):
        super(GraphTransformer2, self).__init__()
        self.output_dim = output_dim

        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden

        self.emb_normlize = model_args.emb_normlize
        self.num_prototypes_per_class = model_args.num_prototypes_per_class
        self.device = torch.device(model_args.device)

        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1

        proto_length=64
        self.dense_dim = self.latent_dim[-1]
        self.pooling_mlp = nn.Linear(proto_length * num_gene, proto_length)

        self.gnn_transformer_layers = nn.ModuleList()
        self.gnn_transformer_layers.append(GCNConv(input_dim, self.latent_dim[0], normalize=model_args.adj_normlize))
        self.gnn_transformer_layers.append(TransformerLayer(self.latent_dim[0],num_heads=model_args.nheads))
        for i in range(1, self.num_gnn_layers):
            self.gnn_transformer_layers.append(GCNConv(self.latent_dim[i - 1], self.latent_dim[i], normalize=model_args.adj_normlize))
            self.gnn_transformer_layers.append(TransformerLayer(self.latent_dim[i],num_heads=model_args.nheads))
        self.gnn_non_linear = nn.ReLU()

        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.dense_dim * len(self.readout_layers),
                                       model_args.mlp_hidden[0]))
            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i - 1], self.mlp_hidden[i]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(self.dense_dim * len(self.readout_layers),
                                       output_dim))
        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

        # prototype layers
        self.enable_prot = model_args.enable_prot
        self.epsilon = 1e-4
        self.prototype_shape = (output_dim * model_args.num_prototypes_per_class, proto_length)

        means = torch.rand(output_dim)
        self.prototype_vectors = torch.empty(output_dim, model_args.num_prototypes_per_class, proto_length)
        for i in range(output_dim):
            self.prototype_vectors[i] = torch.normal(means[i], 1.0, size=(model_args.num_prototypes_per_class, proto_length))
        self.prototype_vectors = nn.Parameter(self.prototype_vectors.reshape(-1, proto_length), requires_grad=True)

        self.num_prototypes = self.prototype_shape[0]
        self.last_layer = nn.Linear(self.num_prototypes, output_dim,
                                    bias=False)  # do not use bias
        assert (self.num_prototypes % output_dim == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def forward(self, data, protgnn_plus=False, similarity=None):
        if protgnn_plus:
            logits = self.last_layer(similarity)
            probs = self.Softmax(logits)
            return logits, probs, None, None, None

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = torch.ones(edge_index.shape[1]).to(self.device)
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        for i in range(self.num_gnn_layers):
            if not self.gnn_transformer_layers[i].normalize:
                x = self.gnn_transformer_layers[i](x, edge_index, edge_attr)
            else:
                x = self.gnn_transformer_layers[i](x, edge_index, edge_attr)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        batch_size = len(set(batch.cpu().numpy()))
        x = x.reshape(batch_size, -1)
        x = self.pooling_mlp(x)
        graph_emb = x

        if self.enable_prot:
            prototype_activations, min_distances = self.prototype_distances(x)
            max_indices = torch.argmax(prototype_activations, dim=1)
            one_hot_result = torch.zeros(batch_size, self.output_dim, device='cuda:0')
            type_index = torch.div(max_indices, self.num_prototypes_per_class, rounding_mode='floor')
            one_hot_result[torch.arange(len(one_hot_result)), type_index] = 1
            logits = self.last_layer(prototype_activations)
            probs = self.Softmax(logits)
            return logits, probs, node_emb, graph_emb, min_distances, None
        else:
            for i in range(self.num_mlp_layers - 1):
                x = self.mlps[i](x)
                x = self.mlp_non_linear(x)
                x = self.dropout(x)

            logits = self.mlps[-1](x)
            probs = self.Softmax(logits)
            return logits, probs, node_emb, graph_emb, [], None