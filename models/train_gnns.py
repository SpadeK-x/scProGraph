import os
import argparse
import random
import time
import pandas as pd
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import threading
import torch.nn as nn
import umap
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from Configures import data_args, train_args, model_args
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from my_mcts import mcts
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def warm_only(model):
    if hasattr(model.model, 'gnn_layers'):
        for p in model.model.gnn_layers.parameters():
            p.requires_grad = True
        model.model.prototype_vectors.requires_grad = True
        for p in model.model.last_layer.parameters():
            p.requires_grad = False


def joint(model):
    if hasattr(model.model, 'gnn_layers'):
        for p in model.model.gnn_layers.parameters():
            p.requires_grad = True
        model.model.prototype_vectors.requires_grad = True
        for p in model.model.last_layer.parameters():
            p.requires_grad = True


def append_record(info):
    f = open('./log/hyper_search', 'a')
    f.write(info)
    f.write('\n')
    f.close()


def concrete_sample(log_alpha, beta=1.0, training=True):
    """ Sample from the instantiation of concrete distribution when training
    \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
    """
    if training:
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()
    else:
        gate_inputs = log_alpha.sigmoid()

    return gate_inputs


def edge_mask(inputs, training=None):
    x, embed, edge_index, prot, tmp = inputs
    nodesize = embed.shape[0]
    feature_dim = embed.shape[1]
    f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
    f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)
    f3 = prot.unsqueeze(0).repeat(nodesize * nodesize, 1)
    # using the node embedding to calculate the edge weight
    f12self = torch.cat([f1, f2, f3], dim=-1)
    h = f12self.to(model_args.device)
    for elayer in elayers:
        h = elayer(h)
    values = h.reshape(-1)
    values = concrete_sample(values, beta=tmp, training=training)
    mask_sigmoid = values.reshape(nodesize, nodesize)

    sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
    edge_mask = sym_mask[edge_index[0], edge_index[1]]

    return edge_mask


def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask


def prototype_subgraph_similarity(x, prototype):
    distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
    similarity = torch.log((distance + 1) / (distance + 1e-4))
    return distance, similarity


elayers = nn.ModuleList()
elayers.append(nn.Sequential(nn.Linear(128 * 3, 64), nn.ReLU()))
elayers.append(nn.Sequential(nn.Linear(64, 8), nn.ReLU()))
elayers.append(nn.Linear(8, 1))
elayers.to(model_args.device)


# train for graphs classification
def train_GC(clst, sep,id1,id2):
    # attention the multi-task here
    print('start loading data====================')
    st_time = time.time()
    dataset = get_dataset(data_args.dataset_dir, 'train', task=data_args.task)

    ed_time = time.time()
    print(f'****graph built accomplished, using {ed_time - st_time}s')
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    num_gene = 200
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)

    y_list = list(torch.cat([batch.y for batch in dataloader['eval']]).numpy())
    class_num = Counter(y_list)

    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, num_gene, model_args)
    ckpt_dir = f"./checkpoint/{data_args.species}_{data_args.tissue}_{id1}_{id2}/"

    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index / 2 :.4f}")
    best_acc = 0.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices
    data_indices = [(i,j.item()) for i,j in zip(data_indices, dataset[data_indices].y)]

    train_loss, eval_loss, eval_acc = [], [], []
    for epoch in range(1,train_args.max_epochs+1):
        loss_list = []
        correct, tot = 0, 0
        # Prototype projection
        if model_args.enable_prot and epoch >= train_args.proj_epochs and epoch % 50 == 0:
            prototype_graph = [None for _ in range(output_dim * model_args.num_prototypes_per_class)]
            gnnNets.eval()
            print('start time: ', time.time())
            for i in range(output_dim * model_args.num_prototypes_per_class):
                best_similarity = 0
                label = i // model_args.num_prototypes_per_class
                proj_prot = gnnNets.model.prototype_vectors.data[i]
                data_indice=[d[0] for d in data_indices if d[1]==label]
                sampled_values = random.sample(data_indice, 10)

                for j in sampled_values:
                    data = dataset[j]
                    if data.y == label:
                        coalition, similarity, prot, adj = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i],
                                                                num_gene)
                        if similarity > best_similarity or best_similarity==0:
                            best_similarity = similarity
                            proj_prot = prot
                            prototype_graph[i] = adj

                gnnNets.model.prototype_vectors.data[i] = proj_prot
                print('Projection of prototype completed')
            print('end time: ', time.time())

            for i in prototype_graph:
                assert i is not None
            prototype_graph = torch.stack(prototype_graph, dim=0)
            print(prototype_graph.shape)
            prototype_graph = prototype_graph.reshape(output_dim, model_args.num_prototypes_per_class,
                                                      prototype_graph.shape[-2], prototype_graph.shape[-1])

            torch.save(prototype_graph,
                       f'./datasets/figs/{data_args.tissue}/{data_args.species}_{data_args.tissue}{data_args.amount}_{epoch}.pt')

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)


        for batch in dataloader['train']:
            batch = batch.to(model_args.device)
            logits, probs, _, a, min_distances, _ = gnnNets(batch)
            X.append(a.detach().cpu().numpy())
            Y.extend(list(batch.y.detach().cpu().numpy()))
            y_onehot = F.one_hot(batch.y, output_dim).float()

            loss = criterion(logits, y_onehot)

            if model_args.enable_prot:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y].bool()).to(
                    model_args.device)

                # cluster loss
                cluster_cost = torch.mean(
                    torch.min(
                        min_distances[prototypes_of_correct_class].reshape(-1, model_args.num_prototypes_per_class),
                        dim=1)[0])

                # seperation loss
                separation_cost = -torch.mean(torch.min(min_distances[~prototypes_of_correct_class].reshape(-1, (
                        output_dim - 1) * model_args.num_prototypes_per_class), dim=1)[0])

                # sparsity loss
                l1_mask = 1 - torch.t(gnnNets.model.prototype_class_identity).to(model_args.device)
                l1 = (gnnNets.model.last_layer.weight * l1_mask).norm(p=1)

                # diversity loss
                ld = 0
                for k in range(output_dim):
                    p = gnnNets.model.prototype_vectors[
                        k * model_args.num_prototypes_per_class: (k + 1) * model_args.num_prototypes_per_class]
                    p = F.normalize(p, p=2, dim=1)
                    matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3
                    matrix2 = torch.zeros(matrix1.shape).to(model_args.device)
                    ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
                loss = loss + clst * cluster_cost + sep * separation_cost + 5e-4 * l1 + 0.05 * ld
                # exit()
            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())

            tot += len(batch.y)
            correct += prediction.eq(batch.y).cpu().numpy().sum()

        acc = correct / tot

        # report train msg
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), acc))
        print(
            f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Acc: {acc:.3f}")
        train_loss.append(np.average(loss_list))

        # report eval msg
        acc_per_class = {}
        for i in class_num:
            acc_per_class[i] = 0
        eval_state, pred = evaluate_GC(dataloader['eval'], gnnNets, criterion, acc_per_class)
        for i in class_num:
            acc_per_class[i] = acc_per_class[i] / class_num[i]
        print(acc_per_class)
        # print(pred[:500])
        # print(y)

        print(
            f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f} | Count: {eval_state['count']}")
        append_record(
            "Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))
        eval_loss.append(eval_state['loss'])
        eval_acc.append(eval_state['acc'])

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)

    print(f"The best validation accuracy is {best_acc}. Accuracy of every label is {acc_per_class}")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best.pt'))
    gnnNets.update_state_dict(checkpoint['net'])

    acc_per_class_test = {}
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion, acc_per_class_test)
    print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    print(f'Accuracy of every label is {acc_per_class}')
    append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))

    with open(f'./datasets/figs/test.csv', 'a') as f:
        f.write(str(test_state['acc']) + ',')


def evaluate_GC(eval_dataloader, gnnNets, criterion, acc_per_class):
    loss_list, pred = [], []
    gnnNets.eval()
    tot, correct = 0, 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(model_args.device)
            logits, probs, _, _, _, _ = gnnNets(batch)

            loss = criterion(logits, batch.y)

            ## record
            _, prediction = torch.max(logits, -1)
            pred.append(prediction.cpu())

            loss_list.append(loss.item())
            for (i, y1) in enumerate(batch.y):
                y1 = int(y1)
                if prediction[i] == y1:
                    if acc_per_class.__contains__(y1):
                        acc_per_class[y1] += 1
                    else:
                        acc_per_class[y1] = 1

            tot += len(batch.y)
            correct += prediction.eq(batch.y).cpu().numpy().sum()

        acc = correct / tot
        pred = np.concatenate(pred).squeeze()

        eval_state = {'loss': np.average(loss_list),
                      'acc': acc,
                      'count': f'{correct}/{tot}'}

    return eval_state, pred


def test_GC(test_dataloader, gnnNets, criterion, acc_per_class):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []

    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            for (i, y1) in enumerate(batch.y):
                y1 = int(y1)
                if prediction[i] == y1:
                    if acc_per_class.__contains__(y1):
                        acc_per_class[y1] += 1
                    else:
                        acc_per_class[y1] = 1
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean())}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach()  # .numpy()
    return test_state, pred_probs, predictions


def predict_GC(test_dataloader, gnnNets):
    """
    return: pred_probs --  np.array : the probability of the graphs class
            predictions -- np.array : the prediction class for each graphs
    """
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)

            ## record
            _, prediction = torch.max(logits, -1)
            predictions.append(prediction)
            pred_probs.append(probs)

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return pred_probs, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pt_name = f"{model_name}_latest.pt"
    best_pt_name = f'{model_name}_best.pt'

    ckpt_path = os.path.join(ckpt_dir, pt_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pt_name))
    gnnNets.to(model_args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--clst', type=float, default=0.05,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.01,
                        help='separation')
    parser.add_argument('--file', type=str, default=None, help='straightly given the file name')
    parser.add_argument('--id1', type=str, default=None)
    parser.add_argument('--id2', type=str, default=None)

    args = parser.parse_args()
    if args.file is not None:
        file = args.file.split('_')
        data_args.species = file[0]
        data_args.amount = args.id1
        data_args.tissue = '_'.join(file[1:-1]).strip(data_args.amount)

    train_GC(args.clst, args.sep,args.id1,args.id2)
