import collections
import shutil
import sys
import os
import time

import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
import anndata
import io
import random
import itertools
import scipy
import anndata as ad
from numpy import inf
from scipy import sparse
from Configures import data_args, train_args, model_args
import math
from tqdm import tqdm


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    df = pd.DataFrame(classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, df


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def lower_matrix(df):
    """Convert the index of the dataframe to lowercase"""
    index = df.index
    index = list(index)
    index2 = []
    for x in index:
        index2.append(x.lower())
    df.index = index2
    return df


def remove_exclusive(label1, threshold):
    count_dict = label1.value_counts().to_dict()
    keep_categories = [key for key in count_dict.keys() if count_dict[key] >= threshold]
    keep_rows = label1.isin(keep_categories)
    label1 = label1[keep_rows]
    return label1


def batch_effect(query_path_M, query_path_L, refer_path_M, refer_path_L, exclusive=None):
    if os.path.exists(query_path_M) and os.path.exists(query_path_L) and os.path.exists(refer_path_M) \
            and os.path.exists(refer_path_L):
        print('File paths are valid.')
    else:
        print('File paths are invalid.')

    train_L = pd.read_table(refer_path_L, sep=',', index_col=1)
    test_L = pd.read_table(query_path_L, sep=',', index_col=1)

    train_M = sc.read_csv(refer_path_M).transpose()
    test_M = sc.read_csv(query_path_M).transpose()

    train_M.obs['cell_type'] = train_L[train_L.columns[1]]
    sc.pp.filter_cells(train_M, min_genes=100)
    sc.pp.filter_genes(train_M, min_cells=50)
    sc.pp.normalize_total(train_M, target_sum=1e4)  # scale the expression levels of each cell to the same total sum
    sc.pp.log1p(train_M)  # perform a log1p transformation on the expression levels

    r_L = train_M.obs['cell_type']
    r_L = remove_exclusive(r_L, min(exclusive, int(0.1 * len(r_L))))

    test_M.obs['cell_type'] = test_L[test_L.columns[1]]
    sc.pp.filter_cells(test_M, min_genes=100)
    sc.pp.filter_genes(test_M, min_cells=50)
    sc.pp.normalize_total(test_M, target_sum=1e4)
    sc.pp.log1p(test_M)

    q_L = test_M.obs['cell_type']
    q_L = remove_exclusive(q_L, min(exclusive, int(0.1 * len(q_L))))

    common_type = set(set(r_L) & set(q_L))
    common_type.discard('Other/Doublet')

    print("common_type:")
    print(common_type)

    r_L = r_L[r_L.isin(common_type)]
    q_L = q_L[q_L.isin(common_type)]

    cell_list_1 = r_L.index.tolist()
    train_M_df = train_M.to_df()
    train_M_df = train_M_df.loc[train_M_df.index.isin(cell_list_1)]
    train_M = anndata.AnnData(X=train_M_df)
    train_M.obs['cell_type'] = r_L

    cell_list_2 = q_L.index.tolist()
    test_M_df = test_M.to_df()
    test_M_df = test_M_df.loc[test_M_df.index.isin(cell_list_2)]
    test_M = anndata.AnnData(X=test_M_df)
    test_M.obs['cell_type'] = q_L
    # # method 1 hvg
    sc.pp.highly_variable_genes(train_M, flavor='seurat_v3', n_top_genes=1000)
    sc.pp.highly_variable_genes(test_M, flavor='seurat_v3', n_top_genes=1000)

    common_highly_variable_genes = list(
        set(train_M.var_names[train_M.var['highly_variable']]) & set(test_M.var_names[test_M.var['highly_variable']]))[
                                   :300]
    print('common_highly_variable_genes: ', len(common_highly_variable_genes))

    train_M = train_M[:, train_M.var_names.isin(common_highly_variable_genes)]
    test_M = test_M[:, test_M.var_names.isin(common_highly_variable_genes)]

    # Common high-variance gene expression matrix.
    train_M_X_df = pd.DataFrame(train_M.X.toarray(), columns=train_M.var_names)
    test_M_X_df = pd.DataFrame(test_M.X.toarray(), columns=test_M.var_names)

    train_M_obs_names = pd.DataFrame(train_M.obs_names, columns=[''])
    Traindataname_fintruefeatures = pd.concat([train_M_obs_names, train_M_X_df], axis=1)
    Traindataname_fintruefeatures.set_index('', inplace=True)
    r_M = Traindataname_fintruefeatures

    test_M_obs_names = pd.DataFrame(test_M.obs_names, columns=[''])
    Testdataname_fintruefeatures = pd.concat([test_M_obs_names, test_M_X_df], axis=1)
    Testdataname_fintruefeatures.set_index('', inplace=True)
    q_M = Testdataname_fintruefeatures

    r_M_DR = None
    q_M_DR = None

    return r_M, r_L, r_M_DR, q_M, q_L, q_M_DR


def prepare_data(dir_path, save_path):
    for file in tqdm(os.listdir(dir_path)):
        if file.endswith('data.csv'):
            pass
        elif file.endswith('celltype.csv'):
            cell_type_data = pd.read_csv(os.path.join(dir_path, file))
            save_file = '_'.join(file.split('_')[:-1]) + '_data.csv'
            data = pd.read_csv(os.path.join(dir_path, save_file), index_col=0)

            type_list = cell_type_data.values[:, 2].tolist()
            label_statistics = dict(collections.Counter(type_list))

            cell_types = list(set(type_list))
            train_total_cell = sum(label_statistics.values())

            for (label, num) in label_statistics.items():
                if num / train_total_cell < data_args.exclude_rate or num < 10 or num / train_total_cell > 0.9:
                    cell_types.remove(label)

            cell_type_data = cell_type_data[cell_type_data['Cell_type'].isin(cell_types)]
            data = data.loc[:, cell_type_data['Cell']]

            cell_type_data.to_csv(os.path.join(save_path, file), index=None, header=['', 'Cell', 'Cell_type'])
            data.to_csv(os.path.join(save_path, save_file))
        else:
            raise NotImplementedError


def diffusion_fun_sparse(A):
    n, m = A.shape
    A_with_selfloop = A + sp.identity(n, format='csc')
    diags = A_with_selfloop.sum(axis=1).flatten()
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


def _normalize_diffusion_matrix(A):
    n, m = A.shape
    A_with_selfloop = A
    diags = A_with_selfloop.sum(axis=1).flatten()

    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


#### return normalized adjcent matrix plus PPMI
def diffusion_fun_improved(A, sampling_num=100, path_len=3,
                           self_loop=True, spars=False):
    shape = A.shape
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")
    # mat is a sparse lil_matrix
    pmi = None
    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)
    A_with_selfloop = A + pmi
    dig = np.sum(A_with_selfloop, axis=1)
    dig = np.squeeze(np.asarray(dig))
    Degree = np.diag(dig)
    Degree_normalized = Degree ** (-0.5)
    Degree_normalized[Degree_normalized == inf] = 0.0
    Diffusion = np.dot(
        np.dot(Degree_normalized, A_with_selfloop), Degree_normalized)
    return Diffusion


def diffusion_fun_improved_ppmi_dynamic_sparsity(A, sampling_num=100, path_len=2,
                                                 self_loop=True, spars=True, k=1.0):
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")
    # mat is a sparse dok_matrix
    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)

    pmi = _shift(pmi, k)
    ans = _normalize_diffusion_matrix(pmi.tocsc())

    return ans


def _shift(mat, k):
    print(k)
    r, c = mat.shape
    x, y = mat.nonzero()
    mat = mat.todok()
    offset = np.log(k)
    print("Offset: " + str(offset))
    for i, j in zip(x, y):
        mat[i, j] = max(mat[i, j] - offset, 0)

    x, y = mat.nonzero()
    sparsity = 1.0 - len(x) / float(r * c)
    print("Sparsity: " + str(sparsity))
    return mat


def _diffusion_fun_sampling(A, sampling_num=100, path_len=3, self_loop=True, spars=False):
    # the will return diffusion matrix
    re = None
    if not spars:
        re = np.zeros(A.shape)
    else:
        re = sparse.dok_matrix(A.shape, dtype=np.float32)

    if self_loop:
        A_with_selfloop = A + sparse.identity(A.shape[0], format="csr")
    else:
        A_with_selfloop = A

    # record each node's neignbors
    dict_nid_neighbors = {}
    for nid in range(A.shape[0]):
        neighbors = np.nonzero(A_with_selfloop[nid])[1]
        dict_nid_neighbors[nid] = neighbors

    # for each node
    for i in range(A.shape[0]):
        # for each sampling iter
        for j in range(sampling_num):
            _generate_path(i, dict_nid_neighbors, re, path_len)
    return re


def _generate_path(node_id, dict_nid_neighbors, re, path_len):
    path_node_list = [node_id]
    for i in range(path_len - 1):
        temp = dict_nid_neighbors.get(path_node_list[-1])
        if len(temp) < 1:
            break
        else:
            path_node_list.append(random.choice(temp))
    # update difussion matrix re
    for pair in itertools.combinations(path_node_list, 2):
        if pair[0] == pair[1]:
            re[pair[0], pair[1]] += 1.0
        else:
            re[pair[0], pair[1]] += 1.0
            re[pair[1], pair[0]] += 1.0


def _PPMI(mat):
    (nrows, ncols) = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        rowMat[i, :] = 0 if rowTotals[i] == 0 else rowMat[i, :] * (1.0 / rowTotals[i])
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:, j] = 0 if colTotals[j] == 0 else colMat[:, j] * (1.0 / colTotals[j])
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows, ncols), dtype=np.float32), np.log(P))
    return P


def _PPMI_sparse(mat):
    # mat is a sparse dok_matrix
    nrows, ncols = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T

    N = float(np.sum(rowTotals))
    rows, cols = mat.nonzero()

    p = sp.dok_matrix((nrows, ncols))
    for i, j in zip(rows, cols):
        _under = rowTotals[0, i] * colTotals[0, j]
        if _under != 0.0:
            log_r = np.log((N * mat[i, j]) / _under)
            if log_r > 0:
                p[i, j] = log_r
    return p


if __name__ == '__main__':
    a, b, _, d, e, _ = batch_effect('./datasets/all/cancer/human_colonA2_data.csv',
                                    './datasets/all/cancer/human_colonA2_celltype.csv',
                                    './datasets/all/cancer/human_colonA2_data.csv',
                                    './datasets/all/cancer/human_colonA2_celltype.csv', 150)
    # return r_M, r_L, r_M_DR, q_M, q_L, q_M_DR
    print('shape: ', a.shape, '   len: ', len(a))
    print('shape: ', b.shape, '   len: ', len(b))
    print('shape: ', d.shape, '   len: ', len(d))
    print('shape: ', e.shape, '   len: ', len(e))
    a.transpose().to_csv('./datasets/SingleCellDataset/train/human/human_colonA2_data.csv')
    b.reset_index().to_csv('./datasets/SingleCellDataset/train/human/human_colonA2_celltype.csv',
                           header=['Cell', 'Cell_type'])
    d.transpose().to_csv('./datasets/SingleCellDataset/test/human/human_colonA2_data.csv')
    d.reset_index().to_csv('./datasets/SingleCellDataset/test/human/human_colonA2_celltype.csv',
                           header=['Cell', 'Cell_type'])
