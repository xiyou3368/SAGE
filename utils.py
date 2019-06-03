import numpy as np
import scipy.io
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from itertools import chain
from collections import defaultdict


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + 1 * sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support,labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def graph_padding(graph_topo_list,graph_tezheng_list,word_pad_length,feature_dimension =3):
  graph_topo_new = []
  graph_tezheng_new = []
  for graph_topo in graph_topo_list:
    len_graph = graph_topo.shape[0]
    to_pad_no = word_pad_length - len_graph
    if to_pad_no <= 0:
      g = graph_topo[range(word_pad_length)][:,range(word_pad_length)]
      graph_topo_new.append(g)
    else:
      graph_topo_tmp = np.zeros(word_pad_length*word_pad_length).reshape(word_pad_length,word_pad_length)
      for i in range(len_graph):
        graph_topo_tmp[i,:len_graph] = graph_topo.todense()[i,:]
      graph_topo_tmp = sp.csr_matrix(graph_topo_tmp,dtype = int)
      graph_topo_new.append(graph_topo_tmp)
  j = 0
  for graph_tezheng in graph_tezheng_list:
    len_graph = graph_tezheng.shape[0]
    to_pad_no = word_pad_length - len_graph
    if to_pad_no <= 0:
      graph_tezheng_new.append(sp.csr_matrix(graph_tezheng[:word_pad_length,:], dtype = int))
    else:
      tezheng_topad = np.array([[0]* feature_dimension for i in range(to_pad_no)])
      try:
        tezheng_topad = sp.csr_matrix(np.vstack((graph_tezheng.todense(),tezheng_topad)), dtype = int)
      except:
        print(j)
      graph_tezheng_new.append(tezheng_topad)
    j += 1
  return graph_topo_new,graph_tezheng_new


def re_order(tuopu_list,feature_list,y_label):
    raw_all_y = y_label
    max_catogory_no = max(sum(y_label),len(y_label) - sum(y_label))
    tag_d = defaultdict(list)
    for x in range(2):
      tag_d["string{0}".format(x)] = []
 
    tag_index = 0
    for tag in raw_all_y:
      tag_d["string{0}".format(tag)].append(tag_index)
      tag_index += 1
    for i in range(2):
      tag_d["string{0}".format(i)] = tag_d["string{0}".format(i)] * int(max_catogory_no/ len(tag_d["string{0}".format(i)]))
      chancha = max_catogory_no - len(tag_d["string{0}".format(i)])
      tag_d["string{0}".format(i)] += tag_d["string{0}".format(i)][:chancha] 
    index_result = [None]*(2 * max_catogory_no)
    for x in range(2):
      index_result[x::2] = tag_d["string{0}".format(x)]
    word_input = list(map(feature_list.__getitem__, index_result))
    tuopu_input = list(map(tuopu_list.__getitem__, index_result))
    y_input = list(map(y_label.__getitem__, index_result))
    return tuopu_input,word_input,y_input


def load_protein_dataset(dataset_name):
    #if dataset_name not in chemical_datasets_list:
    #    print_ext('Dataset doesn\'t exist. Options:', chemical_datasets_list)
    #    return
    mat = scipy.io.loadmat('datasets/%s.mat' % dataset_name)
    
    input = mat[dataset_name]
    labels = mat['l' + dataset_name.lower()]
    labels = labels - min(labels)
    
    node_labels = input['nl']
    v_labels = 0
    for i in range(node_labels.shape[1]):
        v_labels = max(v_labels, max(node_labels[0, i]['values'][0, 0])[0])
    
    e_labels = 1
    # For each sample
    samples_V = []
    samples_A = []
    max_no_nodes = 0
    for i in range(input.shape[1]):
        no_nodes = node_labels[0, i]['values'][0, 0].shape[0]
        max_no_nodes = max(max_no_nodes, no_nodes)
        V = np.ones([no_nodes, v_labels])
        for l in range(v_labels):
            V[..., l] = np.equal(node_labels[0, i]['values'][0, 0][..., 0], l+1).astype(np.float32)
        samples_V.append(V)
        A = np.zeros([no_nodes, no_nodes])
        for j in range(no_nodes):
            for k in range(input[0, i]['al'][j, 0].shape[1]):
                A[j, input[0, i]['al'][j, 0][0, k]-1] = 1
        samples_A.append(A)
    return np.array(samples_V), np.array(samples_A), np.reshape(labels, [-1])
