from utils import *
from scipy import sparse
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold

def load_nci(cv_index):
  dataset = load_protein_dataset('proteins')
  ad_train_list = []
  ad_test_list = []
  fea_train_list = []
  fea_test_list = []
  label_train_list = []
  label_test_list = []
  kf = KFold(n_splits=10)
  kf.get_n_splits(dataset[0])
  j = 0
  for train_index, test_index in kf.split(dataset[0]):
    if j == cv_index:
      i = 0
      for item in dataset[0]:
        if i in test_index:
          fea_test_list.append(sparse.csr_matrix(item))
        else:
          fea_train_list.append(sparse.csr_matrix(item))
        i += 1
 
      i = 0
      for item in dataset[1]:
        if i in test_index:
          ad_test_list.append(nx.adjacency_matrix(nx.from_numpy_matrix(item[:,:])))
        else:
          ad_train_list.append(nx.adjacency_matrix(nx.from_numpy_matrix(item[:,:])))
        i += 1

      i = 0
      for item in dataset[2]:
        if i in test_index:
          label_test_list.append(item)
        else:
          label_train_list.append(item)
        i += 1
    j += 1
  return ad_train_list,fea_train_list,ad_test_list,fea_test_list,label_train_list,label_test_list 

    
