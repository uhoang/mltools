# svm implementation

from settings import *
import numpy as np
import pandas as pd
import nb 
import glob
import os
from settings import *

def svm(X, y, tau = 8, num_outer_loops = 40):
  m = X.shape[0]
  Z = X
  Z[Z > 0] = 1
  squared_Z = np.sum(Z ** 2, axis = 1)
  gram = np.dot(Z, Z.T)

  # find Gaussian radial basis kernel matrix for a train or test point
  # using kernel trick 
  K = np.exp(-(squared_Z.reshape((1, m)) 
            + squared_Z.reshape((m, 1)) 
            - 2 * gram) / (2 ** tau^2))

  alpha = np.zeros(m)/(64*m)
  avg_alpha = np.zeros(m)
  lambda0 = 1/(64 * m)
  grad = np.zeros(m)
  for i in range(num_outer_loops * m):
    print('Iter %d' % i)
    idx = np.random.randint(0, m, 1)
    margin = (y[idx] * K[idx, :] * alpha).reshape(m)
    
    grad = -(margin < 1).astype('float') * y[idx] * (K[idx, :]).reshape(m) \
            + m * lambda0 * np.dot(K[idx, :], alpha)
    alpha -= grad / np.sqrt(i + 1)
    avg_alpha += alpha 
  avg_alpha = avg_alpha / (m * num_outer_loops)
  return avg_alpha


def predict(X, newX, alpha):
  n = newX.shape[0]
  newX[newX > 0] = 1
  preds = np.zeros(n)
  for i in range(n):
    K = np.exp(np.sum(-(X - newX[i, :]) ** 2/(2 * tau **2), axis = 1)) 
    margin = np.mean(K * alpha)
    if margin > 0:
      preds[i] = 1
    else:
      preds[i] = -1
  return preds 

if __name__ == '__main__':
  filenames = []
  data_path_list = glob.iglob(os.path.join(data_path, 'spam_data/MATRIX.TRAIN.'))

  for filename in data_path_list:
    filenames.append(filename)

  train = pd.read_table(os.path.join(data_path, 'spam_data/MATRIX.TRAIN'),
                        skiprows = 2,
                        sep = '\s+')

  train_matrix, train_label = nb.convert_sparse_to_full(train, label_idx = 0)
  train_label[train_label == 0] = -1
  
  test = pd.read_table(os.path.join(data_path, 'spam_data/MATRIX.TEST'),
                        skiprows = 2,
                        sep = '\s+')

  test_matrix, test_label = nb.convert_sparse_to_full(test, label_idx = 0)
  test_label[test_label == 0] = -1

  est_alpha = svm(train_matrix, train_label)