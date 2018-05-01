# svm implementation

from settings import *
import numpy as np
import numpy.matlib
import pandas as pd
import nb 
import glob
import os
import re

def svm(X, y, tau = 8, num_outer_loops = 40):
  m = X.shape[0]
  Z = X
  Z[Z > 0] = 1
  squared_Z = np.sum(Z ** 2, axis = 1).reshape(m, 1)
  gram = np.dot(Z, Z.T)

  # find Gaussian radial basis kernel matrix for a train or test point
  # using kernel trick 
  K = np.exp(-(np.matlib.repmat(squared_Z, 1, m) 
            + np.matlib.repmat(squared_Z.T, m, 1) 
            - 2 * gram) / (2 * tau^2))

  alpha = np.zeros(m)
  avg_alpha = np.zeros(m)
  lambda0 = 1/(64 * m)
  grad = np.zeros(m)
  for i in range(num_outer_loops * m):
    print('Iter %d' % i)
    idx = np.random.randint(0, m, 1)
    margin = (y[idx] * np.dot(K[idx, :], alpha))
    grad = -(margin < 1).astype('float') * y[idx] * (K[:, idx]) \
            + m * lambda0 * K[:, idx] * alpha[idx]
    alpha -= np.squeeze(grad) / np.sqrt(i + 1)
    avg_alpha += alpha 
  avg_alpha = avg_alpha / (m * num_outer_loops)
  return avg_alpha


def predict(Xtrain, Xtest, alpha, tau = 8):

  m = Xtrain.shape[0]
  n = Xtest.shape[0]
  
  Xtrain[Xtrain > 0] = 1
  Xtest[Xtest > 0] = 1

  squared_Xtrain = np.sum(Xtrain ** 2, axis = 1).reshape(m, 1)
  squared_Xtest = np.sum(Xtest ** 2, axis = 1).reshape(n, 1)
  
  gram_test = np.dot(Xtest, Xtrain.T)

  Ktest = np.exp(-(np.matlib.repmat(squared_Xtest, 1, m) 
                + np.matlib.repmat(squared_Xtrain.T, n, 1) 
                - 2 * gram_test) / (2 * tau^2))

  preds =  np.dot(Ktest, alpha)
  return(preds)


if __name__ == '__main__':
  train_file_names = glob.glob(os.path.join(data_path, 'spam_data/MATRIX.TRAIN.*'), recursive = True)

  train = pd.read_table(os.path.join(data_path, 'spam_data/MATRIX.TRAIN'),
                        skiprows = 2,
                        sep = '\s+')

  train_matrix, train_label = nb.convert_sparse_to_full(train, label_idx = 0)
  train_label = 2 * train_label - 1
  
  test = pd.read_table(os.path.join(data_path, 'spam_data/MATRIX.TEST'),
                        skiprows = 2,
                        sep = '\s+')

  test_matrix, test_label = nb.convert_sparse_to_full(test, label_idx = 0)
  test_label = 2 * test_label - 1

  est_alpha = svm(train_matrix, train_label)
  preds = predict(train_matrix, test_matrix, est_alpha)

  test_error_list = np.zeros(len(train_file_names))

  for i in range(len(train_file_names)):
    print('Load', train_file_names[i])
    temp_train = pd.read_table(train_file_names[i],
                              skiprows = 2,
                              sep = '\s+')
    temp_x, temp_label = nb.convert_sparse_to_full(temp_train, label_idx = 0)

    est_alpha = svm(temp_x, temp_label)
    temp_test_preds  = predict(temp_x, test_matrix, est_alpha)

    temp_test_error = np.mean(temp_test_preds != test_label)
    test_error_list[i] = temp_test_error

  sizes = np.array([re.sub('^.*spam_data/MATRIX\.TRAIN\.', '', x) for x in train_file_names]).astype('int64')
  
  import matplotlib.pyplot as plt 
  %matplotlib
  
  
  plt.scatter(sizes[np.argsort(sizes)], test_error_list[np.argsort(sizes)])
  plt.plot(sizes[np.argsort(sizes)], test_error_list[np.argsort(sizes)])
