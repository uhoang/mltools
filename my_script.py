def square(x):
  return x ** 2
for N in range(1, 4):
  print(N, "squared is", square(N))


def qsort(x):
  if len(x) > 1:
    left_seq = x[x < x[0]]
    right_seq = x[x > x[0]]
    return np.hstack([qsort(left_seq), x[0], qsort(right_seq)])
  else:
    return x


def selection_sort(x):
  for i in range(len(x)):
    swap = i + np.argmin(x[i:])
    (x[swap], x[i]) = (x[i], x[swap])
  return x


import urllib
import numpy as np
from numpy.linalg import inv
# x = urllib.urlopen('http://cs229.stanford.edu/ps/ps1/logistic_x.txt').read()

# download the data to the local directory
# urllib.request.urlretrieve('http://cs229.stanford.edu/ps/ps1/logistic_x.txt', 'logistic_x.txt')
# urllib.request.urlretrieve('http://cs229.stanford.edu/ps/ps1/logistic_y.txt', 'logistic_y.txt')

# X = np.loadtxt('logistic_x.txt')
# y = np.loadtxt('logistic_y.txt')

# l
X = pd.read_table('http://cs229.stanford.edu/ps/ps1/logistic_x.txt', 
                header = None,
                sep = '\s+')


X = np.hstack([np.ones((X.shape[0], 1)), X])

m = X.shape[0]

def log_regression(X, y, max_iters):
  '''Implement Newton-Raphson algorithm to estimate MLEs for logistic regression
  @param: X: a (N, 2) matrix of training inputs 
          y: a vector of length N labels {-1, 1}
          max_iters: an integer of number of iterations for
  @return: theta: a (3, ) vector of estimated parameters
           ll: loglikehood for logistic regression
  '''
  m = X.shape[0]
  d = X.shape[1]

  theta = np.zeros((d, ))

  for i in range(max_iters): 
    margins = np.dot(X, theta) * y
    ll = np.mean(np.log(1 + np.exp(-margins)))
    probs = 1/(1 + np.exp(margins))
    grad = -(1/m) * np.sum(X.T * (probs * y), axis = 1)
    scale_X_T = np.dot(X.T, np.diag(probs * (1- probs)))
    H = (1/m) * np.dot(scale_X_T, X)
    theta -= np.dot(inv(H), grad)
    print('The average empirical loss at iter %d: %f' % (i, ll))
    print('The current value of theta:', theta)
  return theta, ll

theta, ll = log_regression(X, y, 10)

%matplotlib
import matplotlib.pyplot as plt 

import pandas as pd

# def join_str(x):
#   return 'y = ' + x.astype('U32')

df = pd.DataFrame(dict(x1 = X[:, 1], x2 = X[:, 2], label = y))
# df['label'] = df.groupby(df.index)['label'].apply(join_str)
groups = df.groupby('label')

# Plot
# plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
# colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

# fig, ax = plt.subplots()
# ax.set_color_cycle(colors)
# ax.margins(0.05)
N
x1 = np.arange(df.x1.values.min(), df.x1.values.max(), 0.1)
x2 = -theta[0]/theta[2] - theta[1]/theta[2] * x1

markers = dict({1: 'o', -1: '+'})
for name, group in groups:
    print(name)
    print(group.head())
    print(markers[name])
    plt.scatter(group.x1, group.x2, 
            marker=markers[name], 
            label='y = {}'.format(name), 
            alpha = .8)

plt.legend(loc = 'upper right')
plt.plot(x1, x2, '-r')


# import urllib
import pandas as pd
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
%matplotlib

# urllib.request.urlretrieve('http://cs229.stanford.edu/ps/ps1/quasar_train.csv', 'quasar_train.csv')
# urllib.request.urlretrieve('http://cs229.stanford.edu/ps/ps1/quasar_test.csv', 'quasar_test.csv')


train = pd.read_csv('http://cs229.stanford.edu/ps/ps1/quasar_train.csv').T

train = pd.read_csv('quasar_train.csv').T

Xtr = np.hstack([np.ones((train.shape[0], 1)), train.index.values[:, np.newaxis].astype('float')])
ytr = train.loc[:, 0].values 
# ytrain = Xtrain.index.values

# Xtr = np.hstack([np.ones((Xtrain.shape[0], 1)), Xtrain.loc[:, 0].values[:, np.newaxis]])

H = np.dot(inv(np.dot(Xtr.T, Xtr)), Xtr.T)
theta = np.dot(H, ytr[:, np.newaxis].astype('float'))

x = np.sort(Xtr[:, 1])
X = np.hstack([np.ones((x.shape[0], 1)), x[:, np.newaxis]])
y = np.dot(X, theta)


plt.scatter(Xtr[:, 1], ytr, marker = '+', label = 'Raw data')
plt.plot(x, y, '-r', label = 'Regression line')
plt.legend(loc = 'upper right')

X = Xtr[:, 1]
newX = x
y = ytr

def local_logistic_regression(X, y, newX, bw = 5):
  '''Implemented locally weighted linear regression
    '''
  m = X.shape[0]
  X = X.reshape((m, 1))
  est_y = np.zeros(newX.shape[0])
  for i in range(newX.shape[0]):
    W = np.diag(np.exp(-(X - newX[i]) ** 2/(2 * bw ** 2)).reshape((m, )))
    M = np.dot(np.dot(X.T, W), X)
    theta = np.dot(np.dot(inv(M), np.dot(X.T, W)), y[:, np.newaxis])
    est_y[i] = newX[i] * theta
  return est_y

x = np.arange(np.min(Xtr[:, 1]), np.max(Xtr[:, 1]), 1)
est_y_1 = local_logistic_regression(Xtr[:, 1], ytr, x, bw = 1)
est_y_10 = local_logistic_regression(Xtr[:, 1], ytr, x, bw = 10)
est_y_100 = local_logistic_regression(Xtr[:, 1], ytr, x, bw = 100)
est_y_100 = local_logistic_regression(Xtr[:, 1], ytr, x, bw = 100)



plt.scatter(Xtr[:, 1], ytr, marker = '+', label = 'Raw data')
plt.plot(x, est_y_1, '-c', label = 'LWR-bw=1')
plt.plot(x, est_y_10, '--k', label = 'LWR-bw=10')
plt.plot(x, est_y_100, '-.r', label = 'LWR-bw=100')
plt.plot(x, est_y_100, ':y', label = 'LWR-bw=1000')

plt.legend(loc = 'upper right')
# x = np.arange()
# train = np.loadtxt('quasar_train.csv', delimiter = ',')

train = pd.read_csv('quasar_train.csv').T
X = train.index.values.astype('float')
f = np.zeros((train.shape))

for i in range(train.shape[1]):
  y = train.loc[:, i].values
  f[:, i] = local_logistic_regression(X, y, X)


test = pd.read_csv('quasar_test.csv').T
newf = np.zeros(test.shape)
for i in range(test.shape[1]):
  y = test.loc[:, i].values
  newf[:, i] = local_logistic_regression(X, y, X)

def local_weighted_regression(f, newf = None, k = 3):

  if newf is None:
    newf = f
  print(newf.shape)
  print(f.shape)
  m = newf.shape[1]
  f_r = f[X >= 1300, :]
  f_l = f[X < 1200, :]

  newf_r = newf[X >= 1300, :]
  newf_l = newf[X < 1200, :]  

  d = np.sum((f_r[:, np.newaxis, :] - newf_r[:, :, np.newaxis]) ** 2, axis = 0)
  d /= np.max(d, axis = 1)[:, np.newaxis]
  kernel = 1 - d
  kernel[kernel < 0] = 0

  m = newf.shape[1]
  neighb = np.argsort(kernel, axis = 1)[:, 0:k]
  est_f_l = np.zeros(newf_l.shape)
  
  for i in range(m):
    w = kernel[i, neighb[i, :]]
    est_f_l[:, i] =  np.sum(f_l[:, neighb[i, :]] * w, axis = 1) / np.sum(w)

  avg_error = np.sum((est_f_l - newf_l) ** 2) / m  
  return avg_error

local_weighted_regression(f)
local_weighted_regression(f, newf = newf)