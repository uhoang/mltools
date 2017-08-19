import numpy as np
import pandas as pd

def find_theta(w, x, y):
  '''Helper function to estimate theta in the weighed liner regression
  @param: w is a vector of diagonal elements in the weight matrix
          x is an input matrix (m, 1) 
          y is an outcome matrix (m, 1) 
  '''
  wx = np.dot(x.T, np.diag(w))
  wxy = np.dot(wx, y)
  theta = np.dot(np.linalg.inv(np.dot(wx, x)), wxy)
  return theta

def local_weighted_linear_regression(x, y, newx = None, bw = 5):
  '''Implemented local weighted linear regression 
  @param: x: a (m, ) vector of inputs 
          y: a (m, ) vector of outcomes 
          newx: a vector of new query points x
          bw: a scalar of bandwidth parameter in the weights' kernel
          function 
          w_{(i)} = exp{(x - x_{(i)})^2/(2 * \tau^2)}
  @return: the smooth curve for y given x
  '''
  m = x.shape[0]
 
  if len(x.shape) == 1:
    x = x.reshape((m, 1))

  if len(y.shape) == 1:
    y = y.reshape((m, 1))

  if newx == None:
    newx = x
  W = np.exp( - (newx[:, np.newaxis] - x[np.newaxis, :]) ** 2 / (2 * bw ** 2))
  theta = np.apply_along_axis(find_theta, 0, W, x = x, y = y)
  est = theta.reshape((m,)) * newx.reshape((m, ))
  return est

def local_weighted_linear_regression_loop(x, y, newx = None, bw = 5):
  '''Implemented local weighted linear regression 
  @param: x: a (m, ) vector of inputs 
          y: a (m, ) vector of outcomes 
          bw: a scalar of bandwidth parameter in the weights' kernel
          function 
          w_{(i)} = exp{(x - x_{(i)})^2/(2 * \tau^2)}
  @return: the smooth curve for y given x
  '''
  m = x.shape[0]
 
  if len(x.shape) == 1:
    x = x.reshape((m, 1))

  if len(y.shape) == 1:
    y = y.reshape((m, 1))

  if newx == None:
    newx = x
  est = np.zeros((m, ))
  for i in range(m):
    w = np.exp(- (newx[i] - x) ** 2/(2 * bw ** 2)).reshape((m, ))
    wx = np.dot(x.T, np.diag(w))
    wxy = np.dot(wx, y)
    theta = np.dot(np.linalg.inv(np.dot(wx, x)), wxy)
    est[i] = np.dot(theta.T, newx[i])

  return est

X = pd.read_csv('http://cs229.stanford.edu/ps/ps1/quasar_train.csv')

# a single header row corresponds integral wavelengths in the interval [1150, 1600]
x = X.columns.values.astype(float)
y = X.loc[0, :].values # the first set of relative flux measurements for given wavelengths
m = x.shape[0]

# fit linear regression to relative flux spectrum for each wavelength
x_it = np.hstack([np.ones((m, 1)), x.reshape(m, 1)])
theta = np.dot(np.linalg.inv(np.dot(x_it.T, x_it)), np.dot(x_it.T, y.reshape(m ,1)))
est_lin_y = np.dot(x_it, theta)


%load_ext line_profiler # load line-by-line profiling with %lprun

%lprun -f local_weighted_linear_regression_loop local_weighted_linear_regression_loop(x, y, bw = 1)
%lprun -f local_weighted_linear_regression local_weighted_linear_regression(x, y, bw = 1)

# fit a smooth curve to relative flux spectrum for given wavelength
est_y_1 = local_weighted_linear_regression(x, y, bw  = 1) 
est_y_5 = local_weighted_linear_regression(x, y, bw = 5) 
est_y_10 = local_weighted_linear_regression(x, y, bw = 10) 
est_y_100 = local_weighted_linear_regression(x, y, bw = 100) 
est_y_1000 = local_weighted_linear_regression(x, y, bw = 1000) 

%matplotlib
import matplotlib.pyplot as plt 

plt.scatter(x, y, label = 'Raw data')
plt.plot(x, est_y_1, '-r', label = 'tau = 1')
plt.plot(x, est_y_5, '--b', label = 'tau = 5')
plt.plot(x, est_y_10, '-.g', label = 'tau = 10')
plt.plot(x, est_y_100, ':c', label = 'tau = 100')
plt.plot(x, est_y_1000, ':k', label = 'tau = 1000')
plt.plot(x, est_lin_y, '--y', label = 'linear regression')

plt.legend(loc = 'upper right')
