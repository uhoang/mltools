import numpy as np
import pandas as pd

def lwlrx(newx, x, y, bw):
  '''Estimate a local weighted linear regression of an outcome for a given input
  @param: x: a (m, 2) matrix of inputs
          y: a (m, ) vector of outcome
          newx: a (1, 2) vector of a new query point
          bw: a scalar of bandwidth parameter in the weight's kernel function
              w_{(i)} = exp{(x - x_{(i)})^2/(2 * \tau^2)}
  @return: a local weighted linear regression of an outcome
  '''
  d = x.shape[1]
  w = np.exp( - (newx[1] - x[:, 1]) ** 2 / (2 * bw **2))
  wx = np.dot(x.T, np.diag(w))
  wxy = np.dot(wx, y)
  theta = np.dot(np.linalg.inv(np.dot(wx, x)), wxy)
  est = np.dot(theta.reshape((1, d)), newx)
  return est
def local_linear_regression(x, y, newx = None, bw = 5):
  '''Implemented local linear regression to smooth the given signal input
  @param: x: a (m, 2) matrix of inputs 
          y: a (m, ) vector of outcomes 
          newx: a (m, 2) matrix of new inputs
          bw: a scalar of bandwidth parameter in the weights' kernel
          function 
          w_{(i)} = exp{(x - x_{(i)})^2/(2 * \tau^2)}
  @return: the smooth curve for y given x
  '''
  m = x.shape[0]
  if newx == None:
    newx = x
  est = np.apply_along_axis(lwlrx, 1, newx, x = x, y= y, bw = bw)
  return np.squeeze(est)

def local_linear_regression_loop(x, y, newx = None, bw = 5):
  '''Implemented local weighted linear regression 
  @param: x: a (m, 2) matrix of inputs 
          y: a (m, ) vector of outcomes
          newx: a (m, 2) matrix of new inputs 
          bw: a scalar of bandwidth parameter in the weights' kernel
          function 
          w_{(i)} = exp{(x - x_{(i)})^2/(2 * \tau^2)}
  @return: the smooth curve for y given x
  '''
  m = x.shape[0]
 
  if newx == None:
    newx = x
  est = np.zeros((m, ))
  for i in range(m):
    w = np.exp(- (newx[i, 1] - x[:, 1]) ** 2/(2 * bw ** 2)).reshape((m, ))
    wx = np.dot(x.T, np.diag(w))
    wxy = np.dot(wx, y)
    theta = np.dot(np.linalg.inv(np.dot(wx, x)), wxy)
    est[i] = np.dot(theta.T, newx[i, :])

  return est

# load train dataset
train = pd.read_csv('http://cs229.stanford.edu/ps/ps1/quasar_train.csv')

# a single header row corresponds integral wavelengths in the interval [1150, 1600]
wavelength = train.columns.values.astype(float)

# extract the first set of relative flux spectrum for each given wavelength
y = train.loc[0, :].values 
m = train.shape[1]

# create an input matrix with a column of 1 for an intercept and wavelength values
X = np.hstack([np.ones((m, 1)), wavelength.reshape(m, 1)])

# fit linear regression to relative flux spectrum
theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y.reshape(m ,1)))
est_lin_y = np.dot(X, theta)

# load line-by-line profiling with %lprun
%load_ext line_profiler 

%lprun -f local_linear_regression_loop local_linear_regression_loop(X, y, bw = 1)
%lprun -f local_linear_regression local_linear_regression(X, y, bw = 1)

# fit a smooth curve to relative flux spectrum for given wavelength
est_y_1 = local_linear_regression(X, y, bw  = 1) 
est_y_5 = local_linear_regression(X, y, bw = 5) 
est_y_10 = local_linear_regression(X, y, bw = 10) 
est_y_100 = local_linear_regression(X, y, bw = 100) 
est_y_1000 = local_linear_regression(X, y, bw = 1000) 

%matplotlib
import matplotlib.pyplot as plt 

# plot raw data and smooth curves for different bandwidth size
plt.scatter(wavelength, y, label = 'Raw data')
plt.plot(wavelength, est_y_1, '-r', label = 'tau = 1')
plt.plot(wavelength, est_y_5, '--b', label = 'tau = 5')
plt.plot(wavelength, est_y_10, '-.g', label = 'tau = 10')
plt.plot(wavelength, est_y_100, ':c', label = 'tau = 100')
plt.plot(wavelength, est_y_1000, ':k', label = 'tau = 1000')
plt.plot(wavelength, est_lin_y, '--y', label = 'linear regression')

plt.legend(loc = 'upper right')

def squared_distance(f1, f2):
  d = np.sum((f1[:, np.newaxis, :] - f2[np.newaxis, :, :]) ** 2, axis = -1)
  return d

def functional_regression(f, newf, k = 3):
  n = newf.shape[0]

  d = np.sum(wavelength < 1200)
  f_right = f[:, wavelength >= 1300]
  f_left = f[:, wavelength < 1200]
  newf_right = newf[:, wavelength >= 1300]
  frdist = squared_distance(f_right, newf_right)
  nearest = np.argsort(frdist, axis = 0)[0:k, :]
  frdist /= np.max(frdist, axis = 0)
  frdist = 1 - frdist
  frdist[frdist < 0] = 0

  est_f_left = np.zeros((n, d))
  for i in range(n):
    nearest_idx = nearest[:, i]
    w = frdist[nearest_idx, i]
    w /= np.sum(w)
    # print(w)
    est_f_left[i, :] = np.sum(f_left[nearest_idx, :] * w[:, np.newaxis], axis = 0)

  return est_f_left

def average_error(outcome, pred):
  error = np.mean(np.sum((outcome - pred) ** 2, axis = 1))
  return error

# label the DataFrame index
train.index.name = 'key'

# apply local linear regression to each set of training examples
# %timeit smooth_curves = X.groupby('key').apply(lambda x: local_linear_regression(wavelength, x.values.reshape((m, ))))
smooth_curves = train.groupby('key').apply(lambda x: local_linear_regression(X, x.values.reshape((m, ))))

# convert series of arrays into 2D numpy array
s = np.array(smooth_curves.tolist())

estimated_f_left = functional_regression(s, s)

avg_train_error = average_error(s[:, wavelength < 1200], estimated_f_left)
print('Average train error:%f' % avg_train_error)

test = pd.read_csv('http://cs229.stanford.edu/ps/ps1/quasar_test.csv')

test.index.name = 'key'
test_smooth_curves = test.groupby('key').apply(lambda x: local_linear_regression(X, x.values.reshape((m, ))))

# convert a pandas series of array to 2-D numpy array
news = np.array(test_smooth_curves.tolist())

# perform functional regression on the test set
est_f_left_test = functional_regression(s, news)

avg_test_error = average_error(news[:, wavelength < 1200], est_f_left_test)
print('Average test error:%f' % avg_test_error)
# np.mean(squared_distance(est_f_left_test, news[:, wavelength < 1200]).diagonal())

plt.plot(wavelength, news[6, :])
plt.plot(wavelength[wavelength < 1200], est_f_left_test[6, :], '-r')