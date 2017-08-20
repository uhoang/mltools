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

# plot raw data and smooth curves for different bandwidth size
plt.scatter(x, y, label = 'Raw data')
plt.plot(x, est_y_1, '-r', label = 'tau = 1')
plt.plot(x, est_y_5, '--b', label = 'tau = 5')
plt.plot(x, est_y_10, '-.g', label = 'tau = 10')
plt.plot(x, est_y_100, ':c', label = 'tau = 100')
plt.plot(x, est_y_1000, ':k', label = 'tau = 1000')
plt.plot(x, est_lin_y, '--y', label = 'linear regression')

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
  frdist = sqaured_distance(f_right, newf_right)
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

# get the wavelength interval
wavelength = X.columns.values.astype(float)
# label the DataFrame index
X.index.name = 'key'

m = len(wavelength)
# apply local_weightd_linear_regression to each row in training dataset
# %timeit smooth_curves = X.groupby('key').apply(lambda x: local_weighted_linear_regression(wavelength, x.values.reshape((m, ))))
smooth_curves = X.groupby('key').apply(lambda x: local_weighted_linear_regression(wavelength, x.values.reshape((m, ))))

# convert series of arrays into 2D numpy array
s = np.array(smooth_curves.tolist())

estimated_f_left = functional_regression(s, s)

avg_train_error = average_error(s[:, wavelength < 1200], estimated_f_left)
print('Average train error:%f' % avg_train_error)

newX = pd.read_csv('http://cs229.stanford.edu/ps/ps1/quasar_test.csv')

newX.index.name = 'key'
# a single header row corresponds integral wavelengths in the interval [1150, 1600]
wavelength = newX.columns.values.astype(float)
n = newX.shape[1]

# find a smooth curve for each spectrum in a test dataset
new_smooth_spectra = newX.groupby('key').apply(lambda x: local_weighted_linear_regression(wavelength, x.values.reshape((n, ))))

# convert a pandas series of array to 2-D numpy array
news = np.array(new_smooth_spectra.tolist())

# perform functional regression on the test set
est_f_left_test = functional_regression(s, news)

avg_test_error = average_error(news[:, wavelength < 1200], est_f_left_test)
print('Average test error:%f' % avg_test_error)
# np.mean(squared_distance(est_f_left_test, news[:, wavelength < 1200]).diagonal())