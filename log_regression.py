import pandas as pd 
import numpy as np

X = pd.read_table('http://cs229.stanford.edu/ps/ps1/logistic_x.txt', 
                header = None,
                sep = '\s+', names = ['x1', 'x2'])

m = X.shape[0]

# add column of 1 correspondings to an intercept to the original X
X = np.hstack([np.ones((m, 1)), X])  

y = pd.read_table('http://cs229.stanford.edu/ps/ps1/logistic_y.txt', 
                header = None,
                sep = '\s+', names = ['y'])

# convert DataFrame to numpy.array and reshape to a vector
y = y.values.reshape((m, )) 

def log_regression(X, y, max_iters = 10):
  ''' Implement Newton-Raphson alg to estimate MLEs for logistic regression
  @param: X: a (N, 3) matrix of training inputs
          y: a vector of length N labels {-1, 1}
          max_iters: a maximum number of iterations to update
  @return: theta: a (3, ) vector of estimated parameters 
            ll: loglikelihood for logistic regression
  '''
  m = X.shape[0]
  d = X.shape[1]

  theta = np.zeros((d, ))

  for i in range(max_iters):
    margins = np.dot(X, theta) * y
    
    # The average of log likelihood
    ll = np.mean(np.log(1 + np.exp(-margins)))
    
    # Probability given inputs X and outcome y for each training case
    probs = 1/(1 + np.exp(margins))
    
    # Find the gradient
    grad = -(1/m) * np.sum(X.T * (probs * y), axis = 1) 

    # Find the Hessian matrix 
    H = (1/m) * np.dot(np.dot(X.T, np.diag(probs * (1 - probs))), X)
    
    # Update theta using N-R alg
    theta -= np.dot(np.linalg.inv(H), grad)
    
    print('At iter %d, the avg. empirical loss: %f' % (i, ll))
    print('The estimated theta:', theta)

  return theta, ll 

theta, ll = log_regression(X, y)

%matplotlib
import matplotlib.pyplot as plt 

df = pd.DataFrame(X[:, 1:3], columns = ['x1', 'x2'])
df.index = y
df.index.name = 'label'
groups = df.groupby('label')

markers = dict({1: 'o', -1:'+'})
for name, group in groups:
  plt.scatter(group.x1, group.x2, 
              marker = markers[name], 
              label = 'y = {}'.format(name))

x1 = np.sort(df['x1'].values)
x2 = -theta[0]/theta[2] - theta[1]/theta[2] * x1
# plot the linear boundary decision in x1, x2
plt.plot(x1, x2, '-r', label = 'Regression line')

plt.legend(loc = 'upper right')