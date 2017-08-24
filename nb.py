import pandas as pd 
import numpy as np

def convert_sparse_to_full(X, label_idx = None):
  '''Convert a sparse matrix to a full matrix
  @params: X - a (N, D) pandas DataFrame
               N represents a total number of documents/emails
               D represents a total number of words in the dictionary
          label_idx - a scalar represents the index of the column label
  @return: a (N, D) array of full matrix and a (N, ) vector of labels
  '''
  n_rows = X.shape[0]
  n_cols = X.shape[1]
  if label_idx != None:
    label = np.array(X.iloc[:, label_idx])
    X = X.drop(X.columns[label_idx], axis = 1)
  else:
    label = None
  full_matrix = np.zeros((n_rows, n_cols))
  
  for i in range(n_rows):

    temp = X.iloc[i, :]
    
    # extract the non NA values in the training set
    # -1 means the end of the line
    temp = temp[(temp != -1) & (temp.isnull() == False)]
    
    # extract a vector of incremental difference in identity 
    # between ordere words appear in the documents/emails
    pos = temp[np.arange(0, temp.shape[0], 2)].astype('int64')
    pos = np.cumsum(pos)

    print('Iter %d, range: (%d, %d)' % (i, np.min(pos), np.max(pos)))
    # words' frequency corresponding to its identity
    val = temp[np.arange(1, temp.shape[0], 2)]

    full_matrix[i, pos] = val

  return full_matrix, label

def naive_bayes(X, y):
  '''Implemented Naive Bayes classifier for multinom event model and Laplace smoothing
  @param: X - a (N, V) matrix of word counts - each row represents a document/email and
          each column represents a bag of in the dictionary. 
          y - a binary vector of length (N, ) - 1 means an email is spam and 0 otherwise
  @return: prob_word_neg - a (V, ) vector of word probabilities for negative cases
           prob_word_pos - a (V, ) vector of word probabilities for positive cases
           prob_class - a (2, ) vector of prior probabilities for each class
  '''
  n_words = X.shape[1]
  neg_bool = y == 0
  pos_bool = y == 1
  n_words_pos = np.sum(X[pos_bool, :])
  n_words_neg = np.sum(X[neg_bool, :])

  # apply Laplace smoothing for probabilities are significantly small
  prob_word_neg = (np.sum(X[neg_bool, :], axis = 0) + 1)/(n_words_neg  + n_words)
  prob_word_pos = (np.sum(X[np.invert(neg_bool),:], axis = 0) + 1)/(n_words_pos + n_words)
  
  val, count = np.unique(y, return_counts = True)
  prob_class = count/np.sum(count)
  return prob_word_neg, prob_word_pos, prob_class

def logl(X, pxy, py, axis = 1):
  '''Find loglikelihood for NBayes classifier in multinom event model
  @param: X - a (N, V) matrix of word counts
          pxy - a (V, ) vector of conditional probabilities of words given a doc/email's class
          py - a prior probability of a class
          axis - by default, 1 means fingding loglikehood at each observation
                 None means finding the overall loglikehood for the entire dataset
  '''
  ll = np.sum(X * np.log(pxy), axis = 1) + np.log(py) 
  if axis == None:
    ll = np.sum(ll)
  return ll


def predict(newX, prob_word_neg, prob_word_pos, prob_class):
  '''Predict the class for NBayes classifier using MAP
  @param: newX - a (M, V) matrix of word counts
          prob_word_neg - a (V, ) vector of conditional prob. of words for a negative case
          prob_word_pos - a (V, ) vector of conditional prob. of words for a positive case
          prob_class - a (2, ) vector of prior prob. of classes
  @return: a (M, ) vector of predicted classes 
  '''
  ll_neg = logl(newX, prob_word_neg, prob_class[0], axis = 1)
  ll_pos = logl(newX, prob_word_pos, prob_class[1], axis = 1)
  pred = np.zeros(newX.shape[0])
  pred[ll_pos > ll_neg] = 1
  return pred

# read the training set, skip the first two rows, separate columns by spaces
train = pd.read_table('spam_data/MATRIX.TRAIN',
                      skiprows = 2,
                      sep = '\s+')

# read the test set, skip the first two rows, separate columns by spaces
test = pd.read_table('spam_data/MATRIX.TEST',
                      skiprows = 2,
                      sep = '\s+')

# convert sparse matrix to full matrix and return label for the training set
train_matrix, train_label = convert_sparse_to_full(train, label_idx = 0)

# convert a sparse matrix to full matrix and return label for the test set
test_matrix, test_label = convert_sparse_to_full(test, label_idx = 0)

# estimate prob. of words for each classes
phi_neg, phi_pos, prob_class = naive_bayes(train_matrix, label)  

# predict classes for test cases
test_preds = predict(test_matrix, phi_neg, phi_pos, prob_class)
test_error = np.mean(test_preds != test_label)
print('Test error: %f' % (test_error))

