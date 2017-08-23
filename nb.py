
import pandas as pd 
import numpy as np
def convert_word_count(x):
  x = x[(x != -1) & (x.isnull() == False)]
  l = x.shape[0]
  pos = x[np.arange(0, l, 2)]
  pos = np.cumsum(pos) + 1
  val = x[np.arange(1, l, 2)]
  return pos, val

train = pd.read_table('spam_data/MATRIX.TRAIN',
                      skiprows = 2,
                      sep = '\s+')

tokenlist = train.columns.values

trainCategory = np.zeros((train.shape[0],))

num_words = train.shape[1] + 1
num_docs = train.shape[0]

fmatrix = np.zeros((num_docs, num_words))

for i in range(num_docs):
  print('Iter:%d' % i)
  x = train.iloc[i, 1:(num_words + 1)]
  pos, val = convert_word_count(x)
  pos = pos.astype('int64')
  fmatrix[i, pos] = val
  trainCategory[i] = train.iloc[i, 0]



# train = train.fillna(0)