# svm implementation

from settings import *
import numpy as np
import pandas as pd
from nb import convert_sparse_to_full


filenames = []

for filename in glob.iglob(os.path.join(data_path, 'spam'_data/MATRIX.TRAIN.'))

train = pd.read_table(os.path.join(data_path, 'spam_data/MATRIX.TRAIN'),
                      skiprows = 2,
                      sep = '\s+')

def svm(X, y):
  Z = K
  Z[Z > 0] = 1