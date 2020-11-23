import random
import sys

import numpy as np

from vector import treeVector
from matrix import treeMatrix

seed = random.randint(0, 10000)
random.seed(seed)


v = treeVector(10)
a = np.zeros((10,))
for _ in range(1000):
  index = random.randint(0, 9)
  value = random.random()
  v.set(index, value)
  a[index] = value
 
print(v.sample) 

m = treeMatrix(20, 10)
b = np.zeros((20, 10))
for _ in range(10000):
  row = random.randint(0, 19)
  col = random.randint(0, 9)
  value = random.random()
  m.set(row, col, value)
  b[row, col] = value

a = m.sample_row(2)