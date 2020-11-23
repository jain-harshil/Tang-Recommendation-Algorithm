import math

from vector import treeVector

class treeMatrix(object):
  def __init__(self, rows, cols):
    self._rows = rows
    self._cols = cols
    self._row_norms = treeVector(rows)
    self._row_trees = [treeVector(cols) for _ in range(rows)]

  def get(self, row, col):
    return self._row_trees[row].get(col)

  def set(self, row, col, value):
    row_tree = self._row_trees[row]
    row_tree.set(col, value)
    self._row_norms.set(row, math.sqrt(row_tree.norm2))

  def get_row_norm(self, row):
    return self._row_norms.get(row)

  def sample_row_norms(self):
    return self._row_norms.sample()

  def sample_row(self, row):
    print(self._row_trees[row].sample)
    return self._row_trees[row].sample()
    
  @property
  def frob_norm2(self):
    return self._row_norms.norm2