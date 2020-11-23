import random

class _treeVectorBase(object):
  def set(self, index, value):
    pass

  def get(self, index):
    pass

  def sample(self):
    pass

  @property
  def norm2(self):
    return self._norm2

class _treeVectorLeaf(_treeVectorBase):
  def __init__(self):
    self.value = 0.0
    self._norm2 = 0.0
    self._dim = 1

  def set(self, index, value):
    self.value = float(value)
    self._update_norm2()

  def get(self, index):
    if index != 0:
      raise IndexError
    return self.value

  def sample(self):
    return 0

  def _update_norm2(self):
    self._norm2 = self.value**2

class _treeVectorNode(_treeVectorBase):
  def __init__(self, dim):
    self._dim = dim
    self._norm2 = 0.0
    self.left = None
    self.right = None

  def set(self, index, value):
    if index < self.cutoff:
      child_side = 'left'
      child_size = self.cutoff
      child_index = index
    else:
      child_side = 'right'
      child_size = self._dim - self.cutoff
      child_index = index - self.cutoff

    if self.__getattribute__(child_side) is None:
      self.__setattr__(child_side, treeVector(child_size))
    child = self.__getattribute__(child_side)
    child.set(child_index, value)
    if child.norm2 == 0.0:
      self.__setattr__(child_side, None)

    self._update_norm2()

  def get(self, index):
    if index >= self._dim:
      raise IndexError

    if index < self.cutoff:
      child_side = 'left'
      child_index = index
    else:
      child_side = 'right'
      child_index = index - self.cutoff

    child = self.__getattribute__(child_side)
    if child is None:
      return 0.0
    return child.get(child_index)

  def sample(self, seed=None):
    if self.norm2 == 0.0:
      raise ValueError("No nonzero entries are available")

    left_norm2 = self.left.norm2 if self.left is not None \
                 else 0.0

    if seed is not None:
      random.seed(seed)

    if random.uniform(0, self.norm2) < left_norm2:
      return self.left.sample()
  
    return self.cutoff + self.right.sample()

  def _update_norm2(self):
    self._norm2 = sum(child.norm2 for child in [self.left, self.right] if child is not None)

  @property
  def cutoff(self):
    return self._dim//2

def treeVector(dim):
  if dim == 1:
    return _treeVectorLeaf()
  else:
    return _treeVectorNode(dim)