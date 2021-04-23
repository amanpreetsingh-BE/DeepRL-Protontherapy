from math import floor
import numpy as np

def get_voxel_index(position, voxel_size, offset):
  index = np.array([0,0,0])
  index[0] = (floor((position[0] - offset[0]) / voxel_size[0]) + 0)
  index[1] = (floor((position[1] - offset[1]) / voxel_size[1]) + 0)
  index[2] = (floor((position[2] - offset[2]) / voxel_size[2]) + 0)
  return index
