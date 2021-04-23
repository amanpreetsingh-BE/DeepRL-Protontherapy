from math import floor, isnan
import numpy as np

def compute_distance_to_interfacee(position, direction, voxel_size, offset):

  position = position - offset

  distance_to_interface = np.array([0.0,0.0,0.0])

  for i in range(0,3):
    with np.errstate(divide='ignore', invalid='ignore'): # disable the warning due to division by zero
      distance = (( floor(position[i] / voxel_size[i]) + np.sign(direction[i])) * voxel_size[i] - position[i]) / direction[i]

    if isnan(distance):
      distance_to_interface[i] = np.inf
    else:
      distance_to_interface[i] = abs(distance)

  return min(distance_to_interface) + 10E-5
