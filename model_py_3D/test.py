from grid import Grid
import numpy as np

xSize = 100
ySize = 100
zSize = 400
voxelSize = 1

# simulation geometry
grid_size = np.array([xSize, ySize, zSize])                           # size of the voxel grid
voxel_size = np.array([voxelSize, voxelSize, voxelSize])              # (mm)
offset = np.array([-50, -50, 0])                                      # coordinates of the first voxel (mm)
density_map = np.ones(grid_size) * 1.0                                # (g/cm3)
material_map = np.ones(grid_size) * 1                                 # 1 = water
scoring_grid = np.zeros(grid_size)                                    # initialize the dose distribution grid
sources = 40000

g = Grid(xSize, ySize, zSize, voxelSize, density_map, material_map, sources)
