from cell import HealthyCell, CancerCell, OARCell, critical_oxygen_level, critical_glucose_level
import numpy as np
import random
import math
import scipy.special
import matplotlib.pyplot as plt


class CellList:
    """Used to hold lists of cells on each voxel while keeping cancer cells and healthy cells sorted
    """

    def __init__(self):
        self.size = 0 # total numbers of cells 
        self.num_c_cells = 0 # number of cancer cell => size - num_c_cells = num_h_cells
        self.cancer_cells = []
        self.healthy_cells = []

    def __iter__(self):
        """Needed to iterate on the list object"""
        self._iter_count = -1
        return self

    def __next__(self):
        """Needed to iterate on the list object"""
        self._iter_count += 1
        if self._iter_count < self.num_c_cells:
            return self.cancer_cells[self._iter_count]
        elif self._iter_count < self.size:
            return self.healthy_cells[self._iter_count - self.num_c_cells]
        else:
            raise StopIteration

    def append(self, cell):
        """Add a cell to the list, keep the API of a Python list"""
        if cell.cell_type() < 0:
            self.cancer_cells.append(cell)
            self.num_c_cells += 1
        else:
            self.healthy_cells.append(cell)
        self.size += 1

    def __len__(self):
        """Return the size of the list, keep the API of a Python list"""
        return self.size

    def __getitem__(self, key):
        if key < self.size:
            if key < self.num_c_cells:
                return self.cancer_cells[key]
            else:
                return self.healthy_cells[key - self.num_c_cells]
        else:
            raise IndexError

    def delete_dead(self):
        """Delete dead cells from the list"""
        self.cancer_cells = [cell for cell in self.cancer_cells if cell.alive]
        self.healthy_cells = [cell for cell in self.healthy_cells if cell.alive]
        self.num_c_cells = len(self.cancer_cells)
        self.size = self.num_c_cells + len(self.healthy_cells)

    def pixel_type(self):
        """Used for observation of types on the grid"""
        if self.size == 0:
            return 0
        elif self.num_c_cells:
            return -1
        else:
            return 1

    def pixel_density(self):
        """Used for observation of densities on the grid"""
        if self.num_c_cells:
            return - self.num_c_cells
        else:
            return self.size




class Grid:
    """ The grid is a 3D phantom of size xSize*ySize*zSize [mm3]. The voxels are cubic and of size defined by voxelSize [mm] 
    It is made out of 3 superimposed layers : one contains the CellLists for each voxel,
    one contains the glucose amount on each voxel and one contains the oxygen amount on each voxel.
    """

    def __init__(self, xSize, ySize, zSize, voxelSize, densityMap, materialMap, sources, oar=None):
        """Constructor of the Grid.

        Parameters :
        xSize : width
        ySize : length
        zSize : height
        voxelSize : size of a voxel (cubic)
        sources : Number of nutrient sources on the grid
        oar : Optional description of an OAR zone on the grid
        """

        self.xSize = xSize
        self.ySize = ySize
        self.zSize = zSize
        self.voxelSize = voxelSize
        self.voxelsNumber = (xSize*ySize*zSize)/voxelSize

        self.glucose = np.full((xSize, ySize, zSize), 100.0)
        self.oxygen = np.full((xSize, ySize, zSize), 1000.0)
        # Helpers are useful because diffusion cannot be done efficiently in place.
        # With a helper array of same shape, we can simply compute the result inside the other and alternate between
        # the arrays.

        self.cells = np.empty((xSize, ySize, zSize), dtype=object)
        for k in range(zSize):
            for i in range(xSize):
                for j in range(ySize):
                    self.cells[i, j, k] = CellList()

        self.num_sources = sources
        self.sources = random_sources(xSize, ySize, zSize, sources) # gets (x,y,z) of sources (int) placed random

        # Neigbor counts contain, for each voxel on the grid, the number of cells on neigboring pixels. They are useful
        # as HealthyCells only reproduce in case of low density. As these counts seldom change after a few hundred
        # simulated hours, it is more efficient to store them than simply recompute them for each pixel while cycling.
        self.neigh_counts = np.zeros((xSize, ySize, zSize), dtype=int)
        #Pixels at the limits of the grid have fewer neighbours

        # "Front" face
        for i in range(xSize): # up and down borders
            self.neigh_counts[i,0,0] += 4
            self.neigh_counts[i, ySize - 1,0] += 4
        for j in range(ySize): # left and right borders
            self.neigh_counts[0, j, 0] += 4 
            self.neigh_counts[xSize - 1, j, 0] += 4

        # "Back" face
        for i in range(xSize): # up and down borders
            self.neigh_counts[i,0,zSize-1] += 4
            self.neigh_counts[i, ySize - 1,zSize-1] += 4
        for j in range(ySize): # left and right borders
            self.neigh_counts[0, j, zSize-1] += 4
            self.neigh_counts[xSize - 1, j, zSize-1] += 4

        # "left" face
        for k in range(zSize): # up and down borders 
            self.neigh_counts[0, 0, k] += 4
            self.neigh_counts[0, ySize-1 ,k] += 4
        for j in range(ySize): # left and right borders 
            self.neigh_counts[0, j, 0] += 4
            self.neigh_counts[0, j, zSize-1] += 4
        
        # "right" face
        for k in range(zSize): # up and down borders 
            self.neigh_counts[xSize-1, 0, k] += 4
            self.neigh_counts[xSize-1, ySize-1 ,k] += 4
        for j in range(ySize): # left and right borders 
            self.neigh_counts[xSize-1, j, 0] += 4
            self.neigh_counts[xSize-1, j, zSize-1] += 4
        

        self.neigh_counts[0, 0, 0] -= 1
        self.neigh_counts[xSize-1, 0, 0] -= 1
        self.neigh_counts[0, 0, zSize-1] -= 1
        self.neigh_counts[xSize-1, ySize-1, 0] -= 1

        self.neigh_counts[0, 0, zSize-1] -= 1
        self.neigh_counts[xSize-1, 0, zSize-1] -= 1
        self.neigh_counts[0, ySize-1, zSize-1] -= 1
        self.neigh_counts[xSize-1, ySize-1, zSize-1] -= 1


        self.oar = oar

        self.center_x = self.xSize // 2
        self.center_y = self.ySize // 2
        self.center_z = self.zSize // 2
    
    def count_neigbors(self):
        """Compute the neigbour counts (the number of cells on neighbouring pixels) for each voxels"""
        
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):
                    self.neigh_counts[i, j, k] = sum(v for _, _, _, v in self.neighbors(i, j, k))

    def fill_source(self, glucose=0, oxygen=0):
        """Sources of nutrients are refilled."""
        for i in range(len(self.sources)):
            self.glucose[self.sources[i][0], self.sources[i][1], self.sources[i][2]] += glucose
            self.oxygen[self.sources[i][0], self.sources[i][1], self.sources[i][2]] += oxygen
            if random.randint(0, 23) == 0:
                self.sources[i] = self.source_move(self.sources[i][0], self.sources[i][1], self.sources[i][2])

    def source_move(self, x, y, z):
        """"Random walk of sources for angiogenesis"""
        if random.randint(0, 50000) < CancerCell.cell_count:  # Move towards tumour center
            if x < self.center_x:
                x += 1
            elif x > self.center_x:
                x -= 1
            if y < self.center_y:
                y += 1
            elif y > self.center_y:
                y -= 1
            if z < self.center_z:
                z += 1
            elif z > self.center_z:
                z -= 1
            return x, y, z
        else:
            return self.rand_neigh(x, y, z)

    def diffuse_glucose(self, drate):
        self.glucose = (1 - drate) * self.glucose + (0.1 * drate) * self.neighbors_glucose()

    def diffuse_oxygen(self, drate):
        self.oxygen = (1 - drate) * self.oxygen + (0.1 * drate) * self.neighbors_oxygen()
    
    def neighbors_glucose(self):
        #Roll array in every direction to diffuse
        down = np.roll(self.glucose, 1, axis=0)
        up = np.roll(self.glucose, -1, axis=0)
        right = np.roll(self.glucose, 1, axis=(0, 1))
        left = np.roll(self.glucose, -1, axis=(0, 1))
        down_right = np.roll(down, 1, axis=(0, 1))
        down_left = np.roll(down, -1, axis=(0, 1))
        up_right = np.roll(up, 1, axis=(0, 1))
        up_left = np.roll(up, -1, axis=(0, 1))

        front = np.roll(self.glucose, 1, axis = 2)
        back = np.roll(self.glucose, -1, axis = 2)

        for i in range(self.ySize):  # Down
            down[0, i, :] = 0
            down_left[0, i, :] = 0
            down_right[0, i, :] = 0
        for i in range(self.ySize):  # Up
            up[self.xSize - 1, i, :] = 0
            up_left[self.xSize - 1, i, :] = 0
            up_right[self.xSize - 1, i, :] = 0
        for i in range(self.xSize):  # Right
            right[i, 0, :] = 0
            down_right[i, 0, :] = 0
            up_right[i, 0, :] = 0
        for i in range(self.xSize):  # Left
            left[i, self.ySize - 1, :] = 0
            down_left[i, self.ySize - 1, :] = 0
            up_left[i, self.ySize - 1, :] = 0
        
        return down + up + right + left + down_left + down_right + up_left + up_right + front + back

    def neighbors_oxygen(self):
        # Roll array in every direction to diffuse
        down = np.roll(self.oxygen, 1, axis=0)
        up = np.roll(self.oxygen, -1, axis=0)
        right = np.roll(self.oxygen, 1, axis=(0, 1))
        left = np.roll(self.oxygen, -1, axis=(0, 1))
        down_right = np.roll(down, 1, axis=(0, 1))
        down_left = np.roll(down, -1, axis=(0, 1))
        up_right = np.roll(up, 1, axis=(0, 1))
        up_left = np.roll(up, -1, axis=(0, 1))

        front = np.roll(oxygen, 1, axis = 2)
        back = np.roll(oxygen, -1, axis = 2)
        
        for i in range(self.ySize):  # Down
            down[0, i, :] = 0
            down_left[0, i, :] = 0
            down_right[0, i, :] = 0
        for i in range(self.ySize):  # Up
            up[self.xSize - 1, i, :] = 0
            up_left[self.xSize - 1, i, :] = 0
            up_right[self.xSize - 1, i, :] = 0
        for i in range(self.xSize):  # Right
            right[i, 0, :] = 0
            down_right[i, 0, :] = 0
            up_right[i, 0, :] = 0
        for i in range(self.xSize):  # Left
            left[i, self.ySize - 1, :] = 0
            down_left[i, self.ySize - 1, :] = 0
            up_left[i, self.ySize - 1, :] = 0

        return down + up + right + left + down_left + down_right + up_left + up_right + back + front

 
    

















def conv(rad, x):
    denom = 3.8 # //sqrt(2) * 2.7
    return math.erf((rad - x)/denom) - math.erf((-rad - x) / denom)

def get_multiplicator(dose, radius):
    return dose / conv(14, 0)

def scale(radius, x, multiplicator):
    return multiplicator * conv(14, x * 10 / radius)


# Creates a list of random positions in the grid where the sources of nutrients (blood vessels) will be
def random_sources(xSize, ySize, zSize, number):
    src = []
    for _ in range(number):
        x = random.randint(0, xSize-1)
        y = random.randint(0, ySize-1)
        z = random.randint(0, zSize-1)
        if (x, y, z) not in src:
            src.append((x,y,z))
    return src
