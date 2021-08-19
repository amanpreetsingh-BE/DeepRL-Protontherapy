'''
    3D GRID -> implement cell proliferations (cell cycle), proton transport (irradiation survival probability)
    @author : singh 
    The entire file is based on Moreau's work (2D version): https://github.com/gregoire-moreau/radio_rl
    Link to my repo : https://github.com/amanpreetsingh-BE/DeepRL-Protontherapy
'''

import numpy as np
import random

from model.cell import HealthyCell, CancerCell
from model.get_voxel_index import get_voxel_index
from model.cdirect import cdirect

# constants (same naming as Fippel document) for proton transport

m_p = 938.272 # [MeV]
m_e = 0.511 # [MeV]
m_o = 16.0 * m_p # [MeV]
r_e = 2.818E-12 # [mm]
z = 1.0 # [/]
X_w = 360.86 # [mm]
Es = 6 # simulation parameter 
Tmin_e = float('inf') # [MeV]

class CellList:
    """ Used to hold lists of cells on each voxel while keeping cancer cells and healthy cells sorted """

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
        """Delete dead cells from the list and update size"""
        self.cancer_cells = [cell for cell in self.cancer_cells if cell.alive]
        self.healthy_cells = [cell for cell in self.healthy_cells if cell.alive]
        self.num_c_cells = len(self.cancer_cells)
        self.size = self.num_c_cells + len(self.healthy_cells)

    def pixel_type(self):
        """Used for observation of types on the grid"""
        if self.size == 0: # if there is no cell 
            return 0
        elif self.num_c_cells: # if number of cancers cells > 1
            return -1
        else:
            return 1

    def pixel_density(self):
        """Used for observation of densities on the grid"""
        if self.num_c_cells:
            return - self.num_c_cells
        else:
            return self.size


class Particle:
    """Proton particle defined by state values """

    def __init__(self, position, direction , energy):
        self.position = position
        self.direction = direction
        self.energy = energy

class Grid:
    """ 
        The grid is a 3D phantom of size xSize*ySize*zSize [mm3]. The voxels are cubic and of size defined by voxelSize [mm] 
        Each voxel contains [CellList, glucose, oxygen, density, scoring]
    """

    def __init__(self, width, height, length, voxelSize, density, materialMap, numSources):
        """ Constructor of the Grid.
            Parameters :
                width : in mm
                height : in mm
                length : in mm
                voxelSize : size of a voxel (always cubic !) in mm, i.e resolution
                densityMap : matrix of size [width, height, length]/voxelSize representing density of tissues for each voxel
                materialMap : materialMap[i,j,k] is a integer representing a tissue, for accessing SPdatabase
                numSources : number of nutrient sources on the grid
        """

        # Geometry parameters :
        self.voxelSize = voxelSize                                                     # Size of voxel
        self.xSize = int(width/voxelSize)                                              # Frontal axis
        self.ySize = int(height/voxelSize)                                             # Longitudinal axis
        self.zSize = int(length/voxelSize)                                             # Sagittal axis
        self.voxelNumbers = self.xSize*self.ySize*self.zSize                           # Number of voxels
        
        # Voxel's data :
        self.glucose = np.full((self.xSize, self.ySize, self.zSize), 100.0)            # oxygen amount of a voxel, with initial scaled value 100
        self.oxygen = np.full((self.xSize, self.ySize, self.zSize), 1000.0)            # glucose amount of a voxel, with initial scaled value 1000
        self.cells = np.empty((self.xSize, self.ySize, self.zSize), dtype=object)      # keep track of cells in a voxel, initialize with empty lists
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):
                    self.cells[i, j, k] = CellList()
        self.scoring = np.zeros((self.xSize, self.ySize, self.zSize), dtype=float)     # keep track of deposited energy in a voxel by proton [MeV]
        self.density = density                                                         # density (bone, water, ... ) matrix of voxels [g/cm3]
        self.materialMap = materialMap                                                 # materialMap[i,j,k] = integer representing a tissue, for accesing SPdatabase
        self.doseMap = None                                                            # keep track of deposited dose in a voxel by proton [MeV]
        
        # Placement of random sources representing endothelial cells :
        self.numSources = numSources
        self.sources = random_sources(self.xSize, self.ySize, self.zSize, numSources)  # list of (x,y,z) positions of numSources placed random

        # Precompute neighboor cells
        self.neigh_counts = np.zeros((self.xSize, self.ySize, self.zSize), dtype=int)  # neigh_counts contains for each voxel on the grid, 
                                                                                       # the number of cells on neigboring pixels

        # "Front" face
        for i in range(self.xSize): # up and down borders
            self.neigh_counts[i,0,0] += 4
            self.neigh_counts[i, self.ySize - 1,0] += 4
        for j in range(self.ySize): # left and right borders
            self.neigh_counts[0, j, 0] += 4 
            self.neigh_counts[self.xSize - 1, j, 0] += 4

        # "Back" face
        for i in range(self.xSize): # up and down borders
            self.neigh_counts[i,0,self.zSize-1] += 4
            self.neigh_counts[i, self.ySize - 1,self.zSize-1] += 4
        for j in range(self.ySize): # left and right borders
            self.neigh_counts[0, j, self.zSize-1] += 4
            self.neigh_counts[self.xSize - 1, j, self.zSize-1] += 4

        # "left" face
        for k in range(self.zSize): # up and down borders 
            self.neigh_counts[0, 0, k] += 4
            self.neigh_counts[0, self.ySize-1 ,k] += 4
        for j in range(self.ySize): # left and right borders 
            self.neigh_counts[0, j, 0] += 4
            self.neigh_counts[0, j, self.zSize-1] += 4
        
        # "right" face
        for k in range(self.zSize): # up and down borders 
            self.neigh_counts[self.xSize-1, 0, k] += 4
            self.neigh_counts[self.xSize-1, self.ySize-1 ,k] += 4
        for j in range(self.ySize): # left and right borders 
            self.neigh_counts[self.xSize-1, j, 0] += 4
            self.neigh_counts[self.xSize-1, j, self.zSize-1] += 4
        
        self.neigh_counts[0, 0, 0] -= 1
        self.neigh_counts[self.xSize-1, 0, 0] -= 1
        self.neigh_counts[0, 0, self.zSize-1] -= 1
        self.neigh_counts[self.xSize-1, self.ySize-1, 0] -= 1

        self.neigh_counts[0, 0, self.zSize-1] -= 1
        self.neigh_counts[self.xSize-1, 0, self.zSize-1] -= 1
        self.neigh_counts[0, self.ySize-1, self.zSize-1] -= 1
        self.neigh_counts[self.xSize-1, self.ySize-1, self.zSize-1] -= 1

        # Compute center of the tumor (for tumor placement, cancer cell at the center initially)
        self.centerX = self.xSize // 2
        self.centerY = self.ySize // 2
        self.centerZ = self.zSize // 2

        # Generate stopping power database (!! path depends on emplacement of running script)
        self.SP_water = np.loadtxt("DeepRL-PT/model/SPdata/SP_water.txt", 'float', '#', None, None, 8)
        self.SP_bone = np.loadtxt("DeepRL-PT/model/SPdata/SP_bone.txt", 'float', '#', None, None, 8) # ADDED ONE FOR TESTING
        self.SP_database = []
        self.SP_database.append([]) # index 0 is empty to be consistant with matlab code
        self.SP_database.append(self.SP_water) # index 1 = water print(SP_database[1][x,y])
        self.SP_database.append(self.SP_bone) # index 2 = bone print(SP_database[2][x,y])
    
    #### cells proliferation functions related : 
    
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
            if x < self.centerX:
                x += 1
            elif x > self.centerX:
                x -= 1
            if y < self.centerY:
                y += 1
            elif y > self.centerY:
                y -= 1
            if z < self.centerZ:
                z += 1
            elif z > self.centerZ:
                z -= 1
            return x, y, z
        else:
            return self.rand_neigh(x, y, z)

    def rand_neigh(self, x, y, z):
        """ take a random voxel neighbor of (x,y,z) """
        ind = []
        for (i, j, k) in [(x - 1, y - 1,z), (x - 1, y,z), (x - 1, y + 1,z), (x, y - 1,z), (x, y + 1,z), (x + 1, y - 1,z), (x + 1, y,z),
                        (x + 1, y + 1,z),(x - 1, y - 1,z-1), (x - 1, y,z-1), (x - 1, y + 1,z-1), (x, y - 1,z-1), (x, y + 1,z-1), (x + 1, y - 1,z-1), (x + 1, y,z-1),
                        (x + 1, y + 1,z-1),(x - 1, y - 1,z+1), (x - 1, y,z+1), (x - 1, y + 1,z+1), (x, y - 1,z+1), (x, y + 1,z+1), (x + 1, y - 1,z+1), (x + 1, y,z+1),
                        (x + 1, y + 1,z+1)]:
            if (i >= 0 and i < self.xSize and j >= 0 and j < self.ySize and k >=0 and k < self.zSize):
                    ind.append((i, j, k))
        
        return random.choice(ind)

    def diffuse_glucose(self, drate):
        """ diffusion of glucose """
        self.glucose = (1 - drate) * self.glucose + (0.083 * drate) * self.neighbors_glucose()

    def diffuse_oxygen(self, drate):
        """ diffusion of oxygen """
        self.oxygen = (1 - drate) * self.oxygen + (0.083 * drate) * self.neighbors_oxygen()
    
    def neighbors_glucose(self):
        """ utility function for diffusion of glucose """
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
        """ utility function for diffusion of oxygen """
        down = np.roll(self.oxygen, 1, axis=0)
        up = np.roll(self.oxygen, -1, axis=0)
        right = np.roll(self.oxygen, 1, axis=(0, 1))
        left = np.roll(self.oxygen, -1, axis=(0, 1))
        down_right = np.roll(down, 1, axis=(0, 1))
        down_left = np.roll(down, -1, axis=(0, 1))
        up_right = np.roll(up, 1, axis=(0, 1))
        up_left = np.roll(up, -1, axis=(0, 1))

        front = np.roll(self.oxygen, 1, axis = 2)
        back = np.roll(self.oxygen, -1, axis = 2)
        
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
    
    def count_neigbors(self):
        """Compute the neigbour counts (the number of cells on neighbouring pixels) for each voxel"""
        for k in range (self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):
                    self.neigh_counts[i, j, k] = sum(v for _, _, _, v in self.neighbors(i, j, k))

    def neighbors(self, x, y, z):
        """Return the positions of every valid voxel in the patch containing x, y, z and its neigbors, and their length"""
        neigh = []
        for (i, j, k) in [(x - 1, y - 1,z), (x - 1, y,z), (x - 1, y + 1,z), (x, y - 1,z), (x, y + 1,z), (x + 1, y - 1,z), (x + 1, y,z),
                        (x + 1, y + 1,z),(x - 1, y - 1,z-1), (x - 1, y,z-1), (x - 1, y + 1,z-1), (x, y - 1,z-1), (x, y + 1,z-1), (x + 1, y - 1,z-1), (x + 1, y,z-1),
                        (x + 1, y + 1,z-1),(x - 1, y - 1,z+1), (x - 1, y,z+1), (x - 1, y + 1,z+1), (x, y - 1,z+1), (x, y + 1,z+1), (x + 1, y - 1,z+1), (x + 1, y,z+1),
                        (x + 1, y + 1,z+1)]:
            if (i >= 0 and i < self.xSize and j >= 0 and j < self.ySize and k >=0 and k < self.zSize):
                neigh.append([i, j, k, len(self.cells[i, j, k])])
        return neigh

    def add_neigh_count(self, x, y, z, v):
        for (i, j, k) in [(x - 1, y - 1,z), (x - 1, y,z), (x - 1, y + 1,z), (x, y - 1,z), (x, y + 1,z), (x + 1, y - 1,z), (x + 1, y,z),
                        (x + 1, y + 1,z),(x - 1, y - 1,z-1), (x - 1, y,z-1), (x - 1, y + 1,z-1), (x, y - 1,z-1), (x, y + 1,z-1), (x + 1, y - 1,z-1), (x + 1, y,z-1),
                        (x + 1, y + 1,z-1),(x - 1, y - 1,z+1), (x - 1, y,z+1), (x - 1, y + 1,z+1), (x, y - 1,z+1), (x, y + 1,z+1), (x + 1, y - 1,z+1), (x + 1, y,z+1),
                        (x + 1, y + 1,z+1)]:
            if (i >= 0 and i < self.xSize and j >= 0 and j < self.ySize and k >=0 and k < self.zSize):
                self.neigh_counts[i, j, k] += v
       
    def rand_min(self, x, y,z, max):
        """ Returns the index of one of the neighboring patches with the lowest density of cells """
        v = 1000000
        ind = []
        for (i, j, k) in [(x - 1, y - 1,z), (x - 1, y,z), (x - 1, y + 1,z), (x, y - 1,z), (x, y + 1,z), (x + 1, y - 1,z), (x + 1, y,z),
                        (x + 1, y + 1,z),(x - 1, y - 1,z-1), (x - 1, y,z-1), (x - 1, y + 1,z-1), (x, y - 1,z-1), (x, y + 1,z-1), (x + 1, y - 1,z-1), (x + 1, y,z-1),
                        (x + 1, y + 1,z-1),(x - 1, y - 1,z+1), (x - 1, y,z+1), (x - 1, y + 1,z+1), (x, y - 1,z+1), (x, y + 1,z+1), (x + 1, y - 1,z+1), (x + 1, y,z+1),
                        (x + 1, y + 1,z+1)]:
            if (i >= 0 and i < self.xSize and j >= 0 and j < self.ySize and k >=0 and k < self.zSize):
                if len(self.cells[i, j, k]) < v:
                    v = len(self.cells[i, j, k])
                    ind = [(i, j, k)]
                elif len(self.cells[i, j,k]) == v:
                    ind.append((i, j, k))
        return random.choice(ind) if v < max else None
    
    
    def cycle_cells(self):
        """Feed every cell, handle mitosis"""
        to_add = []
        tot_count = 0
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):  # For every voxel
                    for cell in self.cells[i, j, k]:
                        res = cell.cycle(self.glucose[i, j, k],  self.neigh_counts[i, j, k], self.oxygen[i, j, k])
                        tot_count += 1
                        if len(res) > 2:  # If there are more than two arguments, a new cell must be created
                            if res[2] == 0: # Mitosis of a healthy cell
                                downhill = self.rand_min(i, j, k, 5)
                                if downhill is not None:
                                    to_add.append((downhill[0], downhill[1], downhill[2], HealthyCell(4)))
                                else:
                                    cell.stage = 4
                            elif res[2] == 1:  # Mitosis of a cancer cell
                                downhill = self.rand_neigh(i, j, k)
                                if downhill is not None:
                                    to_add.append((downhill[0], downhill[1], downhill[2], CancerCell(0)))
                        self.glucose[i, j, k] -= res[0]  # The local variables are updated according to the cell's consumption
                        self.oxygen[i, j, k] -= res[1]
                    count = len(self.cells[i, j, k])
                    self.cells[i, j, k].delete_dead()
                    if len(self.cells[i, j, k]) < count:
                        self.add_neigh_count(i, j, k, len(self.cells[i, j, k]) - count)
        for i, j, k, cell in to_add:
            self.cells[i, j, k].append(cell)
            self.add_neigh_count(i, j, k, 1)
        return tot_count

    def compute_total_c_cells(self):
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):  # For every voxel
                    if self.cells[i, j, k].pixel_type() == -1  :
                        return self.cells[i, j, k].cancer_cells[0].cell_count
        return 0

    def compute_c_voxels_position(self):
        voxelsCancer = []
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):  # For every voxel
                    if self.cells[i, j, k].pixel_type() == -1  :
                        voxelsCancer.append((i,j,k))
        return voxelsCancer

    def compute_mean_tv_dose(self, voxelsCancer):
        doseTV = []
        self.compute_doseMap(500)

        for l in range(0,len(voxelsCancer)):
            doseTV.append(self.doseMap[voxelsCancer[l][0], voxelsCancer[l][1], voxelsCancer[l][2]]) # for each cancer voxel, append the dose
                        
        return np.sum(doseTV)/len(doseTV) # return max dose at tv

    #### proton transport functions related : 

    def omega_square(self, r_e, m_e, z, T_p, m_p, s, Tmin_e, voxel_index):
        """compute variance for energy straggling"""
        rho = self.density[voxel_index[0], voxel_index[1], voxel_index[2]]
        if(rho == 1.0):
            n_el = (6.022E23)*(1/18015.3)* (10) # [#e-/mm3]
        else :
            n_el = (6.022E23)*(1/12010)* (6)
        return 2*np.pi*np.square(r_e)*m_e*n_el*np.square(z/beta(T_p, m_p))*s*Tmax_e(T_p,m_p,m_e)*(1-(np.square(beta(T_p, m_p))/2))

    def polar_angle(self, T_p, m_p, Es, z, s, voxel_index):
        """compute deflection angle for MCS"""
        pc = np.sqrt(np.square(T_p+m_p)-np.square(m_p))
        rho = self.density[voxel_index[0], voxel_index[1], voxel_index[2]]
        if(rho != 1.0): # for different than water
            f_xO = 1.19 + 0.44*np.log(rho-0.44)
            X = ((1*X_w)/(f_xO*rho))*10
        else: # for wate
            X = X_w # [mm]
        return (((Es)/(beta(T_p,m_p)*pc))*z*np.sqrt(s/X))

    def em_interactions(self, particle, ES, MCS, voxel_index, stoppingPower, s) :
        """energy loss calculation"""
        if ES : # Energy Straggling 
            energyStraggling =  np.random.normal(0,np.sqrt(self.omega_square(r_e, m_e, z, particle.energy, m_p, s, Tmin_e, voxel_index)))
        else :
            energyStraggling = 0
            
        dE = ((self.density[voxel_index[0], voxel_index[1], voxel_index[2]])*(stoppingPower)*(s/10)) + energyStraggling

        if (dE > particle.energy) : # in case of last step
            dE = particle.energy
            particle.energy = particle.energy - dE
        else :
            particle.energy = particle.energy - dE
            
        if MCS : # Multiple Coulomb Scattering
            self.scoring[voxel_index[0], voxel_index[1], voxel_index[2]] += dE
            if(particle.energy != 0): # otherwise theta = inf or nan bad values if energy = 0 (last step)
                theta = self.polar_angle(particle.energy, m_p, Es, z, s, voxel_index)
                particle.direction = cdirect(np.cos(np.random.normal(0,theta)), particle.direction[0], particle.direction[1], particle.direction[2])
        else :
            self.scoring[voxel_index[0], voxel_index[1], voxel_index[2]] += dE

    def computeScoring(self, SIPP, SIPE, ES, MCS, particlesNumber, sigmaX, sigmaY, sigmaE, Beam_position, Beam_direction, Beam_mean_energy, offset, step_length) : 
        for i in range(0, particlesNumber): # for each particle of the beam 

            # Initialize a particle with position, direction and energy = STATE of particle i 
            particle = Particle(Beam_position.copy(), Beam_direction.copy(), Beam_mean_energy)

            if SIPP : # Sampling Initial Particle Position
                particle.position[0] = np.random.normal(particle.position[0], sigmaX)
                particle.position[1] = np.random.normal(particle.position[1], sigmaY)
            if SIPE : # Sampling Initial Particle Energy
                particle.energy = np.random.normal(particle.energy, sigmaE)
            #print('CSDA simulation of particle ', (i+1))

            while (particle.energy > 0) : # CSDA
                voxel_index = get_voxel_index(particle.position, np.array([self.voxelSize, self.voxelSize, self.voxelSize]), offset)
                # sanity overflow check
                if(voxel_index[0] >= self.xSize or voxel_index[0] <= -self.xSize or voxel_index[1] >= self.ySize  or voxel_index[1] <= -self.ySize  or voxel_index[2] >= self.zSize  or voxel_index[2] <= -self.zSize ) :
                    break
                stoppingPower = np.interp(particle.energy, self.SP_database[int(self.materialMap[voxel_index[0],voxel_index[1],voxel_index[2]])][:,0], self.SP_database[int(self.materialMap[voxel_index[0],voxel_index[1],voxel_index[2]])][:,1])

                self.em_interactions(particle, ES, MCS, voxel_index, stoppingPower, step_length)
                particle.position = particle.position + ((particle.direction)*(step_length))
            
    def compute_doseMap(self, particlesNumber):
        # Convert scored energy into dose with Gray units
        voxel_volume = self.voxelSize * self.voxelSize * self.voxelSize / 1000 # (cm3)
        self.doseMap = self.scoring / (self.density * voxel_volume) # (MeV / g)
        self.doseMap = self.doseMap * 1000 # (MeV / kg)
        self.doseMap = self.doseMap * 1.602176e-19 # (Gy = J/kg) # mean dose per proton
        self.doseMap = self.doseMap*particlesNumber*1e9 # total dose, adjusted for E9 particles 

    def irradiate(self, SIPP, SIPE, ES, MCS, PLOT, particlesNumber, sigmaX, sigmaY, sigmaE, Beam_position, Beam_direction, Beam_mean_energy, offset, step_length, center=None, rad=-1):
        
        # Compute scoring grid 
        self.computeScoring(SIPP, SIPE, ES, MCS, particlesNumber, sigmaX, sigmaY, sigmaE, Beam_position, Beam_direction, Beam_mean_energy, offset, step_length)
        self.compute_doseMap(particlesNumber)

        oer_m = 3.0
        k_m = 3.0
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):
                    omf = (self.oxygen[i, j, k] / 100.0 * oer_m + k_m) / (self.oxygen[i, j, k] / 100.0 + k_m) / oer_m
                    for cell in self.cells[i, j, k]:
                        cell.radiate(self.doseMap[i, j, k]* omf)
                    count = len(self.cells[i, j, k])
                    self.cells[i, j, k].delete_dead()
                    if len(self.cells[i, j, k]) < count:
                        self.add_neigh_count(i, j, k, len(self.cells[i, j, k]) - count)

def random_sources(xSize, ySize, zSize, number):
    """ returns a list of random positions in the grid where the sources of nutrients (blood vessels) will be """
    src = []
    for _ in range(number):
        x = random.randint(0, xSize-1)
        y = random.randint(0, ySize-1)
        z = random.randint(0, zSize-1)
        if (x, y, z) not in src:
            src.append((x,y,z))
    return src

# utility functions (same naming as Fippel document) for transport functions
def gamma(T_p, m_p) :
    return (T_p+m_p)/m_p

def beta(T_p, m_p) :
    return np.sqrt(1-(1/np.square(gamma(T_p,m_p))))

def Tmax_e(T_p, m_p, m_e) :
    return (2*m_e*np.square(beta(T_p, m_p))*np.square(gamma(T_p, m_p)))/(1+((2*gamma(T_p, m_p))*(m_e/m_p))+np.square(m_e/m_p))


