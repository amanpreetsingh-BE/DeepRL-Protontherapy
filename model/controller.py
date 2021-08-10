from grid_3d import Grid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from cell import HealthyCell, CancerCell
import random
from get_voxel_index import get_voxel_index

class Controller:

    def __init__(self, width, height, length, voxelSize, density, materialMap, numSources):
        """ 
            The controller will instantiate a grid of size width*height*length, 
            with initial glucose and oxygen,
            place randonmly numSources and numHealthyCells with random stage,
        """
        # Instantiate the 3D grid 
        self.grid = Grid(width, height, length, voxelSize, density, materialMap, numSources)        # 3D grid
        
        self.xSize = int(width/voxelSize)                                                           # Frontal axis
        self.ySize = int(height/voxelSize)                                                          # Longitudinal axis
        self.zSize = int(length/voxelSize)                                                          # Sagittal axis

        HealthyCell.cell_count = 0
        CancerCell.cell_count = 0

        self.tick = 0                                                                               # Number of ticks/iterations/hours
        self.numberCancerCells = []                                                                  # Number of cancer cells for each tick
        self.voxelsCancer = []                                                                      # voxels positions before treatment
        self.time = []                                                                              # time tracking
        self.meanTvDose = []                                                                         # keep track of max dose at TV

        # Place one healthy cell
        for k in range(self.zSize):
            for i in range(self.xSize):
                for j in range(self.ySize):
                    #if random.random() < prob:
                    new_cell = HealthyCell(random.randint(0, 4))
                    self.grid.cells[i, j, k].append(new_cell)

        # Place cancer cell at the center and count neighbors
        new_cell = CancerCell(random.randint(0, 3))
        self.grid.cells[self.xSize//2, self.ySize//2, self.zSize//2].append(new_cell)

        # Count neigh
        self.grid.count_neigbors()


    def go(self, steps=1, treatment=False):
        """ Simulates one hour (by default) on the grid : Nutrient diffusion and replenishment, cell cycle"""
        for i in range(steps):

            print('Tick : ', i)

            self.time.append(self.tick)
            self.numberCancerCells.append(self.grid.compute_total_c_cells())
            
            if treatment:
                self.meanTvDose.append(self.grid.compute_mean_tv_dose(self.voxelsCancer))
            else:
                self.meanTvDose.append(0)
            self.plot_tumorMass()
            self.plot_tumor_slicesXY()
            self.plot_glucose_sliceXY()
            self.plot_oxygen_sliceXY()
            self.grid.fill_source(130, 4500)
            self.grid.cycle_cells()
            self.grid.diffuse_glucose(0.3)
            self.grid.diffuse_oxygen(0.3)
            self.tick += 1
        
        if treatment == False:
            self.voxelsCancer = self.grid.compute_c_voxels_position()
            
            

    def irradiate(self, SIPP, SIPE, ES, MCS, PLOT, particlesNumber, sigmaX, sigmaY, sigmaE, Beam_position, Beam_direction, Beam_mean_energy, offset, step_length, center=None, rad=-1):
        """Irradiate XY plane, energy in MeV"""
        self.grid.irradiate(SIPP, SIPE, ES, MCS, PLOT, particlesNumber, sigmaX, sigmaY, sigmaE, Beam_position, Beam_direction, Beam_mean_energy, offset, step_length)
        if PLOT:
            self.plot_beam_slices(Beam_position, self.grid.voxelSize, offset)
            self.plot_beam_3D()


    def plot_tumorMass(self):
        """ Scatter cancer cells -> 3D volume """
        if self.tick % 75 == 0 : # every 75 hours, save figure

            density = np.empty((self.xSize, self.ySize, self.zSize))

            for k in range(zSize):
                for i in range(xSize):
                    for j in range(ySize):
                        if(self.grid.cells[i, j, k].num_c_cells != 0):
                            density[i, j, k] = self.grid.cells[i, j, k].num_c_cells
                        else:
                            density[i, j, k] = np.NaN

            ccmap = LinearSegmentedColormap.from_list('mycmap', ['red', 'darkred', 'black'])
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(40, -60)
            # Add x, y gridlines
            ax.grid(b = True, color ='grey',
                    linestyle ='-.', linewidth = 0.1,
                    alpha = 0.9)
            X, Y, Z = np.mgrid[:xSize, :ySize, :zSize]
            sctt = ax.scatter3D(X, Y, Z, c=density.ravel(), alpha = 1, marker ='o', cmap = ccmap, s=30)
            ax.set_xlabel('Width [mm]', fontweight ='bold')
            ax.set_ylabel('Height [mm]', fontweight ='bold')
            ax.set_zlabel('Depth [mm]', fontweight ='bold')
            fig.colorbar(sctt, ax = ax)
            ax.set_zlim(0, zSize)
            plt.xlim(0, xSize)
            plt.ylim(0, ySize)
            plt.title('Time step = '+str(self.tick)+' [h]', pad=40)
            plt.savefig('tumorMass@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
            plt.close()
        
    def plot_tumor_slicesXY(self):
        """ Plot XY planes from center of the tumor """
        if self.tick % 75 == 0: # every 75 hours, save figure
            fig, axs = plt.subplots(1, 5,figsize=(18,5))
            fig.subplots_adjust(left=0.03,bottom=0,right=0.99, top=1,wspace=0.37,hspace=0.21)
            fig.suptitle('Time step = '+str(self.tick)+' [h]')
            count = 0
            for offset in [-4, -2, 0, 2,4]:
                axs[count].imshow(
                            [[patch_type_color(self.grid.cells[i][j][int((self.zSize/2)+offset)]) for j in range(self.grid.ySize)] for i in
                            range(self.grid.xSize)])
                axs[count].set_title('@depth = '+str(int((self.zSize/2)+offset))+' [mm]')
                axs[count].set_xlabel('Width [mm]')
                axs[count].set_ylabel('Height [mm]')
                count +=1
            
            plt.savefig('xyPlanes@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
            plt.close()

    def plot_glucose_sliceXY(self):
        """ Plot glucose at middle depth (center of the tumor) every 75h """
        if self.tick % 75 == 0: # every 75 hours, save figure
            plt.figure()
            plt.imshow(self.grid.glucose[:,:,int(self.zSize//2)], cmap='inferno')
            plt.title('@time '+str(self.tick)+ ' [h]')
            plt.xlabel('Width [mm]')
            plt.ylabel('Height [mm]')
            plt.title('Time step = '+str(self.tick)+' [h]')
            plt.colorbar()
            plt.savefig('glucose@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
            plt.close()     

    def plot_oxygen_sliceXY(self):
        """ Plot oxygen at middle depth (center of the tumor) every 75h """
        if self.tick % 75 == 0: # every 75 hours, save figure
            plt.figure()
            plt.imshow(self.grid.glucose[:,:,int(self.zSize//2)], cmap='inferno')
            plt.title('@time '+str(self.tick)+ ' [h]')
            plt.xlabel('Width [mm]')
            plt.ylabel('Height [mm]')
            plt.title('Time step = '+str(self.tick)+' [h]')
            plt.colorbar()
            plt.savefig('oxygen@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
            plt.close()

    def plot_beam_slices(self, Beam_position, voxelSize, offset):
        """ Plot slices of beam at middle depth (center of the tumor) """
        voxel_size = np.array([voxelSize, voxelSize, voxelSize])          
        voxel_index = get_voxel_index(Beam_position, voxel_size, offset)
        voxel_volume = self.grid.voxelSize * self.grid.voxelSize * self.grid.voxelSize / 1000 # (cm3)
        dose = self.grid.scoring / (self.grid.density * voxel_volume) # (MeV / g)
        dose = dose * 1000 # (MeV / kg)
        dose = dose * 1.602176e-19 # (Gy = J/kg)

        plt.figure(figsize=(15,4))
        plt.subplot(1, 4, 1)
        plt.imshow(dose[:,:,int(round(self.grid.zSize/2))], cmap="jet")
        plt.title("Dose map (XY slice)")
        plt.xlabel("X (voxels)")
        plt.ylabel("Y (voxels)")

        plt.subplot(1, 4, 2)
        plt.imshow(dose[voxel_index[0],:,:], cmap="jet", aspect=4)
        plt.title("Dose map (YZ slice)")
        plt.xlabel("Z (voxels)")
        plt.ylabel("Y (voxels)")

        plt.subplot(1, 4, 3)
        plt.imshow(dose[:,voxel_index[1],:], cmap="jet", aspect=4)
        plt.title("Dose map (XZ slice)")
        plt.xlabel("Z (voxels)")
        plt.ylabel("X (voxels)")

        plt.subplot(1, 4, 4)
        z = np.arange(offset[2], self.grid.zSize*voxel_size[2], voxel_size[2])
        plt.plot(z,np.sum(dose, axis=(0,1)))
        plt.ylabel('Integrated dose (Gy = J/kg)')
        plt.xlabel('Depth (voxels)')
        plt.xlim(0.1, 200)
        plt.ylim(0, max(np.sum(dose, axis=(0,1))))
        plt.savefig('beam@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
        plt.close()

    def plot_beam_3D(self):
        """ Plot beam in 3D """       
        voxel_volume = self.grid.voxelSize * self.grid.voxelSize * self.grid.voxelSize / 1000 # (cm3)
        dose = self.grid.scoring / (self.grid.density * voxel_volume) # (MeV / g)
        dose = dose * 1000 # (MeV / kg)
        dose = dose * 1.602176e-19 # (Gy = J/kg)
        for k in range(dose.shape[2]):
            for i in range(dose.shape[0]):
                for j in range(dose.shape[1]):
                    if(dose[i, j, k] <= 0.0):
                        dose[i, j, k] = np.NaN

        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(40, -60)
        # Add x, y gridlines
        ax.grid(b = True, color ='grey',
                linestyle ='-.', linewidth = 0.1,
                alpha = 0.9)
        X, Y, Z = np.mgrid[:dose.shape[0], :dose.shape[1], :dose.shape[2]]
        sctt = ax.scatter3D(X, Y, Z, c=dose.ravel(), alpha = 0.8, marker ='.', cmap="jet", s=0.1)
        ax.set_xlabel('Width [mm]', fontweight ='bold')
        ax.set_ylabel('Height [mm]', fontweight ='bold')
        ax.set_zlabel('Depth [mm]', fontweight ='bold')
        fig.colorbar(sctt, ax = ax)
        ax.set_zlim(0, dose.shape[2])
        plt.xlim(0, dose.shape[0])
        plt.ylim(0, dose.shape[1])
        plt.savefig('beam3D.png')
        plt.close()

def patch_type_color(patch):
    """Color of voxel in function of density and type of cell (red -> cancer) ; (vert -> healthy)"""
    if len(patch) == 0:
        return 0, 0, 0
    else:
        if(patch[0].cell_type() == 1):
            return 0, 255 - int(1.15*(patch.size - patch.num_c_cells)), 0
        elif(patch[0].cell_type() == -1):
            return 255-int(1.15*patch.num_c_cells), 0, 0
        else:
            return patch[0].cell_color()

if __name__ == '__main__':
    random.seed(4775)

    # geometry [mmm]
    width = 30
    height = 30
    length = 100 
    voxelSize = 1

    xSize = int(width/voxelSize)
    ySize = int(height/voxelSize)
    zSize = int(length/voxelSize)

    # arround 1 voxel out of 2 is initialized with 1 healthy cell and 2% of voxels are sources
    numSources = int((xSize*ySize*zSize)*0.1)  

    # only one type of tissue, in practice a map of tissue (e.g 1 for water, 2 for bone, etc.)
    materialMap = np.ones((xSize, ySize, zSize), dtype=int)

    # water everywhere (1 g/cm3), in practice values depends on materialMap
    density = np.ones((xSize, ySize, zSize), dtype=float)

    controller = Controller(width, height, length, voxelSize, density, materialMap, numSources)
    controller.go(101) # cycles for 375 hours, capture every 75h

    for i in range(0,10):
        controller.irradiate(True, True, True, True, True, 500, 2, 2, 1, np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1.0]), 80.0, np.array([-15, -15, 0]), 1)
        controller.go(24, treatment=True)
        print('Day number ', i)

    fig, ax1 = plt.subplots()
    plt.title('Effect of irradiation and rest')

    color = 'tab:red'
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Number of cancer cells', color=color)
    ax1.plot(controller.time, controller.numberCancerCells, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Mean dose at TV', color=color)  # we already handled the x-label with ax1
    ax2.plot(controller.time, controller.meanTvDose, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("treatment.png")
    plt.close()
    # irradiate XY plane with 200 MeV protons beam