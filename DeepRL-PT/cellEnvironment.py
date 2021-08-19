from deer.base_classes import Environment
from model.controller import Controller
from model.cell import CancerCell, HealthyCell
from model.grid_3d import Grid
import numpy as np
import cv2

class CellEnvironment(Environment):
    def __init__(self, obs_type, resize, reward, action_type, special_reward):
        """Constructor of the environment
        Parameters:
        obs_type : Type of observations provided to the agent (segmentation or densities)
        resize : True if the observations should be resized to 15 * 15 arrays
        reward : Type of reward function used ('dose' to minimize the total dose, 'killed' to maximize damage to cancer
                 cells while miniizing damage to healthy tissue and 'oar' to minimize damage to the Organ At Risk
        action_type : 'DQN' means that we have a discrete action domain and 'AC' means that it is continuous
        special_reward : True if the agent should receive a special reward at the end of the episode.
        """

        self.controller = Controller(30, 30, 50, 1, np.ones((30, 30, 50), dtype=float), np.ones((30, 30, 50), dtype=int), int((30*30*50)*0.1))
        self.controller.go(180)
        self.init_hcell_count = HealthyCell.cell_count
        self.obs_type = obs_type
        self.resize = resize
        self.reward = reward
        self.action_type = action_type
        self.special_reward = special_reward
        self.dose_map = self.controller.grid.doseMap
        self.end_type = ""

    def get_tick(self):
        return self.controller.tick

    def init_dose_map(self):
        self.dose_map = np.zeros((30, 30, 50), dtype=float)
        self.dataset = [[], [], [], []]
        self.dose_maps = []
        self.tumor_images = []

    def reset(self, mode):
        self.controller = Controller(30, 30, 50, 1, np.ones((30, 30, 50), dtype=float), np.ones((30, 30, 50), dtype=int), int((30*30*50)*0.1))
        self.controller.go(180)
        self.init_hcell_count = HealthyCell.cell_count # initHcells (cfr thesis)
        if mode == -1:
            self.verbose = False
        else :
            self.verbose = True

        self.total_dose = 0
        self.num_doses = 0
        self.radiation_h_killed = 0 # healthyCellsKilled (cfr thesis)

        if self.dose_map is not None:
            self.dose_maps.append((self.controller.tick - 180, np.copy(self.dose_map)))
            self.tumor_images.append((self.controller.tick - 180,
                                      self.controller.observeDensity()))
        return self.observe()

    def observe(self):
        """return observation array of type density or segmentation """
        if self.obs_type == 'densities':
            cells = (np.array(self.controller.observeDensity(), dtype=np.float32)) / 10.0
        else:
            cells = (np.array(self.controller.observeSegmentation(), dtype=np.float32) + 1.0) / 2.0 #  Obs from 0 to 1
        if self.resize:
            cells = cv2.resize(cells, dsize=(15,15,25), interpolation=cv2.INTER_CUBIC)
        return [cells]

    def act(self, action):

        listEnergy = np.arange(50,85,5)     # radiation energy 50->80 MeV
        listRest = np.arange(12,28,4)       # rest hours 12->24 hours
        listXPosition = np.arange(-5,10,5)  # x position 10 15 20 in mm (arround xCenter = 15)
        listYPosition = np.arange(-5,10,5)  # y position 10 15 20 in mm (arround yCenter = 15)
        
        meanBeamEnergy = 0
        rest = 0
        xPosition = 0
        yPosition = 0

        if self.action_type == 'DQN':
            meanBeamEnergy = listEnergy[action] # depends on agent
            rest = listRest[0] # fixed
            xPosition = listXPosition[1] # fixed
            yPosition = listYPosition[1] # fixed 
        else:
            meanBeamEnergy = listEnergy[0]+(listEnergy[len(listEnergy)-1]-listEnergy[0])*action[0] 
            rest = listRest[0]#int(round(listEnergy[0]+(listEnergy[len(listEnergy)-1]-listEnergy[0])*action[1]))
            xPosition = 2*action[2]-1             
            yPosition = 2*action[3]-1
        if self.verbose:
            print("Energy choosed, ", meanBeamEnergy)
       
        voxel_volume = (self.controller.grid.xSize*self.controller.grid.ySize*self.controller.grid.zSize) / 1000 # (cm3)
        dose = meanBeamEnergy / (self.controller.grid.density[0,0,0] * voxel_volume) # (MeV / g)
        dose = dose * 1000 # (MeV / kg)
        dose = dose * 1.602176e-19 # (Gy = J/kg) # mean dose per proton
        dose = dose*500*1e9 # total dose, adjusted for many particles 

        pre_hcell = HealthyCell.cell_count  #healthyCells before radiation
        pre_ccell = CancerCell.cell_count   #cancerCells before radiation
        self.total_dose += dose
        self.num_doses += 1 if dose > 0 else 0
        self.controller.irradiate(True, True, True, True, True, 500, 2, 2, 1, np.array([xPosition, yPosition, 0.0]), np.array([0, 0, 1.0]), meanBeamEnergy, np.array([-15, -15, 0]), 1)     
        self.radiation_h_killed += (pre_hcell - HealthyCell.cell_count)
        if self.dose_map is not None:
            self.dataset[0].append(self.controller.tick - 180)
            self.dataset[1].append((pre_ccell, CancerCell.cell_count, pre_hcell, HealthyCell.cell_count))
            self.dataset[2].append(meanBeamEnergy)
            self.dataset[3].append((xPosition, yPosition))
            self.dose_maps.append((self.controller.tick - 180, np.copy(self.dose_map)))
            self.tumor_images.append((self.controller.tick - 180, self.controller.observeDensity()))

        p_hcell = HealthyCell.cell_count
        p_ccell = CancerCell.cell_count
        self.controller.go(rest, treatment=True)
        post_hcell = HealthyCell.cell_count
        post_ccell = CancerCell.cell_count

        reward = self.adjust_reward(dose, pre_ccell - post_ccell, pre_hcell-min(post_hcell, p_hcell))
        if self.verbose:
                print("Radiation dose :", dose, "Gy ", "remaining :", post_ccell,  "time =", rest, "reward=", reward)
        return reward
    
    def adjust_reward(self, dose, ccell_killed, hcells_lost):
        if self.special_reward and self.inTerminalState() or False:
            if self.end_type == "L" or self.end_type == "T":
                return -1
            else:
                if self.reward == 'dose':
                    return - dose / 400 + 0.5 - (self.init_hcell_count - HealthyCell.cell_count) / 3000
                else:
                    return (self.init_hcell_count - HealthyCell.cell_count) / 5000#(cppCellModel.HCellCount() / self.init_hcell_count) - 0.5 - (2 * hcells_lost/2500)
        else:
            if self.reward == 'dose' or self.reward == 'oar':
                return - dose / 400 + (ccell_killed - 5 * hcells_lost)/100000
            elif self.reward == 'killed':
                return (ccell_killed - 3 * hcells_lost)/10000

    def inTerminalState(self):
        if CancerCell.cell_count <= 0 :
            if self.verbose:
                print("No more cancer")
            self.end_type = 'W'
            return True
        elif HealthyCell.cell_count < 10:
            if self.verbose:
                print("Cancer wins")
            self.end_type = "L"
            return True
        elif self.controller.tick > 420:
            if self.verbose:
                print("Time out!")
            self.end_type = "T"
            return True
        else:
            return False

    def inputDimensions(self):
        """History size 1 with observations of shape 15*15*25 if resize else 30*30*50"""
        if self.resize:
            tab = [(1, 15, 15, 25)]
        else:
            tab = [(1, 30, 30, 50)]
        return tab

    def nActions(self):
        """discrete 7 actions for DQN and 4 types of actions between 0 and 1"""
        if self.action_type == 'DQN':
            return 7
        elif self.action_type == 'DDPG':
            return [[0, 1], [0, 1], [0, 1], [0, 1]] # [beam values, rest values, x values, y values] mapped from 0->1

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        print(test_data_set)
    #def end(self):
    #    """called at the end of all epochs"""
    #    del self.controller