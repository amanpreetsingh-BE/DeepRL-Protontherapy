# The code is based on the template from Monte Carlo lab (LGBIO2070)

import matplotlib.pyplot as plt
import numpy as np
from get_voxel_index import get_voxel_index

# stopping power database (water, bone)
SP_water = np.loadtxt("data/SP_water.txt", 'float', '#', None, None, 8)
SP_bone = np.loadtxt("data/SP_bone.txt", 'float', '#', None, None, 8)
SP_database = []
SP_database.append([]) # index 0 is empty to be consistant with matlab code
SP_database.append(SP_water) # energy = SP_database[1][:,0] ; sp = SP_database[1][:,1] ; 132x2 dimension
SP_database.append(SP_bone) 

# simulation geometry
grid_size = np.array([100, 100, 400])            # water tank of size 10x10x40 [cm]
voxel_size = np.array([1, 1, 1])                 # voxel size 1x1x1 [mm]
offset = np.array([-50, -50, 0])                 # coordinates of the first voxel (origin) [mm]
density_map = np.ones(grid_size) * 1.0           # density of water (g/cm3)
material_map = np.ones(grid_size) * 1            # 1 = water
scoring_grid = np.zeros(grid_size)               # initialize the dose distribution grid, i.e beamlet 

# simulation parameters
Beam_position = np.array([0.0, 0.0, 0.0])        # position of the beam [mm]
Beam_direction = np.array([0, 0, 1])             # direction of the beam [mm]
Beam_mean_energy = 200.0                         # mean energy of the beam [MeV]
step_length = 1.0                                # step length [mm]

# proton particle object
class Particle:
  def __init__(self, position, direction , energy):
    self.position = position
    self.direction = direction
    self.energy = energy

# constants 
m_p = 938.272                                    # proton mass [MeV]
m_e = 0.511                                      # electron mass [MeV]
m_o = 16.0 * m_p                                 # oxygen mass [MeV]
r_e = 2.818E-12                                  # radius of electron [mm]
z = 1.0                                          # atomic number of proton [/]
X_w = 360.86                                     # radiation length of water [mm]
Es = 18.3                                        # MCSquare simulation parameter 
Tmin_e = float('inf')                            # [MeV]

# utility functions
def gamma(T_p, m_p) :
    return (T_p+m_p)/m_p

def beta(T_p, m_p) :
    return np.sqrt(1-(1/np.square(gamma(T_p,m_p))))

def Tmax_e(T_p, m_p, m_e) :
    return (2*m_e*np.square(beta(T_p, m_p))*np.square(gamma(T_p, m_p)))/(1+((2*gamma(T_p, m_p))*(m_e/m_p))+np.square(m_e/m_p))

def omega_square(r_e, m_e, z, T_p, m_p, s, Tmin_e, voxel_index):
    rho = density_map[voxel_index[0], voxel_index[1], voxel_index[2]]
    if(rho == 1.0):
        n_el = (6.022E23)*(1/18015.3)* (10) # [#e-/mm3]
    else :
        n_el = (6.022E23)*(1/12010)* (6)
    return 2*np.pi*np.square(r_e)*m_e*n_el*np.square(z/beta(T_p, m_p))*s*Tmax_e(T_p,m_p,m_e)*(1-(np.square(beta(T_p, m_p))/2))

def csda(particle, ES, voxel_index, s) :

    stoppingPower = np.interp(particle.energy, SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,0], SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,1])

    if ES : # Energy Straggling 
        energyStraggling =  np.random.normal(0,np.sqrt(omega_square(r_e, m_e, z, particle.energy, m_p, s, Tmin_e, voxel_index)))
    else :
        energyStraggling = 0
        
    dE = ((density_map[voxel_index[0], voxel_index[1], voxel_index[2]])*(stoppingPower)*(s/10)) + energyStraggling

    if (dE > particle.energy) : # in case of last step
        dE = particle.energy
        particle.energy = particle.energy - dE
    else :
        particle.energy = particle.energy - dE
        
    scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += dE
 

def radiate(SIPP, SIPE, ES, particlesNumber, sigmaX, sigmaY, sigmaE) : 


    for i in range(0, particlesNumber):
        print('CSDA simulation of particle ', (i+1))

        # Initialize a particle with position, direction and energy = STATE of particle i 
        particle = Particle(Beam_position.copy(), Beam_direction.copy(), Beam_mean_energy)

        if SIPP : # Sampling Initial Particle Position
            particle.position[0] = np.random.normal(particle.position[0], sigmaX)
            particle.position[1] = np.random.normal(particle.position[1], sigmaY)

        if SIPE : # Sampling Initial Particle Energy
            particle.energy = np.random.normal(particle.energy, sigmaE)

        while (particle.energy > 0) : # CSDA
            voxel_index = get_voxel_index(particle.position, voxel_size, offset)

            csda(particle, ES, voxel_index, step_length)

            particle.position = particle.position + ((particle.direction)*(step_length))

    print('End of radiation')

# RUN Simulation to encode scoring_grid with following parameters (from MCSquare for comparison)
Num_particles = 500 # numbers of particles
SIPP = True # Sampling Initial Particle Position
SIPE = True # Sampling Initial Particle Energy
ES = True # Energy Straggling
sigmaX = 1.0 # Spot size
sigmaY = 1.0 # Spot size
sigmaE = 1.0 # Energy beam spread

radiate(SIPP, SIPE, ES, Num_particles, sigmaX, sigmaY, sigmaE) 


# Convert scored energy into dose with Gray units
scoring_grid # (MeV)
voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000 # (cm3)
dose = scoring_grid / (density_map * voxel_volume) # (MeV / g)
dose = dose * 1000 # (MeV / kg)
dose = dose * 1.602176e-19 # (Gy = J/kg)

# plot result
voxel_index = get_voxel_index(Beam_position, voxel_size, offset)

plt.figure(figsize=(15,4))
plt.subplot(1, 4, 1)
plt.imshow(np.transpose(dose[:,:,int(round(grid_size[2]/2))]), cmap="jet")
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
z = np.arange(offset[2], grid_size[2]*voxel_size[2], voxel_size[2])
plt.plot(z,np.sum(dose, axis=(0,1)))
plt.ylabel('Integrated dose (Gy = J/kg)')
plt.xlabel('Depth (voxels)')
plt.xlim(0, 400)
plt.ylim(0, max(np.sum(dose, axis=(0,1))))
plt.show()

