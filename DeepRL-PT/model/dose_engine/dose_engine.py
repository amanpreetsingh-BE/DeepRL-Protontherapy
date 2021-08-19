import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from get_voxel_index import get_voxel_index
from cdirect import cdirect

# Generate stopping power database
SP_water = np.loadtxt("data/SP_water.txt", 'float', '#', None, None, 8)
SP_bone = np.loadtxt("data/SP_bone.txt", 'float', '#', None, None, 8) # ADDED ONE FOR TESTING
SP_database = []
SP_database.append([]) # index 0 is empty to be consistant with matlab code
SP_database.append(SP_water) # index 1 = water print(SP_database[1][x,y])
SP_database.append(SP_bone) # index 2 = bone print(SP_database[2][x,y])

# simulation geometry
grid_size = np.array([100, 100, 400])            # size of the voxel grid
voxel_size = np.array([1, 1, 1])                 # (mm)
offset = np.array([-50, -50, 0])                 # coordinates of the first voxel (mm)
density_map = np.ones(grid_size) * 1.0 # (g/cm3)
material_map = np.ones(grid_size) * 1  # 1 = water
scoring_grid = np.zeros(grid_size)     # initialize the dose distribution grid

# simulation parameters
Beam_position = np.array([0.0, 0.0, 0.0])    # (mm)
Beam_direction = np.array([0, 0, 1.0])
Beam_mean_energy = 200.0       # (MeV)
step_length = 1.0              # (mm)

#############################
# IMPLEMENTATION START HERE #
#############################

# particle object
class Particle:
  def __init__(self, position, direction , energy):
    self.position = position
    self.direction = direction
    self.energy = energy

# constants (same naming as Fippel document)
m_p = 938.272 # [MeV]
m_e = 0.511 # [MeV]
m_o = 16.0 * m_p # [MeV]
r_e = 2.818E-12 # [mm]
z = 1.0 # [/]
X_w = 360.86 # [mm]
Es = 6 # simulation parameter 
Tmin_e = float('inf') # [MeV]

# utility functions (same naming as Fippel document) for transport functions
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

def polar_angle(T_p, m_p, Es, z, s, voxel_index):
    pc = np.sqrt(np.square(T_p+m_p)-np.square(m_p))
    rho = density_map[voxel_index[0], voxel_index[1], voxel_index[2]]
    if(rho != 1.0): # for different than water
        f_xO = 1.19 + 0.44*np.log(rho-0.44)
        X = ((1*X_w)/(f_xO*rho))*10
    else: # for wate
        X = X_w # [mm]
    return (((Es)/(beta(T_p,m_p)*pc))*z*np.sqrt(s/X))

def em_interactions(particle, ES, MCS, voxel_index, stoppingPower, s) :
    
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
        
    if MCS : # Multiple Coulomb Scattering
        scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += dE
        if(particle.energy != 0): # otherwise theta = inf or nan bad values if energy = 0 (last step)
            theta = polar_angle(particle.energy, m_p, Es, z, s, voxel_index)
            particle.direction = cdirect(np.cos(np.random.normal(0,theta)), particle.direction[0], particle.direction[1], particle.direction[2])
    else :
        scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += dE

def simulate(SIPP, SIPE, ES, MCS, particlesNumber, sigmaX, sigmaY, sigmaE) : 
    for i in range(0, particlesNumber): # for each particle of the beam 

        # Initialize a particle with position, direction and energy = STATE of particle i 
        particle = Particle(Beam_position.copy(), Beam_direction.copy(), Beam_mean_energy)

        if SIPP : # Sampling Initial Particle Position
            particle.position[0] = np.random.normal(particle.position[0], sigmaX)
            particle.position[1] = np.random.normal(particle.position[1], sigmaY)
        if SIPE : # Sampling Initial Particle Energy
            particle.energy = np.random.normal(particle.energy, sigmaE)
        print('CSDA simulation of particle ', (i+1))

        while (particle.energy > 0) : # CSDA
            voxel_index = get_voxel_index(particle.position, voxel_size, offset)
            # sanity overflow check
            if(voxel_index[0] >= grid_size[0] or voxel_index[0] <= -grid_size[0] or voxel_index[1] >= grid_size[1]  or voxel_index[1] <= -grid_size[1]  or voxel_index[2] >= grid_size[2]  or voxel_index[2] <= -grid_size[2] ) :
                break
            stoppingPower = np.interp(particle.energy, SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,0], SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,1])

            em_interactions(particle, ES, MCS, voxel_index, stoppingPower, step_length)
            particle.position = particle.position + ((particle.direction)*(step_length))

def plot_beam(dose, voxel_index):
    for k in range(dose.shape[2]):
        for i in range(dose.shape[0]):
            for j in range(dose.shape[1]):
                if(dose[i, j, k] <= 0.0):
                    dose[i, j, k] = np.NaN

    ccmap = LinearSegmentedColormap.from_list('mycmap', ['red', 'darkred', 'black'])
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
    #plt.title('3D protons beam 200MeV', pad=40)
    #plt.savefig('tumorMass@_'+str(self.tick)+'_'+str(xSize)+'x'+str(ySize)+'x'+str(zSize)+'.png')
    plt.show()

# RUN Simulation to encode scoring_grid with following parameters (from MCSquare for comparison)
Num_particles = 10000 # numbers of particles

SIPP = True # Sampling Initial Particle Position
SIPE = True # Sampling Initial Particle Energy
ES = True # Energy Straggling
MCS = True # Multi Coulomb Scattering

sigmaX = 1.3 # Spot size
sigmaY = 1.3 # Spot size
sigmaE = 1   # Energy beam spread

simulate(SIPP, SIPE, ES, MCS, Num_particles, sigmaX, sigmaY, sigmaE) 


# Convert scored energy into dose with Gray units
scoring_grid # (MeV)
voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000 # (cm3)
dose = scoring_grid / (density_map * voxel_volume) # (MeV / g)
dose = dose * 1000 # (MeV / kg)
dose = dose * 1.602176e-19 # (Gy = J/kg)

# plot result

'''
voxel_index = get_voxel_index(Beam_position, voxel_size, offset)
plt.figure(figsize=(15,4))
plt.subplot(1, 4, 1)
plt.imshow(dose[:,:,int(round(grid_size[2]/2))], cmap="jet")
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
plt.xlim(0.1, 400)
plt.ylim(0, max(np.sum(dose, axis=(0,1))))
plt.show()

plot_beam(dose, voxel_index)'''