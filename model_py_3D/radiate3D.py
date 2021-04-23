import matplotlib.pyplot as plt
import numpy as np
from get_voxel_index import get_voxel_index
from compute_distance_to_interface import compute_distance_to_interfacee
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

# for testing part (comment or uncomment)
#density_map[49:99,49:99,:] = 2.0 # for part testing
#material_map[49:99,49:99,:] = 2 # for part testing


# simulation parameters
Beam_position = np.array([0.0, 0.0, 0.0])    # (mm)
Beam_direction = np.array([0.0, 0.0, 1.0])
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
Es = 18.3 # MCSquare simulation parameter 
Tmin_e = float('inf') # [MeV]

# utility functions (same naming as Fippel document)
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

def mean_free_path(T_p, rho):
    lamba_tot = 0.0 
    p_pp = 0.0
    p_el = 0.0
    p_in = 0.0
    s_nucl = 0.0
    
    # [1/mm]
    sigma_pp = ((0.315*np.power(T_p,-1.126)) + (3.78E-6*T_p))*(rho/10) # elastic proton - proton cross-section [1/mm]
    sigma_el = ((1.88/T_p) + (4E-5*T_p) - 0.01475)*(rho/10) # elastic proton - oxygen cross-section [1/mm]
    sigma_in = (0.001*( (1.64*(T_p-7.9)*np.exp((-0.064*T_p)+(7.85/T_p))) + 9.86))*(rho/10) # inelastic proton - oxygen cross-section [1/mm]
    
    if (T_p >= 7 and T_p <= 10) :   
        lamba_tot=1/sigma_in
        p_in = sigma_in*lamba_tot
        s_nucl = (-lamba_tot)*np.log(np.random.uniform(0,1))
    elif (T_p > 10 and T_p <= 50) :
        lamba_tot = 1/(sigma_in+sigma_pp)
        p_pp = sigma_pp*lamba_tot
        p_in = sigma_in*lamba_tot
        s_nucl = (-lamba_tot)*np.log(np.random.uniform(0,1)) 
    elif (T_p > 50 and T_p <= 250) :
        lamba_tot = 1/(sigma_pp+sigma_el+sigma_in)
        p_pp = sigma_pp*lamba_tot
        p_el = sigma_el*lamba_tot
        p_in = sigma_in*lamba_tot
        s_nucl = (-lamba_tot)*np.log(np.random.uniform(0,1))
    elif (T_p > 250 and T_p <= 300) :
        lamba_tot = 1/sigma_pp
        p_pp = sigma_pp*lamba_tot
        s_nucl = (-lamba_tot)*np.log(np.random.uniform(0,1))
    else:
        s_nucl = np.inf # no nuclear interaction
        
    return s_nucl, p_pp, p_el, p_in

def convert_CM_to_lab(cos_theta_CM,T_p,M1C2,M2C2) :
    beta_CM = (np.sqrt(T_p*(T_p+(2*M1C2))))/(T_p+M1C2+M2C2)
    tau = np.sqrt(np.square(M1C2/M2C2)*(1-np.square(beta_CM))+np.square(beta_CM))
    gamma_CM = 1/(np.sqrt(1-np.square(beta_CM)))
    sin_theta_cm_square = 1-np.square(cos_theta_CM)
    cos_theta = (cos_theta_CM+tau)/(np.sqrt( np.square(cos_theta_CM+tau) + (sin_theta_cm_square/(np.square(gamma_CM)))))
    return cos_theta

def energy_transfered_T0(T_p) :
    epsilon = np.random.uniform(0.0,1.0)
    TO_mean = (0.65*np.exp(-0.0013*T_p))-(0.71*np.exp(-0.0177*T_p)) # positif

    TO_max = Tmax_e(T_p, m_p, m_o)
        
    TO = -TO_mean * np.log(epsilon)

    while TO > TO_max :
        epsilon = np.random.uniform(0,1)
        TO = -TO_mean * np.log(epsilon)
            
    return TO

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


def nuclear_interactions(particle, secondaryParticles, voxel_index, p_pp, p_el, p_in, s) :
    epsilon = np.random.uniform(0,1) 
  
    if (epsilon < p_pp) : # simulate proton - proton elastic
        print('Proton - Proton Elastic occured')
        # init new particle
        secondaryParticle = Particle((particle.position).copy(),(particle.direction).copy(),particle.energy)
        E_incident = particle.energy
        cos_theta_cm1 = (2.0*np.random.uniform(0,1))-1
        cos_theta_cm2 = (-1.0)*cos_theta_cm1
        w = (E_incident*(1-cos_theta_cm1))/2 # cos_theta_cm1 or cos_theta1 ?
        if (w > particle.energy) : # in case of last step
            w = particle.energy
            particle.energy = particle.energy - w
        else :
            particle.energy = particle.energy - w
        
        secondaryParticle.energy = w

        cos_theta1 = convert_CM_to_lab(cos_theta_cm1, E_incident, m_p, m_p) # particle.energy or particle.energy-w ?
        cos_theta2 = convert_CM_to_lab(cos_theta_cm2, E_incident, m_p, m_p) # or particle.energy or w ?

        particle.direction = cdirect(cos_theta1, particle.direction[0], particle.direction[1], particle.direction[2])
        secondaryParticle.direction = cdirect(cos_theta2, secondaryParticle.direction[0], secondaryParticle.direction[1], secondaryParticle.direction[2])
        secondaryParticles.append(secondaryParticle) # add to queue

    
    elif (epsilon < (p_pp+p_el)) : # simulate proton - oxygen elastic (no loss but transfer !)
        print('Proton - Oxygen Elastic occured')
        T0 = energy_transfered_T0(particle.energy)
        E_incident = particle.energy
        if (T0 > particle.energy) : # in case of last step
            T0 = particle.energy
            particle.energy = particle.energy-T0
        else :
            particle.energy = particle.energy-T0
        
        scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += T0 # not enough to make O moves
        cos_theta_cm = 1-(T0/(np.square(beta(E_incident, m_p)*gamma(E_incident, m_p))*m_o))
        cos_theta = convert_CM_to_lab(cos_theta_cm, E_incident, m_p, m_o)
        particle.direction = cdirect(cos_theta, particle.direction[0], particle.direction[1], particle.direction[2])
  
    else : # simulate proton - oxygen inelastic
        print('Proton - Oxygen Inelastic occured')
        E_min = 3.0
        E_binding = 100.0
        
        E_system = particle.energy - E_binding
        E_incident = particle.energy
            
        while(E_system > E_min):
            E_secondary = np.random.uniform(E_min,E_system)
            # init new particle 
            secondaryParticle = Particle((particle.position).copy(),(particle.direction).copy(),E_secondary)
            random_number = np.random.uniform(0.0,1.0)
            if (random_number < 0.5) : # 50% probability of protons emitted 
                if (E_secondary > particle.energy) : # in case of last step
                    E_secondary = particle.energy # avoid negative
                    scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += E_secondary
                    cos_theta = np.random.uniform((2*(E_secondary/E_incident))-1,1)
                else :
                    scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += E_secondary
                    cos_theta = np.random.uniform((2*(E_secondary/E_incident))-1,1)

                secondaryParticle.direction = cdirect(cos_theta, secondaryParticle.direction[0], secondaryParticle.direction[1], secondaryParticle.direction[2])
                secondaryParticles.append(secondaryParticle) # add to queue
            if(random_number > 0.965) : # 3.5% probability short range particles absorbed
                #particle.energy = particle.energy - E_secondary
                scoring_grid[voxel_index[0], voxel_index[1], voxel_index[2]] += E_secondary
            
            E_system = E_system - E_secondary - E_binding
        


def simulate(SIPP, SIPE, ES, MCS, NUCL, particlesNumber, sigmaX, sigmaY, sigmaE) : 
  
    secondaryParticles = [] # stack of secondary particles generated by nuclear interactionss

    for i in range(0, particlesNumber):

        # Initialize a particle with position, direction and energy = STATE of particle i 
        particle = Particle(Beam_position.copy(), Beam_direction.copy(), Beam_mean_energy)

        if(len(secondaryParticles) != 0) : # if there are secondary particles, simulate them 
            particle = secondaryParticles.pop()
            print('CSDA simulation of a secondary particle !')
            i = i -1
        else:
            if SIPP : # Sampling Initial Particle Position
                particle.position[0] = np.random.normal(particle.position[0], sigmaX)
                particle.position[1] = np.random.normal(particle.position[1], sigmaY)
            if SIPE : # Sampling Initial Particle Energy
                particle.energy = np.random.normal(particle.energy, sigmaE)
            print('CSDA simulation of particle ', (i+1))

        while (particle.energy > 0) : # CSDA
            voxel_index = get_voxel_index(particle.position, voxel_size, offset)
            # sanity overflow check
            if(voxel_index[0] >= 100 or voxel_index[0] <= -100 or voxel_index[1] >= 100 or voxel_index[1] <= -100 or voxel_index[2] >= 400 or voxel_index[2] <= -400 ) :
                break
            stoppingPower = np.interp(particle.energy, SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,0], SP_database[int(material_map[voxel_index[0],voxel_index[1],voxel_index[2]])][:,1])
            s_nucl, p_pp, p_el, p_in = mean_free_path(particle.energy, density_map[voxel_index[0], voxel_index[1], voxel_index[2]])

            if(NUCL and (s_nucl < step_length)): # EM and Nuclear
                em_interactions(particle, ES, MCS, voxel_index, stoppingPower, s_nucl)
                nuclear_interactions(particle, secondaryParticles, voxel_index, p_pp, p_el, p_in, s_nucl)
                particle.position = particle.position + ((particle.direction)*(s_nucl))

                
            
            else: # EM only
                em_interactions(particle, ES, MCS, voxel_index, stoppingPower, step_length)
                particle.position = particle.position + ((particle.direction)*(step_length))
        

# RUN Simulation to encode scoring_grid with following parameters (from MCSquare for comparison)
Num_particles = 1000 # numbers of particles
SIPP = True # Sampling Initial Particle Position
SIPE = True # Sampling Initial Particle Energy
ES = True # Energy Straggling
MCS = True # Multi Coulomb Scattering
NUCL = False # Nuclear interactions
sigmaX = 1.0 # Spot size
sigmaY = 3.0 # Spot size
sigmaE = 1.0 # Energy beam spread

simulate(SIPP, SIPE, ES, MCS, NUCL, Num_particles, sigmaX, sigmaY, sigmaE) 


# Convert scored energy into dose with Gray units
scoring_grid # (MeV)
voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2] / 1000 # (cm3)
dose = scoring_grid / (density_map * voxel_volume) # (MeV / g)
dose = dose * 1000 # (MeV / kg)
dose = dose * 1.602176e-19 # (Gy = J/kg)