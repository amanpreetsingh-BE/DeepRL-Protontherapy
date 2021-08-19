#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

'''
    @author : singh
    The entire file is highly inspired on Moreau's work : https://github.com/gregoire-moreau/radio_rl
    Link to my repo : https://github.com/amanpreetsingh-BE/DeepRL-Protontherapy
    Usage of agent using DQN and DDPG in a tumoral environment in 3D  to get treatment plan
'''

parser = argparse.ArgumentParser(description='Using trained agent')
parser.add_argument('--obs_type', choices=['densities', 'segmentation'], default='densities')
parser.add_argument('--resize', action='store_true')
parser.add_argument('-t', action='store_true', dest='tumor_radius')
parser.add_argument('-n', '--network', choices=['DDPG', 'DQN'], dest='network', required=True)
parser.add_argument('-r', '--reward', choices=['dose', 'killed'], dest='reward', default='killed')
parser.add_argument('--no_special', action='store_false', dest='special')
parser.add_argument('-c', '--center', action='store_true')
parser.add_argument('-e', '--epochs', nargs=2, type=int, default=[30, 100])
parser.add_argument('--naive_treat', choices=['True', 'False'], dest='naive', default=False)
args = parser.parse_args()
print(args)


from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
from deer.learning_algos.AC_net_keras import MyACNetwork
from deer.policies import EpsilonGreedyPolicy
from cellEnvironment import CellEnvironment

#------ END OF IMPORTS

# RANDOM SEED
rng = np.random.RandomState(123456) # seed

# IMPORT ENVIRONMENT 
env = CellEnvironment(args.obs_type, args.resize, args.reward, args.network, args.special) 

# NAIVE TREATMENT 
if args.naive :
    random.seed(4775)

    for i in range(0,10):
        env.controller.irradiate(True, True, True, True, True, 500, 2, 2, 1, np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1.0]), 60.0, np.array([-15, -15, 0]), 1)
        env.controller.go(24, treatment=True)

    hcell_pre = env.controller.numberHealthyCells[179] 
    hcell_post = env.controller.numberHealthyCells[len(env.controller.numberHealthyCells)-1]
    ccell_post = env.controller.numberCancerCells[len(env.controller.numberCancerCells)-1]
    
    print('Killed Healthy cells : ', hcell_pre-hcell_post)
    print('Number Cancer cells post : ', ccell_post)
    env.controller.plot_treatment()
    print('DONE')

# AGENT TREATMENT
else:

    # LOAD DQN NETWORK
    if args.network == 'DQN':
        network = MyQNetwork(
            environment=env,
            batch_size=32,
            double_Q=True,
            random_state=rng)

    # LOAD DDPG NETWORKS
    elif args.network == 'DDPG':
        network = MyACNetwork(
            environment=env,
            batch_size=32,
            random_state=rng)
        #args.epochs[0] = 8 #trained on 8 epoch for tested network

    agent = NeuralAgent(
            env,
            network,
            train_policy=EpsilonGreedyPolicy(network, env.nActions(), rng, 0.0),
            replay_memory_size=int(args.epochs[0]*args.epochs[1] * 1.1),
            batch_size=32,
            random_state=rng)
            
    if args.network == 'DQN': 
        agent.setNetwork('DQN_agent') 
    else:
        agent.setNetwork('nnet.epoch=27')
        print('nnet 27')

    if __name__ == '__main__':

        env.init_dose_map()
        agent._runEpisode(20) # run an entire episode for 20 experiences

        env.controller.plot_treatment()
        cells = env.dataset[0]
        rests = [r+180 for r in env.dataset[0]]
        positions = env.dataset[3]
        energies = env.dataset[2]

        if args.network == 'DQN': 
            plt.figure()
            rest_DQN = ([i*12 for i in range(0,15)])+rests
            beam_DQN = ([0] * 15)+energies

            plt.plot(rest_DQN, beam_DQN, '-o')
            plt.legend()
            plt.xlabel('Time [h]')
            plt.ylabel('Mean beam energy [MeV]')
            plt.savefig("DeepRL-PT/out/misc/dqnmbe.png")
            plt.close()

        else:
            plt.figure(figsize=(8, 6))
            colors = ["black", "dimgray", "bisque", "darkorange", "indianred", "deeppink", "brown", "red", "forestgreen", "limegreen", "greenyellow", "slategrey", "lightsteelblue", "cornflowerblue", "royalblue", "deepskyblue", "lavender", "mediumpurple", "darkviolet", "indigo"] # normally size of len(positions) !!!! but flemme
            for i in range(0,len(positions)):
                plt.scatter(positions[i][0],positions[i][1], marker = 'x', color=colors[i], label= 'E_'+str(rests[i])+ ' = '+ str(energies[i]) + ' MeV')
            
            plt.legend(bbox_to_anchor=(1, 1))
            plt.xlabel('X offset around the center')
            plt.ylabel('Y offset around the center')
            plt.ylim([-2,2])
            plt.xlim([-2,2])
            plt.tight_layout()
            plt.savefig("DeepRL-PT/out/misc/ddpgxy.png")
            plt.close()
        
        hcell_pre = env.controller.numberHealthyCells[179] 
        hcell_post = env.controller.numberHealthyCells[len(env.controller.numberHealthyCells)-1]
        ccell_post = env.controller.numberCancerCells[len(env.controller.numberCancerCells)-1]
    
        print('Killed Healthy cells : ', hcell_pre-hcell_post)
        print('Number Cancer cells post : ', ccell_post)
        
        print('DONE')