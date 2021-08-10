#!/usr/bin/env python3

'''
    @author : singh
    The entire file is highly inspired on Moreau's work : https://github.com/gregoire-moreau/radio_rl
    Link to my repo : https://github.com/amanpreetsingh-BE/DeepRL-Protontherapy
'''

'''
import argparse
import datetime
print(datetime.datetime.now())
parser = argparse.ArgumentParser(description='Start training of an agent')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--obs_type', choices=['segmentation', 'densities', 'scalars'], default='densities')
parser.add_argument('--resize', action='store_true')
parser.add_argument('-n', '--network', choices=['DDPG', 'DQN'], dest='network', required=True)
parser.add_argument('-r', '--reward', choices=['dose', 'killed', 'oar'], dest='reward', required=True)
parser.add_argument('--no_special', action='store_false', dest='special')
parser.add_argument('-l', '--learning_rate', nargs=3, type=float, default=[0.0001, 0.75,5])
parser.add_argument('--fname', default='nnet')
parser.add_argument('-e', '--epochs', nargs=2, type=int, default=[20, 2500])
parser.add_argument('--exploration', choices=['epsilon', 'gauss'], dest='exploration', default='epsilon')

args = parser.parse_args()
print(args)

if args.network == 'DQN' and args.exploration == 'gauss':
    raise Exception("Can't use Gaussian Noise with DQN, use it with DDPG")

if args.gpu:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # GPU cluster
''' 
import numpy as np
from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
from deer.learning_algos.AC_net_keras import MyACNetwork
import deer.experiment.base_controllers as bc
from deer.policies import EpsilonGreedyPolicy
