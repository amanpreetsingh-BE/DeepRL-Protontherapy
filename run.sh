#!/bin/bash -l

#SBATCH --job-name=aman_masterthesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=500
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=amanpreet.singh@student.uclouvain.be

module load cuda/11.0.2
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load OpenCV/4.2.0-foss-2019b-Python-3.7.4

# COMMENT/UNCOMMENT one of the following command : 

# train DQN agent : 
#python3 DeepRL-PT/trainAgent.py -n 'DQN'

# train DDPG agent : 
#python3 DeepRL-PT/trainAgent.py -n 'DDPG'

# run DQN agent : 
#python3 DeepRL-PT/runAgent.py -n 'DQN'

# run DDPG agent : 
python3 DeepRL-PT/runAgent.py -n 'DDPG'

# run naive treatment : 
#python3 DeepRL-PT/runAgent.py -n 'DQN' --naive 'True'