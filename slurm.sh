#!/bin/bash                                                                                                                                                                               
#SBATCH --account=def-arguinj
#SBATCH --gres=gpu:4       # Number of GPU(s) per node
#SBATCH --mem=186G         # memory (per node)
#SBATCH --time=06:00:00    # time (DD-HH:MM) 
#SBATCH --job-name=classifier
#SBATCH --output=classifier.out
module load singularity/2.6
singularity shell --nv --bind /project/def-arguinj/dgodin /project/def-arguinj/dgodin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif classifier.sh

