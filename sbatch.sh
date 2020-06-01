#!/bin/bash


##SBATCH --account=def-arguinj                          # uncomment on Beluga
##SBATCH --time=0-03:00     # time (DD-HH:MM)           # uncomment on Beluga
##SBATCH --mem=128G         # memory (per node)         # uncomment on Beluga
##SBATCH --cpus-per-task=8  # CPU threads               # uncomment on Beluga
#SBATCH --gres=gpu:4       # Number of GPU(s) per node
#SBATCH --job-name=el-id
#SBATCH --output=outputs/log_files/%x_%A_%a.out         # directory must exist
#SBATCH --array=0


export VAR=$SLURM_ARRAY_TASK_ID
export SCRIPT_VAR


# TRAINING ON LPS
SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
singularity shell --nv --bind /lcg,/opt $SIF classifier.sh $VAR $SCRIPT_VAR
#singularity shell      --bind /lcg,/opt $SIF presampler.sh


# TRAINING ON BELUGA
#SIF=/project/def-arguinj/dgodin/sing_images/tf-2.1.0-gpu-py3_sing-3.5.sif
#module load singularity/3.5
#singularity shell --nv --bind /project/def-arguinj/dgodin $SIF < classifier.sh $VAR $SCRIPT_VAR
