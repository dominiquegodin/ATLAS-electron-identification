#!/bin/bash
##SBATCH --account=def-arguinj
##SBATCH --time=12:00:00    # time (DD-HH:MM)
##SBATCH --mem=186G         # memory (per node)
##SBATCH --cpus-per-task=12 # CPU cores/threads
#SBATCH --gres=gpu:4       # Number of GPU(s) per node
#SBATCH --job-name=el-id
#SBATCH --output=outputs/log_files/%x_%A_%a.out
#SBATCH --array=1
export VAR=$SLURM_ARRAY_TASK_ID
export SCRIPT_VAR


# TRAINING ON LPS
SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
singularity shell --nv --bind /lcg,/opt $SIF classifier.sh $VAR $SCRIPT_VAR
# PRESAMPLE ON LPS
#singularity shell      --bind /lcg,/opt $SIF presampler.sh


# TRAINING ON BELUGA
#SIF=/project/def-arguinj/dgodin/sing_images/tf-2.1.0-gpu-py3_sing-3.5.sif
#module load singularity/3.5
#singularity shell --nv --bind /project/def-arguinj/dgodin $SIF classifier.sh $VAR $SCRIPT_VAR
