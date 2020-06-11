#!/bin/bash
#SBATCH --account=def-stelzer
#SBATCH --time=12:00:00    # time (DD-HH:MM)
#SBATCH --mem=250G         # memory (per node)
#SBATCH --gres=gpu:p100l:4
#SBATCH --job-name=el-id
##SBATCH --output=outputs/log_files/%x_%A_%a.out
#SBATCH --array=1
export VAR=$SLURM_ARRAY_TASK_ID


# TRAINING ON LPS
#singularity shell --nv --bind /lcg,/opt \
#/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif classifier.sh $VAR


# SAMPLE GENERATION ON LPS
#singularity shell      --bind /lcg,/opt \
#/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif presampler.sh


# TRAINING ON BELUGA
#module load singularity/2.6
#singularity shell --nv --bind /project/def-stelzer/edreyer \
#/home/edreyer/project/edreyer/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif /project/6001319/edreyer/el_classifier/classifier.sh $VAR
#singularity shell --nv /project/6001319/edreyer/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif /project/6001319/edreyer/el_classifier/classifier.sh $VAR


module load python/3.6
module load scipy-stack
source ~/ENV/bin/activate
source /project/6001319/edreyer/el_classifier/classifier.sh $VAR
