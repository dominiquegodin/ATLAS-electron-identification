#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=00-12:00         #time limit (DD-HH:MM)
#SBATCH --nodes=1               #number of nodes
#SBATCH --mem=128G              #memory per node (on Beluga)
#SBATCH --cpus-per-task=8       #number of CPU threads per node
#SBATCH --gres=gpu:4            #number of GPU(s) per node
#SBATCH --job-name=e-ID
#SBATCH --output=outputs/log_files/%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

export SBATCH_VAR=$SLURM_ARRAY_TASK_ID
export HOST_NAME=$SLURM_SUBMIT_HOST
export NODE_DIR=$SLURM_TMPDIR
export SCRIPT_VAR

if [[ $HOST_NAME == *atlas* ]]
then
    # TRAINING ON LPS
    SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
    #singularity shell      --bind /lcg,/opt $SIF presampler.sh
    singularity shell --nv --bind /lcg,/opt $SIF classifier.sh $SBATCH_VAR $HOST_NAME $SCRIPT_VAR
else
    # TRAINING ON BELUGA
    if [[ -n "$NODE_DIR" ]]
    then
	cp -r /project/def-arguinj/shared/e-ID_data/{0.0-1.3,1.3-1.6,1.6-2.5} $NODE_DIR
	echo "COPYING DATA FILES TO LOCAL NODE"
    fi
    module load singularity/3.6
    SIF=/project/def-arguinj/shared/sing_images/tf-2.1.0-gpu-py3_sing-3.5.sif
    singularity shell --nv --bind /project,$NODE_DIR $SIF < classifier.sh $SBATCH_VAR $NODE_DIR $SCRIPT_VAR
fi
