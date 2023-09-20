#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=07-00:00         #time limit (DD-HH:MM)
#SBATCH --nodes=1               #number of nodes
##SBATCH --mem=128G              #memory per node (on Beluga)
#SBATCH --cpus-per-task=4       #number of CPU threads per node
#SBATCH --gres=gpu:1            #number of GPU(s) per node
#SBATCH --job-name=e-ID_CNN
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

export SBATCH_VAR=$SLURM_ARRAY_TASK_ID
export  HOST_NAME=$SLURM_SUBMIT_HOST
export INPUT_PATH=$SLURM_TMPDIR
export SCRIPT_VAR

if [[ $HOST_NAME == *atlas* ]]
then
    # TRAINING ON LPS
    if [[ -d "/nvme1" ]] ; then
	PATHS=/lcg,/opt,/nvme1
    else
	PATHS=/lcg,/opt
    fi
    SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
    if  [ -z ${PRESAMPLER+x} ] || [ $PRESAMPLER != True ]
    then
	singularity shell --nv --bind $PATHS $SIF classifier.sh $SBATCH_VAR $HOST_NAME $SCRIPT_VAR
    else
	singularity shell      --bind $PATHS $SIF presampler.sh
    fi
else
    # TRAINING ON BELUGA
    if [[ -n "$INPUT_PATH" ]]
    then
	echo "COPYING DATA FILES TO LOCAL NODE"
	cp -r /project/def-arguinj/shared/e-ID_data/{0.0-1.3,1.3-1.6,1.6-2.5,0.0-2.5} $INPUT_PATH
    fi
    module load singularity/3.6
    PATHS=/project/def-arguinj,$INPUT_PATH
    SIF=/project/def-arguinj/shared/sing_images/tf-2.1.0-gpu-py3_sing-3.5.sif
    singularity shell --nv --bind $PATHS $SIF < classifier.sh $SBATCH_VAR $HOST_NAME $INPUT_PATH
fi

mkdir -p outputs/log_files
mv *.out outputs/log_files 2>/dev/null