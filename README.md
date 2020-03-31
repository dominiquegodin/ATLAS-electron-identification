# Introduction
This is a TensorFlow framework for the identification of ATLAS electrons by using neural networks.


# Training at LPS  
1) ssh -Y atlas16  
(login to atlas16 for GPU's avaibility)	  
2) cd /opt/tmp/$USER  
(change to user directory)  
3) ln -s /opt/tmp/godin/el_data/2020-03-24/el_data.h5 .  
(link data file to user directory)  
4) git clone https://github.com/dominiquegodin/el_classifier.git  
(clone framework from GitHub)  
5) cd el_classifier  
(enter framework directory)
6) singularity shell --nv --bind /opt /opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif  
(activate the virtual environment of TensorFlow2.1.0+Python3.6.8 Singularity image)  
(use the flag --nv or not to wether run on GPUs or CPUs)
7) python batch_classifier.py [OPTIONS]  
(start training; see options below)
8) nvidia-smi  
(for monitoring NVIDIA GPU devices, e.g. memory and power usage, temperature, fan speed, etc.)


# Training on Beluga Cluster
1) ssh -Y $USER@beluga.calculquebec.ca  
(login to Beluga cluster)	  
2) cd /home/$USER  
(change to user directory)  
3) ln -s /project/def-arguinj/dgodin/el_data/2020-03-24/el_data.h5 .  
(link data file to user directory)  
4) git clone https://github.com/dominiquegodin/el_classifier.git  
(clone framework from GitHub)  
5) cd el_classifier  
(enter framework directory)  


# Using Slurm jobs manager
1) sbatch slurm.sh  
(run classifier.sh script and sent jobs to Slurm batch system)  
2) squeue or sview 
(report status of job) 
3) scancel 'job_id' 
(cancel job) 
4) srun --jobid 'job_id' --pty watch -n 2 nvidia-smi  
(monitor jobs GPU usage at 2s interval)  
4) salloc --time=00:30:00 --gres=gpu:1 --mem=16G --x11 --account=def-arguinj  
(use Slurm interactively and request appropriate ressources)


# classifier.py Options
--n_train     = number of training electrons (default=1e5)

--n_valid     = number of testing electrons (default=1e5)

--batch_size  = size of training batches (default=1000)

--n_epochs    = number of training epochs (default=100)

--n_classes   = number of classes (default=2)

--n_gpus      = number of gpus for multiprocessing (default=4)

--NN_type     = CNN or FCN specify the type of neural networks (default=CNN)

--plotting    = ON plots accuracy history, distributions separation and ROC curves (default=OFF)

--weight_file = name of h5 weight file from a previous training checkpoint (requires .h5 extension)  

--scaling     = ON applies Quantile transform to scalar variables (fit performed on train sample
	        and applied to whole sample)  

--checkpoint  = name of h5 checkpoint file used for saving and updating the model best weights

--weight_type = name of weighting method, either of 'none' (default), 'match2b', 'match2s', 'flattening' should be given

--cuts        = applied cuts on physics variables 

--output_dir  = name of output directory, which should be useful when you run jobs in parallel

# Explanations
1) For each epoch where the validation performance (either accuracy or loss function) has reached its best so far, the training model will automatically be saved to a h5 file checkpoint. 
2) An early stopping callback allows the training to stop automatically when the validation performance has stop improving for a pre-determined number of epochs (default=10).  
3) Finished or aborted trainings can be resumed from where they were stopped by using previously trained weights of other same-model
h5 file checkpoints (see --weight_file option).
4) All plots, weights and models are saved by default in the "outputs" directory.
5) To use pre-trained weights for generating plots without training, just specifiy n_epochs = 0.
6) In order to optimize data transfer rate, datasets should physically be present on the same server of the GPU's. Significant access speed gain is achieved by simply linking to "/opt/tmp/godin/el_data" as shown above. 
