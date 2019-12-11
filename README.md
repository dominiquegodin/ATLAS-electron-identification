# Introduction
This is a TensorFlow framework for the identification of ATLAS electrons by using neural networks.

# Getting Started at LPS
1) ssh -Y atlas16  
(login to atlas16 for GPU's avaibility)	  
2) cd /opt/tmp/$USER  
(change to user directory)											  
3) ln -s /opt/tmp/godin/el_data/2019-12-10/el_data.h5 .  
(link datasets to user directory)  
4) git clone https://github.com/dominiquegodin/el_classifier.git  
(clone framework from GitHub)  
5) cd el_classifier  
(enter framework directory)
6) singularity shell --nv --bind /opt /opt/tmp/godin/sing_images/tf_2.0.0-gpu-py3.sif  
(activate the virtual environment of TensorFlow2.0.0+Python3.6.8 Singularity image)  
(use the flag --nv or not to wether run on GPUs or CPUs)
7) python classifier.py [OPTIONS]  
(start training; see options below)
8) nvidia-smi  
(for monitoring NVIDIA GPU devices, e.g. memory and power usage, temperature, fan speed, etc.)

# classifier.py Options
--generator=ON  (enables batches generator; default=OFF) NOTE: DISABLED FOR NOW!

--plotting=ON  (enables plotting of accuracy history, distributions separations ans ROC curve; default=OFF)

--cal_images=ON  (peforms no training and plots random calorimeter images for each layer; default=OFF)  

--checkpoint=file_name (h5 checkpoint file name used for saving and updating the model best weights)

--load_weights=file_name (uses weights from a previous training h5 file checkpoint; requires .h5 extension)  

--epochs=number_of_training_epochs  (default=100)

--batch_size=training_batch_size  (default=500) 

--n_gpus=number of gpus for multiprocessing (default=4)

--n_classes=number of classes (default=2)

--n_e=number of electrons (use ALL for full sample; default=100000)


# Explanations
1) For each epoch where the validation performance (either accuracy or loss function) has reached its best so far, the training model will automatically be saved to a h5 file checkpoint. 
2) An early stopping callback allows the training to stop automatically when the validation performance has stop improving for a pre-determined number of epochs (default=20).  
3) Although trainings with batches generator start without delay, they are unfortunately bottlenecked by hard disks data transfer rate and will therefore incur some slowing down. It is however recommended to use the generator for very big datasets that risk to overload the memory. It could also be used to speed up the start of small trainings for trials and errors or when playing with architectures and hyper-parameters.
4) Finished or aborted trainings can be resumed from where they were stopped by using previously trained weights of other same-model
h5 file checkpoints (see --load_weights option).
5) All plots, weights and models are saved by default in the "outputs" directory.
5) Data augmentation (with images inversions) can for the moment only be provided when the generator is not in use. 
7) Multi-GPU is enabled by default when the batches generator is OFF. Due to relatively slow hard disks transfer rates, GPU multi-processing will not result in any training speed improvements while using the generator.
8) In order to optimize data transfer rate, datasets should physically be present on the same server of the GPU's. Significant access speed gain is achieved by simply linking to "/opt/tmp/godin/el_data" as shown above. 
