#!/bin/bash


# SINGLE TRAINING
python classifier.py --n_train=1e6   --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
                     --n_classes=2   --n_tracks=5   --l2=1e-8         --dropout=0.05     \
                     --verbose=2     --NN_type=CNN  --plotting=ON     --weight_type=None \
                     --output_dir=outputs/test



################################################################################################################
#### PICK A SHELL SCRIPT FROM BELOW ############################################################################
################################################################################################################
exit


# SINGLE TRAINING
python classifier.py --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
                     --n_classes=2   --n_tracks=5   --l2=1e-8         --dropout=0.05     \
                     --verbose=2     --NN_type=CNN  --plotting=ON     --weight_type=None \
                     --output_dir=outputs/test


# TRAINING (array mode: multiple trainings)
python classifier.py --n_train=10e6      --n_valid=15e6       --batch_size=5e3   --n_epochs=100  --n_classes=2 \
                     --n_tracks=${VAR}   --l2=1e-8            --dropout=0.05     --verbose=2     --NN_type=CNN \
                     --plotting=OFF      --weight_type=None   --scalars=OFF      --FCN_neurons 200 200         \
                     --output_dir  outputs/test               --model_out   model_${VAR}-tracks.h5             \
                     --scaler_out  scaler_${VAR}-tracks.pkl   --results_out results_${VAR}-tracks.pkl


# CROSS-VALIDATION TRAINING (array mode)
n_e=1e6; n_epochs=2; n_classes=2; n_tracks=$VAR; n_folds=10; verbose=2; scalars=OFF; NN_type=CNN
output_dir=outputs/test/${VAR}-track
for ((fold = 1; fold <= $n_folds;  fold++)) do
python classifier.py  --n_train=$n_e        --batch_size=5e3     --n_epochs=$n_epochs --n_classes=$n_classes   \
                      --n_tracks=$n_tracks  --n_folds=${n_folds} --verbose=$verbose   --l2=1e-8 --dropout=0.05 \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'                      \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'                      \
                      --scalars=$scalars    --NN_type=$NN_type   --plotting=OFF       --weight_type=None       \
                      --output_dir $output_dir      --scaler_out scaler_${fold}.pkl                            \
		      --model_out model_${fold}.h5  --results_out results_${fold}.pkl
done
python classifier.py  --n_valid=$n_e     --n_classes=$n_classes  --n_tracks=$n_tracks --n_folds=${n_folds}     \
                      --verbose=$verbose --NN_type=$NN_type      --scalars=$scalars   --cross_valid=ON         \
                      --plotting=ON      --output_dir $output_dir


# USING RESULTS FOR PLOTTING
python classifier.py --output_dir=outputs/test --results_in=results.pkl