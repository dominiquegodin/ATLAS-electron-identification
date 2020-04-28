#!/bin/bash #echo $VAR


# SINGLE TRAINING
#python classifier.py --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
#                     --n_classes=2   --n_tracks=10  --l2=1e-8         --dropout=0.05     \
#                     --verbose=2     --NN_type=CNN  --plotting=ON     --weight_type=None \
#                     --output_dir=outputs/test      --sbatch_var=$VAR1


# ARRAY TRAINING
#python classifier.py --n_train=10e6      --n_valid=15e6       --batch_size=5e3   --n_epochs=100  --n_classes=2 \
#                     --n_tracks=${VAR}   --l2=1e-8            --dropout=0.05     --verbose=2     --NN_type=CNN \
#                     --plotting=OFF      --weight_type=None   --scalars=OFF      --FCN_neurons 200 200         \
#                     --output_dir  outputs/n_tracks/test      --model_out   model_${VAR}-tracks.h5             \
#                     --scaler_out  scaler_${VAR}-tracks.pkl   --results_out results_${VAR}-tracks.pkl


# FOLD TRAINING (array mode)
n_folds=10
for ((fold = 1; fold <= $n_folds;  fold++)) do
python classifier.py  --n_train=15e6                         --batch_size=5e3  --n_epochs=100  --n_classes=2  \
                      --n_tracks=${VAR}  --l2=1e-8           --dropout=0.05    --verbose=2     --NN_type=CNN  \
                      --plotting=OFF     --weight_type=None  --scalars=OFF     --n_folds=${n_folds}           \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'         \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'         \
                      --output_dir  outputs/n_tracks/cross-valid/2c_10m_kernel-1x1/${VAR}-tracks  \
                      --scaler_out   scaler_${fold}.pkl  \
                      --model_out     model_${fold}.h5   \
                      --results_out results_${fold}.h5
done


# CROSS-VALIDATION (array mode)
n_folds=10
python classifier.py  --n_valid=15e6  --n_classes=2   --n_tracks=${VAR}  --n_folds=${n_folds}  --verbose=2  \
                      --NN_type=CNN   --scalars=OFF   --plotting=ON      --cross_valid=ON                   \
                      --output_dir outputs/n_tracks/cross-valid/2c_10m_kernel-1x1/${VAR}-tracks


# USING RESULTS
#python classifier.py --output_dir=outputs/6c_cross-val/plots_0_vs_1 --results_in=results.pkl