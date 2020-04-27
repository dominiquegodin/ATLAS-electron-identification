#!/bin/bash #echo $VAR1

# SINGLE TRAINING
python classifier.py --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
                     --n_classes=2   --n_tracks=10  --l2=1e-8         --dropout=0.05     \
                     --verbose=2     --NN_type=CNN  --plotting=ON     --weight_type=None \
                     --output_dir=outputs/test      --sbatch_var=$VAR1

# ARRAY TRAINING
#python classifier.py --n_train=10e6      --n_valid=15e6       --batch_size=5e3   --n_epochs=100  --n_classes=2 \
#                     --n_tracks=${VAR1}  --l2=1e-8            --dropout=0.05     --verbose=2     --NN_type=CNN \
#                     --plotting=OFF      --weight_type=None   --scalars=OFF      --FCN_neurons 200 200         \
#                     --output_dir  outputs/n_tracks/test      --model_out   model_${VAR1}-tracks.h5            \
#                     --scaler_out  scaler_${VAR1}-tracks.pkl  --results_out results_${VAR1}-tracks.pkl

# CROSS-VALIDATION TRAINING
#n_folds=10
#fold_number=$VAR1
#python classifier.py --n_train=1e6      --n_valid=1e6       --batch_size=5e3   --n_epochs=100  --n_classes=2 \
#                     --n_tracks=${VAR2} --l2=1e-8           --dropout=0.05     --verbose=2     --NN_type=CNN \
#                     --plotting=OFF     --weight_type=None  --scalars=OFF      --n_folds=${n_folds}          \
#                     --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='${fold_number}')'        \
#                     --valid_cuts '(sample["eventNumber"]%'${n_folds}'!='${fold_number}')'        \
#                     --output_dir  outputs/n_tracks/cross-valid/2c_10m_kernel-1x1/${VAR2}-tracks  \
#                     --scaler_out  scaler_fold-${VAR1}.pkl  --model_out model_fold-${VAR1}.h5     \
#                     --results_out results_fold-${VAR1}.h5

# POST CROSS-VALIDATION
#python classifier.py --output_dir=outputs/6c_cross-val/plots_0_vs_1 --result_file=results.pkl