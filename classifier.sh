#!/bin/bash

#python batch_classifier.py --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
#                           --n_classes=2   --n_tracks=4   --n_folds=1       --fold_number=$VAR \
#                           --verbose=2     --NN_type=CNN  --plotting=OFF    --weight_type=None \
#                           --output_dir=outputs/2c_10m_4tracks-images

python batch_classifier.py --n_train=15e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
                           --n_classes=6   --n_tracks=5   --n_folds=10      --fold_number=$VAR \
                           --verbose=2     --NN_type=CNN  --plotting=OFF    --weight_type=None \
                           --output_dir=outputs/6c_cross_val

#--checkpoint=checkpoint_$VAR.h5