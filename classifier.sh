#!/bin/bash

python cross_classifier.py --n_train=15e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100     \
                           --n_classes=2   --n_tracks=10  --n_folds=10      --fold_number=$VAR \
                           --verbose=2     --NN_type=CNN  --plotting=OFF    --weight_type=None \
                           --output_dir=test --Beluga_nod=OFF

#--checkpoint=checkpoint_$VAR.h5