#!/bin/bash
python batch_classifier.py --n_train=10e6 --n_valid=1e6 --batch_size=5e3  --n_epochs=100 \
                           --n_classes=2  --n_tracks=15  --NN_type=CNN    --plotting=ON  \
                           --weight_file=outputs/checkpoint
