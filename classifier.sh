#!/bin/bash
cd /home/dgodin/el_classifier
python batch_classifier.py --n_train=10e6 --n_test=10e6 --batch_size=5e3 --n_epochs=100 --n_classes=2 --NN_type=CNN --plotting=ON
