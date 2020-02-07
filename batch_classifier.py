# PACKAGES IMPORTS
import tensorflow as tf, numpy as np, h5py, os, time, sys
from   argparse   import ArgumentParser
from   utils      import make_data, make_labels, balance_sample, filter_sample, show_matrix
from   models     import multi_CNN
from   plots      import test_accuracy, plot_history, plot_distributions, plot_ROC_curves


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_train'    , default=1e5, type=float )
parser.add_argument( '--n_test'     , default=1e5, type=float )
parser.add_argument( '--batch_size' , default=1e3, type=float )
parser.add_argument( '--n_epochs'   , default=100, type=int   )
parser.add_argument( '--n_classes'  , default=2  , type=int   )
parser.add_argument( '--n_gpus'     , default=4  , type=int   )
parser.add_argument( '--NN_type'    , default='CNN'           )
parser.add_argument( '--plotting'   , default='OFF'           )
parser.add_argument( '--weight_file', default='OFF'           )
parser.add_argument( '--rebalance'  , default='OFF'           )
parser.add_argument( '--checkpoint' , default='checkpoint.h5' )
args = parser.parse_args() ; float16  = tf.__version__ >= '2.1.0'


# TRAINING VARIABLES
images    = ['em_barrel_Lr0',   'em_barrel_Lr1_fine', 'em_barrel_Lr2', 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2',    'tile_barrel_Lr3']
tracks    = ['tracks' ]
scalars   = ['p_Eratio', 'p_Reta', 'p_Rhad', 'p_Rphi', 'p_TRTPID', 'p_d0', 'p_d0Sig', 'p_dPOverP',
             'p_deltaPhiRescaled2', 'p_deltaEta1', 'p_f1', 'p_f3', 'p_numberOfSCTHits', 'p_weta2']
others    = ['p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose', 'p_e']
train_var = {'images':images, 'tracks':tracks, 'scalars':scalars}
#train_var = {'images':[]    , 'tracks':[]    , 'scalars':scalars}
total_var = {**train_var, 'others':others}
if train_var['images'] == []: args.NN_type = 'FCN'


# DATAFILE PATH
if not os.path.isdir('outputs'): os.mkdir('outputs')
checkpoint_file = 'outputs/'+args.checkpoint
data_file       = '/opt/tmp/godin/el_data/2019-12-10/el_data.h5'
#data_file       = '/project/def-arguinj/dgodin/el_data/2019-12-10/el_data.h5'


# TEST AND TRAIN SAMPLES LIMITS
n_max   = len(h5py.File(data_file, 'r')['p_TruthType'])
n_train = [0          , int(args.n_train)                       ]
n_test  = [n_train[-1], min(n_train[-1]+int(args.n_test), n_max)]


# TEST SAMPLE GENERATION
print('\nCLASSIFIER: loading test sample', n_test, end=' ... ', flush=True)
start_time  = time.time()
test_data   = make_data(data_file, total_var, n_test, float16)
test_labels = make_labels(test_data, args.n_classes)
print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)')
test_data, test_labels = filter_sample(test_data, test_labels)


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
tf.debugging.set_log_device_placement(False)
strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
with strategy.scope():
    if float16: tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    model = multi_CNN(args.n_classes, args.NN_type, test_data, **train_var)
    print() ; model.summary()
    if '.h5' in args.weight_file:
        print('\nCLASSIFIER: loading weights from', args.weight_file)
        model.load_weights(args.weight_file)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# TRAINING
if args.n_epochs >= 1:
    print('\nCLASSIFIER: train sample:'   , format(n_train[1] -n_train[0], '8.0f'), 'e')
    print(  'CLASSIFIER:  test sample:'   , format(n_test [1] -n_test [0], '8.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__                             )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)'                           )
    print('\nCLASSIFIER: using'           , args.NN_type, 'architecture with', end=' ' )
    print([group for group in train_var if train_var[group] != []]                     )
    Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True,
                                                          monitor='val_accuracy', verbose=1)
    Early_Stopping   = tf.keras.callbacks.EarlyStopping  (patience=10, restore_best_weights=True,
                                                          monitor='val_accuracy', verbose=1)
    for idx_1, idx_2 in list(zip(n_train[:-1], n_train[1:])):
        print('\nCLASSIFIER: loading train sample', n_train, end=' ... ', flush=True)
        start_time   = time.time()
        train_data   = make_data(data_file, total_var, [idx_1,idx_2], float16)
        train_labels = make_labels(train_data, args.n_classes)
        print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)')
        train_data, train_labels = filter_sample(train_data, train_labels)
        show_matrix(train_labels, test_labels)
        if args.rebalance == 'ON':
            print('CLASSIFIER: rebalancing train sample', end=' ... ', flush=True)
            start_time   = time.time()
            train_data, train_labels = balance_sample(train_data, train_labels, args.n_classes)
            print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)\n')
        class_ratios = [np.sum(train_labels==m)/len(train_labels) for m in np.arange(args.n_classes)]
        class_weight = {m:1/(class_ratios[m]*args.n_classes)    for m in np.arange(args.n_classes)}
        training = model.fit( train_data, train_labels, validation_data=(test_data,test_labels),
                              callbacks=[Model_Checkpoint, Early_Stopping], verbose=1,
                              class_weight=class_weight,
                              batch_size=max(1,n_gpus)*int(args.batch_size), epochs=args.n_epochs )


# PLOTTING SECTION
model.load_weights(checkpoint_file)
print('\nCLASSIFIER: test sample', n_test, 'class predictions')
test_prob = model.predict(test_data, batch_size=20000, verbose=1)
if args.n_epochs < 1: train_labels = test_labels
show_matrix(train_labels, test_labels, test_prob)
print('TEST SAMPLE ACCURACY:', format(100*test_accuracy(test_labels, test_prob), '.2f'), '%\n')
if args.plotting == 'ON' and args.n_classes == 2:
    if args.n_epochs > 1: plot_history(training)
    plot_distributions(test_labels, test_prob)
    plot_ROC_curves(test_data, test_labels, test_prob, ROC_type=1)
    plot_ROC_curves(test_data, test_labels, test_prob, ROC_type=2)
