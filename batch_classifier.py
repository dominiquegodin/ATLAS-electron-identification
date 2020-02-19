# PACKAGES IMPORTS
import tensorflow as tf, numpy as np, h5py, os, time, sys
from   argparse   import ArgumentParser
from   utils      import make_sample, make_labels, filter_sample
from   utils      import analyze_sample, balance_sample, transform_sample, show_matrix
from   plots      import valid_accuracy, plot_history, plot_distributions, plot_ROC_curves
from   plots      import plot_scalars, plot_tracks
from   models     import multi_CNN


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_train'    , default=1e5, type=float )
parser.add_argument( '--n_valid'    , default=1e5, type=float )
parser.add_argument( '--batch_size' , default=1e3, type=float )
parser.add_argument( '--n_epochs'   , default=100, type=int   )
parser.add_argument( '--n_classes'  , default=2  , type=int   )
parser.add_argument( '--n_gpus'     , default=4  , type=int   )
parser.add_argument( '--NN_type'    , default='CNN'           )
parser.add_argument( '--scalars'    , default='OFF'           )
parser.add_argument( '--plotting'   , default='OFF'           )
parser.add_argument( '--weight_file', default='OFF'           )
parser.add_argument( '--transform'  , default='ON'            )
parser.add_argument( '--rebalance'  , default='OFF'           )
parser.add_argument( '--checkpoint' , default='checkpoint.h5' )
args = parser.parse_args(); float16 = tf.__version__ >= '2.1.0'
print('\nCLASSIFIER OPTIONS:')
for key, val in vars(args).items(): print(format(key,'11s'), '-->', int(val) if type(val)==float else val)


# TRAINING VARIABLES
images    = ['em_barrel_Lr0'  , 'em_barrel_Lr1_fine', 'em_barrel_Lr2'  , 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2'   , 'tile_barrel_Lr3', 'tracks_image']
tracks    = ['tracks' ]
scalars   = ['p_Eratio', 'p_Reta', 'p_Rhad', 'p_Rphi', 'p_TRTPID', 'p_d0', 'p_d0Sig', 'p_dPOverP',
             'p_deltaPhiRescaled2', 'p_deltaEta1', 'p_f1', 'p_f3', 'p_numberOfSCTHits', 'p_weta2']
others    = ['p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose', 'p_e', 'mcChannelNumber']
train_var = {'images':images, 'tracks':[], 'scalars':scalars}
#train_var = {'images':[], 'tracks':[], 'scalars':scalars}
total_var = {**train_var, 'others':others}
if train_var['images'] == []: args.NN_type = 'FCN'
print('\nTRAINING VARIABLES:'); n = 1
for var in np.sum(list(train_var.values())): print(format(n, '2.0f' )+'-', var); n+=1


# DATAFILE PATH
if not os.path.isdir('outputs'): os.mkdir('outputs')
checkpoint_file = 'outputs/'+args.checkpoint
data_file       = '/opt/tmp/godin/el_data/2020-02-18/el_data.h5'
#data_file       = '/project/def-arguinj/dgodin/el_data/2019-12-10/el_data.h5'


# TEST AND TRAIN SAMPLES LIMITS
n_max   = len(h5py.File(data_file, 'r')['p_TruthType'])
n_train = [0          , min(n_max, int(args.n_train))            ]
n_valid = [n_train[-1], min(n_train[-1]+int(args.n_valid), n_max)]


# TEST SAMPLE GENERATION
print('\nCLASSIFIER: loading test sample', n_valid, end=' ... ', flush=True)
start_time   = time.time()
valid_sample = make_sample(data_file, total_var, n_valid, float16)
valid_labels = make_labels(valid_sample, args.n_classes)
print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
valid_sample, valid_labels = filter_sample(valid_sample, valid_labels)


# SAMPLE ANALYSIS AND PLOTS
#analyze_sample(valid_sample, scan=True); sys.exit()
#plot_tracks(valid_sample['tracks_image'], valid_labels, 'deta'); sys.exit()
#valid_sample_trans = valid_sample.copy()
#valid_sample_trans = transform_sample(valid_sample_trans, valid_sample_trans, scalars)[0]
#for key in train_var['scalars']: plot_scalars(valid_sample, valid_sample_trans, key); sys.exit()


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
tf.debugging.set_log_device_placement(False)
strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
with strategy.scope():
    if float16: tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    model = multi_CNN(args.n_classes, args.NN_type, valid_sample, **train_var)
    print(); model.summary()
    if '.h5' in args.weight_file:
        print('\nCLASSIFIER: loading weights from', args.weight_file)
        model.load_weights(args.weight_file)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# TRAINING
if args.n_epochs >= 1:
    print('\nCLASSIFIER: train sample:'   , format(n_train[1] -n_train[0], '8.0f'), 'e')
    print(  'CLASSIFIER:  test sample:'   , format(n_valid[1] -n_valid[0], '8.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__                             )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)'                           )
    print('\nCLASSIFIER: using'           , args.NN_type, 'architecture with', end=' ' )
    print([group for group in train_var if train_var[group] != []]                     )
if args.n_epochs >= 1 or args.transform == 'ON':
    for idx in list(zip(n_train[:-1], n_train[1:])):
        print('\nCLASSIFIER: loading train sample', n_train, end=' ... ', flush=True)
        start_time   = time.time()
        train_sample = make_sample(data_file, total_var, idx, float16)
        train_labels = make_labels(train_sample, args.n_classes)
        print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        train_sample, train_labels = filter_sample(train_sample, train_labels)
        analyze_sample(train_sample, scan=False)
        if args.rebalance == 'ON':
            train_sample, train_labels = balance_sample(train_sample, train_labels, args.n_classes)
        if args.transform == 'ON' and train_var['scalars'] != []:
            train_sample, valid_sample = transform_sample(train_sample, valid_sample, train_var['scalars'])
if args.n_epochs >= 1:
    show_matrix(train_labels, valid_labels)
    class_ratios = [np.sum(train_labels==m)/len(train_labels) for m in np.arange(args.n_classes)]
    class_weight = {m:1/(class_ratios[m]*args.n_classes)      for m in np.arange(args.n_classes)}
    Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True,
                                                          monitor='val_accuracy', verbose=1)
    Early_Stopping   = tf.keras.callbacks.EarlyStopping  (patience=10, restore_best_weights=True,
                                                          monitor='val_accuracy', verbose=1)
    training = model.fit( train_sample, train_labels, validation_data=(valid_sample,valid_labels),
                          callbacks=[Model_Checkpoint, Early_Stopping], verbose=1,
                          class_weight=class_weight,
                          batch_size=max(1,n_gpus)*int(args.batch_size), epochs=args.n_epochs )
    model.load_weights(checkpoint_file)


# PLOTTING SECTION
print('\nCLASSIFIER: test sample', n_valid, 'class predictions')
valid_prob = model.predict(valid_sample, batch_size=20000, verbose=1)
if args.n_epochs < 1: train_labels = valid_labels
show_matrix(train_labels, valid_labels, valid_prob)
print('TEST SAMPLE ACCURACY:', format(100*valid_accuracy(valid_labels, valid_prob), '.2f'), '%\n')
if args.plotting == 'ON' and args.n_classes == 2:
    if args.n_epochs > 1: plot_history(training)
    plot_distributions(valid_labels, valid_prob)
    plot_ROC_curves(valid_sample, valid_labels, valid_prob, ROC_type=1)
    plot_ROC_curves(valid_sample, valid_labels, valid_prob, ROC_type=2)
