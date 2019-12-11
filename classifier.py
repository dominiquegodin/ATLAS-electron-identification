# PACKAGES IMPORTS
import tensorflow as tf, numpy as np, h5py, multiprocessing, time, os, sys
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import make_sample, make_labels, label_sizes, check_values, Batch_Generator
from   models    import CNN_multichannel
from   plots     import val_accuracy,  plot_history, plot_distributions, plot_ROC_curves, cal_images
from   sklearn.model_selection import train_test_split


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--generator'   , default='OFF'             )
parser.add_argument( '--plotting'    , default='OFF'             )
#parser.add_argument( '--cal_images'  , default='OFF'             )
parser.add_argument( '--checkpoint'  , default='checkpoint.h5'   )
parser.add_argument( '--load_weights', default='OFF'             )
parser.add_argument( '--epochs'      , default=100   ,  type=int )
parser.add_argument( '--batch_size'  , default=1000  ,  type=int )
parser.add_argument( '--random_state', default=0     ,  type=int )
parser.add_argument( '--n_gpus'      , default=4     ,  type=int )
parser.add_argument( '--n_classes'   , default=2     ,  type=int )
parser.add_argument( '--n_cpus'      , default=24    ,  type=int )
parser.add_argument( '--n_e'         , default=100000,  type=int )
args = parser.parse_args()


# TRAINING FEATURES
images    = ['em_barrel_Lr0',   'em_barrel_Lr1',   'em_barrel_Lr2', 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']
tracks    = ['tracks' ]
scalars   = ['p_Eratio', 'p_Reta', 'p_Rhad', 'p_Rphi', 'p_TRTPID', 'p_d0', 'p_d0Sig', 'p_dPOverP',
             'p_deltaPhiRescaled2', 'p_deltaEta1', 'p_f1', 'p_f3', 'p_numberOfSCTHits', 'p_weta2']
others    = ['p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose', 'p_e']
features  = {'images':images, 'tracks':tracks, 'scalars':scalars}
#features  = {'images':[], 'tracks':[], 'scalars':scalars}
complete  = np.sum(list(features.values())) + others


# DATAFILE PATH
data_file  =  '/opt/tmp/godin/el_data/2019-12-10/el_data.h5'


# TRAIN AND TEST INDICES GENERATION
#n_e = 5000000
if args.n_e=='ALL': n_e = len(h5py.File(data_file, 'r')['p_TruthType'])
else:               n_e = args.n_e
images_shape = h5py.File(data_file, 'r')['em_barrel_Lr1'].shape[1:]
tracks_shape = h5py.File(data_file, 'r')['tracks'       ].shape[1:]
train_indices, test_indices = train_test_split(np.arange(n_e), test_size=0.1,
                              random_state=args.random_state, shuffle=True)

# MULTIPROCESSING
for n in np.arange( min(args.n_cpus, multiprocessing.cpu_count()), 0, -1):
    if n_e % n == 0: n_cpus = n ; break


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
architecture = 'CNN_multichannel'
if args.generator == 'ON':
    n_gpus = min(1,len(tf.config.experimental.list_physical_devices('GPU')))
    model = CNN_multichannel(images_shape, tracks_shape, args.n_classes, **features)
    print() ; model.summary()
    if '.h5' in args.load_weights:
        print('\nCLASSIFIER: loading weights from', args.load_weights)
        model.load_weights(args.load_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
    devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
    tf.debugging.set_log_device_placement(False)
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
    with mirrored_strategy.scope():
        model = CNN_multichannel(images_shape, tracks_shape, args.n_classes, **features)
        print() ; model.summary()
        if '.h5' in args.load_weights:
            print('\nCLASSIFIER: loading weights from', args.load_weights)
            model.load_weights(args.load_weights)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# DATA SAMPLES PREPARATION
'''
if args.generator == 'ON':
    train_batch_size, test_batch_size = args.batch_size, int(test_indices.size/n_cpus)
    train_generator = Batch_Generator(data_files, train_indices, train_batch_size, **features)
    test_generator  = Batch_Generator(data_files,  test_indices,  test_batch_size, **features)
'''
print('\nCLASSIFIER: data generator is OFF\nCLASSIFIER: loading data', end='    ... ', flush=True)
start_time   = time.time()
train_data   = make_sample(data_file, images, complete, n_e, index=0)
print('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
print('CLASSIFIER: processing data', end=' ... ', flush=True)
start_time   = time.time()
test_data    = dict([key, np.take(train_data[key],  test_indices, axis=0)] for key in complete)
train_data   = dict([key, np.take(train_data[key], train_indices, axis=0)] for key in complete)
test_labels  = make_labels( test_data, args.n_classes)
train_labels = make_labels(train_data, args.n_classes)
test_LLH     = dict([key,   test_data[key]] for key in ['p_LHTight', 'p_LHMedium', 'p_LHLoose'])
test_data    = [np.float32( test_data[key]) for key in np.sum(list(features.values()))]
train_data   = [np.float32(train_data[key]) for key in np.sum(list(features.values()))]
print('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)\n')

#print('Train Data, Test Data:')
#for i in zip(train_data, test_data): print(i[0].shape,i[1].shape)
#print('Train Labels, Test Labels:')
#print(train_labels.shape, test_labels.shape)
#print(np.sum(list(features.values())))


# CALLBACKS
monitored_value  = 'val_accuracy'
checkpoint_file  = 'outputs/'+args.checkpoint
Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True,
                                                      monitor=monitored_value, verbose=1)
Early_Stopping   = tf.keras.callbacks.EarlyStopping  (patience=10, restore_best_weights=True,
                                                      monitor=monitored_value, verbose=1)
callbacks_list   = [Model_Checkpoint, Early_Stopping]


# TRAINING AND TESTING
print('\nCLASSIFIER: training sample:',  format(train_indices.size, '8.0f'), 'e')
print(  'CLASSIFIER: testing sample: ' , format( test_indices.size, '8.0f'), 'e')
label_sizes(train_labels, test_labels, args.n_classes)
print('\nCLASSIFIER: using TensorFlow', tf.__version__  )
print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)')
print('\nCLASSIFIER: using'           , architecture    )
print(  'CLASSIFIER: starting training ...\n'           )
if args.generator == 'ON':
    print('CLASSIFIER: using batches generator with', n_cpus, 'CPUs'                          )
    print('CLASSIFIER: training batches:',  len(train_generator), 'x', train_batch_size, 'e'  )
    print('CLASSIFIER: testing batches:  ', len(test_generator ), 'x',  test_batch_size, 'e\n')
    training = model.fit_generator ( generator       = train_generator,
                                     validation_data =  test_generator,
                                     callbacks=callbacks_list,
                                     workers=n_cpus, use_multiprocessing=True,
                                     epochs=args.epochs, shuffle=True, verbose=1 )
else:
    training = model.fit           ( train_data, train_labels,
                                     validation_data=(test_data,test_labels),
                                     callbacks=callbacks_list, epochs=args.epochs,
                                     workers=n_cpus, use_multiprocessing=True,
                                     batch_size=max(1,n_gpus)*args.batch_size, verbose=1 )

#PLOTTING SECTION
if args.plotting == 'ON':
    model.load_weights(checkpoint_file)
    if args.generator == 'ON':
        generator = Batch_Generator(data_files, test_indices, test_batch_size, transforms, **features)
        print('\nCLASSIFIER: recovering truth labels for plotting functions (generator batches:',
               len(generator), 'x', test_batch_size, 'e)')
        y_true = np.concatenate([ generator[i][1] for i in np.arange(0,len(generator)) ])
        y_prob = model.predict_generator(generator, verbose=1, workers=n_cpus, use_multiprocessing=True)
        pool   = multiprocessing.Pool(n_cpus)
    else:
        y_true = test_labels
        y_prob = model.predict(test_data)
    print('\nCLASSIFIER: last checkpoint validation accuracy:', val_accuracy(y_true,y_prob), '\n')
    plot_history(training)
    plot_distributions(y_true, y_prob)
    plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=1)
    plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=2)
    #plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=3)
