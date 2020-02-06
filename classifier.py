# PACKAGES IMPORTS
import tensorflow as tf, numpy as np, h5py, multiprocessing, time, os, sys
from argparse  import ArgumentParser
from utils     import make_data, class_filter, make_labels, class_matrix#, generator_sample, Batch_Generator
from models    import multi_CNNs
from plots     import test_accuracy, plot_history, plot_distributions, plot_ROC_curves
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from collections import Counter


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--generator'   , default='OFF'            )
parser.add_argument( '--plotting'    , default='OFF'            )
parser.add_argument( '--checkpoint'  , default='checkpoint.h5'  )
parser.add_argument( '--weight_file', default='OFF'            )
parser.add_argument( '--n_type'      , default='CNN'            )
parser.add_argument( '--n_epochs'    , default=100 , type=int   )
parser.add_argument( '--batch_size'  , default=1000, type=int   )
parser.add_argument( '--random_state', default=0   , type=int   )
parser.add_argument( '--n_classes'   , default=2   , type=int   )
parser.add_argument( '--n_gpus'      , default=4   , type=int   )
parser.add_argument( '--n_cpus'      , default=12  , type=int   )
parser.add_argument( '--n_e'         , default=1e5 , type=float )
args = parser.parse_args()


# TRAINING VARIABLES
images    = ['em_barrel_Lr0',   'em_barrel_Lr1_fine', 'em_barrel_Lr2', 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2',    'tile_barrel_Lr3']
tracks    = ['tracks' ]
scalars   = ['p_Eratio', 'p_Reta', 'p_Rhad', 'p_Rphi', 'p_TRTPID', 'p_d0', 'p_d0Sig', 'p_dPOverP',
             'p_deltaPhiRescaled2', 'p_deltaEta1', 'p_f1', 'p_f3', 'p_numberOfSCTHits', 'p_weta2']
others    = ['p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose', 'p_e']
train_var = {'images':images, 'tracks':tracks, 'scalars':scalars}
#train_var = {'images':[], 'tracks':[], 'scalars':scalars}
all_var   = np.sum(list(train_var.values())) + others
if train_var['images'] == []: args.n_type = 'FCN'


# DATAFILE PATH
if not os.path.isdir('outputs'): os.mkdir('outputs')
data_file = '/opt/tmp/godin/el_data/2019-12-10/el_data.h5'


# TRAIN AND TEST INDICES GENERATION
'''
max_e = len(h5py.File(data_file, 'r')['p_TruthType'])
train_indices, test_indices = train_test_split(np.arange(max_e), train_size=int(args.n_e),
                              random_state=args.random_state, shuffle=True)
train_indices, test_indices = train_test_split(train_indices, test_size=0.1,
                              random_state=args.random_state, shuffle=True)
start_time   = time.time()
train_data   = generator_sample(data_file, all_var, sorted(train_indices))
#tracks_shape = dict([key, train_data[key].shape[1:]] for key in train_var['tracks'])
#image_shapes = dict([key, train_data[key].shape[1:]] for key in train_var['images'])
print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)')
print(sorted(train_indices))
#print(test_indices)
sys.exit()
'''
shuffle = False if args.generator=='ON' else True
n_e = int(max(1e2, min(args.n_e, len(h5py.File(data_file, 'r')['p_TruthType']))))
train_indices, test_indices = train_test_split(np.arange(n_e), test_size=0.1,
                              random_state=args.random_state, shuffle=False)
#print(train_indices)
#print(test_indices)
#b_size = 10
#index = 1
#idx_1, idx_2 = train_indices[0] + index*b_size, train_indices[0] +(index+1)*b_size
#print(idx_1,idx_2) 
#sys.exit()

'''
# Pipeline
#data    = np.arange(10)
train_data = make_sample(data_file, 10, all_var, train_var['images'], upscale=False)
data1 = train_data['p_d0'] ; data2 =train_data['tracks']
dataset1 = tf.data.Dataset.from_tensor_slices(data1)
dataset2 = tf.data.Dataset.from_tensor_slices(data2)
dataset  = tf.data.Dataset.zip((dataset1,dataset2))
print(type(dataset))
for elem in dataset: print(elem[0].numpy())
print(dataset.element_spec)
sys.exit()
'''

# MULTIPROCESSING
for n in np.arange( min(args.n_cpus, multiprocessing.cpu_count()), 0, -1):
    if n_e % n == 0: n_cpus = n ; break


# DATA SAMPLES PREPARATION
if args.generator == 'ON':
    train_batch_size = test_batch_size = args.batch_size
    train_generator = Batch_Generator(data_file,    args.n_classes, train_var,
                                      all_var, train_indices , train_batch_size)
    test_generator  = Batch_Generator(data_file,    args.n_classes, train_var,
                                      all_var,  test_indices ,  test_batch_size)
    train_data   = make_data(data_file, 1, all_var, train_var['images'], upscale=False)
    tracks_shape = dict([key, train_data[key].shape[1:]] for key in train_var['tracks'])
    image_shapes = dict([key, train_data[key].shape[1:]] for key in train_var['images'])
else:
    print('\nCLASSIFIER: data generator is OFF\nCLASSIFIER: loading data', end='    ... ', flush=True)
    start_time   = time.time()
    train_data   = make_data(data_file, all_var, train_var['images'], [0, n_e], floats16=False,  upscale=False)
    tracks_shape = dict([key, train_data[key].shape[1:]] for key in train_var['tracks'])
    image_shapes = dict([key, train_data[key].shape[1:]] for key in train_var['images'])
    print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)')
    print('CLASSIFIER: processing data', end=' ... ', flush=True)
    start_time   = time.time()
    test_data    = dict([key, np.take(train_data[key],  test_indices, axis=0)] for key in all_var)
    train_data   = dict([key, np.take(train_data[key], train_indices, axis=0)] for key in all_var)
    test_labels  = make_labels( test_data, args.n_classes)
    train_labels = make_labels(train_data, args.n_classes)
    test_LLH     = dict([key,   test_data[key]] for key in ['p_LHTight', 'p_LHMedium', 'p_LHLoose'])
    #test_data    = [np.float32( test_data[key]) for key in np.sum(list(train_var.values()))]
    #train_data   = [np.float32(train_data[key]) for key in np.sum(list(train_var.values()))]
    #test_data    = [ test_data[key] for key in np.sum(list(train_var.values()))]
    #train_data   = [train_data[key] for key in np.sum(list(train_var.values()))]
    train_data, train_labels = class_filter(train_data, train_labels)
    test_data, test_labels = class_filter(test_data, test_labels)
    print('(', '\b'+format(time.time() - start_time,'2.1f'), '\b'+' s)\n')


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
if args.generator == 'ON' or n_e < 1e6:
    n_gpus = min(1,len(tf.config.experimental.list_physical_devices('GPU')))
    model = multi_CNNs(args.n_classes, args.n_type, image_shapes, tracks_shape, **train_var)
    print() ; model.summary()
    if '.h5' in args.weight_file:
        print('\nCLASSIFIER: loading weights from', args.weight_file)
        model.load_weights(args.weight_file)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
    devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
    tf.debugging.set_log_device_placement(False)
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
    with mirrored_strategy.scope():
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        model = multi_CNNs(args.n_classes, args.n_type, image_shapes, tracks_shape, **train_var)
        print() ; model.summary()
        if '.h5' in args.weight_file:
            print('\nCLASSIFIER: loading weights from', args.weight_file)
            model.load_weights(args.weight_file)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# CALLBACKS
monitored_value  = 'val_accuracy'
checkpoint_file  = 'outputs/'+args.checkpoint
Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True,
                                                      monitor=monitored_value, verbose=1)
Early_Stopping   = tf.keras.callbacks.EarlyStopping  (patience=10, restore_best_weights=True,
                                                      monitor=monitored_value, verbose=1)
callbacks_list   = [Model_Checkpoint, Early_Stopping]


# TRAINING AND TESTING
print('\nCLASSIFIER: training sample:', format(train_indices.size, '8.0f'), 'e')
print(  'CLASSIFIER: testing sample: ', format( test_indices.size, '8.0f'), 'e')
if args.generator != 'ON': class_matrix(train_labels, test_labels)
print(  'CLASSIFIER: using TensorFlow', tf.__version__  )
print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)')
print('\nCLASSIFIER: using', args.n_type, 'architecture with', end=' ')
print([group for group in train_var.keys() if train_var[group] != []])
print(  'CLASSIFIER: starting training ...\n')
if args.generator == 'ON':
    print('CLASSIFIER: using batches generator with', n_cpus, 'CPUs'                           )
    print('CLASSIFIER: training batches:' , len(train_generator),  'x', train_batch_size, 'e'  )
    print('CLASSIFIER: testing batches:  ', len( test_generator ), 'x',  test_batch_size, 'e\n')
    training = model.fit_generator ( generator       = train_generator,
                                     validation_data =  test_generator,
                                     callbacks=callbacks_list,
                                     workers=n_cpus, use_multiprocessing=True,
                                     epochs=args.n_epochs, shuffle=True, verbose=1 )
else:
    #training = model.train_on_batch( train_data, train_labels, reset_metrics=True )
    #class_weight = {0:1., 1:2.}
    training = model.fit           ( train_data, train_labels,
                                     validation_data=(test_data,test_labels),
                                     callbacks=callbacks_list, epochs=args.n_epochs,
                                     #class_weight=class_weight,
                                     workers=n_cpus, use_multiprocessing=True,
                                     batch_size=max(1,n_gpus)*args.batch_size, verbose=1 )


# PLOTTING SECTION
model.load_weights(checkpoint_file)
if args.generator == 'ON':
    print('\nCLASSIFIER: recovering truth labels (generator batches:',
          len(test_generator), 'x', test_batch_size, 'e)')
    y_true = np.concatenate([ test_generator[i][1] for i in np.arange(0,len(test_generator)) ])
    y_prob = model.predict_generator(test_generator, verbose=1, workers=n_cpus, use_multiprocessing=True)
else:
    y_true = test_labels
    y_prob = model.predict(test_data)
print('\nCLASSIFIER: best test sample accuracy:', format(100*test_accuracy(y_true, y_prob), '.2f'), '%')
class_matrix(train_labels, y_true, y_prob)
if args.plotting == 'ON':
    plot_history(training)
    plot_distributions(y_true, y_prob)
    plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=1)
    plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=2)
