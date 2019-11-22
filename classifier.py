# PACKAGES IMPORTS
import tensorflow as tf, numpy as np, multiprocessing, time, os, sys
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import make_indices, load_files, call_generator, Batch_Generator
from   models    import CNN_multichannel
from   plots     import val_accuracy,  plot_history, plot_distributions, plot_ROC_curves, cal_images


parser = ArgumentParser()
parser.add_argument( '--generator'   , default='OFF'            )
parser.add_argument( '--plotting'    , default='OFF'            )
parser.add_argument( '--symmetries'  , default='OFF'            )
parser.add_argument( '--cal_images'  , default='OFF'            )
parser.add_argument( '--checkpoint'  , default='checkpoint.h5'  )
parser.add_argument( '--load_weights', default='OFF'            )
parser.add_argument( '--epochs'      , default=100,  type=int   )
parser.add_argument( '--batch_size'  , default=500,  type=int   )
parser.add_argument( '--random_state', default=None, type=int   )
parser.add_argument( '--gpus'        , default=2,    type=int   )
args = parser.parse_args()


# DATAFILES DEFINITIONS
if not os.path.isdir('outputs'): os.mkdir('outputs')
file_path  = '/opt/tmp/godin/el_data/2019-11-05/'
data_files = [file_path+file for file in os.listdir(file_path) if '.h5' in file]


# FEATURES AND ARCHITECTURES
images       = [ 's0','s1', 's2', 's3','s4','s5','s6' ] #+ [ 'tracks' ]
tracks       = [ 'tracks' ]
scalars      = [ 'Eratio', 'Reta', 'Rhad', 'Rphi', 'TRTPID', 'd0', 'd0Sig', 'dPhiRes2',
                 'dPoverP', 'deltaEta1', 'e', 'eta', 'f1', 'f3', 'mu', 'nSCT', 'weta2']
#features     = {'images':[], 'tracks':[], 'scalars':['pt','phi']}
features     = {'images':images, 'tracks':tracks, 'scalars':scalars}
architecture = 'CNN_multichannel'


# IF ACTIVATED: PLOTTING CALORIMETER IMAGES AND EXITING
if args.cal_images == 'ON': cal_images(data_files, images)


# RANDOM TRAIN AND TEST INDICES GENERATION
train_indices, test_indices = make_indices(data_files, random_state=args.random_state)


# CPU COUNT FOR MULTIPROCESSING
for n in np.arange( multiprocessing.cpu_count(),0,-1):
    if test_indices[0].shape % n == 0: n_cpus = n ; break


# IMAGES SHAPES AND TRANSFORMS
single_sample = load_files(data_files, train_indices, batch_size=len(train_indices), index=0)
images_shape  = single_sample[0]['s4'].shape
tracks_shape  = single_sample[0]['tracks'].shape
transforms    = {'target_shape':images_shape, 'normalize':False}


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
if args.generator == 'ON':
    n_gpus = min(1,len(tf.config.experimental.list_physical_devices('GPU')))
    model = CNN_multichannel(images_shape, tracks_shape, len(data_files), **features)
    print() ; model.summary()
    if '.h5' in args.load_weights:
        print('\nCLASSIFIER: loading weights from', args.load_weights)
        model.load_weights(args.load_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    n_gpus  = min(args.gpus, len(tf.config.experimental.list_physical_devices('GPU')))
    devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
    tf.debugging.set_log_device_placement(False)
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
    with mirrored_strategy.scope():
        model = CNN_multichannel(images_shape, tracks_shape, len(data_files), **features)
        print() ; model.summary()
        if '.h5' in args.load_weights:
            print('\nCLASSIFIER: loading weights from', args.load_weights)
            model.load_weights(args.load_weights)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# DATA SAMPLES PREPARATION
if args.generator == 'ON':
    train_batch_size, test_batch_size = args.batch_size, int(test_indices.size/n_cpus)
    train_generator = Batch_Generator(data_files, train_indices, train_batch_size, transforms, **features)
    test_generator  = Batch_Generator(data_files,  test_indices,  test_batch_size, transforms, **features)
else:
    pooler       = multiprocessing.Pool(n_cpus)
    print('\nCLASSIFIER: data generator is OFF')
    print(  'CLASSIFIER: loading data and transforming images')
    print(  'CLASSIFIER: multiprocessing with', n_cpus, 'CPUs')
    start_time   = time.time()
    train_pool   = partial(call_generator, data_files, train_indices,
                           train_indices.size/n_cpus, transforms, features)
    test_pool    = partial(call_generator, data_files,  test_indices,
                            test_indices.size/n_cpus, transforms, features)
    train_sample = pooler.map(train_pool, np.arange(0,n_cpus))
    test_sample  = pooler.map( test_pool, np.arange(0,n_cpus))
    train_data   = [ np.concatenate([n[0][feature] for n in train_sample])
                     for feature in np.arange(0,len(train_sample[0][0])) ]
    test_data    = [ np.concatenate([n[0][feature] for n in  test_sample])
                     for feature in np.arange(0,len( test_sample[0][0])) ]
    train_labels =   np.concatenate([n[1] for n in train_sample])
    test_labels  =   np.concatenate([n[1] for n in  test_sample])
    print('CLASSIFIER: loading and processing time:', format(time.time() - start_time,'.0f'), 's')


# CALLBACKS
monitored_value  = 'val_accuracy'
checkpoint_file  = 'outputs/'+args.checkpoint
Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint( checkpoint_file, save_best_only=True,
                                                       monitor=monitored_value, verbose=1 )
Early_Stopping   = tf.keras.callbacks.EarlyStopping  ( patience=20, restore_best_weights=True,
                                                       monitor=monitored_value, verbose=1 )
callbacks_list   = [Model_Checkpoint, Early_Stopping]


# TRAINING AND TESTING
print('\nCLASSIFIER: training Sample:',  train_indices.size, 'e')
print(  'CLASSIFIER: testing Sample:  ',  test_indices.size, 'e')
print('\nCLASSIFIER: using TensorFlow',      tf.__version__     )
print(  'CLASSIFIER: using',                n_gpus, 'GPU(s)'    )
print('\nCLASSIFIER: using',                   architecture     )
print(  'CLASSIFIER: starting training\n'                       )
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
        pooler = multiprocessing.Pool(n_cpus)
    else:
        y_true = test_labels
        y_prob = model.predict(test_data)
    print('\nCLASSIFIER: last checkpoint validation accuracy:', val_accuracy(y_true,y_prob), '\n')
    test_pool  = partial(load_files, data_files, test_indices, test_indices.size/n_cpus)
    test_data  = np.concatenate( pooler.map(test_pool, np.arange(0,n_cpus)) )
    plot_history(training)
    plot_distributions(y_true, y_prob)
    plot_ROC_curves(data_files, test_data, test_indices, y_true, y_prob, ROC_type=1)
    plot_ROC_curves(data_files, test_data, test_indices, y_true, y_prob, ROC_type=2)
    #plot_ROC_curves(data_files, test_data, test_indices, y_true, y_prob, ROC_type=3)
