# IMPORTS
import tensorflow as tf, numpy as np, multiprocessing, time, os, sys
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import make_indices, load_files, inverse_images
from   utils     import process_images, Batch_Generator, Get_Batch
from   models    import CNN_multichannel
from   plots     import accuracy, plot_accuracy, plot_distributions
from   plots     import plot_ROC1_curve, plot_ROC2_curve, cal_images


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
args = parser.parse_args()


# DATAFILES DEFINITIONS
file_path  = '/lcg/storage16/atlas/godin/el_data/2019-11-05/'
#file_path  = '/opt/tmp/godin/el_data/2019-11-05/'
file_names = [file_path+file for file in os.listdir(file_path) if '.h5' in file]


# FEATURES AND ARCHITECTURES
images       = [ 's0','s1', 's2', 's3','s4','s5','s6' ] #+ [ 'tracks' ]
tracks       = [ 'tracks' ]
scalars      = [ 'Eratio', 'Reta', 'Rhad', 'Rphi', 'TRTPID', 'd0', 'd0Sig', 'dPhiRes2',
                 'dPoverP', 'deltaEta1', 'e', 'eta', 'f1', 'f3', 'mu', 'nSCT', 'weta2']
architecture = 'CNN_multichannel'
#features     = {'images':[], 'tracks':[], 'scalars':['pt','phi']}
features     = {'images':images, 'tracks':tracks, 'scalars':scalars}


# IF ACTIVATED: PLOTTING CALORIMETER IMAGES AND EXITING
if args.cal_images == 'ON': cal_images(file_names, images)


# RANDOM TRAIN AND TEST INDICES GENERATION
train_indices, test_indices = make_indices(file_names, random_state=args.random_state)


# CPU COUNT FOR MULTIPROCESSING
for n in np.arange( multiprocessing.cpu_count(),0,-1):
    if test_indices[1].shape % n == 0: n_cpus = n ; break


# ARCHITECTURE SELECTION AND MULTI-GPU PROCESSING
single_sample = load_files(file_names, train_indices, batch_size=len(train_indices), index=0)
images_shape, tracks_shape = single_sample[0]['s1'].shape, single_sample[0]['tracks'].shape
if args.generator == 'ON':
    n_gpus = min(1,len(tf.config.experimental.list_physical_devices('GPU')))
    model = CNN_multichannel(images_shape, tracks_shape, len(file_names), **features)
    print() ; model.summary()
    if '.h5' in args.load_weights:
        print('\nCLASSIFIER: loading weights from', args.load_weights)
        model.load_weights(args.load_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    tf.debugging.set_log_device_placement(False)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = CNN_multichannel(images_shape, tracks_shape, len(file_names), **features)
        print() ; model.summary()
        if '.h5' in args.load_weights:
            print('\nCLASSIFIER: loading weights from', args.load_weights)
            model.load_weights(args.load_weights)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# DATA SAMPLES GENERATION
if args.generator == 'ON':
    train_batch_size, test_batch_size = args.batch_size, 500
    train_generator = Batch_Generator(file_names, train_indices, train_batch_size, **features)
    test_generator  = Batch_Generator(file_names,  test_indices,  test_batch_size, **features)
else:
    print('\nCLASSIFIER: data generator is OFF\nCLASSIFIER: loading data using', n_cpus, 'CPUs...')
    start_time    = time.time()
    #train_sample  = load_files(file_names, train_indices, batch_size=train_indices.size, index=0)
    #test_sample   = load_files(file_names,  test_indices, batch_size= test_indices.size, index=0)
    pooler        = multiprocessing.Pool(n_cpus)
    pool_func     = partial(load_files, file_names, train_indices, train_indices.size/n_cpus)
    train_sample  = np.concatenate( pooler.map(pool_func, np.arange(0,n_cpus)) )
    pool_func     = partial(load_files, file_names,  test_indices,  test_indices.size/n_cpus)
    test_sample   = np.concatenate( pooler.map(pool_func, np.arange(0,n_cpus)) )
    print('CLASSIFIER: loading time:', format(time.time() - start_time,'.0f'), 's\n')

    print('CLASSIFIER: processing images...')
    start_time    = time.time()
    #train_images  = pad_images(train_sample, features['images'], padding=True)
    #test_images   = pad_images(test_sample,  features['images'], padding=True)
    train_images  = process_images(train_sample, features['images'], normalize=False)
    test_images   = process_images(test_sample,  features['images'], normalize=False)
    print('CLASSIFIER: processing time:', format(time.time() - start_time,'.0f'), 's')

    train_tracks  = [ train_sample[f] for f in features['tracks' ] ]
    train_scalars = [ train_sample[f] for f in features['scalars'] ]
    train_data    = train_images + train_tracks + train_scalars
    test_data     =  test_images + [ test_sample[f] for f in features['tracks']+features['scalars']]
    train_labels  = train_sample['truthmode']
    test_labels   =  test_sample['truthmode']


# DATA AUGMENTATION WITH IMAGES SYMMETRIES
if args.symmetries == 'ON' and args.generator != 'ON':
    print('\nCLASSIFIER: using data augmentation with images symmetries\n')
    train_data, train_labels = inverse_images(train_images, train_tracks, train_scalars, train_labels)


# CALLBACKS
monitored_value  = 'val_accuracy'
checkpoint_file  = 'outputs/'+args.checkpoint
Model_Checkpoint = tf.keras.callbacks.ModelCheckpoint( checkpoint_file, save_best_only=True,
                                                       monitor=monitored_value, verbose=1 )
Early_Stopping   = tf.keras.callbacks.EarlyStopping  ( patience=20, restore_best_weights=True,
                                                       monitor=monitored_value, verbose=1 )
callbacks_list   = [Model_Checkpoint, Early_Stopping]
if args.generator == 'ON': callbacks_list.append(Get_Batch(train_generator))


# TRAINING AND TESTING
print('\nCLASSIFIER: training Sample:',  train_indices.size, 'e')
print(  'CLASSIFIER: testing Sample:  ',  test_indices.size, 'e')
print('\nCLASSIFIER: using TensorFlow',   tf.__version__        )
print(  'CLASSIFIER: using',              n_gpus, 'GPU(s)'      )
print('\nCLASSIFIER: using model',        architecture          )
print(  'CLASSIFIER: starting training\n'                       )
if args.generator == 'ON':
    print('CLASSIFIER: using batches generator with', n_cpus, 'CPUs'                          )
    print('CLASSIFIER: training batches:',  len(train_generator), 'x', train_batch_size, 'e'  )
    print('CLASSIFIER: testing batches:  ', len(test_generator ), 'x',  test_batch_size, 'e\n')
    model_history = model.fit_generator( generator       = train_generator,
                                         validation_data =  test_generator,
                                         callbacks=callbacks_list,
                                         workers=n_cpus, use_multiprocessing=True,
                                         epochs=args.epochs, shuffle=True, verbose=1 )
else:
    model_history = model.fit          ( train_data, train_labels,
                                         validation_data=(test_data,test_labels),
                                         callbacks=callbacks_list, epochs=args.epochs,
                                         workers=n_cpus, use_multiprocessing=True,
                                         batch_size=max(1,n_gpus)*args.batch_size, verbose=1 )


#PLOTTING SECTION
if args.plotting == 'ON':
    model.load_weights(checkpoint_file)
    if args.generator == 'ON':
        generator = Batch_Generator(file_names, test_indices, test_batch_size, **features)
        print('\nCLASSIFIER: recovering truth labels for plotting functions (generator batches:',
               len(generator), 'x', test_batch_size, 'e)')
        y_true = np.concatenate([ generator[i][1] for i in np.arange(0,len(generator)) ])
        y_prob = model.predict_generator(generator, verbose=1, workers=n_cpus, use_multiprocessing=True)
    else:
        y_true = test_labels
        y_prob = model.predict(test_data)
    print('\nCLASSIFIER: last checkpoint validation accuracy:', accuracy(y_true,y_prob), '\n')
    plot_accuracy(model_history)
    plot_distributions(y_true, y_prob)
    plot_ROC1_curve(file_names, test_indices, y_true, y_prob)
    plot_ROC2_curve(file_names, test_indices, y_true, y_prob)
