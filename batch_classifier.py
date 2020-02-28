# IMPORT PACKAGES AND FUNCTIONS
import tensorflow as tf, numpy as np, multiprocessing, time, os, sys, h5py
import tensorflow.keras.callbacks as cb
from   argparse   import ArgumentParser
from   tabulate   import tabulate
from   utils      import valid_data, train_data, compo_matrix, class_weights
from   plots      import valid_accuracy, plot_history, plot_distributions
from   plots      import plot_ROC_curves, plot_scalars, plot_tracks
from   models     import multi_CNN


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'    ,  default =  1e5,  type = float )
parser.add_argument( '--n_valid'    ,  default =  1e5,  type = float )
parser.add_argument( '--batch_size' ,  default =  1e3,  type = float )
parser.add_argument( '--n_epochs'   ,  default =  100,  type = int   )
parser.add_argument( '--n_classes'  ,  default =    2,  type = int   )
parser.add_argument( '--n_tracks'   ,  default =   10,  type = int   )
parser.add_argument( '--n_gpus'     ,  default =    4,  type = int   )
parser.add_argument( '--l2'         ,  default = 1e-6,  type = float )
parser.add_argument( '--dropout'    ,  default =  0.2,  type = float )
parser.add_argument( '--alpha'      ,  default =    0,  type = float )
parser.add_argument( '--NN_type'    ,  default = 'CNN'               )
parser.add_argument( '--images'     ,  default = 'ON'                )
parser.add_argument( '--scalars'    ,  default = 'ON'                )
parser.add_argument( '--plotting'   ,  default = 'ON'                )
parser.add_argument( '--scaling'    ,  default = 'ON'                )
parser.add_argument( '--resampling' ,  default = 'OFF'               )
parser.add_argument( '--metrics'    ,  default = 'val_accuracy'      )
parser.add_argument( '--checkpoint' ,  default = 'checkpoint.h5'     )
parser.add_argument( '--weight_file',  default =  None               )
parser.add_argument( '--pickle_file',  default = 'scaler.pkl'        )
parser.add_argument( '--scaler_file',  default = 'scaler.pkl'        )
parser.add_argument( '--cuts'       ,  default =  None               )
args = parser.parse_args()
#for key, val in vars(args).items(): vars(args)[key]= int(val) if type(val)==float else val
if '.h5' not in args.weight_file and args.n_epochs < 1:
    print('\nCLASSIFIER: no valid weight file -> exiting program\n'); sys.exit()
if '.h5' not in args.weight_file: args.weight_file = None


# DATAFILE PATH AND SAMPLES SIZES
if not os.path.isdir('outputs'): os.mkdir('outputs')
checkpoint_file = 'outputs/' + args.checkpoint
data_file       = '/opt/tmp/godin/el_data/2020-02-28/el_data.h5'
#data_file       = '/project/def-arguinj/dgodin/el_data/2019-12-10/el_data.h5'
n_max        = len(h5py.File(data_file, 'r')['mcChannelNumber'])
args.n_train = [0               , min(n_max, int(args.n_train)                 )]
args.n_valid = [args.n_train[-1], min(args.n_train[-1]+int(args.n_valid), n_max)]


# TRAINING VARIABLES
images    = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image']
tracks    = ['tracks']
scalars   = ['p_Eratio', 'p_Reta', 'p_Rhad', 'p_Rphi', 'p_TRTPID', 'p_d0',
             'p_d0Sig', 'p_dPOverP', 'p_deltaPhiRescaled2', 'p_deltaEta1',
             'p_f1', 'p_f3', 'p_numberOfSCTHits', 'p_weta2', 'p_nTracks']
others    = ['p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose',
             'p_e', 'p_eta', 'p_et_calo']
train_var = {'images' :images  if args.images =='ON' else [], 'tracks':[],
             'scalars':scalars if args.scalars=='ON' else []}
all_var   = {**train_var, 'others':others}; scalars = train_var['scalars']


# ARGUMENTS AND VARIABLES SUMMARY
print('\nCLASSIFIER OPTIONS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [key            for key in train_var if train_var[key]!=[]]
table   = [train_var[key] for key in train_var if train_var[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql'))
if train_var['images'] == []: args.NN_type = 'FCN'
args.scaling = args.scaling == 'ON' and scalars != []


# IFF AND PHYSICS CUTS ON SAMPLES
#args.cuts = 'sample["p_et_calo"]  >= 20'
#args.cuts = '(sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'
#args.cuts = 'abs(sample["p_eta"]) <=0.6' #args.cuts = 'abs(sample["p_eta"])>0.6'


# TEST SAMPLE GENERATION
print('\nCLASSIFIER: loading test sample', args.n_valid, end=' ... ', flush=True)
#for key, val in vars(args).items(): exec(key + '= val')
arguments = (data_file, all_var, scalars, args.n_valid, args.n_tracks, args.n_classes,
             args.scaling, args.pickle_file, args.weight_file, args.cuts)
valid_sample, valid_labels = valid_data(*arguments)


# ARCHITECTURE SELECTION AND MULTI-GPU DISTRIBUTION
n_gpus  = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
tf.debugging.set_log_device_placement(False)
strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
with strategy.scope():
    if tf.__version__ >= '2.1.0': tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    model = multi_CNN(args.n_classes, args.NN_type, valid_sample, args.l2, args.dropout, args.alpha, **train_var)
    print(); model.summary()
    if args.weight_file != None:
        print('\nCLASSIFIER: loading pre-trained weights from: outputs/' + args.weight_file)
        model.load_weights('outputs/' + args.weight_file)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# TRAINING LOOP
if args.n_epochs >= 1:
    print('\nCLASSIFIER: train sample:'   , format(args.n_train[1] -args.n_train[0], '8.0f'), 'e')
    print(  'CLASSIFIER:  test sample:'   , format(args.n_valid[1] -args.n_valid[0], '8.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__                                       )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)'                                     )
    print('\nCLASSIFIER: using'           , args.NN_type, 'architecture with', end=' '           )
    print([group for group in train_var if train_var[group] != [ ]]                              )
    for idx in list(zip(args.n_train[:-1], args.n_train[1:])):
        print('\nCLASSIFIER: loading train sample', args.n_train, end=' ... ', flush=True)
        arguments = (data_file, valid_sample, all_var, scalars, idx, args.n_tracks,
                     args.n_classes, args.resampling, args.scaling, args.scaler_file,
                     args.pickle_file, args.weight_file, args.cuts)
        train_sample, valid_sample, train_labels = train_data(*arguments)
    compo_matrix(train_labels, valid_labels); print()
    checkpoint = cb.ModelCheckpoint(checkpoint_file, save_best_only=True, monitor=args.metrics, verbose=1)
    early_stop = cb.EarlyStopping(patience=10, restore_best_weights=True, monitor=args.metrics, verbose=1)
    training   = model.fit( train_sample, train_labels, validation_data=(valid_sample,valid_labels),
                            callbacks=[checkpoint, early_stop], epochs=args.n_epochs, verbose=1,
                            class_weight=class_weights(train_labels),
                            batch_size=max(1,n_gpus)*int(args.batch_size) )
    model.load_weights(checkpoint_file)


# RESULTS AND PLOTTING SECTION
print('\nCLASSIFIER: test sample', args.n_valid, 'class predictions')
valid_prob = model.predict(valid_sample, batch_size=20000, verbose=1)
if args.n_epochs < 1: train_labels = valid_labels
compo_matrix(train_labels, valid_labels, valid_prob)
print('TEST SAMPLE ACCURACY:', format(100*valid_accuracy(valid_labels, valid_prob), '.2f'), '%\n')
if args.plotting == 'ON' and args.n_classes == 2:
    if args.n_epochs > 1: plot_history(training)
    plot_distributions(valid_labels, valid_prob)
    plot_ROC_curves(valid_sample, valid_labels, valid_prob, ROC_type=1)
    plot_ROC_curves(valid_sample, valid_labels, valid_prob, ROC_type=2)
