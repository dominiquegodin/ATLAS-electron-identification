# IMPORT PACKAGES AND FUNCTIONS
import tensorflow as tf, tensorflow.keras.callbacks as cb
import numpy      as np, multiprocessing as mp, os, sys, h5py, pickle
from   argparse   import ArgumentParser
from   tabulate   import tabulate
from   itertools  import accumulate
from   utils      import make_sample, sample_composition, balance_sample, apply_scaler, load_scaler
from   utils      import compo_matrix, class_weights, cross_validation, valid_results, sample_analysis
from   utils      import sample_weights, get_bin_indices
from   models     import multi_CNN


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'     ,  default =  1e5,  type = float )
parser.add_argument( '--n_valid'     ,  default =  1e5,  type = float )
parser.add_argument( '--batch_size'  ,  default =  5e3,  type = float )
parser.add_argument( '--n_epochs'    ,  default =  100,  type = int   )
parser.add_argument( '--n_classes'   ,  default =    2,  type = int   )
parser.add_argument( '--n_tracks'    ,  default =    5,  type = int   )
parser.add_argument( '--n_folds'     ,  default =    1,  type = int   )
parser.add_argument( '--fold_number' ,  default =    0,  type = int   )
parser.add_argument( '--n_gpus'      ,  default =    4,  type = int   )
parser.add_argument( '--verbose'     ,  default =    1,  type = int   )
parser.add_argument( '--l2'          ,  default = 1e-8,  type = float )
parser.add_argument( '--dropout'     ,  default = 0.05,  type = float )
parser.add_argument( '--alpha'       ,  default =    0,  type = float )
parser.add_argument( '--CNN_neurons' ,  default = [200, 200]          )
parser.add_argument( '--FCN_neurons' ,  default = [200, 200]          )
parser.add_argument( '--weight_type' ,  default = None                )
parser.add_argument( '--NN_type'     ,  default = 'CNN'               )
parser.add_argument( '--images'      ,  default = 'ON'                )
parser.add_argument( '--scalars'     ,  default = 'ON'                )
parser.add_argument( '--cuts'        ,  default = ''                  )
parser.add_argument( '--metrics'     ,  default = 'val_accuracy'      )
parser.add_argument( '--resampling'  ,  default = 'OFF'               )
parser.add_argument( '--scaling'     ,  default = 'ON'                )
parser.add_argument( '--cross_valid' ,  default = 'OFF'               )
parser.add_argument( '--plotting'    ,  default = 'ON'                )
parser.add_argument( '--input'       ,  default = ''                  )
parser.add_argument( '--output_dir'  ,  default = 'outputs'           )
parser.add_argument( '--scaler_file' ,  default = 'scaler.pkl'        )
parser.add_argument( '--checkpoint'  ,  default = ''                  )
parser.add_argument( '--result_file' ,  default = ''                  )
args = parser.parse_args()


# OBTAINING PERFORMANCE FROM EXISTING VALIDATION RESULTS
if '.pkl' in args.result_file:
    result_file = args.output_dir+'/'+args.result_file
    if os.path.isfile(result_file):
        print('\nLOADING VALIDATION RESULTS FROM', result_file, '\n')
        sample, labels, probs = pickle.load(open(result_file, 'rb'))
        valid_results(sample, labels, probs, [], None, args.output_dir, args.plotting)
    sys.exit()


# PROGRAM ARGUMENTS VERIFICATIONS
for key in ['n_train', 'n_valid', 'batch_size']: vars(args)[key] = int(vars(args)[key])
#for key, val in vars(args).items(): vars(args)[key]= int(val) if type(val)==float else val
#for key, val in vars(args).items(): exec(key + '= val')
if args.weight_type not in ['flattening', 'match2s', 'match2b']: args.weight_type = None
if '.h5' not in args.checkpoint and args.n_epochs < 1:
    print('\nERROR: weight file required with n_epochs < 1 -> exiting program\n'); sys.exit()
if args.cross_valid == 'ON' and args.n_folds <= 1:
    print('\nERROR: n_folds must be greater than 1 for cross-validation -> exiting program\n'); sys.exit()
if args.n_folds > 1 and args.fold_number >= args.n_folds:
    print('\nERROR: fold_number must be smaller than n_folds -> exiting program\n'); sys.exit()


# DATAFILE AND PATHS
args.scaler_file = args.scaler_file if '.pkl' in args.scaler_file else ''
args.checkpoint  = args.checkpoint  if '.h5'  in args.checkpoint  else ''
scaler_file      =  args.output_dir+'/'+args.scaler_file
checkpoint       =  args.output_dir+'/'+args.checkpoint

for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
    if not os.path.isdir(path): os.mkdir(path)
#data_file = '/project/def-arguinj/dgodin/el_data/2020-03-24/el_data.h5'
data_file = '/opt/tmp/godin/el_data/2020-03-24/el_data.h5'
if args.input!='': data_file= args.input


# TRAINING VARIABLES
images    = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3',
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image']
tracks    = ['tracks']
scalars   = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'  ,
             'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2',
             'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge']
#scalars  += ['p_mean_efrac'  , 'p_mean_deta'   , 'p_mean_dphi'  , 'p_mean_d0'  , 'p_mean_z0'     ,
#             'p_mean_charge' , 'p_mean_vertex' , 'p_mean_chi2'  , 'p_mean_ndof', 'p_mean_pixhits',
#             'p_mean_scthits', 'p_mean_trthits', 'p_mean_sigmad0']
others    = ['eventNumber', 'p_TruthType', 'p_iffTruth', 'p_LHTight', 'p_LHMedium', 'p_LHLoose',
             'p_eta', 'p_et_calo','p_LHValue']
train_var = {'images' :images  if args.images =='ON' else [], 'tracks':[],
             'scalars':scalars if args.scalars=='ON' else []}
all_var   = {**train_var, 'others':others}; scalars = train_var['scalars']


# ARCHITECTURE SELECTION AND MULTI-GPU DISTRIBUTION
n_gpus    = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
devices   = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
tf.debugging.set_log_device_placement(False)
strategy  = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
sample, _ = make_sample(data_file, all_var, [0,1], args.n_tracks, args.n_classes)
with strategy.scope():
    if tf.__version__ >= '2.1.0': tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    model = multi_CNN(args.n_classes, args.NN_type, sample, args.l2, args.dropout,
                      args.alpha, args.CNN_neurons, args.FCN_neurons, **train_var)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# SAMPLES SIZES AND APPLIED CUTS ON PHYSICS VARIABLES
sample_size  = len(h5py.File(data_file, 'r')['eventNumber'])
args.n_train = [0, min(sample_size, args.n_train)] if args.cross_valid == 'OFF' else [0,0]
args.n_valid = [args.n_train[-1], min(args.n_train[-1]+args.n_valid, sample_size )]
if args.cross_valid == 'OFF' and args.n_folds > 1:
    args.n_valid = args.n_train
    args.cuts   += 'sample["eventNumber"]%'+str(args.n_folds)+' == '+str(args.fold_number)
#args.cuts += '(sample["p_et_calo"] >= 20)'
#args.cuts += '(sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'
#args.cuts += '(abs(sample["p_eta"]) > 0.6)'


# ARGUMENTS AND VARIABLES TABLES
args.NN_type = 'FCN' if train_var['images'] == [] else args.NN_type
args.scaling = (args.scaling == 'ON' and scalars != [])
print('\nPROGRAM ARGUMENTS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [          key  for key in train_var if train_var[key]!=[]]
table   = [train_var[key] for key in train_var if train_var[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql')); print()


# GENERATING VALIDATION SAMPLE AND LOADING PRE-TRAINED WEIGHTS
print('CLASSIFIER: loading valid sample', args.n_valid, end=' ... ', flush=True)
func_args = data_file, all_var, args.n_valid, args.n_tracks, args.n_classes, args.cuts
valid_sample, valid_labels = make_sample(*func_args)
#sample_analysis(valid_sample, valid_labels, scalars, scaler_file); sys.exit()
if args.cross_valid == 'OFF' and args.checkpoint != '':
    print('CLASSIFIER: loading pre-trained weights from', checkpoint, '\n')
    model.load_weights(checkpoint)
    if args.scaling: valid_sample = load_scaler(valid_sample, scalars, scaler_file)


# TRAINING LOOP
if args.cross_valid == 'OFF' and args.n_epochs >= 1:
    print(  'CLASSIFIER: train sample:'   , format(args.n_train[1] -args.n_train[0], '8.0f'), 'e')
    print(  'CLASSIFIER: valid sample:'   , format(args.n_valid[1] -args.n_valid[0], '8.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__ )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)')
    print('\nCLASSIFIER: using'           , args.NN_type, 'architecture with', end=' ')
    print([group for group in train_var if train_var[group] != [ ]])
    print('\nCLASSIFIER: loading train sample', args.n_train, end=' ... ', flush=True)
    weight_file = args.output_dir+'/checkpoint.h5'; scaler_out = args.output_dir+'/scaler.pkl'
    if args.n_folds > 1:
        weight_file = args.output_dir+'/checkpoint_'+str(args.fold_number)+'.h5'
        scaler_out  = args.output_dir+'/scaler_'    +str(args.fold_number)+'.pkl'
        args.cuts   = 'sample["eventNumber"]%' + str(args.n_folds)+' != '+str(args.fold_number)
    func_args = (data_file, all_var, args.n_train, args.n_tracks, args.n_classes, args.cuts)
    train_sample, train_labels = make_sample(*func_args); sample_composition(train_sample)
    if args.resampling == 'ON': train_sample, train_labels = balance_sample(train_sample, train_labels)
    if args.scaling:
        if args.checkpoint != '': train_sample = load_scaler(train_sample, scalars, scaler_file)
        else: train_sample, valid_sample = apply_scaler(train_sample, valid_sample, scalars, scaler_out)
    compo_matrix(valid_labels, train_labels=train_labels); print()
    check_point = cb.ModelCheckpoint(weight_file,     save_best_only=True, monitor=args.metrics, verbose=1)
    early_stop  = cb.EarlyStopping(patience=10, restore_best_weights=True, monitor=args.metrics, verbose=1)
    training = model.fit( train_sample, train_labels, validation_data=(valid_sample,valid_labels),
                          callbacks=[check_point,early_stop], epochs=args.n_epochs, verbose=args.verbose,
                          class_weight=None if args.n_classes==2 else class_weights(train_labels),
                          sample_weight=sample_weights(train_sample, train_labels, args.n_classes,
                          args.weight_type, args.output_dir), batch_size=max(1,n_gpus)*int(args.batch_size) )
    model.load_weights(weight_file)
else: train_labels = []; training = None


# RESULTS AND PLOTTING SECTION
if args.cross_valid == 'ON':
    valid_probs = cross_validation(valid_sample, valid_labels, scalars, model, args.output_dir, args.n_folds)
    print('MERGING ALL FOLDS AND PREDICTING CLASSES ...')
if args.cross_valid == 'OFF':
    print('\nValidation sample', args.n_valid, 'class predictions:')
    valid_probs = model.predict(valid_sample, batch_size=20000, verbose=args.verbose); print()
valid_results(valid_sample, valid_labels, valid_probs, train_labels, training, args.output_dir, args.plotting)
if args.n_folds <= 1:
    print('Saving validation results to', args.output_dir+'/'+'valid_results.pkl', '\n')
    valid_sample = {key:valid_sample[key] for key in others}
    pickle.dump((valid_sample,valid_labels,valid_probs), open(args.output_dir+'/'+'valid_results.pkl','wb'))
