# IMPORT PACKAGES AND FUNCTIONS
import tensorflow      as tf
import numpy           as np
import os, sys, h5py, pickle
from   argparse  import ArgumentParser
from   tabulate  import tabulate
from   itertools import accumulate
from   utils     import get_dataset, validation, make_sample, merge_samples, sample_composition
from   utils     import compo_matrix, get_sample_weight, get_class_weight, gen_weights, sample_weights
from   utils     import cross_valid, valid_results, sample_analysis, sample_histograms, Batch_Generator
from   utils     import feature_removal, feature_ranking, fit_scaler, apply_scaler
from   models    import callback, create_model
#os.system('nvidia-modprobe -u -c=0') # for atlas15


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'        , default =  1e6,  type = float )
parser.add_argument( '--n_eval'         , default =    0,  type = float )
parser.add_argument( '--n_valid'        , default =  1e6,  type = float )
parser.add_argument( '--batch_size'     , default =  5e3,  type = float )
parser.add_argument( '--n_epochs'       , default =  100,  type = int   )
parser.add_argument( '--n_classes'      , default =    2,  type = int   )
parser.add_argument( '--n_tracks'       , default =    5,  type = int   )
parser.add_argument( '--bkg_ratio'      , default =    2,  type = float )
parser.add_argument( '--n_folds'        , default =    1,  type = int   )
parser.add_argument( '--n_gpus'         , default =    4,  type = int   )
parser.add_argument( '--verbose'        , default =    1,  type = int   )
parser.add_argument( '--patience'       , default =   10,  type = int   )
parser.add_argument( '--sbatch_var'     , default =    0,  type = int   )
parser.add_argument( '--node_dir'       , default = ''                  )
parser.add_argument( '--host_name'      , default = 'lps'               )
parser.add_argument( '--l2'             , default = 1e-7,  type = float )
parser.add_argument( '--dropout'        , default =  0.1,  type = float )
parser.add_argument( '--FCN_neurons'    , default = [200,200], type = int, nargs='+')
parser.add_argument( '--weight_type'    , default = 'none'              )
parser.add_argument( '--train_cuts'     , default = ''                  )
parser.add_argument( '--valid_cuts'     , default = ''                  )
parser.add_argument( '--NN_type'        , default = 'CNN'               )
parser.add_argument( '--images'         , default = 'ON'                )
parser.add_argument( '--scalars'        , default = 'ON'                )
parser.add_argument( '--scaling'        , default = 'ON'                )
parser.add_argument( '--plotting'       , default = 'OFF'               )
parser.add_argument( '--generator'      , default = 'OFF'               )
parser.add_argument( '--sep_bkg'        , default = 'OFF'               )
parser.add_argument( '--metrics'        , default = 'val_accuracy'      )
parser.add_argument( '--eta_region'     , default = ''                  )
parser.add_argument( '--output_dir'     , default = 'outputs'           )
parser.add_argument( '--model_in'       , default = ''                  )
parser.add_argument( '--model_out'      , default = 'model.h5'          )
parser.add_argument( '--scaler_in'      , default = ''                  )
parser.add_argument( '--scaler_out'     , default = 'scaler.pkl'        )
parser.add_argument( '--results_in'     , default = ''                  )
parser.add_argument( '--results_out'    , default = ''                  )
parser.add_argument( '--runDiffPlots'   , default = 0, type = int       )
parser.add_argument( '--feature_removal', default = 'OFF'               )
parser.add_argument( '--correlations'   , default = 'OFF'               )
args = parser.parse_args()


# VERIFYING ARGUMENTS
for key in ['n_train', 'n_eval', 'n_valid', 'batch_size']: vars(args)[key] = int(vars(args)[key])
if args.weight_type not in ['bkg_ratio', 'flattening', 'match2s', 'match2b', 'match2max', 'none']:
    print('\nweight_type: \"', args.weight_type, '\" not recognized, resetting it to none!!!')
    args.weight_type = 'none'
if '.h5' not in args.model_in and args.n_epochs < 1 and args.n_folds==1:
    print('\nERROR: weights file required with n_epochs < 1 --> aborting\n'); sys.exit()


# CNN PARAMETERS
CNN = {(56,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (4,1) , (2,1) ]},
        (7,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (1,1) , (1,1) ]},
        #(7,11):{'maps':[200,200], 'kernels':[(3,5,7),(3,5,1)], 'pools':[(1,1,1),(1,1,1)]},
      'tracks':{'maps':[200,200], 'kernels':[ (1,1) , (1,1) ], 'pools':[ (1,1) , (1,1) ]}}


# TRAINING VARIABLES
scalars = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'           ,
           'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2'         ,
           'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge'         ,
           'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_EoverP', 'p_wtots1' , 'p_numberOfInnermostPixelHits']
images  = [ 'em_barrel_Lr0',   'em_barrel_Lr1',   'em_barrel_Lr2',   'em_barrel_Lr3', 'em_barrel_Lr1_fine',
                                'tile_gap_Lr1',
            'em_endcap_Lr0',   'em_endcap_Lr1',   'em_endcap_Lr2',   'em_endcap_Lr3', 'em_endcap_Lr1_fine',
           'lar_endcap_Lr0',  'lar_endcap_Lr1',  'lar_endcap_Lr2',  'lar_endcap_Lr3',
                             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'      ]
others  = ['mcChannelNumber', 'eventNumber', 'p_TruthType', 'p_iffTruth'   , 'p_TruthOrigin', 'p_LHValue' ,
           'p_LHTight'      , 'p_LHMedium' , 'p_LHLoose'  , 'p_ECIDSResult', 'p_eta'        , 'p_et_calo' ,
           'p_vertexIndex'  , 'p_charge'   , 'p_firstEgMotherTruthType'    , 'p_firstEgMotherTruthOrigin' ,
           'correctedAverageMu', 'p_firstEgMotherPdgId'                                                   ]


# FEATURE REMOVAL IMPORTANCE RANKING
if args.feature_removal == 'ON':
    groups = [('em_barrel_Lr1','em_barrel_Lr1_fine'), ('em_barrel_Lr0','em_barrel_Lr2','em_barrel_Lr3')]
    scalars, images, removed_feature = feature_removal(scalars, images, groups=[], index=args.sbatch_var)
    args.output_dir += '/'+removed_feature


# DATASET AND TRAINING DICTIONARY
data_files = get_dataset(args.host_name, args.node_dir, args.eta_region)
keys       = set().union(*[h5py.File(data_file,'r').keys() for data_file in data_files])
images     = [key for key in images  if key in keys or key=='tracks_image']
scalars    = [key for key in scalars if key in keys]
others     = [key for key in others  if key in keys]
if args.scalars != 'ON': scalars=[]
if args.images  != 'ON': images =[]
if images == []: args.NN_type = 'FCN'
train_data = {'scalars':scalars, 'images':images}
input_data = {**train_data, 'others':others}


# SAMPLES SIZES AND APPLIED CUTS ON PHYSICS VARIABLES
sample_size  = sum([len(h5py.File(data_file,'r')['eventNumber']) for data_file in data_files])
args.n_train = [0, min(sample_size, args.n_train)]
args.n_valid = [args.n_train[1], min(args.n_train[1]+args.n_valid, sample_size)]
if args.n_valid[0] == args.n_valid[1]: args.n_valid = args.n_train
if args.n_eval != 0: args.n_eval = [args.n_valid[0], min(args.n_valid[1],args.n_valid[0]+args.n_eval)]
else               : args.n_eval = args.n_valid
#args.train_cuts = '(abs(sample["eta"]) > 0.8) & (abs(sample["eta"]) < 1.15)'
#args.valid_cuts = '(sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'
#args.valid_cuts = '(sample["p_et_calo"] >= 5) & (sample["p_vertexIndex"] != -999)'


# OBTAINING PERFORMANCE FROM EXISTING VALIDATION RESULTS
if os.path.isfile(args.output_dir+'/'+args.results_in) or os.path.islink(args.output_dir+'/'+args.results_in):
    if args.eta_region in ['0.0-1.3', '1.3-1.6', '1.6-2.5']:
        eta_1, eta_2    = args.eta_region.split('-')
        args.valid_cuts = '(abs(sample["eta"]) >= '+str(eta_1)+') & (abs(sample["eta"]) <= '+str(eta_2)+')'
    inputs = {'scalars':scalars, 'images':[], 'others':others}
    validation(args.output_dir, args.results_in, args.plotting, args.n_valid, data_files,
               inputs, args.valid_cuts, args.sep_bkg, args.runDiffPlots)
elif args.results_in != '': print('\nOption [--results_in] not matching any file --> aborting\n')
if   args.results_in != '': sys.exit()


# MODEL CREATION AND MULTI-GPU DISTRIBUTION
n_gpus = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
train_batch_size = max(1,n_gpus) * args.batch_size
valid_batch_size = max(1,n_gpus) * int(1e4)
sample = make_sample(data_files[0], [0,1], input_data, args.n_tracks, args.n_classes)[0]
model  = create_model(args.n_classes, sample, args.NN_type, args.FCN_neurons, CNN,
                      args.l2, args.dropout, train_data, n_gpus)


# ARGUMENTS AND VARIABLES SUMMARY
args.scaling = args.scaling == 'ON' and scalars != []
if args.NN_type == 'CNN':
    print('\nCNN ARCHITECTURES:')
    for shape in [shape for shape in CNN if shape in [sample[key].shape[1:] for key in sample]]:
        print(format(str(shape),'>8s')+':', str(CNN[shape]))
print('\nPROGRAM ARGUMENTS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [           key  for key in train_data if train_data[key]!=[]]
table   = [train_data[key] for key in train_data if train_data[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql')); print()
args.model_in   = args.output_dir+'/'  +args.model_in; args.model_out    = args.output_dir+'/'  +args.model_out
args.scaler_in  = args.output_dir+'/' +args.scaler_in; args.scaler_out   = args.output_dir+'/' +args.scaler_out
args.results_in = args.output_dir+'/'+args.results_in; args.results_out  = args.output_dir+'/'+args.results_out


# GENERATING VALIDATION SAMPLE AND LOADING PRE-TRAINED WEIGHTS
if os.path.isfile(args.model_in):
    print('CLASSIFIER: loading pre-trained weights from', args.model_in, '\n')
    model.load_weights(args.model_in)
if args.scaling and os.path.isfile(args.scaler_in):
    print('CLASSIFIER: loading scalars scaler from', args.scaler_in, '\n')
    scaler = pickle.load(open(args.scaler_in, 'rb'))
else: scaler = None
print('CLASSIFIER: LOADING', np.diff(args.n_valid)[0], 'VALIDATION SAMPLES')
if args.generator == 'ON':
    valid_gen = Batch_Generator(data_files, args.n_valid, input_data, args.n_tracks, args.n_classes,
                                valid_batch_size, args.valid_cuts, scaler, shuffle='OFF')
inputs = {'scalars':scalars, 'images':[], 'others':others} if args.generator == 'ON' else input_data
valid_sample, valid_labels, _ = merge_samples(data_files, args.n_valid, inputs, args.n_tracks, args.n_classes,
                                              args.valid_cuts, None if args.generator=='ON' else scaler)
#sample_analysis(valid_sample, valid_labels, scalars, scaler, args.output_dir); sys.exit()


# EVALUATING FEATURES CORRELATIONS
if args.correlations == 'ON':
    from importance import correlations
    correlations(images, scalars, valid_sample, valid_labels, args.eta_region, args.output_dir, args.images)


# TRAINING LOOP
if args.n_epochs > 0:
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    print(  'CLASSIFIER: train sample:'   , format(np.diff(args.n_train)[0], '9.0f'), 'e')
    print(  'CLASSIFIER: valid sample:'   , format(np.diff(args.n_valid)[0], '9.0f'), 'e')
    print('\nCLASSIFIER: using TensorFlow', tf.__version__ )
    print(  'CLASSIFIER: using'           , n_gpus, 'GPU(s)')
    print(  'CLASSIFIER: using'           , args.NN_type, 'architecture with', end=' ')
    print([key for key in train_data if train_data[key] != []], '\n')
    print('CLASSIFIER: LOADING', np.diff(args.n_train)[0], 'TRAINING SAMPLES')
    train_sample, train_labels, weight_idx = merge_samples(data_files, args.n_train, inputs, args.n_tracks,
                                                           args.n_classes, args.train_cuts, scaler=None)
    if args.scaling:
        if not os.path.isfile(args.scaler_in):
            scaler = fit_scaler(train_sample, scalars, args.scaler_out)
            if args.generator != 'ON': valid_sample = apply_scaler(valid_sample, scalars, scaler, verbose='OFF')
        if args.generator != 'ON'    : train_sample = apply_scaler(train_sample, scalars, scaler, verbose='ON')
    sample_composition(train_sample); compo_matrix(valid_labels, train_labels=train_labels); print()
    #sample_weight = sample_weights(train_sample,train_labels,args.n_classes,args.weight_type,args.output_dir)
    sample_weight = get_sample_weight(train_sample, train_labels, args.weight_type, args.bkg_ratio, hist='2D')
    sample_histograms(valid_sample, valid_labels, train_sample, train_labels, sample_weight, args.output_dir)
    callbacks = callback(args.model_out, args.patience, args.metrics); print()
    if args.generator == 'ON':
        del(train_sample)
        if np.all(sample_weight) != None: sample_weight = gen_weights(args.n_train, weight_idx, sample_weight)
        print('CLASSIFIER: LAUNCHING GENERATOR FOR', np.diff(args.n_train)[0], 'TRAINING SAMPLES')
        eval_gen  = Batch_Generator(data_files, args.n_eval, input_data, args.n_tracks, args.n_classes,
                                    valid_batch_size, args.valid_cuts, scaler, shuffle='OFF')
        train_gen = Batch_Generator(data_files, args.n_train, input_data, args.n_tracks, args.n_classes,
                                    train_batch_size, args.train_cuts, scaler, sample_weight, shuffle='ON')
        training  = model.fit( train_gen, validation_data=eval_gen, max_queue_size=max(1,n_gpus),
                               callbacks=callbacks, workers=1, epochs=args.n_epochs, verbose=args.verbose )
    else:
        eval_sample = {key:valid_sample[key][args.n_eval[0]:args.n_eval[1]] for key in valid_sample}
        eval_labels = valid_labels[args.n_eval[0]:args.n_eval[1]]
        training    = model.fit( train_sample, train_labels, validation_data=(eval_sample,eval_labels),
                                 callbacks=callbacks, sample_weight=sample_weight, batch_size=train_batch_size,
                                 epochs=args.n_epochs, verbose=args.verbose )
    model.load_weights(args.model_out); print()
else: train_labels = []; training = None


# RESULTS AND PLOTTING SECTION
if args.n_folds > 1:
    valid_probs = cross_valid(valid_sample, valid_labels, scalars, args.output_dir, args.n_folds, data_files,
                              args.n_valid, input_data, args.n_tracks, args.valid_cuts, model, args.generator)
else:
    print('Validation sample', args.n_valid, 'class predictions:')
    if args.generator == 'ON': valid_probs = model.predict(valid_gen, verbose=args.verbose)
    else: valid_probs = model.predict(valid_sample, batch_size=valid_batch_size, verbose=args.verbose)
bkg_rej = valid_results(valid_sample, valid_labels, valid_probs, train_labels, training,
                        args.output_dir, args.plotting, args.sep_bkg, args.runDiffPlots)
if '.pkl' in args.results_out:
    if args.feature_removal == 'ON':
        args.results_out = args.output_dir[0:args.output_dir.rfind('/')]+'/'+args.results_out.split('/')[-1]
        try: pickle.dump({removed_feature:bkg_rej}, open(args.results_out,'ab'))
        except IOError: print('FILE ACCESS CONFLICT FOR', removed_feature, '--> SKIPPING FILE ACCESS\n')
        feature_ranking(args.output_dir, args.results_out, scalars, images, groups=[])
    else:
        if args.n_folds > 1 and False: valid_data = (valid_probs,)
        else: valid_data = ({key:valid_sample[key] for key in others+['eta','pt']}, valid_labels, valid_probs)
        pickle.dump(valid_data, open(args.results_out,'wb'))
    print('\nValidation results saved to:', args.results_out, '\n')
