# PACKAGES IMPORTS
import numpy as np, multiprocessing, time, os, h5py
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import make_samples, merge_samples


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'   , default=None, type=int )
parser.add_argument( '--cpus'  , default=12  , type=int )
parser.add_argument( '--output', default='el_data.h5'   )
parser.add_argument( '--merge' , default='ON'           )
args = parser.parse_args()


# DATAFILES DEFINITIONS
file_path   = '/opt/tmp/godin/el_data/2019-06-20/'
if not os.path.isdir(file_path+'output'): os.mkdir(file_path+'output')
data_files  = sorted([file_path+h5_file for h5_file in os.listdir(file_path) if '.h5' in h5_file])
output_path = file_path + 'output/'


# ELECTRONS FEATURES
images  = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2', 'em_barrel_Lr3',
           'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3'                                       ]
tracks  = ['tracks_pt'      , 'tracks_phi'     , 'tracks_eta'   , 'tracks_d0'                            ]
scalars = ['p_TruthType'    , 'p_iffTruth'     , 'p_truth_pt'   , 'p_truth_phi'  , 'p_truth_eta'        ,
           'p_truth_E'      , 'p_et_calo'      , 'p_pt_track'   , 'p_Eratio'     , 'p_phi'              ,
           'p_eta'          , 'p_e'            , 'p_Rhad'       , 'p_Rphi'       , 'p_Reta'             ,
           'p_d0Sig'        , 'p_dPOverP'      , 'p_d0'         , 'p_f1'         , 'p_deltaPhiRescaled2',
           'p_f3'           , 'p_weta2'        , 'p_TRTPID'     , 'p_deltaEta1'  , 'p_numberOfSCTHits'  ,
           'p_LHTight'      , 'p_LHMedium'     , 'p_LHLoose'    , 'p_LHValue'    , 'eventNumber'         ]
int_var = ['p_TruthType'    , 'p_iffTruth'     , 'eventNumber'  ,
           'p_LHTight'      , 'p_LHMedium'     , 'p_LHLoose'    , 'p_numberOfSCTHits'                    ]


# POSSIBLE NOMENCLATURE CHANGE (disabled right now)
'''
images  = {'em_barrel_Lr0'  :'s0', 'em_barrel_Lr1'  :'s1', 'em_barrel_Lr2'  :'s2', 'em_barrel_Lr3':'s3',
           'tile_barrel_Lr1':'s4', 'tile_barrel_Lr2':'s5', 'tile_barrel_Lr3':'s6'}
scalars = {'p_TruthType'        :'truth_type', 'p_iffTruth'       :'truth_IFF' , 'p_truth_pt' :'truth_pt' ,
           'p_truth_phi'        :'truth_phi' , 'p_truth_eta'      :'truth_eta' , 'p_truth_E'  :'truth_e'  ,
           'p_et_calo'          :'pt'        , 'p_pt_track'       :'pt_track'  , 'p_Eratio'   :'e_ratio'  ,
           'p_phi'              :'phi'       , 'p_eta'            :'eta'       , 'p_e'        :'e'        ,
           'p_Rhad'             :'r_had'     , 'p_Rphi'           :'r_phi'     , 'p_Reta'     :'r_eta'    ,
           'p_d0Sig'            :'d0_sig'    , 'p_dPOverP'        :'dp_over_p' , 'p_d0'       :'d0'       ,
           'p_f1'               :'f1'        , 'p_f3'             :'f3'        , 'p_weta2'    :'w_eta2'   ,
           'p_LHTight'          :'LLH_tight' , 'p_LHMedium'       :'LLH_medium', 'p_LHLoose'  :'LLH_loose',
           'p_LHValue'          :'LLH_value' , 'p_deltaEta1'      :'d_eta1'    , 'p_TRTPID'   :'TRT_PID'  ,
           'p_deltaPhiRescaled2':'dphi_res2' , 'p_numberOfSCTHits':'nSCT'      , 'eventNumber':'eventNumber'}
'''


# REMOVING TEMPORARY FILES (if any)
temp_files = sorted([h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file])
for h5_file in temp_files: os.remove(output_path+h5_file)


# STARTING SAMPLING AND COLLECTING DATA
print('\nSTARTING ELECTRONS COLLECTION:')
n_cpus = min(multiprocessing.cpu_count(), args.cpus)
max_e  = [len(h5py.File(h5_file, 'r')['train']['eventNumber']) for h5_file in data_files]
if args.n_e != None: max_e = [args.n_e//(len(data_files))] * len(data_files)
max_e  = np.array(max_e)//n_cpus*n_cpus
pool   = multiprocessing.Pool(n_cpus)
sum_e  = 0
for h5_file in data_files:
    start_time = time.time()
    batch_size = max_e[data_files.index(h5_file)]//n_cpus
    print('Collecting', batch_size*n_cpus, 'e from:', h5_file.split('/')[-1], end=' ... ', flush=True)
    pool   = multiprocessing.Pool(n_cpus)
    func   = partial(make_samples, h5_file, output_path, batch_size, sum_e, images, tracks, scalars, int_var)
    sample = pool.map(func, np.arange(n_cpus))
    sum_e += batch_size
    print ('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
pool.close()


# MERGING FILES
if args.merge=='ON':
    start_time = time.time()
    merge_samples(sum_e, n_cpus, output_path, args.output)
    print (' (', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')


# TOTAL ELECTRONS COLLECTED
print('TOTAL ELECTRONS COLLECTED:', sum_e*n_cpus, '\n')
