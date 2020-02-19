# PACKAGES IMPORTS
import numpy     as np, multiprocessing, time, os, h5py
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import presample, merge_presamples


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'    , default=None, type=float )
parser.add_argument( '--n_tasks', default=12  , type=int   )
parser.add_argument( '--output' , default='el_data.h5'     )
parser.add_argument( '--merge'  , default='ON'             )
args = parser.parse_args()


# DATAFILES DEFINITIONS
file_path   = '/opt/tmp/godin/el_data/2019-06-20/'
if not os.path.isdir(file_path+'output'): os.mkdir(file_path+'output')
output_path = file_path + 'output/'
data_files  = sorted([file_path+h5_file for h5_file in os.listdir(file_path) if '.h5' in h5_file])


# ELECTRONS FEATURES
images  = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2', 'em_barrel_Lr3'  ,
           'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3'                                        ]
tracks  = ['tracks_pt'      , 'tracks_phi'     , 'tracks_eta'   , 'tracks_d0'      , 'tracks_z0'          ]
scalars = ['p_TruthType'    , 'p_iffTruth'     , 'p_truth_pt'   , 'p_truth_phi'    , 'p_truth_eta'        ,
           'p_truth_E'      , 'p_et_calo'      , 'p_pt_track'   , 'p_Eratio'       , 'p_phi'              ,
           'p_eta'          , 'p_e'            , 'p_Rhad'       , 'p_Rphi'         , 'p_Reta'             ,
           'p_d0Sig'        , 'p_dPOverP'      , 'p_d0'         , 'p_f1'           , 'p_deltaPhiRescaled2',
           'p_f3'           , 'p_weta2'        , 'p_TRTPID'     , 'p_deltaEta1'    , 'p_numberOfSCTHits'  ,
           'p_LHTight'      , 'p_LHMedium'     , 'p_LHLoose'    , 'p_LHValue'      , 'mcChannelNumber'    ]
int_var = ['p_TruthType'    , 'p_iffTruth'     , 'eventNumber'  , 'mcChannelNumber',
           'p_LHTight'      , 'p_LHMedium'     , 'p_LHLoose'    , 'p_numberOfSCTHits'                     ]


# POSSIBLE NOMENCLATURE CHANGE (disabled right now)
'''
images  = {'em_barrel_Lr0'      :'ecal_L0'   , 'em_barrel_Lr1'  :'ecal_L1' ,
           'em_barrel_Lr2'      :'ecal_L2'   , 'em_barrel_Lr3'  :'ecal_L3' ,
           'tile_barrel_Lr1'    :'hcal_L1'   , 'tile_barrel_Lr2':'hcal_L2'   , 'tile_barrel_Lr3':'hcal_L3'}
scalars = {'p_TruthType'        :'MC_type'   , 'p_iffTruth'     :'IFF_type'  , 'p_truth_pt'     :'truth_pt' ,
           'p_truth_phi'        :'truth_phi' , 'p_truth_eta'    :'truth_eta' , 'p_truth_E'      :'truth_e'  ,
           'p_et_calo'          :'pt_calo'   , 'p_pt_track'     :'pt_track'  , 'p_Eratio'       :'e_ratio'  ,
           'p_phi'              :'phi'       , 'p_eta'          :'eta'       , 'p_e'            :'e'        ,
           'p_Rhad'             :'r_had'     , 'p_Rphi'         :'r_phi'     , 'p_Reta'         :'r_eta'    ,
           'p_d0Sig'            :'d0_sig'    , 'p_dPOverP'      :'dp_over_p' , 'p_d0'           :'d0'       ,
           'mcChannelNumber'    :'MC_channel', 'p_f1'           :'f1'        , 'p_f3'           :'f3'       ,
           'p_LHTight'          :'LLH_tight' , 'p_LHMedium'     :'LLH_medium', 'p_LHLoose'      :'LLH_loose',
           'p_numberOfSCTHits'  :'nSCT'      , 'p_LHValue'      :'LLH_value' , 'p_deltaEta1'    :'deta1'    ,
           'p_deltaPhiRescaled2':'dphi_res2' , 'p_TRTPID'       :'TRT_PID'   , 'p_weta2'        :'w_eta2'}
'''


# REMOVING TEMPORARY FILES (if any)
temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
for h5_file in temp_files: os.remove(output_path+h5_file)


# STARTING SAMPLING AND COLLECTING DATA
n_tasks = min(12, multiprocessing.cpu_count(), args.n_tasks)
max_e   = [len(h5py.File(h5_file, 'r')['train']['eventNumber']) for h5_file in data_files]
if args.n_e != None: max_e = [min(np.sum(max_e),int(args.n_e))//(len(data_files))] * len(data_files)
max_e   = np.array(max_e)//n_tasks*n_tasks; sum_e = 0
print('\nSTARTING ELECTRONS COLLECTION (', '\b'+str(sum(max_e)), end=' ', flush=True)
print('electrons from', len(data_files),'files, using', n_tasks,'threads):')
pool = multiprocessing.Pool(n_tasks)
for h5_file in data_files:
    batch_size = max_e[data_files.index(h5_file)]//n_tasks; start_time = time.time()
    print('Collecting', batch_size*n_tasks, 'e from:', h5_file.split('/')[-1], end=' ... ', flush=True)
    pool   = multiprocessing.Pool(n_tasks)
    func   = partial(presample, h5_file, output_path, batch_size, sum_e, images, tracks, scalars, int_var)
    sample = pool.map(func, np.arange(n_tasks))
    sum_e += batch_size
    print ('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
pool.close()


# MERGING FILES
if args.merge=='ON': merge_presamples(sum_e, n_tasks, output_path, args.output)
#print('TOTAL ELECTRONS COLLECTED:', sum_e*n_tasks, '\n')
