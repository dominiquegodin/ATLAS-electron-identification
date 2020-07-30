# IMPORT PACKAGES AND FUNCTIONS
import numpy     as np, multiprocessing, time, os, sys, h5py
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import presample, merge_presamples


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'         , default = None        , type=float )
parser.add_argument( '--n_tasks'     , default = 12          , type=int   )
parser.add_argument( '--n_files'     , default = 20          , type=int   )
parser.add_argument( '--output_file' , default = 'el_data.h5'             )
parser.add_argument( '--sampling'    , default = 'ON'                     )
parser.add_argument( '--merging'     , default = 'ON'                     )
parser.add_argument( '--file_path'   , default = ''                       )
args = parser.parse_args()


#data_path = '/opt/tmp/godin/el_data/2019-06-20'
#data_path = '/opt/tmp/godin/el_data/2020-05-08/0.0_1.3'
data_path = '/opt/tmp/godin/el_data/2020-05-08/1.3_1.6'
#data_path = '/opt/tmp/godin/el_data/2020-05-08/1.6_2.5'


# DATAFILES DEFINITIONS
if args.file_path == '': args.file_path = data_path
if not os.path.isdir(args.file_path+'/'+'output'): os.mkdir(args.file_path+'/'+'output')
output_path = args.file_path+'/'+'output'
data_files  = [args.file_path+'/'+h5_file for h5_file in os.listdir(args.file_path) if '.h5' in h5_file]
data_files  = sorted(data_files)[0:args.n_files]


#data = h5py.File(data_files[0], 'r')
#for key in data: print(key)
#for key, val in data['train'].items(): print(key, val.shape)
#sys.exit()


# MERGING FILES/ NO PRESAMPLING
if args.sampling == 'OFF' and args.merging == 'ON':
    temp_files = [output_path+'/'+h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
    sum_e      = [len(h5py.File(h5_file, 'r')['mcChannelNumber']) for h5_file in temp_files]
    print(); merge_presamples(sum_e[0], len(temp_files), output_path, args.output_file)
    print(sum(sum_e), 'ELECTRONS COLLECTED\n'); sys.exit()
if args.sampling == 'OFF' and args.merging == 'OFF': sys.exit()


# ELECTRONS VARIABLES
images  =  ['em_barrel_Lr0'   , 'em_barrel_Lr1'   , 'em_barrel_Lr2'   , 'em_barrel_Lr3'                   ,
            'em_endcap_Lr0'   , 'em_endcap_Lr1'   , 'em_endcap_Lr2'   , 'em_endcap_Lr3'                   ,
            'lar_endcap_Lr0'  , 'lar_endcap_Lr1'  , 'lar_endcap_Lr2'  , 'lar_endcap_Lr3'                  ,
            'tile_barrel_Lr1' , 'tile_barrel_Lr2' , 'tile_barrel_Lr3' , 'tile_gap_Lr1'                    ]
tracks  =  ['tracks_pt'       , 'tracks_phi'      , 'tracks_eta'      , 'tracks_d0'        , 'tracks_z0'  ,
            'p_tracks_pt'     , 'p_tracks_phi'    , 'p_tracks_eta'    , 'p_tracks_d0'      , 'p_tracks_z0',
            'p_tracks_charge' , 'p_tracks_vertex' , 'p_tracks_chi2'   , 'p_tracks_ndof'    ,
            'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits', 'p_tracks_sigmad0'                ]
scalars =  ['p_truth_pt'      , 'p_truth_phi'     , 'p_truth_eta'     , 'p_truth_E'        , 'p_truth_e'  ,
            'p_et_calo'       , 'p_pt_track'      , 'p_EoverP'        , 'p_Eratio'         , 'p_phi'      ,
            'p_eta'           , 'p_e'             , 'p_Rhad'          , 'p_Rphi'           , 'p_Reta'     ,
            'p_d0'            , 'p_d0Sig'         , 'p_sigmad0'       , 'p_dPOverP'        , 'p_f1'       ,
            'p_f3'            , 'p_weta2'         , 'p_TRTPID'        , 'p_deltaEta1'      , 'p_LHValue'  ,
            'p_chi2'          , 'p_ndof'          , 'p_ECIDSResult'   , 'p_wtots1'         , 'p_EptRatio' ,
            'correctedAverageMu', 'p_deltaPhiRescaled2'                                                   ]
integers = ['p_truthType'     , 'p_TruthType'     , 'p_iffTruth'      , 'p_nTracks'        , 'p_charge'   ,
            'mcChannelNumber' , 'eventNumber'     , 'p_LHTight'       , 'p_LHMedium'       , 'p_LHLoose'  ,
            'p_truthOrigin'   , 'p_TruthOrigin'   , 'p_numberOfSCTHits', 'p_numberOfInnermostPixelHits'   ,
            'p_firstEgMotherTruthType', 'p_firstEgMotherTruthOrigin'                                      ]


# REMOVING TEMPORARY FILES (if any)
temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
for h5_file in temp_files: os.remove(output_path+'/'+h5_file)


# STARTING SAMPLING AND COLLECTING DATA
n_tasks = min(multiprocessing.cpu_count(), args.n_tasks)
max_e   = [len(h5py.File(h5_file, 'r')['train']['mcChannelNumber']) for h5_file in data_files]
if args.n_e != None: max_e = [min(max_e+[int(args.n_e)//len(data_files)])] * len(data_files)
max_e = np.array(max_e)//n_tasks*n_tasks; sum_e = 0
print('\nSTARTING ELECTRONS COLLECTION (', '\b'+str(sum(max_e)), end=' ', flush=True)
print('electrons from', len(data_files),'files, using', n_tasks,'threads):')
pool = multiprocessing.Pool(n_tasks, maxtasksperchild=1)
for h5_file in data_files:
    batch_size = max_e[data_files.index(h5_file)]//n_tasks; start_time = time.time()
    print('Collecting', batch_size*n_tasks, 'e from:', h5_file.split('/')[-1], end=' ... ', flush=True)
    func   = partial(presample, h5_file, output_path, batch_size, sum_e, images, tracks, scalars, integers)
    sample = pool.map(func, np.arange(n_tasks))
    sum_e += batch_size
    print ('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
pool.close()


# MERGING FILES
if args.merging=='ON': merge_presamples(sum_e, n_tasks, output_path, args.output_file); print()
