from utils import print_importances
import pickle
import numpy as np

images   = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
            'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine']
images  += ['lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3']                             # Removes the empty images temporarily
images  += ['tile_gap_Lr1']
images  += ['tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']                                               # Removes the empty images temporarily
images  += ['tracks_image']
scalars  = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'  ,
            'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2',
            'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge',
            'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_wtots1', 'p_numberOfInnermostPixelHits']

feats = 'full' + images + scalars + ['group_{}'.format(g) for g in range(12)]
output_dir = '/scratch/odenis/2c_10m/none/barrel'

for feat in feats:
    try:
        file = output_dir + '/removal_importance/' + feat
        print('Opening:',file)
        f, bkg_rej = print_importances(file + '/removal_importance.pkl')
        pickle.dump((f,np.array([bkg_rej])), open(file  + '/importance.pkl', 'wb'))
    except OSError:
        print(OSError)
