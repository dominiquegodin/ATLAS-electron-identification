import pickle
from utils import plot_importances

feats = [
            'full',
            'em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3', 'em_barrel_Lr1_fine', 'tile_barrel_Lr1',
            'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'    ,   'p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  ,
            'p_TRTPID' , 'p_numberOfSCTHits' , 'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1' , 'p_f3' , 'p_deltaPhiRescaled2',
            'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge' , 'p_eta'   , 'p_et_calo'
        ]

bkg_rej = dict()
for file in feats:
    with open('/scratch/odenis/removal_importance/' + file + '/removal_importance.pkl', 'rb') as rfp
        bkg_rej.update(pickle.load(rfp))

print(bkg_rej)
imp = dict()
for feat in feats[1:]:
    imp[feat] = bkg_rej['full'][0]/bkg_rej[feat][0]
print(imp)

path = '/scratch/odenis/removal_importance/rm_imp.png'
title = 'Feature removal importance without reweighting'
plot_importances(imp, path, title)
