import pickle
from   argparse   import ArgumentParser
from utils import plot_importances

parser = ArgumentParser()
parser.add_argument('--region', default='barrel')
args = parser.parse_args()

feats = [
            'full',
            'em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
            'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine',
            'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3', 'tile_gap_Lr1'      ,
            'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'  , 'p_Eratio', 'p_Reta',
            'p_Rhad'   , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits' , 'p_ndof'  , 'p_dPOverP', 'p_deltaEta1',
            'p_f1'     , 'p_f3'    , 'p_deltaPhiRescaled2','p_weta2'  , 'p_d0'    , 'p_d0Sig'  , 'p_qd0Sig',
            'p_nTracks', 'p_sct_weight_charge' , 'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_wtots1', 'p_numberOfInnermostPixelHits'
            'group_0'  , 'group_1', 'group_2', 'group_3', 'group_4', 'group_5', 'group_6', 'group_7','group_8',  'group_9', 'group_10'
        ]

eta ={'barrel': r'$0<\eta<1.3$'}

bkg_rej = {}
absent = []
for folder in feats:
    try:
        with open('/scratch/odenis/removal_importance/' + args.region + folder + '/removal_importance.pkl', 'rb') as rfp:
            bkg_tup = pickle.load(rfp)
            key = bkg_tup[0].replace(' ', '_')
            bkg_rej[key] = bkg_tup[1]
    except:
        print(folder + ' not in directory')
        absent.append(folder)
        continue
print('\n', bkg_rej)
imp = {}
for feat in [f for f in feats if f not in absent + ['full']]:
    imp[feat] = bkg_rej['full']/bkg_rej[feat], 0.05
path = '/scratch/odenis/removal_importance/{}/rm_imp.pdf'.format(args.region)
title = r'Feature removal importance without reweighting ({})'.format(eta[args.region])
plot_importances(imp, path, title)
