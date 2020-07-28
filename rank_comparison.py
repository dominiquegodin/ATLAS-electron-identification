import numpy as np
import matplotlib.pyplot as plt
import pickle

feats = [
            'em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3', 'em_barrel_Lr1_fine', 'tile_barrel_Lr1',
            'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'    ,   'p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  ,
            'p_TRTPID' , 'p_numberOfSCTHits' , 'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1' , 'p_f3' , 'p_deltaPhiRescaled2',
            'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge' , 'p_eta'   , 'p_et_calo'
        ]
file1 = 'outputs/2c_10m/bkg_ratio_2d/importances.pkl'

imp = {}
imp['permutation'] = []
with open(file1,'rb') as rfp:
    while True:
        try:
            imp['permutation'].append(pickle.load(rfp))
        except EOFError:
            break
imp['permutation'] = sorted(imp['permutation'], key = lambda lst: lst[1], reverse=True)

imp['removal'] = []
for feat in feats:
    with open('removal_importance/'+feat+'/removal_importance.pkl','rb') as rfp:
        while True:
            try:
                tup = pickle.load(rfp)
                lst = [tup[0],1685.52/tup[1]]
                imp['removal'].append(lst)
                print(imp['removal'])
            except EOFError:
                break
imp['removal'] = sorted(imp['removal'], key = lambda lst: lst[1], reverse = True)

perm = imp['permutation']
rm = imp['removal']


data = []
for i in range(len(perm)):
    data.append([round(perm[i][1],2), perm[i][0], rm[i][0], round(rm[i][1],2)])

print(data)

fig, ax = plt.subplots(1)

collabel = ['importance', 'permutation', 'removal', 'importance' ]
#ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=data,colLabels=collabel,loc='center')
plt.tight_layout()
plt.savefig('results/rank_comparison_perm_rm.png')
plt.show()
