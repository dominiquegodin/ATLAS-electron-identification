import numpy as np
import matplotlib.pyplot as plt
import pickle

feats = [
            'em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3', 'em_barrel_Lr1_fine', 'tile_barrel_Lr1',
            'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'    ,   'p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  ,
            'p_TRTPID' , 'p_numberOfSCTHits' , 'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1' , 'p_f3' , 'p_deltaPhiRescaled2',
            'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge' , 'p_eta'   , 'p_et_calo'
        ]

files = ('outputs/2c_10m/bkg_ratio_2d/importances.pkl', 'outputs/2c_10m/match2s_2d/importances.pkl')

imp = {}
for f in files:
    rwt = f.split('/')[-2][:-3] # Type of reweighting
    imp[rwt] = []
    with open(f,'rb') as rfp:
        while True:
            try:
                imp[rwt].append(pickle.load(rfp))
            except EOFError:
                break
    imp[rwt] = sorted(imp[rwt], key = lambda lst: lst[1], reverse=True)

noRwt = imp['bkg_ratio']
match2s = imp['match2s']

data = []
for i in range(len(noRwt)):
    data.append([round(noRwt[i][1],2), noRwt[i][0], match2s[i][0], round(match2s[i][1],2)])

print(data)

fig, ax = plt.subplots(1)

collabel = ['importance', files[0].split('/')[-2], files[1].split('/')[-2], 'importance' ]
#ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=collabel,loc='center')
plt.tight_layout()
plt.savefig('results/RC_noweight_match2s_perm/full.png')

for feat in feats:
    fig, ax = plt.subplots(1)
    colors = [['lime' if feat == cell else 'w' for cell in row] for row in data]
    ax.axis('off')
    ax.table(cellText=data,colLabels=collabel,cellColours=colors,loc='center')
    plt.tight_layout()
    plt.savefig('results/RC_noweight_match2s_perm/{}.png'.format(feat))


plt.show()
