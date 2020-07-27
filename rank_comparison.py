import numpy as np
import matplotlib.pyplot as plt
import pickle

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
the_table = ax.table(cellText=data,colLabels=collabel,loc='center')
plt.tight_layout()
plt.savefig('results/rank_comparison_noweight_match2s.png')
plt.show()
