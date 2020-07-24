import numpy as np
import matplotlib.pyplot as plt
import pickle

files = ('outputs/2c_10m/bkg_ratio_2d/importances.pkl', 'outputs/2c_10m/match2s_2d/importances.pkl')

imp = {}
for f in files:
    key = f.split('/')[-2]
    print(key, '\n')
    imp[key] = []
    with open(f,'rb') as rfp:
        while True:
            try:
                imp[key].append(pickle.load(rfp))
            except EOFError:
                break
print(imp)



#fig, ax = plt.subplots(1)

#collabel = ['']+[f.split('/')[-2] for f in files]
#ax.axis('tight')
#ax.axis('off')
#the_table = ax.table(cellText=data,colLabels=collabel,loc='center')

#plt.savefig('results/rank_comparison_noweight_match2s.png')
