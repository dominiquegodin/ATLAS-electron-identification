import numpy as np
import matplotlib.pyplot as plt
import pickle

files = 'outputs/2c_10m/match2s_2d/importances.pkl', '/outputs/2c_10m/match2s_2d/importances.pkl'

data = list()
with open(files[0],'rb') as rfp1, open(files[1],'rb') as rfp2:
    while True:
        try:
            imp1 = pickle.load(rfp1)
            imp2 = pickle.load(rfp2)
            data.append([imp1[0],imp1[1],imp2[0],imp2[1]])
        except EOFError:
            break
print(data)

fig, ax = plt.subplots(1)


collabel = [f.split('/')[-2] for f in files]
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=data,colLabels=collabel,loc='center')

#ax[1].plot(clust_data[:,0],clust_data[:,1])
plt.show()
