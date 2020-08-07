import tensorflow as tf, matplotlib.pyplot as plt
import numpy      as np, multiprocessing as mp, os, sys, h5py, pickle, time
import pandas as pd
from pandas.plotting import scatter_matrix
from   sklearn    import metrics, utils, preprocessing
from   tabulate   import tabulate
from   skimage    import transform
from   plots_DG   import valid_accuracy, plot_history, plot_distributions_DG, plot_ROC_curves
from   plots_KM   import plot_distributions_KM, differential_plots
from   copy       import deepcopy
rdm = np.random


def find_bin(array,binning):

    binarized_bin_indices=list()
    for i in range(len(binning)-1):
        binarized_bin_indices.append( ((binning[i]<array) & (array<=binning[i+1])).astype(float) )
        #tmp_array = (binning[i]<array) & (array<binning[i+1])
        #print(tmp_array)
        pass

    #print(np.shape(binarized_bin_indices[0]))

    return binarized_bin_indices
#def find_bin(element,binning):
#
#    binNum=-1 # underflow and overflow
#    for i in range(len(binning)-1):
#        if binning[i]<element and element<binning[i+1]: binNum=i
#        pass
#
#    return binNum

def get_bin_indices(p_var,boundaries):
    bin_indices=list()
    #print("hi=",boundaries[0],np.where( p_var<boundaries[0] )[0])
    bin_indices.append (np.where( p_var<=boundaries[0] )[0])
    for idx in range(len(boundaries)-1):
        #print("lo, hi=",boundaries[idx],":",boundaries[idx+1])
        bin_indices.append (np.where( (boundaries[idx]<p_var) & (p_var<=boundaries[idx+1]) )[0])
        #bin_indices.append (np.where( (boundaries[idx]<=p_var) & (p_var<boundaries[idx+1]) & (isnan(p_var)) )[0])
        pass
    #print("lo=",boundaries[len(boundaries)-1],np.where( boundaries[len(boundaries)-1]<p_var )[0])
    bin_indices.append (np.where( boundaries[len(boundaries)-1]<p_var )[0])
    #print(len(bin_indices),len(boundaries))

    tmp_idx=0
    total=0
    #total=len(bin_indices[0])
    #print("hi=",boundaries[0],bin_indices[0],len(bin_indices[0]),total)

    debug=False
    for bin_idx in bin_indices:
        total+=len(bin_idx)

        if debug:
            if tmp_idx==0:
                print("hi=",boundaries[tmp_idx],bin_idx,len(bin_idx),total)
            elif tmp_idx==len(bin_indices)-1:
                print("lo=",boundaries[tmp_idx-1],bin_idx,len(bin_idx),total)
            else:
                print("lo,hi=[",boundaries[tmp_idx-1],",",boundaries[tmp_idx],"]",bin_idx,len(bin_idx),total)
                pass
            pass

        tmp_idx+=1
        pass
    total+=len(bin_indices[-1])
    #print("lo=",boundaries[-1],bin_indices[-1],len(bin_indices[-1]),total)

    return bin_indices

def getMaxContents(binContents):

    maxContents = np.full(len(binContents[0]),-1.)
    for i_bin in range(len(binContents[0])):
        for i in range(len(binContents)):
            if binContents[i][i_bin] > maxContents[i_bin]: maxContents[i_bin] = binContents[i][i_bin]
            pass
        pass
    #print("maxContens=",maxContents)

    return maxContents

#def generate_weights(train_data,train_labels,nClass,weight_type='none',ref_var='pt',output_dir='outputs/'):
def sample_weights(train_data,train_labels,nClass,weight_type,output_dir='outputs/',ref_var='pt'):
    if weight_type=="none": return None

    print("-------------------------------")
    print("generate_weights: sample weight mode \"",weight_type,"\" designated. Generating weights.",)
    print("-------------------------------\n")

    binning=[0,10,20,30,40,60,80,100,130,180,250,500]
    labels=['sig','bkg']
    colors=['blue','red']
    binContents=[0,0]
    if nClass==6:
        #below 2b implemented
        labels=['sig','chf','conv','hf','eg','lf']
        colors=['blue','orange','green','red','purple','brown']
        binContents=[0,0,0,0,0,0]
        pass

    variable=list()                          #only for specific label
    variable_array = train_data['p_et_calo'] #entire set
    if   ref_var=='eta'  : variable_array = train_data['p_eta']
    #elif ref_var=='pteta': variable_array = train_data['p_eta']

    for i_class in range(nClass):
        variable.append( variable_array[ train_labels==i_class ] )
        (binContents[i_class],bins,patches)=plt.hist(variable[i_class],bins=binning,weights=np.full(len(variable[i_class]),1/len(variable[i_class])),label=labels[i_class],histtype='step',facecolor=colors[i_class])
        #(binContents[i_class],bins,patches)=plt.hist(variable[i_class],bins=binning,weights=np.full(len(variable[i_class]),1/len(train_labels)),label=labels[i_class],histtype='step',facecolor=colors[i_class])
        pass

        if nClass>2: plt.yscale("log")
        plt.savefig(output_dir+'/'+ref_var+"_bfrReweighting.png")
    plt.clf() #clear figure

    weights=list() #KM: currently implemented for the 2-class case only
    if weight_type=="flattening":
        for i in range(nClass): weights.append(np.average(binContents[i])/binContents[i] )
    elif weight_type=="match2max": #shaping to whichever that has max in the corresponding bin
        #for i in range(nClass): print("bincontens[",i,"]=",binContents[i])
        maxContents = getMaxContents(binContents)#print(maxContents)
        for i in range(nClass): weights.append( maxContents/binContents[i] )
    elif weight_type=="match2b": #shaping sig to match the bkg, using pt,or any other designated variable
        for i in range(nClass-1): weights.append(binContents[nClass-1]/binContents[i])
        weights.append(np.ones(len(binContents[5])))
    elif weight_type=="match2s": #shaping bkg to match the sig, using pt,or any other designated variable
        weights.append(np.ones(len(binContents[0])))
        for i in range (1, nClass): weights.append(binContents[0]/binContents[i])
        pass

    #KM: to replce inf with 0
    for i in range(nClass): weights[i]=np.where(weights[i]==np.inf,0,weights[i]) #np.where(array1==0, 1, array1)

    debug=0
    if debug:
        tmp_i=0
        for weight in weights:
            print("weights[",labels[tmp_i],"]=",weight)
            tmp_i+=1
        #print(weights[0])
        #print(weights[1])
        pass

    #KM: Generates weights for all events
    #    This is not very efficient, to be improved

    #final_weights*=weights[train_labels][1]

    class_weight=list()
    for i in range(nClass): class_weight.append( np.full(len(variable_array),0,dtype=float) )
    final_weights= np.full(len(variable_array),0,dtype=float)
    #sig_weight=np.full(len(variable_array),0,dtype=float)
    #bkg_weight=np.full(len(variable_array),0,dtype=float)

    #To produce vectors of 0 or 1 for given pt ranges
    bin_indices0or1=find_bin(variable_array,binning)
    tmp_i=0 # pt bin index
    for vec01 in bin_indices0or1:
        #KM: this line is the most important to calculate the weights for al events
        for i in range(nClass): class_weight[i] += (vec01 * (train_labels==i) )* weights[i][tmp_i]
        #sig_weight += (vec01 * (train_labels==0) )* weights[0][tmp_i]
        #bkg_weight += (vec01 * (train_labels==1) )* weights[1][tmp_i]
        tmp_i+=1
        pass

    if debug:
        print()
        #print(sig_weight,"\n", bkg_weight)
        tmp_i=0
        for i in range(nClass):
            print("class_weight[",tmp_i,"]=",class_weight[i])
            tmp_i+=1
            pass
        #print("final_weights=",final_weights) #print(final_weights,final_weights.all()==1) # w
        print("train_labels=",train_labels)
        pass

    for i in range(nClass): final_weights+=class_weight[i]

    if debug:
        print("variable_array=",variable_array)
        print(final_weights, len(final_weights), "any element is zero?",final_weights.any()==0)
        pass

    #KM: below only for plotting
    for i_class in range(nClass):
        plt.hist(variable[i_class],bins=binning,weights=final_weights[ train_labels==i_class ],label=labels[i_class],histtype='step',facecolor=colors[i_class])
        #weights = final_weights[ train_labels==i_class ]/len(train_labels)
        #plt.hist(variable[i_class],bins=binning, weights=weights, label=labels[i_class],histtype='step',facecolor=colors[i_class])
        pass
    if nClass>2: plt.yscale("log")
    plt.savefig(output_dir+'/'+ref_var+"_aftReweighting.png")
    plt.clf() #clear plot

    return final_weights




#################################################################################
##### classifier.py functions ###################################################
#################################################################################


def balance_sample(sample, labels, sampling_type=None, bkg_ratio=None, hist='2d', get_weights=True):
    if sampling_type not in ['bkg_ratio', 'flattening', 'match2s', 'match2b', 'match2max']:
        return sample, labels, None
    pt  =     sample['pt']  ;  pt_bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
    eta = abs(sample['eta']); eta_bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37]
    if hist == 'pt' : eta_bins = [eta_bins[0], eta_bins[-1]]
    if hist == 'eta':  pt_bins = [ pt_bins[0],  pt_bins[-1]]
    pt_ind   = np.digitize(pt , pt_bins , right=True) -1
    eta_ind  = np.digitize(eta, eta_bins, right=True) -1
    hist_sig = np.histogram2d(pt[labels==0], eta[labels==0], bins=[pt_bins,eta_bins])[0]
    hist_bkg = np.histogram2d(pt[labels!=0], eta[labels!=0], bins=[pt_bins,eta_bins])[0]
    if bkg_ratio == None: bkg_ratio = np.sum(hist_bkg)/np.sum(hist_sig)
    if   sampling_type == 'bkg_ratio':
        total_sig = hist_sig * max(1, np.sum(hist_bkg)/np.sum(hist_sig)/bkg_ratio)
        total_bkg = hist_bkg * max(1, np.sum(hist_sig)/np.sum(hist_bkg)*bkg_ratio)
    elif sampling_type == 'flattening':
        total_sig = np.minimum(1, hist_sig) * max(np.max(hist_sig), np.max(hist_bkg)/bkg_ratio)
        total_bkg = np.minimum(1, hist_bkg) * max(np.max(hist_bkg), np.max(hist_sig)/bkg_ratio)
    elif sampling_type == 'match2s':
        total_sig = hist_sig * max(1, np.max(hist_bkg/hist_sig)/bkg_ratio)
        total_bkg = hist_sig * max(1, np.max(hist_bkg/hist_sig)/bkg_ratio) * bkg_ratio
    elif sampling_type == 'match2b':
        total_sig = hist_bkg * max(1, np.max(hist_sig/hist_bkg)*bkg_ratio) / bkg_ratio
        total_bkg = hist_bkg * max(1, np.max(hist_sig/hist_bkg)*bkg_ratio)
    elif sampling_type == 'match2max':
        total_sig = np.maximum(hist_sig, hist_bkg/bkg_ratio)
        total_bkg = np.maximum(hist_bkg, hist_sig*bkg_ratio)
    if get_weights or hist != 'pt':
        weights_sig = total_sig/hist_sig * len(labels)/np.sum(total_sig+total_bkg)
        weights_bkg = total_bkg/hist_bkg * len(labels)/np.sum(total_sig+total_bkg)
        return sample, labels, np.where(labels==0, weights_sig[pt_ind,eta_ind], weights_bkg[pt_ind,eta_ind])
    return upsampling(sample, labels, pt_bins, pt_ind, hist_sig, hist_bkg, total_sig, total_bkg) + (None,)


def upsampling(sample, labels, bins, indices, hist_sig, hist_bkg, total_sig, total_bkg):
    new_sig = np.int_(np.around(total_sig)) - hist_sig
    new_bkg = np.int_(np.around(total_bkg)) - hist_bkg
    ind_sig = [np.where((indices==n) & (labels==0))[0] for n in np.arange(len(bins)-1)]
    ind_bkg = [np.where((indices==n) & (labels!=0))[0] for n in np.arange(len(bins)-1)]
    np.random.seed(0)
    ind_sig = [np.append(ind_sig[n], np.random.choice(ind_sig[n], new_sig[n],
               replace = len(ind_sig[n])<new_sig[n])) for n in np.arange(len(bins)-1)]
    ind_bkg = [np.append(ind_bkg[n], np.random.choice(ind_bkg[n], new_bkg[n],
               replace = len(ind_bkg[n])<new_bkg[n])) for n in np.arange(len(bins)-1)]
    indices = np.concatenate(ind_sig + ind_bkg); np.random.shuffle(indices)
    return {key:np.take(sample[key], indices, axis=0) for key in sample}, np.take(labels, indices)


def downsampling(sample, labels, bkg_ratio=None):
    pt = sample['p_et_calo']; bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
    indices  = np.digitize(pt, bins, right=True) -1
    hist_sig = np.histogram(pt[labels==0], bins)[0]
    hist_bkg = np.histogram(pt[labels!=0], bins)[0]
    if bkg_ratio == None: bkg_ratio = np.sum(hist_bkg)/np.sum(hist_sig)
    total_sig = np.int_(np.around(np.minimum(hist_sig, hist_bkg/bkg_ratio)))
    total_bkg = np.int_(np.around(np.minimum(hist_bkg, hist_sig*bkg_ratio)))
    ind_sig   = [np.where((indices==n) & (labels==0))[0][:total_sig[n]] for n in np.arange(len(bins)-1)]
    ind_bkg   = [np.where((indices==n) & (labels!=0))[0][:total_bkg[n]] for n in np.arange(len(bins)-1)]
    valid_ind = np.concatenate(ind_sig+ind_bkg); np.random.seed(0); np.random.shuffle(valid_ind)
    train_ind = list(set(np.arange(len(pt))) - set(valid_ind))
    valid_sample = {key:np.take(sample[key], valid_ind, axis=0) for key in sample}
    valid_labels = np.take(labels, valid_ind)
    extra_sample = {key:np.take(sample[key], train_ind, axis=0) for key in sample}
    extra_labels = np.take(labels, train_ind)
    return valid_sample, valid_labels, extra_sample, extra_labels


def match_distributions(sample, labels, target_sample, target_labels):
    bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 501]
    pt = sample['p_et_calo']; target_pt = target_sample['p_et_calo']
    indices         = np.digitize(pt, bins, right=False) -1
    hist_sig        = np.histogram(       pt[labels==0]       , bins)[0]
    hist_bkg        = np.histogram(       pt[labels!=0]       , bins)[0]
    hist_sig_target = np.histogram(target_pt[target_labels==0], bins)[0]
    hist_bkg_target = np.histogram(target_pt[target_labels!=0], bins)[0]
    total_sig   = hist_sig_target * np.max(np.append(hist_sig/hist_sig_target, hist_bkg/hist_bkg_target))
    total_bkg   = hist_bkg_target * np.max(np.append(hist_sig/hist_sig_target, hist_bkg/hist_bkg_target))
    weights_sig = total_sig/hist_sig * len(labels)/np.sum(total_sig+total_bkg)
    weights_bkg = total_bkg/hist_bkg * len(labels)/np.sum(total_sig+total_bkg)
    return np.where(labels==0, weights_sig[indices], weights_bkg[indices])


def class_weights(labels, bkg_ratio):
    n_e = len(labels); n_classes = max(labels) + 1
    if bkg_ratio == 0 and n_classes == 2: return None
    if bkg_ratio == 0 and n_classes != 2: bkg_ratio = 1
    ratios = {**{0:1}, **{n:bkg_ratio for n in np.arange(1, n_classes)}}
    return {n:n_e/np.sum(labels==n)*ratios[n]/sum(ratios.values()) for n in np.arange(n_classes)}


def validation(output_dir, results_in, plotting, n_valid, data_file, variables, diff_plots, valid_cuts=''):
    print('\nLOADING VALIDATION RESULTS FROM', output_dir+'/'+results_in)
    valid_data = pickle.load(open(output_dir+'/'+results_in, 'rb'))
    if len(valid_data) > 1: sample, labels, probs   = valid_data
    else:                                  (probs,) = valid_data
    n_e = min(len(probs), int(n_valid[1]-n_valid[0]))
    if False or len(valid_data) == 1: #add variables to the results
        print('CLASSIFIER: loading valid sample', n_e, end=' ... ', flush=True)
        sample, labels = make_sample(data_file, variables, n_valid, n_tracks=5, n_classes=probs.shape[1])
        n_e = len(labels)
    sample, labels, probs = {key:sample[key][:n_e] for key in sample}, labels[:n_e], probs[:n_e]
    if False: #save the added variables to the results file
        print('Saving validation data to:', output_dir+'/'+'valid_data.pkl', '\n')
        pickle.dump((sample, labels, probs), open(output_dir+'/'+'valid_data.pkl','wb')); sys.exit()
    print('GENERATING PERFORMANCE RESULTS FOR', n_e, 'ELECTRONS', end=' ...', flush=True)
    #valid_cuts = '(labels==0) & (probs[:,0]<=0.05)'
    #valid_cuts = '(sample["p_et_calo"]  < 20)'
    #valid_cuts = '(sample["p_et_calo"] >= 20) & (sample["p_et_calo"] <= 80)'
    #valid_cuts = '(sample["p_et_calo"]  > 80)'
    cuts = n_e*[True] if valid_cuts == '' else eval(valid_cuts)
    sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]
    if False: #generate calorimeter images
        layers = ['em_barrel_Lr0'  , 'em_barrel_Lr1_fine'  , 'em_barrel_Lr2', 'em_barrel_Lr3',
                  'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']
        from plots_DG import cal_images
        cal_images(sample, labels, layers, output_dir, mode='mean', soft=False)
    def text_line(n_cut): return ' ('+str(n_cut)+' selected = '+format(100*n_cut/n_e,'0.2f')+'%)'
    print(text_line(len(labels)) if len(labels) < n_e else '', '\n')
    valid_results(sample, labels, probs, [], None, output_dir, plotting, diff_plots)


def make_sample(data_file, variables, idx, n_tracks, n_classes, cuts='', p='p_', process=False):
    var_list = np.sum(list(variables.values())); start_time = time.time()
    with h5py.File(data_file, 'r') as data:
        sample = {key:data[key][idx[0]:idx[1]] for key in var_list if key != 'tracks_image'}
        sample.update({'eta':sample['p_eta'], 'pt':sample['p_et_calo']})
        if 'tracks_image' in var_list or 'tracks' in var_list:
            n_tracks    = min(n_tracks, data[p+'tracks'].shape[1])
            tracks_data = data[p+'tracks'][idx[0]:idx[1]][:,:n_tracks,:]
            tracks_data = np.concatenate((abs(tracks_data[...,0:5]), tracks_data[...,5:13]), axis=2)
    if 'tracks_image' in var_list: sample['tracks_image'] = tracks_data
    if 'tracks'       in var_list: sample['tracks'      ] = tracks_data
    if tf.__version__ < '2.1.0' or len(variables['images']) == 0:
        for key in set(sample) - set(variables['others']): sample[key] = np.float32(sample[key])
    if False:
        for n in variables['images']: sample[n] = resize_images(np.float32(sample[n]),target_shape=(56,11))
    labels = make_labels(sample, n_classes)
    if idx[1]-idx[0] > 1:
        print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        sample, labels = sample_cuts(sample, labels, cuts)
        if process: sample = process_images(sample, variables['images'])
    return sample, labels


def make_labels(sample, n_classes):
    MC_type = sample['p_TruthType']; IFF_type = sample['p_iffTruth']
    if   n_classes == 2:
        labels = np.where(IFF_type <= 1                               , -1, IFF_type)
        labels = np.where(IFF_type == 2                               ,  0, labels  )
        return   np.where(IFF_type >= 3                               ,  1, labels  )
    elif n_classes == 6:
        labels = np.where(np.logical_or (IFF_type <= 1, IFF_type == 4), -1, IFF_type)
        labels = np.where(np.logical_or (IFF_type == 6, IFF_type == 7), -1, labels  )
        labels = np.where(IFF_type == 2                               ,  0, labels  )
        labels = np.where(IFF_type == 3                               ,  1, labels  )
        labels = np.where(IFF_type == 5                               ,  2, labels  )
        labels = np.where(np.logical_or (IFF_type == 8, IFF_type == 9),  3, labels  )
        labels = np.where(np.logical_and(IFF_type ==10,  MC_type == 4),  4, labels  )
        labels = np.where(np.logical_and(IFF_type ==10,  MC_type ==16),  4, labels  )
        labels = np.where(np.logical_and(IFF_type ==10,  MC_type ==17),  5, labels  )
        return   np.where(  labels == 10                              , -1, labels  )
    elif n_classes == 9:
        labels = np.where(IFF_type == 9                               ,  4, IFF_type)
        return   np.where(IFF_type ==10                               ,  6, labels  )
    else: print('\nERROR:', n_classes, 'classes not supported -> exiting program\n'); sys.exit()


def sample_cuts(sample, labels, cuts):
    if np.sum(labels==-1) != 0:
        length = len(labels)
        sample = {key:sample[key][labels!=-1] for key in sample}; labels = labels[labels!=-1]
        print('CLASSIFIER: applying IFF labels cuts -->', format(len(labels),'8d'), 'e conserved', end='')
        print(' (' + format(100*len(labels)/length, '.2f') + ' %)')
    if cuts != '':
        length = len(labels)
        labels = labels[eval(cuts)]; sample = {key:sample[key][eval(cuts)] for key in sample}
        print('CLASSIFIER: applying properties cuts -->', format(len(labels),'8d') ,'e conserved', end='')
        print(' (' + format(100*len(labels)/length, '.2f') + ' %)')
        print('CLASSIFIER: applied cuts:', cuts)
    print(); return sample, labels


def process_images(sample, image_list, n_tasks=16):
    def rotation(images, indices, return_dict):
        images = images[indices[0]:indices[1]].T
        #images  = abs(images)                               # negatives to positives
        #images -= np.minimum(0, np.min(images, axis=(0,1))) # shift to positive domain
        images = np.maximum(0, images)                       # clips negative values
        mean_1 = np.mean(images[:images.shape[0]//2   ], axis=(0,1))
        mean_2 = np.mean(images[ images.shape[0]//2:-1], axis=(0,1))
        return_dict[indices] = np.where(mean_1 > mean_2, images[::-1,::-1,:], images).T
    n_samples = len(sample['eventNumber'])
    idx_list  = [task*(n_samples//n_tasks) for task in np.arange(n_tasks)] + [n_samples]
    idx_list  = list( zip(idx_list[:-1], idx_list[1:]) ); start_time = time.time()
    print('CLASSIFIER: preprocessing images for best axis', end=' ... ', flush=True)
    for cal_image in [key for key in image_list if 'tracks' not in key]:
        images    = sample[cal_image]; manager = mp.Manager(); return_dict = manager.dict()
        processes = [mp.Process(target=rotation, args=(images, idx, return_dict)) for idx in idx_list]
        for job in processes: job.start()
        for job in processes: job.join()
        sample[cal_image] = np.concatenate([return_dict[idx] for idx in idx_list])
    print(' ('+format(time.time()-start_time,'.1f'), '\b'+' s)\n')
    return sample


def sample_composition(sample):
    MC_type, IFF_type  = sample['p_TruthType']    , sample['p_iffTruth']
    MC_list, IFF_list  = np.arange(max(MC_type)+1), np.arange(max(IFF_type)+1)
    ratios = np.array([ [np.sum(MC_type[IFF_type==IFF]==MC) for MC in MC_list] for IFF in IFF_list ])
    IFF_sum, MC_sum = 100*np.sum(ratios, axis=0)/len(MC_type), 100*np.sum(ratios, axis=1)/len(MC_type)
    ratios = np.round(1e4*ratios/len(MC_type))/100
    MC_empty, IFF_empty = np.where(np.sum(ratios, axis=0)==0)[0], np.where(np.sum(ratios, axis=1)==0)[0]
    MC_list,  IFF_list  = list(set(MC_list)-set(MC_empty))      , list(set(IFF_list)-set(IFF_empty))
    print('IFF AND MC TRUTH CLASSIFIERS SAMPLE COMPOSITION (', '\b'+str(len(MC_type)), 'e)')
    dash = (26+7*len(MC_list))*'-'
    print(dash, format('\n| IFF \ MC |','10s'), end='')
    for col in MC_list:
        print(format(col, '7.0f'), end='   |  Total  | \n' + dash + '\n' if col==MC_list[-1] else '')
    for row in IFF_list:
        print('|', format(row, '5.0f'), '   |', end='' )
        for col in MC_list:
            print(format(ratios[row,col], '7.0f' if ratios[row,col]==0 else '7.2f'), end='', flush=True)
        print('   |' + format(MC_sum[row], '7.2f')+ '  |')
        if row != IFF_list[-1]: print('|' + 10*' ' + '|' + (3+7*len(MC_list))*' ' + '|' + 9*' ' + '|')
    print(dash + '\n|   Total  |', end='')
    for col in MC_list: print(format(IFF_sum[col], '7.2f'), end='')
    print('   |  100 %  |\n' + dash +'\n')


def apply_scaler(train_sample, valid_sample, scalars, scaler_out):
    print('CLASSIFIER: applying quantile transform to scalar variables', end=' ... ', flush=True)
    start_time    = time.time()
    train_scalars = np.hstack([np.expand_dims(train_sample[key], axis=1) for key in scalars])
    valid_scalars = np.hstack([np.expand_dims(valid_sample[key], axis=1) for key in scalars])
    scaler        = preprocessing.QuantileTransformer(output_distribution='normal',
                                                      n_quantiles=10000, random_state=0)
    train_scalars = scaler.fit_transform(train_scalars)
    valid_scalars = scaler.transform(valid_scalars)
    for n in np.arange(len(scalars)):
        train_sample[scalars[n]] = train_scalars[:,n]
        valid_sample[scalars[n]] = valid_scalars[:,n]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('CLASSIFIER: saving transformed scalars in ' + scaler_out + '\n')
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return train_sample, valid_sample


def load_scaler(sample, scalars, scaler_file):
    print('CLASSIFIER: loading quantile transform from ' + scaler_file)
    scaler         = pickle.load(open(scaler_file, 'rb'))
    start_time     = time.time()
    scalars_scaled = np.hstack([np.expand_dims(sample[key], axis=1) for key in scalars])
    print('CLASSIFIER: applying quantile transform to scalar variables', end=' ... ', flush=True)
    scalars_scaled = scaler.transform(scalars_scaled)
    for n in np.arange(len(scalars)): sample[scalars[n]] = scalars_scaled[:,n]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def class_ratios(labels):
    def get_ratios(labels, n, return_dict): return_dict[n] = 100*np.sum(labels==n)/len(labels)
    manager   =  mp.Manager(); return_dict = manager.dict(); n_classes = max(labels) + 1
    processes = [mp.Process(target=get_ratios, args=(labels, n, return_dict)) for n in np.arange(n_classes)]
    for job in processes: job.start()
    for job in processes: job.join()
    return [return_dict[n] for n in np.arange(n_classes)]


def compo_matrix(valid_labels, train_labels=[], valid_probs=[]):
    valid_pred   = np.argmax(valid_probs, axis=1) if valid_probs != [] else valid_labels
    matrix       = metrics.confusion_matrix(valid_labels, valid_pred)
    matrix       = 100*matrix.T/matrix.sum(axis=1); n_classes = len(matrix)
    classes      = ['CLASS '+str(n) for n in np.arange(n_classes)]
    valid_ratios = class_ratios(valid_labels)
    train_ratios = class_ratios(train_labels) if train_labels != [] else n_classes*['n/a']
    if valid_probs == []:
        print('+---------------------------------------+\n| CLASS DISTRIBUTIONS'+19*' '+'|')
        headers = ['CLASS #', 'TRAIN (%)', 'VALID (%)']
        table   = zip(classes, train_ratios, valid_ratios)
        print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f"))
    else:
        if n_classes > 2:
            headers = ['CLASS #', 'TRAIN', 'VALID'] + classes
            table   = [classes] + [train_ratios] + [valid_ratios] + matrix.T.tolist()
            table   = list(map(list, zip(*table)))
            print_dict[2]  = '+'+31*'-'+'+'+35*'-'+12*(n_classes-3)*'-'+'+\n| CLASS DISTRIBUTIONS (%)'
            print_dict[2] += '       | VALID SAMPLE PREDICTIONS (%)      '+12*(n_classes-3)*' '+ '|\n'
        else:
            headers = ['CLASS #', 'TRAIN (%)', 'VALID (%)', 'ACC. (%)']
            table   = zip(classes, train_ratios, valid_ratios, matrix.diagonal())
            print_dict[2]  = '+----------------------------------------------------+\n'
            print_dict[2] += '| CLASS DISTRIBUTIONS AND VALID SAMPLE ACCURACIES    |\n'
        valid_accuracy = np.array(valid_ratios) @ np.array(matrix.diagonal())/100
        print_dict[2] += tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f")+'\n'
        print_dict[2] += 'VALIDATION SAMPLE ACCURACY: '+format(valid_accuracy,'.2f')+' %\n'


def cross_valid(valid_sample, valid_labels, scalars, model, output_dir, n_folds, verbose=1):
    print('\n########################################################################'  )
    print(  '#### STARTING CROSS-VALIDATION #########################################'  )
    print(  '########################################################################\n')
    valid_probs  = np.full(valid_labels.shape + (max(valid_labels)+1,), -1.)
    event_number = valid_sample['eventNumber']
    for fold_number in np.arange(1, n_folds+1):
        print('FOLD', fold_number, 'EVALUATION ('+str(n_folds)+'-fold cross-validation)')
        weight_file = output_dir+'/model_' +str(fold_number)+'.h5'
        scaler_file = output_dir+'/scaler_'+str(fold_number)+'.pkl'
        print('CLASSIFIER: loading pre-trained weights from', weight_file)
        model.load_weights(weight_file); start_time = time.time()
        indices =               np.where(event_number%n_folds==fold_number-1)[0]
        labels  =           valid_labels[event_number%n_folds==fold_number-1]
        sample  = {key:valid_sample[key][event_number%n_folds==fold_number-1] for key in valid_sample}
        if scalars != [] and os.path.isfile(scaler_file): sample = load_scaler(sample, scalars, scaler_file)
        print('CLASSIFIER:', weight_file.split('/')[-1], 'class predictions for', len(labels), 'e')
        probs = model.predict(sample, batch_size=20000, verbose=verbose)
        print('FOLD', fold_number, 'ACCURACY:', format(100*valid_accuracy(labels, probs), '.2f'), end='')
        print(' % (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
        for n in np.arange(len(indices)): valid_probs[indices[n],:] = probs[n,:]
    return valid_probs


def binarization(sample, labels, probs, class_1=['bkg'], class_0=[0], normalization=True, LR=False):
    from functools import reduce
    if class_1==['bkg'] or class_1==class_0: class_1 = set(np.arange(max(labels)+1)) - set(class_0)
    print_dict[1] += 'BINARIZATION: CLASS 0 = '+str(set(class_0))+' vs CLASS 1 = '+str(set(class_1))+'\n'
    ratios  = class_ratios(labels) if LR else (max(labels)+1)*[1]
    labels  = np.array([0 if label in class_0 else 1 if label in class_1 else -1 for label in labels])
    probs_0 = reduce(np.add, [ratios[n]*probs[:,n] for n in class_0])[labels!=-1]
    probs_1 = reduce(np.add, [ratios[n]*probs[:,n] for n in class_1])[labels!=-1]
    sample  = {key:sample[key][labels!=-1] for key in sample}; labels = labels[labels!=-1]
    if normalization:
        probs_0 = np.where(probs_0!=probs_1, probs_0, 0.5)
        probs_1 = np.where(probs_0!=probs_1, probs_1, 0.5)
        return sample, labels, (np.vstack([probs_0, probs_1])/(probs_0+probs_1)).T
    else: return sample, labels, np.vstack([probs_0, probs_1]).T


def bkg_separation(sample, labels, probs, bkg):
    if bkg == 'bkg': return sample, labels, probs
    print_dict[1] += 'BACKGROUND SEPARATION: CLASS 0 = {0} VS CLASS 1 = {'+str(bkg)+'}\n'
    multi_labels = make_labels(sample, n_classes=6)
    cuts = np.logical_or(multi_labels==0, multi_labels==bkg)
    return {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]


def print_performance(labels, probs, sig_eff=[90, 80, 70]):
    fpr, tpr, _ = metrics.roc_curve(labels, probs[:,0], pos_label=0)
    for val in sig_eff:
        print_dict[3] += 'BACKGROUND REJECTION AT '+str(val)+'%: '
        print_dict[3] += format(1/fpr[np.argwhere(tpr>=val/100)[0]][0],'>6.0f')+'\n'


def print_results(sample, labels, probs, plotting, output_dir, bkg, return_dict, separation=False):
    if max(labels) > 1: sample, labels, probs =   binarization(sample, labels, probs, [bkg])
    else              : sample, labels, probs = bkg_separation(sample, labels, probs,  bkg )
    if False: pickle.dump((sample,labels,probs), open(output_dir+'/'+'results_0_vs_'+str(bkg)+'.pkl','wb'))
    if plotting == 'ON':
        folder = output_dir+'/'+'class_0_vs_'+str(bkg)
        if not os.path.isdir(folder): os.mkdir(folder)
        arguments  = (sample, labels, probs, folder, separation and bkg=='bkg', bkg)
        processes  = [mp.Process(target=plot_distributions_DG, args=arguments)]
        arguments  = [(sample, labels, probs, ROC_type, folder) for ROC_type in [1,2,3]]
        processes += [mp.Process(target=plot_ROC_curves, args=arg) for arg in arguments]
        for job in processes: job.start()
        for job in processes: job.join()
    else:
        compo_matrix(labels, [], probs); print_performance(labels, probs)
    return_dict[bkg] = print_dict


def valid_results(sample, labels, probs, train_labels, training, output_dir, plotting, diff_plots):
    global print_dict; print_dict = {n:'' for n in np.arange(1,4)}
    compo_matrix(labels, train_labels, probs); print(print_dict[2])
    manager   = mp.Manager(); return_dict = manager.dict(); bkg_list  = ['bkg'] #+[1, 2, 3, 4, 5]
    arguments = [(sample, labels, probs, plotting, output_dir, bkg, return_dict) for bkg in bkg_list]
    processes = [mp.Process(target=print_results, args=arg) for arg in arguments]
    if training != None: processes += [mp.Process(target=plot_history, args=(training, output_dir,))]
    for job in processes: job.start()
    for job in processes: job.join()
    if plotting=='OFF':
        for bkg in bkg_list: print("".join(list(return_dict[bkg].values())))
    # DIFFERENTIAL PLOTS
    if plotting == 'ON' and diff_plots:
        eta_boundaries  = [-1.6, -0.8, 0, 0.8, 1.6]
        pt_boundaries   = [10, 20, 30, 40, 60, 100, 200, 500] #60, 80, 120, 180, 300, 500]
        eta, pt         = sample['eta'], sample['pt']
        eta_bin_indices = get_bin_indices(eta, eta_boundaries)
        pt_bin_indices  = get_bin_indices(pt , pt_boundaries)
        plot_distributions_KM(labels, eta, 'eta', output_dir=output_dir)
        plot_distributions_KM(labels, pt , 'pt' , output_dir=output_dir)
        tmp_llh      = sample['p_LHValue']
        tmp_llh_pair = np.zeros(len(tmp_llh))
        prob_LLH     = np.stack((tmp_llh,tmp_llh_pair),axis=-1)
        print('\nEvaluating differential performance in eta')
        differential_plots(sample, labels, probs, eta_boundaries,
                           eta_bin_indices, varname='eta', output_dir=output_dir)
        print('\nEvaluating differential performance in pt')
        differential_plots(sample, labels, probs, pt_boundaries,
                           pt_bin_indices,  varname='pt',  output_dir=output_dir)
        differential_plots(sample, labels, prob_LLH   , pt_boundaries,
                           pt_bin_indices,  varname='pt',  output_dir=output_dir, evalLLH=True)


def verify_sample(sample):
    def scan(sample, batch_size, index, return_dict):
        idx1, idx2 = index*batch_size, (index+1)*batch_size
        return_dict[index] = sum([np.sum(np.isfinite(sample[key][idx1:idx2])==False) for key in sample])
    n_e = len(list(sample.values())[0]); start_time = time.time()
    print('SCANNING', n_e, 'ELECTRONS FOR ERRORS ...', end=' ', flush=True)
    for n in np.arange(min(12, mp.cpu_count()), 0, -1):
        if n_e % n == 0: n_tasks = n; batch_size = n_e//n_tasks; break
    manager   =  mp.Manager(); return_dict = manager.dict()
    processes = [mp.Process(target=scan, args=(sample, batch_size, index, return_dict))
                for index in np.arange(n_tasks)]
    for job in processes: job.start()
    for job in processes: job.join()
    print(sum(return_dict.values()), 'ERRORS FOUND', end=' ', flush=True)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
    if sum(return_dict.values()) != 0:
        for key in sample: print(key, np.where(np.isfinite(sample[key])==False))


def NN_weights(image_shape, CNN_dict, FCN_neurons, n_classes):
    kernels = np.array(CNN_dict[image_shape]['kernels']); n_maps = CNN_dict[image_shape]['maps']
    K = [image_shape[2] if len(image_shape)==3 else 1] + n_maps + FCN_neurons + [n_classes]
    A = [np.prod([image_shape[n] + sum(1-kernels)[n] for n in [0,1]])]
    A = [np.prod(kernels[n]) for n in np.arange(len(n_maps))] + A + len(FCN_neurons)*[1]
    return sum([(K[l]*A[l]+1)*K[l+1] for l in np.arange(len(K)-1)])


def order_kernels(image_shape, n_maps, FCN_neurons, n_classes):
    x_dims  = [(x1,x2) for x1 in np.arange(1,image_shape[0]+1) for x2 in np.arange(1,image_shape[0]-x1+2)]
    y_dims  = [(y1,y2) for y1 in np.arange(1,image_shape[1]+1) for y2 in np.arange(1,image_shape[1]-y1+2)]
    par_tuple = []
    for kernels in [[(x[0],y[0]),(x[1],y[1])] for x in x_dims for y in y_dims]:
        CNN_dict   = {image_shape:{'maps':n_maps, 'kernels':kernels}}
        par_tuple += [(NN_weights(image_shape, CNN_dict, FCN_neurons, n_classes), kernels)]
    return sorted(par_tuple)[::-1]


def print_channels(sample, col=0, reverse=False):
    def getkey(item): return item[col]
    channel_dict= {301535:'Z -> ee+y 10-35'   , 301536:'Z -> mumu+y 10-35', 301899:'Z -> ee+y 35-70'   ,
                   301900:'Z -> ee+y 70-140'  , 301901:'Z -> ee+y 140'    , 301902:'Z -> mumu+y 35-70' ,
                   301903:'Z -> mumu+y 70-140', 301904:'Z -> mumu+y 140'  , 361020:'dijet JZ0W'        ,
                   361021:'dijet JZ1W'        , 361022:'dijet JZ2W'       , 361023:'dijet JZ3W'        ,
                   361024:'dijet JZ4W'        , 361025:'dijet JZ5W'       , 361026:'dijet JZ6W'        ,
                   361027:'dijet JZ7W'        , 361028:'dijet JZ8W'       , 361029:'dijet JZ9W'        ,
                   361100:'W+ -> ev'          , 361101:'W+ -> muv'        , 361102:'W+ -> tauv'        ,
                   361103:'W- -> ev'          , 361104:'W- -> muv'        , 361105:'W- -> tauv'        ,
                   361106:'Z -> ee'           , 361108:'Z -> tautau'      , 410470:'ttbar nonhad'      ,
                   410471:'ttbar allhad'      , 410644:'s top s-chan.'    , 410645:'s top s-chan.'     ,
                   410646:'s top Wt-chan.'    , 410647:'s top Wt-chan.'   , 410658:'s top t-chan.'     ,
                   410659:'s top t-chan.'     , 423099:'y+jets 8-17'      , 423100:'y+jets 17-35'      ,
                   423101:'y+jets 35-50'      , 423102:'y+jets 50-70'     , 423103:'y+jets 70-140'     ,
                   423104:'y+jets 140-280'    , 423105:'y+jets 280-500'   , 423106:'y+jets 500-800'    ,
                   423107:'y+jets 800-1000'   , 423108:'y+jets 1000-1500' , 423111:'y+jets 2500-3000'  ,
                   423112:'y+jets 3000'       , 423200:'direct:Jpsie3e3'  , 423201:'direct:Jpsie3e8'   ,
                   423202:'direct:Jpsie3e13'  , 423211:'np:bb -> Jpsie3e8', 423212:'np:bb -> Jpsie3e13',
                   423300:'JF17'              , 423301:'JF23'             , 423302:'JF35'              }
    channels = sample['mcChannelNumber']; headers = ['Channel', 'Process', 'Number']
    channels = sorted([[n, channel_dict[n], int(np.sum(channels==n))] for n in set(channels)], key=getkey)
    print(tabulate(channels[::-1] if reverse else channels, headers=headers, tablefmt='psql'))


def sample_analysis(sample, labels, scalars, scaler_file, output_dir):
    for key in sample: print(key, sample[key].shape)
    sys.exit()
    verify_sample(sample); sys.exit()
    # CALORIMETER IMAGES
    from plots_DG import cal_images
    layers = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2', 'em_barrel_Lr3',
              'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']
    cal_images(sample, labels, layers, output_dir, mode='random')
    # TRACKS DISTRIBUTIONS
    #from plots_DG import plot_tracks
    #arguments = [(sample['tracks_image'], labels, key,) for key in ['efrac','deta','dphi','d0','z0']]
    #processes = [mp.Process(target=plot_tracks, args=arg) for arg in arguments]
    #for job in processes: job.start()
    #for job in processes: job.join()
    # SCALARS DISTRIBUTIONS
    #from plots_DG import plot_scalars
    #sample_trans = sample.copy()
    #sample_trans = load_scaler(sample_trans, scalars, scaler_file)#[0]
    #for key in ['p_qd0Sig', 'p_sct_weight_charge']: plot_scalars(sample, sample_trans, key)


#################################################################################
#####    presampler.py functions    #############################################
#################################################################################


def presample(h5_file, output_path, batch_size, sum_e, images, tracks, scalars, integers, index):
    idx = index*batch_size, (index+1)*batch_size
    with h5py.File(h5_file, 'r') as data:
        images   = list(set(data['train']) & set(images))
        tracks   = list(set(data['train']) & set(tracks))
        scalars  = list(set(data['train']) & set(scalars+integers))
        sample = {key:data['train'][key][idx[0]:idx[1]] for key in images+tracks+scalars}
    for key in images: sample[key] = sample[key]/(sample['p_e'][:, np.newaxis, np.newaxis])
    for key in ['em_barrel_Lr1', 'em_endcap_Lr1']:
        try: sample[key+'_fine'] = sample[key]
        except KeyError: pass
    for key in images: sample[key] = resize_images(sample[key])
    for key in images+scalars: sample[key] = np.float16(sample[key])
    try: sample['p_TruthType'] = sample.pop('p_truthType')
    except KeyError: pass
    try: sample['p_TruthOrigin'] = sample.pop('p_truthOrigin')
    except KeyError: pass
    tracks_list = [np.expand_dims(get_tracks(sample,n,50     ), axis=0) for n in np.arange(batch_size)]
    sample['tracks'] = np.concatenate(tracks_list)
    tracks_list = [np.expand_dims(get_tracks(sample,n,20,'p_'), axis=0) for n in np.arange(batch_size)]
    sample['p_tracks'] = np.concatenate(tracks_list)
    tracks_list = [np.expand_dims(get_tracks(sample,n,20,'p_',True), axis=0) for n in np.arange(batch_size)]
    tracks_list = np.concatenate(tracks_list)
    tracks_dict = {'p_mean_efrac'  :0 , 'p_mean_deta'   :1 , 'p_mean_dphi'   :2 , 'p_mean_d0'          :3 ,
                   'p_mean_z0'     :4 , 'p_mean_charge' :5 , 'p_mean_vertex' :6 , 'p_mean_chi2'        :7 ,
                   'p_mean_ndof'   :8 , 'p_mean_pixhits':9 , 'p_mean_scthits':10, 'p_mean_trthits'     :11,
                   'p_mean_sigmad0':12, 'p_qd0Sig'      :13, 'p_nTracks'     :14, 'p_sct_weight_charge':15}
    #sample.update({key:tracks_list[:,tracks_dict[key]] for key in tracks_dict})
    for key in tracks_dict:
        if np.any(tracks_list[:,tracks_dict[key]]!=0): sample[key] = tracks_list[:,tracks_dict[key]]
    for key in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']: sample[key] = np.where(sample[key]==0, 1, 0)
    sample['true_m'] = np.float16(get_truth_m(sample))
    for key in tracks + ['p_truth_E', 'p_truth_e']:
        try: sample.pop(key)
        except KeyError: pass
    with h5py.File(output_path+'/'+'temp_'+'{:=02}'.format(index)+'.h5', 'w' if sum_e==0 else 'a') as data:
        for key in sample:
            shape = (sum_e+batch_size,) + sample[key].shape[1:]
            if sum_e == 0:
                maxshape = (None,) + sample[key].shape[1:]; dtype = 'i4' if key in integers else 'f2'
                data.create_dataset(key, shape, dtype=dtype, maxshape=maxshape, chunks=shape)
            else: data[key].resize(shape)
        for key in sample: data[key][sum_e:sum_e+batch_size,...] = utils.shuffle(sample[key],random_state=0)


def resize_images(images_array, target_shape=(7,11)):
    if images_array.shape[1:] == target_shape: return images_array
    else: return transform.resize(images_array, ( (len(images_array),) + target_shape))


def get_tracks(sample, idx, max_tracks=20, p='', scalars=False):
    tracks_p    = np.cosh(sample[p+'tracks_eta'][idx]) * sample[p+'tracks_pt' ][idx]
    tracks_deta =         sample[p+'tracks_eta'][idx]  - sample[  'p_eta'     ][idx]
    tracks_dphi =         sample[p+'tracks_phi'][idx]  - sample[  'p_phi'     ][idx]
    tracks_d0   =         sample[p+'tracks_d0' ][idx]
    tracks_z0   =         sample[p+'tracks_z0' ][idx]
    tracks_dphi = np.where(tracks_dphi < -np.pi, tracks_dphi + 2*np.pi, tracks_dphi )
    tracks_dphi = np.where(tracks_dphi >  np.pi, tracks_dphi - 2*np.pi, tracks_dphi )
    tracks      = [tracks_p/sample['p_e'][idx], tracks_deta, tracks_dphi, tracks_d0, tracks_z0]
    p_tracks    = ['p_tracks_charge' , 'p_tracks_vertex' , 'p_tracks_chi2'   , 'p_tracks_ndof',
                   'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits', 'p_tracks_sigmad0']
    #if p == 'p_': tracks += [sample[key][idx] for key in p_tracks]
    if p == 'p_':
        for key in p_tracks:
            try: tracks += [sample[key][idx]]
            except KeyError: tracks += [np.zeros(sample['p_tracks_charge'][idx].shape)]
    tracks      = np.float16(np.vstack(tracks).T)
    tracks      = tracks[np.isfinite(np.sum(abs(tracks), axis=1))][:max_tracks,:]
    if p == 'p_' and scalars:
        tracks_means       = np.mean(tracks,axis=0) if len(tracks)!=0 else tracks.shape[1]*[0]
        qd0Sig             = sample['p_charge'][idx] * sample['p_d0'][idx] / sample['p_sigmad0'][idx]
        sct_weight_charge  = sample['p_tracks_charge'][idx] @     sample['p_tracks_scthits'][idx]
        sct_weight_charge *= sample['p_charge'       ][idx] / sum(sample['p_tracks_scthits'][idx])
        return np.concatenate([tracks_means, np.array([qd0Sig, len(tracks), sct_weight_charge])])
    else:
        return np.vstack([tracks, np.zeros((max(0, max_tracks-len(tracks)), tracks.shape[1]))])


def get_truth_m(sample, new=True, m_e=0.511, max_eta=4.9):
    truth_eta = np.float64(np.vectorize(min)(abs(sample['p_truth_eta']), max_eta))
    try:             truth_e = np.float64(sample['p_truth_E' ])
    except KeyError: truth_e = np.float64(sample['p_truth_e' ])
    truth_pt  = np.float64(sample['p_truth_pt'])
    truth_s   = truth_e**2 - (truth_pt*np.cosh(truth_eta))**2
    if new: return np.where(truth_eta == max_eta, -1, np.sqrt(np.vectorize(max)(m_e**2, truth_s)))
    else:   return np.where(truth_eta == max_eta, -1, np.sign(truth_s)*np.sqrt(abs(truth_s)) )


def merge_presamples(n_e, n_tasks, output_path, output_file):
    temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file and '.h5' in h5_file]
    np.random.seed(0); np.random.shuffle(temp_files)
    os.rename(output_path+'/'+temp_files[0], output_path+'/'+output_file)
    dataset = h5py.File(output_path+'/'+output_file, 'a')
    GB_size = n_tasks*sum([np.float16(dataset[key]).nbytes for key in dataset])/(1024)**2/1e3
    print('MERGING TEMPORARY FILES (', '\b{:.1f}'.format(GB_size),'GB) IN:', end=' ')
    print('output/'+output_file, end=' .', flush=True); start_time = time.time()
    for key in dataset: dataset[key].resize((n_e*n_tasks,) + dataset[key].shape[1:])
    for h5_file in temp_files[1:]:
        data  = h5py.File(output_path+'/'+h5_file, 'r')
        index = temp_files.index(h5_file)
        for key in dataset: dataset[key][index*n_e:(index+1)*n_e] = data[key]
        data.close(); os.remove(output_path+'/'+h5_file)
        print('.', end='', flush=True)
    print(' (', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')


    #################################################################################
    #####  FEATURE IMPORTANCE  ######################################################
    #################################################################################

def LaTeXizer(names=[]):
    vars  = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
             'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine',
             'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3', 'tile_gap_Lr1'      ,
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'                        ]
    vars += ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'  ,
             'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2',
             'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge',
             'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_wtots1', 'p_numberOfInnermostPixelHits', 'p_EoverP' ]
    vars += ['group_0', 'group_1', 'group_2', 'group_3', 'group_4', 'group_5', 'group_6', 'group_7',
             'group_8',  'group_9', 'group_10', 'group_11']

    Lvars =  ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
              'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine',
              'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3', 'tile_gap_Lr1'      ,
              'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'                        ]
    Lvars += [r'$E_{ratio}$', r'$R_{\eta}$', r'$R_{had}$', r'$R_{\phi}$' , r'TRTPID' ,   r'Nb of SCT hits',
              'ndof', r'$\Delta p/p$', r'$\Delta \eta_1$', r'$f_1$'    ,  r'$f_3$' , r'$\Delta \phi _{res}$',
              r'$w_{\eta 2}$',  r'$d_0$', r'$d_0/{\sigma(d_0)}$' , r'qd0Sig'   , r'$n_{Tracks}$',
              r'sct wt charge',r'$\eta$'      , r'$p_t$', r'$E/p_T$'    , r'$w_{stot}$', r'$n_{Blayer}$',r'$E/p$']
    Lvars += ['em_barrel_Lr1 variables', 'em_barrel variables', 'em_endcap variables', 'em_endcap_Lr1 variables',
              'lar_endcap variables', 'tile variables', r'$d_0$ variables 1', r'$d_0$ variables 2', r'$f_1$ and $f_3$',
              r'$n_{Tracks}$ and sct wt charge',  r'$n_{Tracks}$ and $p_t$', 'group 11']

    converter = {var:Lvar for var,Lvar in zip(vars,Lvars)}
    Lnames = [converter[name] for name in names]
    return converter,Lnames


def feature_permutation(model, valid_sample, labels, valid_probs, feats, g , n_rep, file):      # feats must be a list
    features = ' + '.join(feats)
    print('PERMUTATION DE : ' + features)
    bkg_rej = np.empty(n_rep)
    fpr, tpr, _ = metrics.roc_curve(labels, valid_probs[:,0], pos_label=0)
    bkg_rej_full = 1/fpr[np.argwhere(tpr>=0.7)[0]][0]
    shuffled_sample = {key:value for (key,value) in valid_sample.items() if key not in feats}
    for feat in feats:
        shuffled_sample[feat] = deepcopy(valid_sample[feat])                                # Copy of the feature to be shuffled in order to keep valid_sample intact

    for k in range(n_rep):                                                                  # Reshuffling loop
        print('PERMUTATION DE ' + features + " " + str(k+1))
        for feat in feats:
            rdm.shuffle(shuffled_sample[feat])                                              # Shuffling of one feature
        probs = model.predict(shuffled_sample, batch_size=20000, verbose=1)                 # Prediction with only one feature shuffled
        fpr, tpr, _ = metrics.roc_curve(labels, probs[:,0], pos_label=0)
        bkg_rej[k] = 1/fpr[np.argwhere(tpr>=0.7)[0]][0]                                     # Background rejection with one feature shuffled

    name = [feats[0],'group_{}'.format(g)][bool(g+1)]
    importance = bkg_rej_full / bkg_rej                                                     # Comparison with the unshuffled sample
    imp_tup = name, np.mean(importance), np.std(importance)
    with open(file,'ab') as afp:                                                            # Saving the results in a pickle
        pickle.dump(imp_tup, afp)

def print_importances(file):
    with open(file,'rb') as rfp:
        record = dict()
        while True:
            try:
                imp = pickle.load(rfp)
                record[imp[0]] = imp[1:]
            except EOFError:
                break
    print(record)
    return record

def plot_importances(results, path, title):
    sortedResults = sorted(results.items(), key = lambda lst: lst[1][0], reverse=True)
    labels = [tup[0] for tup in sortedResults]
    newLabels = LaTeXizer(labels)[1]
    data = [tup[1][0] for tup in sortedResults]
    try :
        error = [tup[1][1] for tup in sortedResults]
    except:
        error = np.zeros(len(sortedResults))
    data = np.array(data)
    error = np.array(error)

    fig, ax = plt.subplots(figsize=(18.4, 10))
    ax.invert_yaxis()
    widths = data
    print(labels)
    color = ['g' if 'variables' in label or 'and' in label else 'tab:blue' for label in newLabels]
    print(color)
    ax.barh(newLabels, widths, height=0.75, xerr=error, capsize=5, color=color)
    xcenters = widths / 2

    text_color = 'white'
    for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(round(c,3)), ha='center', va='center', color=text_color)

    plt.axvline(1,color='r', ls=':')
    plt.title(title, fontsize=20)
    ax.set_xlabel(r'$\frac{bkg\_rej\_full}{bkg\_rej}$', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    plt.savefig(path)
    return fig, ax

def removal_bkg_rej(model,valid_probs,labels,feat,file):
    fpr, tpr, _ = metrics.roc_curve(labels, valid_probs[:,0], pos_label=0)
    bkg_rej_tup = feat, 1/fpr[np.argwhere(tpr>=0.7)[0]][0]                                  # Background rejection with one feature removed
    with open(file,'wb') as wfp:                                                            # Saving the results in a pickle
        pickle.dump(bkg_rej_tup, wfp)


def correlations(sample, dir, scatter=False, LaTeX=True, frmt = '.pdf', mode='', fname=''):
    data = pd.DataFrame(sample)
    if LaTeX:
        print("LaTeX : ", "ON" if LaTeX else 'OFF')
        data = data.rename(columns = LaTeXizer()[0])
    names = data.columns
    correlations = data.corr()

    # plot scatter plot matrix
    if scatter == 'SCATTER':
        print('Plotting scatter plot matrix')
        scatter_matrix(data, figsize = (18,18))
        plt.suptitle('Scatter plot matrix' + mode, fontsize = 20)
        plt.yticks(rotation=-90)
        plt.tight_layout()
        plt.savefig(dir + 'scatter_plot_matrix' + fname + '.png')

    # plot correlation matrix
    else:
        print('Plotting correlation matrix')
        fig = plt.figure(figsize=(20,18))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        for (i, j), z in np.ndenumerate(correlations):
            ax.text(j, i, '{:0.1f}'.format(z) if abs(z) > 0.15 and z != 1.0 else '', ha='center', va='center', fontsize=8)
        ticks = np.arange(0,len(names),1)
        xtcks = np.arange(0,len(names),1, dtype = 'float64')
        try :
            xtcks[[5,17]] += 0.35
        except:
            pass
        ax.set_xticks(xtcks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names, fontsize = 14)
        ax.set_yticklabels(names, fontsize = 14)
        plt.xticks(rotation=[30,90][fname == '_with_tracks'])
        plt.title('Correlation matrix' + mode, fontsize = 20)
        plt.tight_layout()
        plt.savefig(dir + 'corr_matrix' + fname + frmt)


        #################################################################################
        #####  UNDER DEVELOPMENT   ######################################################
        #################################################################################

class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_name, n_classes, train_features, all_features, indices, batch_size):
        self.file_name  = file_name ; self.train_features = train_features
        self.indices    = indices   ; self.all_features   = all_features
        self.batch_size = batch_size; self.n_classes      = n_classes
    def __len__(self):
        "number of batches per epoch"
        return int(self.indices.size/self.batch_size)
    def __getitem__(self, index):
        data   = generator_sample(self.file_name, self.all_features, self.indices, self.batch_size, index)
        labels = make_labels(data, self.n_classes)
        data   = [np.float32(data[key]) for key in np.sum(list(self.train_features.values()))]
        return data, labels
