import tensorflow            as tf, numpy as np, h5py, os, time, sys
from   sklearn.metrics       import confusion_matrix
from   sklearn.utils         import shuffle
from   sklearn.preprocessing import QuantileTransformer
from   tabulate              import tabulate
from   skimage               import transform
import matplotlib.pyplot as plt

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

def generate_weights(train_data,train_labels,nClass,weight_type='none',ref_var='pt',output_dir='outputs/'):
    if weight_type=="none": return None

    print("-------------------------------")
    print("generate_weights: sample weight mode \"",weight_type,"\" designated. Generating weights.",)
    print("-------------------------------")

    binning=[0,10,20,30,40,60,80,100,130,180,250,500]
    labels=['sig','bkg']
    colors=['blue','red']
    binContents=[0,0]
    if nClass>2:
        #below 2b implemented
        labels=['sig','bkg']
        colors=['blue','red']
        binContents=[0,0]
        pass

    variable=list()                          #only for specific label
    variable_array = train_data['p_et_calo'] #entire set
    if   ref_var=='eta'  : variable_array = train_data['p_eta']
    #elif ref_var=='pteta': variable_array = train_data['p_eta']

    for i_class in range(nClass):
        variable.append( variable_array[ train_labels==i_class ] )
        (binContents[i_class],bins,patches)=plt.hist(variable[i_class],bins=binning,weights=np.full(len(variable[i_class]),1/len(variable[i_class])),label=labels[i_class],histtype='step',facecolor=colors[i_class])
        pass

        plt.savefig(output_dir+ref_var+"_bfrReweighting.png")
    plt.clf() #clear figure

    weights=list() #KM: currently implemented for the 2-class case only
    if weight_type=="flattening":
        weights.append(np.average(binContents[0])/binContents[0] )
        weights.append(np.average(binContents[1])/binContents[1] )
    elif weight_type=="match2b": #shaping sig to match the bkg, using pt,or any other designated variable
        weights.append(binContents[1]/binContents[0])
        weights.append(np.ones(len(binContents[1])))
    elif weight_type=="match2s": #shaping bkg to match the sig, using pt,or any other designated variable
        weights.append(np.ones(len(binContents[0])))
        weights.append(binContents[0]/binContents[1])

    debug=0
    if debug:
        print(weights[0])
        print(weights[1])
        pass

    #KM: Generates weights for all events
    #    This is not very efficient, to be improved

    #final_weights*=weights[train_labels][1]

    sig_weight=np.full(len(variable_array),0,dtype=float)
    bkg_weight=np.full(len(variable_array),0,dtype=float)

    bin_indices0or1=find_bin(variable_array,binning)
    tmp_i=0
    for vec01 in bin_indices0or1:
        sig_weight += (vec01 * (train_labels==0) )* weights[0][tmp_i]
        bkg_weight += (vec01 * (train_labels==1) )* weights[1][tmp_i]
        tmp_i+=1
        pass

    if debug:
        print()
        print(sig_weight,"\n", bkg_weight)
        print(sig_weight+bkg_weight,(sig_weight+bkg_weight).all()==1) # w
        print(train_labels)
        pass

    final_weights = sig_weight+bkg_weight

    if debug:
        print(variable_array)
        print(final_weights, len(final_weights), "any element is zero?",final_weights.any()==0)
        pass

    #KM: below only for plotting
    for i_class in range(nClass):
        plt.hist(variable[i_class],bins=binning,weights=final_weights[ train_labels==i_class ],label=labels[i_class],histtype='step',facecolor=colors[i_class])
        pass
    plt.savefig(output_dir+ref_var+"_aftReweighting.png")
    plt.clf() #clear plot

    return final_weights

def make_sample(data_file, var_dict, idx, float16=True, n_tracks=15, upscale=False, denormalize=False):
    data  , var_list = h5py.File(data_file, 'r'), np.sum(list(var_dict.values()))
    images, tracks   = var_dict['images']       , var_dict['tracks']
    sample = dict([key, data[key][idx[0]:idx[1]]] for key in var_list if key != 'tracks_image')
    if 'tracks_image' in var_list:
        n_tracks = min(n_tracks, data['tracks'].shape[1])
        sample.update({'tracks_image':data['tracks'][idx[0]:idx[1]][:,:n_tracks,0:5]})
    if not float16: sample = {key:np.float32(sample[key]) for key in sample}
    if denormalize:
        for n in images: sample[n]        = sample[n]        * sample['p_e'][:, np.newaxis, np.newaxis]
        for n in tracks: sample[n][:,:,0] = sample[n][:,:,0] * sample['p_e'][:, np.newaxis]
    if upscale:
        for n in images: sample[n]        = resize_images(np.float32(sample[n]), target_shape=(56,11))
    return sample


def resize_images(images_array, target_shape=(7,11)):
    if images_array.shape[1:] == target_shape: return images_array
    else: return transform.resize(images_array, ( (len(images_array),) + target_shape))


def make_labels(sample, n_classes, MC_truth=False):
    MC_type, IFF_type = sample['p_TruthType'], sample['p_iffTruth']
    if n_classes == 2 and MC_truth:
        return   np.where(np.logical_or(MC_type==2, MC_type==4),    0,    1       )
    elif n_classes == 2:
        labels = np.where(IFF_type<=1                            , -1,    IFF_type)
        labels = np.where(IFF_type>=4                            ,  1,    labels  )
        return   np.where(np.logical_or(IFF_type==2, IFF_type==3),  0,    labels  )
    elif n_classes == 6:
        labels = np.where(np.logical_or (IFF_type<= 1, IFF_type== 4), -1, IFF_type)
        labels = np.where(np.logical_or (IFF_type== 6, IFF_type== 7), -1, labels  )
        labels = np.where(IFF_type==2                               ,  0, labels  )
        labels = np.where(IFF_type==3                               ,  1, labels  )
        labels = np.where(IFF_type==5                               ,  2, labels  )
        labels = np.where(np.logical_or (IFF_type== 8, IFF_type== 9),  3, labels  )
        labels = np.where(np.logical_and(IFF_type==10,  MC_type== 4),  4, labels  )
        labels = np.where(np.logical_and(IFF_type==10,  MC_type==16),  4, labels  )
        labels = np.where(np.logical_and(IFF_type==10,  MC_type==17),  5, labels  )
        return   np.where(  labels==10                              , -1, labels  )
    elif n_classes == 9:
        labels = np.where(IFF_type== 9                              ,  4, IFF_type)
        return   np.where(IFF_type==10                              ,  6, labels  )
    else: print('\nCLASSIFIER: classes not supported -> exiting program\n'); sys.exit()


def filter_sample(sample, labels):
    label_rows  = np.where(labels!=-1)[0]
    n_conserved = 100*len(label_rows)/len(labels)
    if n_conserved == 100: print(); return sample, labels
    for key in sample: sample[key] = np.take(sample[key], label_rows, axis=0)
    print('CLASSIFIER: filtering sample ->', format(n_conserved, '.2f'),'% conserved\n')
    return sample, np.take(labels, label_rows)


def analyze_sample(sample, scan=False):
    MC_type,  IFF_type  = np.int8(sample['p_TruthType']), np.int8(sample['p_iffTruth'])
    MC_list,  IFF_list  = np.arange(max( MC_type)+1)    , np.arange(max(IFF_type)+1)
    ratios = np.array([ [np.sum(MC_type[IFF_type==IFF]==MC) for MC in MC_list] for IFF in IFF_list ])
    IFF_sum, MC_sum = 100*np.sum(ratios, axis=0)/len(MC_type), 100*np.sum(ratios, axis=1)/len(MC_type)
    ratios = np.round(1e4*ratios/len(MC_type))/100
    MC_empty, IFF_empty = np.where(np.sum(ratios, axis=0)==0)[0], np.where(np.sum(ratios, axis=1)==0)[0]
    MC_list,  IFF_list  = list(set(MC_list)-set(MC_empty))      , list(set(IFF_list)-set(IFF_empty))
    print('\nIFF AND MC TRUTH CLASSIFIERS SAMPLE COMPOSITION (', '\b'+str(len(MC_type)), 'e)')
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
    print('   |  100 %  |\n' + dash)
    if scan:
        print('SCANNING FOR VALUE ERRORS IN SAMPLE ...', end=' ', flush=True)
        bad_values = [np.sum(np.isfinite(sample[key])==False) for key in sample]
        print(sum(bad_values), 'ERRORS FOUND\n')


def balance_sample(sample, labels, n_classes):
    print('CLASSIFIER: rebalancing train sample', end=' ... ', flush=True)
    start_time = time.time()
    class_size = int(len(labels)/n_classes)
    label_rows = [np.where(labels==m)[0] for m in np.arange(n_classes)]
    label_rows = [np.random.choice(label_rows[m], class_size, replace=len(label_rows[m])<class_size)
                  for m in np.arange(n_classes)]
    label_rows = np.concatenate(label_rows); np.random.shuffle(label_rows)
    for key in sample: sample[key] = np.take(sample[key], label_rows, axis=0)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
    return sample, np.take(labels, label_rows)


def transform_sample(train_sample, valid_sample, scalars):
    print('\nCLASSIFIER: applying Quantile transform to scalars', end=' ... ', flush=True)
    start_time    = time.time()
    train_scalars = np.hstack([np.expand_dims(train_sample[key], axis=1) for key in scalars])
    valid_scalars = np.hstack([np.expand_dims(valid_sample[key], axis=1) for key in scalars])
    quantile      = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=0)
    train_scalars = quantile.fit_transform(train_scalars)
    valid_scalars = quantile.transform(valid_scalars)
    for n in np.arange(len(scalars)):
        train_sample[scalars[n]] = train_scalars[:,n]
        valid_sample[scalars[n]] = valid_scalars[:,n]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
    return train_sample, valid_sample


def show_matrix(train_labels, valid_labels, valid_prob=[]):
    if valid_prob == []: valid_pred = valid_labels
    else:                valid_pred = np.argmax(valid_prob, axis=1)
    matrix      = confusion_matrix(valid_labels, valid_pred)
    matrix      = 100*matrix.T/matrix.sum(axis=1)
    n_classes   = len(matrix)
    valid_sizes = [100*np.sum(valid_labels==n)/len(valid_labels) for n in np.arange(n_classes)]
    train_sizes = [100*np.sum(train_labels==n)/len(train_labels) for n in np.arange(n_classes)]
    classes     = ['CLASS '+str(n) for n in np.arange(n_classes)]
    if valid_prob == []:
        print('+--------------------------------------+')
        print('| CLASS DISTRIBUTIONS                  |')
        headers = ['CLASS #', 'TRAIN (%)', 'TEST (%)']
        table   = zip(classes, train_sizes, valid_sizes)
    else:
        if n_classes > 2:
            headers = ['CLASS #', 'TRAIN', 'TEST'] + classes
            table   = [classes] + [train_sizes] + [valid_sizes] + matrix.T.tolist()
            table   = list(map(list, zip(*table)))
            print('\n+'+30*'-'+'+'+35*'-'+12*(n_classes-3)*'-'+'+\n', '\b| CLASS DISTRIBUTIONS (%)',
                  '    ', '| TEST SAMPLE PREDICTIONS (%)       '+12*(n_classes-3)*' '+ '|')
        else:
            headers = ['CLASS #', 'TRAIN (%)', 'TEST (%)', 'ACC. (%)']
            table   = zip(classes, train_sizes, valid_sizes, matrix.diagonal())
            print('\n+---------------------------------------------------+')
            print(  '| CLASS DISTRIBUTIONS AND TEST SAMPLE ACCURACIES    |')
    print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f"), '\n')


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


def presample(h5_file, output_path, batch_size, sum_e, images, tracks, scalars, int_var, index):
    data    = h5py.File(h5_file, 'r'); idx = index*batch_size, (index+1)*batch_size
    energy  =                data['train']['p_e'][idx[0]:idx[1]][:, np.newaxis, np.newaxis        ]
    sample  = [resize_images(data['train'][ key ][idx[0]:idx[1]])/energy for key in images        ]
    sample += [              data['train'][ key ][idx[0]:idx[1]]         for key in tracks+scalars]
    sample  = dict(zip(images+tracks+scalars, sample))
    sample.update({'em_barrel_Lr1_fine':data['train']['em_barrel_Lr1'][idx[0]:idx[1]]/energy})
    tracks_list = [np.expand_dims(get_tracks(sample, n), axis=0) for n in np.arange(batch_size)]
    sample.update({'tracks':np.concatenate(tracks_list), 'true_m':get_truth_m(sample)})
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']: sample[wp] = get_LLH(sample, wp)
    for var in tracks + ['p_truth_E', 'p_LHValue']: sample.pop(var)
    data = h5py.File(output_path+'temp_'+'{:=02}'.format(index)+'.h5', 'w' if sum_e==0 else 'a')
    for key in sample:
        shape = (sum_e+batch_size,) + sample[key].shape[1:]
        if sum_e == 0:
            maxshape =  (None,) + sample[key].shape[1:]
            dtype    = 'i4' if key in int_var else 'f2' if key=='tracks' else 'f2'
            data.create_dataset(key, shape, dtype=dtype, maxshape=maxshape, chunks=shape)
        else: data[key].resize(shape)
    for key in sample: data[key][sum_e:sum_e+batch_size,...] = shuffle(sample[key], random_state=0)


def get_tracks(sample, idx, max_tracks=100):
    tracks_efrac = (np.cosh(sample['tracks_eta'][idx]) * sample['tracks_pt' ][idx]) / sample['p_e'][idx]
    #tracks_deta  =      abs(sample['tracks_eta'][idx]  - sample['p_eta'     ][idx])
    tracks_deta  =          sample['tracks_eta'][idx]  - sample['p_eta'     ][idx]
    tracks_dphi  =          sample['p_phi'     ][idx]  - sample['tracks_phi'][idx]
    tracks_d0    =          sample['tracks_d0' ][idx]
    tracks_z0    =          sample['tracks_z0' ][idx]
    tracks_efrac = np.where(tracks_efrac >  65000, 65000                , tracks_efrac)
    tracks_dphi  = np.where(tracks_dphi  < -np.pi, tracks_dphi + 2*np.pi, tracks_dphi )
    tracks_dphi  = np.where(tracks_dphi  >  np.pi, tracks_dphi - 2*np.pi, tracks_dphi )
    tracks       = np.vstack([tracks_efrac, tracks_deta, tracks_dphi, tracks_d0, tracks_z0]).T
    numbers      = list(set(np.arange(len(tracks))) - set(np.where(np.isfinite(tracks)==False)[0]))
    tracks       = np.take(tracks, numbers, axis=0)[:max_tracks,:]
    return         np.vstack([tracks, np.zeros((max(0, max_tracks-len(tracks)), tracks.shape[1]))])


def get_LLH(sample, wp):
    LLH =  np.where(sample[wp] == 0,                         2, 0  )
    return np.where((LLH == 2) & (sample['p_LHValue'] < -1), 1, LLH)


def get_truth_m(sample, new=True, m_e=0.511, max_eta=4.9):
    truth_eta = np.vectorize(min)(abs(sample['p_truth_eta']), max_eta)
    truth_s   = sample["p_truth_E"]**2 - (sample['p_truth_pt']*np.cosh(truth_eta))**2
    if new: return np.where(truth_eta == max_eta, -1, np.sqrt(np.vectorize(max)(m_e**2, truth_s)))
    else:   return np.where(truth_eta == max_eta, -1, np.sign(truth_s)*np.sqrt(abs(truth_s)) )


def merge_presamples(n_e, n_files, output_path, output_file):
    temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
    np.random.seed(0); np.random.shuffle(temp_files)
    os.rename(output_path+temp_files[0], output_path+output_file)
    dataset = h5py.File(output_path+output_file, 'a')
    GB_size = n_files*sum([np.float16(dataset[key]).nbytes for key in dataset])/(1024)**2/1e3
    print('MERGING TEMPORARY FILES (', '\b{:.1f}'.format(GB_size),'GB) INTO:', end=' ')
    print('output/'+output_file, end=' .', flush=True); start_time = time.time()
    for key in dataset: dataset[key].resize((n_e*n_files,) + dataset[key].shape[1:])
    for h5_file in temp_files[1:]:
        data  = h5py.File(output_path+h5_file, 'r')
        index = temp_files.index(h5_file)
        for key in dataset: dataset[key][index*n_e:(index+1)*n_e] = data[key]
        data.close(); os.remove(output_path+h5_file)
        print('.', end='', flush=True)
    print(' (', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)\n')
