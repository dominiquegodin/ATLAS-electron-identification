import tensorflow            as tf, numpy as np, h5py, os, time, sys
from   sklearn.metrics       import confusion_matrix
from   sklearn.utils         import shuffle
from   sklearn.preprocessing import QuantileTransformer
from   tabulate              import tabulate
from   skimage               import transform


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
