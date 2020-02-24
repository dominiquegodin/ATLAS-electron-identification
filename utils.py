import tensorflow            as tf, numpy as np, h5py, os, time, sys
from   sklearn.metrics       import confusion_matrix
from   sklearn.utils         import shuffle
from   sklearn.preprocessing import QuantileTransformer, StandardScaler
from   pickle                import dump, load
from   tabulate              import tabulate
from   skimage               import transform
from   skimage.util          import img_as_int
from   scipy.sparse          import csr_matrix


def valid_data(data_file, all_var, scalars, idx, float16, n_tracks,
               n_classes, scaling, pickle_file, weight_file):
    start_time   = time.time()
    valid_sample = make_sample(data_file, all_var, idx, n_tracks, float16)
    valid_labels = make_labels(valid_sample, n_classes)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    valid_sample, valid_labels = filter_sample(valid_sample, valid_labels)
    if weight_file != None and scaling:
        valid_sample = load_scaler(valid_sample, scalars, pickle_file)
    return valid_sample, valid_labels


def train_data(data_file, valid_sample, all_var, scalars, idx, n_tracks, n_classes,
               float16, resampling, scaling, scaler_file, pickle_file, weight_file):
    start_time   = time.time()
    train_sample = make_sample(data_file, all_var, idx, n_tracks, float16)
    train_labels = make_labels(train_sample, n_classes)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    train_sample, train_labels = filter_sample(train_sample, train_labels)
    analyze_sample(train_sample, scan=False); print()
    if resampling == 'ON':
        train_sample, train_labels = balance_sample(train_sample, train_labels, n_classes)
    if weight_file == None and scaling:
        train_sample, valid_sample = run_scaler(train_sample, valid_sample, scalars, scaler_file)
    if weight_file != None and scaling:
        train_sample = load_scaler(train_sample, scalars, pickle_file)
    return train_sample, valid_sample, train_labels


def make_sample(data_file, all_var, idx, n_tracks, float16, p='', upscale=False, denormalize=False):
    var_list = np.sum(list(all_var.values())); images = all_var['images']; tracks = all_var['tracks']
    with h5py.File(data_file, 'r') as data:
        sample = dict([key, data[key][idx[0]:idx[1]]] for key in var_list if key != 'tracks_image')
        if 'tracks_image' in var_list:
            n_tracks = min(n_tracks, data[p+'tracks'].shape[1])
            tracks_image = data[p+'tracks'][idx[0]:idx[1]][:,:n_tracks,:]
            tracks_image = np.concatenate((abs(tracks_image[...,0:5]), tracks_image[...,5:]), axis=2)
            sample.update({'tracks_image':tracks_image})
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
        return   np.where(np.logical_or ( MC_type== 2,  MC_type== 4),  0, 1       )
    elif n_classes == 2:
        labels = np.where(IFF_type<=1                               , -1, IFF_type)
        labels = np.where(IFF_type>=4                               ,  1, labels  )
        return   np.where(np.logical_or (IFF_type== 2, IFF_type== 3),  0, labels  )
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
    if  len(label_rows) == len(labels): print(); return sample, labels
    for key in sample: sample[key] = np.take(sample[key], label_rows, axis=0)
    print('CLASSIFIER: filtering sample ->', len(label_rows), 'e conserved', end='')
    print(' (' + format(100*len(label_rows)/len(labels), '.2f') + ' %)\n')
    return sample, np.take(labels, label_rows)


def analyze_sample(sample, scan=False):
    MC_type,  IFF_type  = sample['p_TruthType'], sample['p_iffTruth']
    MC_list,  IFF_list  = np.arange(max( MC_type)+1)    , np.arange(max(IFF_type)+1)
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
    print('   |  100 %  |\n' + dash)
    if scan:
        print('SCANNING FOR VALUE ERRORS IN SAMPLE ...', end=' ', flush=True)
        bad_values = [np.sum(np.isfinite(sample[key])==False) for key in sample]
        #for key in sample: print(key, np.where(np.isfinite(sample[key])==False))
        print(sum(bad_values), 'ERRORS FOUND')


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


def run_scaler(train_sample, valid_sample, scalars, scaler_file):
    print('CLASSIFIER: applying scaler transform to scalar variables', end=' ... ', flush=True)
    start_time    = time.time()
    train_scalars = np.hstack([np.expand_dims(train_sample[key], axis=1) for key in scalars])
    valid_scalars = np.hstack([np.expand_dims(valid_sample[key], axis=1) for key in scalars])
    scaler        = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=0)
    train_scalars = scaler.fit_transform(train_scalars)
    valid_scalars = scaler.transform(valid_scalars)
    for n in np.arange(len(scalars)):
        train_sample[scalars[n]] = train_scalars[:,n]
        valid_sample[scalars[n]] = valid_scalars[:,n]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('CLASSIFIER: saving fitted data in scaler: outputs/' + scaler_file + '\n')
    dump(scaler, open('outputs/' + scaler_file, 'wb'))
    return train_sample, valid_sample


def load_scaler(sample, scalars, scaler_file):
    print('CLASSIFIER: loading fitted data from scaler: outputs/' + scaler_file)
    scaler         = load(open('outputs/' + scaler_file, 'rb'))
    start_time     = time.time()
    scalars_scaled = np.hstack([np.expand_dims(sample[key], axis=1) for key in scalars])
    print('CLASSIFIER: applying scaler transform to scalar variables', end=' ... ', flush=True)
    scalars_scaled = scaler.transform(scalars_scaled)
    for n in np.arange(len(scalars)): sample[scalars[n]] = scalars_scaled[:,n]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
    return sample


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
    print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f"))


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
    idx = index*batch_size, (index+1)*batch_size
    with h5py.File(h5_file, 'r') as data:
        sample = {key:data['train'][key][idx[0]:idx[1]] for key in images + tracks + scalars + int_var}
    for key in images: sample[key] = sample[key]/(sample['p_e'][:, np.newaxis, np.newaxis])
    for key in set(images)-set(['em_barrel_Lr1']): sample[key] = resize_images(sample[key])
    for key in images+scalars: sample[key] = np.float16(sample[key])
    tracks_list = [np.expand_dims(get_tracks(sample,n,50     ), axis=0) for n in np.arange(batch_size)]
    sample.update({'tracks'  :np.concatenate(tracks_list)})
    tracks_list = [np.expand_dims(get_tracks(sample,n,30,'p_'), axis=0) for n in np.arange(batch_size)]
    sample.update({'p_tracks':np.concatenate(tracks_list)})
    #sample.update({'true_m':np.float16(get_truth_m(sample))})
    for key in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']: sample[key] = get_LLH(sample, key)
    for key in tracks + ['p_truth_E', 'p_LHValue']: sample.pop(key)
    with h5py.File(output_path+'temp_'+'{:=02}'.format(index)+'.h5', 'w' if sum_e==0 else 'a') as data:
        for key in sample:
            shape = (sum_e+batch_size,) + sample[key].shape[1:]
            if sum_e == 0:
                maxshape = (None,) + sample[key].shape[1:]; dtype = 'i4' if key in int_var else 'f2'
                data.create_dataset(key, shape, dtype=dtype, maxshape=maxshape, chunks=shape)
            else: data[key].resize(shape)
        for key in sample: data[key][sum_e:sum_e+batch_size,...] = shuffle(sample[key], random_state=0)
    '''
    temp_file = 'pid_' + h5_file.split('.')[-2].split('_')[-1] + '_temp_{:=02}'.format(index) + '.h5'
    with h5py.File(output_path + temp_file, 'w') as data:
        for key in sample:
            shape = (batch_size,) + sample[key].shape[1:]
            dtype    = 'i4' if key in int_var else 'f2'
            data.create_dataset(key, shape, dtype=dtype, chunks=shape)
        for key in sample: data[key][0:batch_size] = shuffle(sample[key], random_state=0)
    '''


def get_tracks(sample, idx, max_tracks=20, p=''):
    tracks_p    = np.cosh(sample[p+'tracks_eta'][idx]) * sample[p+'tracks_pt' ][idx]
    tracks_deta =         sample[p+'tracks_eta'][idx]  - sample[  'p_eta'     ][idx]
    tracks_dphi =         sample[p+'tracks_phi'][idx]  - sample[  'p_phi'     ][idx]
    tracks_d0   =         sample[p+'tracks_d0' ][idx]
    tracks_z0   =         sample[p+'tracks_z0' ][idx]
    tracks_dphi = np.where(tracks_dphi  < -np.pi, tracks_dphi + 2*np.pi, tracks_dphi )
    tracks_dphi = np.where(tracks_dphi  >  np.pi, tracks_dphi - 2*np.pi, tracks_dphi )
    tracks      = [tracks_p/sample['p_e'][idx], tracks_deta, tracks_dphi, tracks_d0, tracks_z0]
    p_tracks    = ['p_tracks_charge' , 'p_tracks_theta'  , 'p_tracks_sigmad0',
                   'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits']
    tracks      = tracks + [sample[key][idx] for key in p_tracks] if p=='p_' else tracks
    tracks      = np.float16(np.vstack(tracks).T)
    tracks      = tracks[np.isfinite(np.sum(abs(tracks), axis=1))][:max_tracks,:]
    return        np.vstack([tracks, np.zeros((max(0, max_tracks-len(tracks)), tracks.shape[1]))])


def get_LLH(sample, wp):
    LLH =  np.where(sample[wp] == 0                               , 2, 0  )
    return np.where(np.logical_and(LLH==2, sample['p_LHValue']<-1), 1, LLH)


def get_truth_m(sample, new=True, m_e=0.511, max_eta=4.9):
    truth_eta = np.float64(np.vectorize(min)(abs(sample['p_truth_eta']), max_eta))
    truth_e   = np.float64(sample['p_truth_E' ])
    truth_pt  = np.float64(sample['p_truth_pt'])
    truth_s   = truth_e**2 - (truth_pt*np.cosh(truth_eta))**2
    if new: return np.where(truth_eta == max_eta, -1, np.sqrt(np.vectorize(max)(m_e**2, truth_s)))
    else:   return np.where(truth_eta == max_eta, -1, np.sign(truth_s)*np.sqrt(abs(truth_s)) )


def merge_presamples(n_e, n_files, output_path, output_file):
    temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file and '.h5' in h5_file]
    np.random.seed(0); np.random.shuffle(temp_files)
    os.rename(output_path+temp_files[0], output_path+output_file)
    dataset = h5py.File(output_path+output_file, 'a')
    GB_size = n_files*sum([np.float16(dataset[key]).nbytes for key in dataset])/(1024)**2/1e3
    print('MERGING TEMPORARY FILES (', '\b{:.1f}'.format(GB_size),'GB) IN:', end=' ')
    print('output/'+output_file, end=' .', flush=True); start_time = time.time()
    for key in dataset: dataset[key].resize((n_e*n_files,) + dataset[key].shape[1:])
    for h5_file in temp_files[1:]:
        data  = h5py.File(output_path+h5_file, 'r')
        index = temp_files.index(h5_file)
        for key in dataset: dataset[key][index*n_e:(index+1)*n_e] = data[key]
        data.close(); os.remove(output_path+h5_file)
        print('.', end='', flush=True)
    print(' (', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
    print(n_e*n_files, 'ELECTRONS COLLECTED\n')


def make_images(tracks):
    import matplotlib.pyplot as plt
    from functools import reduce
    #var_dict = {'efrac':{'idx':0},
    #            'deta' :{'idx':1, 'lim':(0, 0.4), 'dim':100}, 'dphi' :{'idx':2, 'lim':(0, 0.4), 'dim':100},
    #            'd0'   :{'idx':3, 'lim':(0,  10), 'dim':100}, 'z0'   :{'idx':4, 'lim':(0, 300), 'dim':100}}
    var_dict = {'efrac':{'idx':0},
                'deta' :{'idx':1, 'lim':(-0.4, 0.4), 'dim':100}, 'dphi' :{'idx':2, 'lim':(-0.4, 0.4), 'dim':100},
                'd0'   :{'idx':3, 'lim':(-10,  10), 'dim':100}, 'z0'   :{'idx':4, 'lim':(-300, 300), 'dim':100}}
    def indices(x, lim, dim)     : return np.round((dim-1)*(x-lim[0])/(lim[1]-lim[0]))
    def keep(x, xdim, y, ydim, z): return z[reduce(np.logical_and,[x>=0, x<=xdim-1, y>=0, y<=ydim-1])]
    #start_time = time.time()
    #print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print(tracks[10])
    print(max(tracks[10,4]))
    n_e      = len(tracks)
    n_tracks = np.sum(abs(tracks), axis=2)
    n_tracks = np.array([len(np.where(n_tracks[n,:]!=0)[0]) for n in np.arange(n_e)])
    #print( len(n_tracks[n_tracks==0]) )
    #print( np.where(n_tracks==100) )
    #print( np.float16(tracks[167]) )
    #print( tracks[167].shape )
    tracks = [tracks[n,:n_tracks[n],:] for n in np.arange(len(n_tracks))]
    #print( tracks[168].shape )
    #print( tracks[168] )
    #print(n_tracks)
    sizes = np.array([n.size for n in tracks])
    print( len(sizes[sizes==0]) )
    print( n_e, len(tracks) )

    for track in tracks[10:11]:
        efrac, deta, dphi, d0, z0 = [ track[:,n] for n in np.arange(5)]
        deta = indices(deta, var_dict['deta']['lim'], var_dict['deta']['dim'])
        dphi = indices(dphi, var_dict['dphi']['lim'], var_dict['dphi']['dim'])
        d0   = indices(d0  , var_dict['d0'  ]['lim'], var_dict['d0'  ]['dim'])
        z0   = indices(z0  , var_dict['z0'  ]['lim'], var_dict['z0'  ]['dim'])
        deta, dphi, efrac = [keep(deta, var_dict['deta']['dim'], dphi, var_dict['dphi']['dim'], x)
                             for x in [deta, dphi, efrac]]
        fig_efrac = csr_matrix((efrac, (deta, dphi)),
                               shape=(var_dict['deta']['dim'], var_dict['dphi']['dim'])).toarray()
        d0, z0, ones      = [keep(d0, var_dict['d0']['dim'], z0, var_dict['z0']['dim'], x)
                             for x in [d0, z0, np.ones(len(d0))]]
        fig_d0z0  = csr_matrix((ones, (d0, z0)),
                               shape=(var_dict['d0']['dim'], var_dict['z0']['dim'])).toarray()
        print(fig_efrac.shape)
        print(fig_d0z0.shape, np.sum(fig_d0z0))
        #print(np.float16(fig_efrac))
        print(np.float16(fig_d0z0))
    fig = plt.figure(figsize=(18,8))

    plt.subplot(1,2,1)
    plt.imshow(fig_efrac, cmap='Reds')
    plt.subplot(1,2,2)
    plt.imshow(fig_d0z0, cmap='Reds')

    file_name = 'outputs/images.png'
    fig.savefig(file_name)
