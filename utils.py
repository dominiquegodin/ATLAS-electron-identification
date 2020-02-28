import tensorflow            as tf, numpy as np, multiprocessing, time, os, sys, h5py
from   sklearn.metrics       import confusion_matrix
from   sklearn.utils         import shuffle
from   sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from   pickle                import dump, load
from   tabulate              import tabulate
from   skimage               import transform
from   skimage.util          import img_as_int
from   scipy.sparse          import csr_matrix



#################################################################################
##### Batch_classifier.py functions #############################################
#################################################################################


def sample_checks(sample, labels):
    #for key in sample: print(key, sample[key].shape)
    make_images(sample['tracks_image']); sys.exit()
    analyze_sample(sample, scan=True); sys.exit()
    # trackss distributions
    arguments = [(sample['tracks_image'], labels, key,) for key in ['efrac','deta','dphi','d0','z0']]
    processes = [multiprocessing.Process(target=plot_tracks, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    # scalars distributions
    sample_trans = sample.copy()
    sample_trans = scale_sample(sample_trans, sample_trans, scalars)[0]
    for key in train_var['scalars']: plot_scalars(sample, sample_trans, key); sys.exit()


def valid_data(data_file, all_var, scalars, idx, n_tracks, n_classes,
               scaling, pickle_file, weight_file, cuts):
    start_time   = time.time()
    valid_sample = make_sample(data_file, all_var, idx, n_tracks)
    valid_labels = make_labels(valid_sample, n_classes)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    valid_sample, valid_labels = sample_cuts(valid_sample, valid_labels, cuts)
    # validation sample diagnosis and plots
    #sample_checks(valid_sample, valid_labels); sys.exit()
    if weight_file != None and scaling:
        valid_sample = load_scaler(valid_sample, scalars, pickle_file)
    return valid_sample, valid_labels


def train_data(data_file, valid_sample, all_var, scalars, idx, n_tracks, n_classes,
               resampling, scaling, scaler_file, pickle_file, weight_file, cuts):
    start_time   = time.time()
    train_sample = make_sample(data_file, all_var, idx, n_tracks)
    train_labels = make_labels(train_sample, n_classes)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    train_sample, train_labels = sample_cuts(train_sample, train_labels, cuts)
    analyze_sample(train_sample)
    if resampling == 'ON':
        train_sample, train_labels = balance_sample(train_sample, train_labels, n_classes)
    if weight_file == None and scaling:
        train_sample, valid_sample = apply_scaler(train_sample, valid_sample, scalars, scaler_file)
        #train_sample, valid_sample = tracks_scaler(train_sample, valid_sample, n_tracks)
        #print(train_sample['tracks_image']) #print(valid_sample['tracks_image']);
    if weight_file != None and scaling:
        train_sample = load_scaler(train_sample, scalars, pickle_file)
    return train_sample, valid_sample, train_labels


def make_sample(data_file, all_var, idx, n_tracks, p='p_', upscale=False, denormalize=False):
    var_list = np.sum(list(all_var.values())); images = all_var['images']; tracks = all_var['tracks']
    with h5py.File(data_file, 'r') as data:
        sample = {key:data[key][idx[0]:idx[1]] for key in var_list if key != 'tracks_image'}
        if 'tracks_image' in var_list or 'tracks' in var_list:
            n_tracks    = min(n_tracks, data[p+'tracks'].shape[1])
            tracks_data = data[p+'tracks'][idx[0]:idx[1]][:,:n_tracks,:]
            tracks_data = np.concatenate((abs(tracks_data[...,0:5]), tracks_data[...,5:13]), axis=2)
    if 'tracks_image' in var_list: sample.update({'tracks_image':tracks_data})
    if 'tracks'       in var_list: sample['tracks'] = tracks_data
    #tracks = {'efrac' :0 , 'deta'  :1 , 'dphi'     :2 , 'd0'       :3,  'z0'       :4,  'p_efrac'  :5 ,
    #          'p_deta':6 , 'p_dphi':7 , 'p_d0'     :8 , 'p_z0'     :9,  'p_charge' :10, 'p_vertex' :11,
    #          'p_chi2':12, 'p_ndof':13, 'p_pixhits':14, 'p_scthits':15, 'p_trthits':16, 'p_sigmad0':17}
    if tf.__version__ < '2.1.0': sample = {key:np.float32(sample[key]) for key in sample}
    if denormalize:
        for n in images: sample[n]        = sample[n]        * sample['p_e'][:, np.newaxis, np.newaxis]
        for n in tracks: sample[n][:,:,0] = sample[n][:,:,0] * sample['p_e'][:, np.newaxis]
    if upscale:
        for n in images: sample[n]        = resize_images(np.float32(sample[n]), target_shape=(56,11))
    return sample


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


def sample_cuts(sample, labels, cuts):
    if sum(labels==-1) != 0:
        length = len(labels)
        sample = {key:sample[key][labels!=-1] for key in sample}; labels = labels[labels!=-1]
        print('CLASSIFIER: applying IFF labels cuts -->', format(len(labels),'7d'), 'e conserved', end='')
        print(' (' + format(100*len(labels)/length, '.2f') + ' %)')
    if cuts != None:
        length = len(labels)
        labels = labels[eval(cuts)]; sample = {key:sample[key][eval(cuts)] for key in sample}
        print('CLASSIFIER: applying properties cuts -->', format(len(labels),'7d') ,'e conserved', end='')
        print(' (' + format(100*len(labels)/length, '.2f') + ' %)')
        print('CLASSIFIER: cuts:', cuts)
    print(); return sample, labels


def analyze_sample(sample):
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


def scaled_tracks(tracks, n_tracks, max_tracks):
    tracks =  np.split(tracks, np.cumsum(n_tracks)[0:-1], axis=0)
    tracks = [np.vstack([n, np.zeros((max(0, max_tracks-len(n)), n.shape[1]))]) for n in tracks]
    return    np.concatenate([np.expand_dims(n, axis=0) for n in tracks])


def tracks_scaler(train_sample, valid_sample, max_tracks):
    print('CLASSIFIER: applying scaler transform to tracks variables', end=' ... ', flush=True)
    start_time    = time.time()
    valid_tracks   = valid_sample['tracks_image']
    train_tracks   = train_sample['tracks_image']
    n_valid_tracks = np.sum(abs(valid_tracks), axis=2)
    n_train_tracks = np.sum(abs(train_tracks), axis=2)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)', end=' ... ', flush=True)
    n_valid_tracks = np.array([sum(n_valid_tracks[n,:]!=0) for n in np.arange(len(valid_tracks))])
    n_train_tracks = np.array([sum(n_train_tracks[n,:]!=0) for n in np.arange(len(train_tracks))])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)', end=' ... ', flush=True)
    valid_tracks   = [valid_tracks[n,:n_valid_tracks[n],:] for n in np.arange(len(valid_tracks))]
    train_tracks   = [train_tracks[n,:n_train_tracks[n],:] for n in np.arange(len(train_tracks))]
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)', end=' ... ', flush=True)
    scaler         = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=0)
    train_tracks   = scaler.fit_transform(np.concatenate(train_tracks, axis=0))
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)', end=' ... ', flush=True)
    valid_tracks   = scaler.transform(np.concatenate(valid_tracks, axis=0))
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)', end=' ... ', flush=True)
    train_sample['tracks_image'] = scaled_tracks(train_tracks, n_train_tracks, max_tracks)
    valid_sample['tracks_image'] = scaled_tracks(valid_tracks, n_valid_tracks, max_tracks)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return train_sample, valid_sample


def apply_scaler(train_sample, valid_sample, scalars, scaler_file):
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


def compo_matrix(train_labels, valid_labels, valid_prob=[]):
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


def class_weights(labels):
    n_classes = max(labels) + 1
    return {m:len(labels)/sum(labels==m)/n_classes for m in np.arange(n_classes)}


def scan_sample(sample):
    def scan(sample, batch_size, index, return_dict):
        idx1, idx2 = index*batch_size, (index+1)*batch_size
        return_dict[index] = sum([np.sum(np.isfinite(sample[key][idx1:idx2])==False) for key in sample])
    n_e = len(list(sample.values())[0]); start_time = time.time()
    print('SCANNING', n_e, 'ELECTRONS FOR ERRORS ...', end=' ', flush=True)
    for n in np.arange(min(12, multiprocessing.cpu_count()), 1, -1):
        if n_e  % n == 0: n_tasks = n; batch_size = n_e//n_tasks; break
    manager   =  multiprocessing.Manager(); return_dict = manager.dict()
    processes = [multiprocessing.Process(target=scan, args=(sample, batch_size, job, return_dict))
                for job in np.arange(n_tasks)]
    for job in processes: job.start()
    for job in processes: job.join()
    #for key in sample: print(key, np.where(np.isfinite(sample[key])==False))
    print(sum(return_dict.values()), 'ERRORS FOUND', end=' ', flush=True)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')




#################################################################################
#####    presampler.py functions    #############################################
#################################################################################


def presample(h5_file, output_path, batch_size, sum_e, images, tracks, scalars, integers, index):
    idx = index*batch_size, (index+1)*batch_size
    with h5py.File(h5_file, 'r') as data:
        sample = {key:data['train'][key][idx[0]:idx[1]] for key in images + tracks + scalars + integers}
    for key in images: sample[key] = sample[key]/(sample['p_e'][:, np.newaxis, np.newaxis])
    for key in set(images)-set(['em_barrel_Lr1']): sample[key] = resize_images(sample[key])
    for key in images+scalars: sample[key] = np.float16(sample[key])
    tracks_list = [np.expand_dims(get_tracks(sample,n,50     ), axis=0) for n in np.arange(batch_size)]
    sample.update({'tracks'  :np.concatenate(tracks_list)})
    tracks_list = [np.expand_dims(get_tracks(sample,n,20,'p_'), axis=0) for n in np.arange(batch_size)]
    sample.update({'p_tracks':np.concatenate(tracks_list)})
    sample.update({'true_m':np.float16(get_truth_m(sample))})
    for key in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']: sample[key] = get_LLH(sample, key)
    for key in tracks + ['p_truth_E', 'p_LHValue']: sample.pop(key)
    with h5py.File(output_path+'temp_'+'{:=02}'.format(index)+'.h5', 'w' if sum_e==0 else 'a') as data:
        for key in sample:
            shape = (sum_e+batch_size,) + sample[key].shape[1:]
            if sum_e == 0:
                maxshape = (None,) + sample[key].shape[1:]; dtype = 'i4' if key in integers else 'f2'
                data.create_dataset(key, shape, dtype=dtype, maxshape=maxshape, chunks=shape)
            else: data[key].resize(shape)
        for key in sample: data[key][sum_e:sum_e+batch_size,...] = shuffle(sample[key], random_state=0)


def resize_images(images_array, target_shape=(7,11)):
    if images_array.shape[1:] == target_shape: return images_array
    else: return transform.resize(images_array, ( (len(images_array),) + target_shape))


def get_tracks(sample, idx, max_tracks=20, p=''):
    tracks_p    = np.cosh(sample[p+'tracks_eta'][idx]) * sample[p+'tracks_pt' ][idx]
    tracks_deta =         sample[p+'tracks_eta'][idx]  - sample[  'p_eta'     ][idx]
    tracks_dphi =         sample[p+'tracks_phi'][idx]  - sample[  'p_phi'     ][idx]
    tracks_d0   =         sample[p+'tracks_d0' ][idx]
    tracks_z0   =         sample[p+'tracks_z0' ][idx]
    tracks_dphi = np.where(tracks_dphi  < -np.pi, tracks_dphi + 2*np.pi, tracks_dphi )
    tracks_dphi = np.where(tracks_dphi  >  np.pi, tracks_dphi - 2*np.pi, tracks_dphi )
    tracks      = [tracks_p/sample['p_e'][idx], tracks_deta, tracks_dphi, tracks_d0, tracks_z0]
    p_tracks    = ['p_tracks_charge' , 'p_tracks_vertex' , 'p_tracks_chi2'   , 'p_tracks_ndof',
                   'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits', 'p_tracks_sigmad0']
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




#################################################################################
#####  under development functions  #############################################
#################################################################################


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
