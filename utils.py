import tensorflow as tf, numpy as np, h5py, os, sys
from sklearn.metrics import confusion_matrix
from tabulate        import tabulate
from skimage         import transform


def make_sample(data_file, batch_size, all_features, images, upscale=False, denormalize=False, index=0):
    data = h5py.File(data_file, 'r')
    idx_1, idx_2 = index*batch_size, (index+1)*batch_size
    sample_dict  = dict([key, data[key][idx_1:idx_2]] for key in all_features)
    if images != [] and denormalize:
        energy = sample_dict['p_e']
        for key in images: sample_dict[key] = sample_dict[key] * energy[:, np.newaxis, np.newaxis]
        sample_dict['tracks'][:,:,0] = sample_dict['tracks'][:,:,0] * energy[:, np.newaxis]
    if images != [] and upscale:
        for i in images: sample_dict[i] = resize_images(np.float32(sample_dict[i]), target_shape=(56,11))
    return sample_dict


def generator_sample(data_file, all_features, indices, batch_size=None, index=0):
    data     = h5py.File(data_file, 'r')
    if batch_size != None:
        batch    = np.arange(index*batch_size,(index+1)*batch_size)
        indices  = np.take(indices, batch)
    sample_dict  = dict([key, np.take(data[key], indices, axis=0)] for key in features)
    return sample_dict


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_name, n_classes, train_features, all_features, indices, batch_size):
        self.file_name  = file_name  ; self.train_features = train_features
        self.indices    = indices    ; self.all_features   = all_features
        self.batch_size = batch_size ; self.n_classes      = n_classes
    def __len__(self):
        "number of batches per epoch"
        return int(self.indices.size/self.batch_size)
    def __getitem__(self, index):
        data   = generator_sample(self.file_name, self.all_features, self.indices, self.batch_size, index)
        labels = make_labels(data, self.n_classes)
        data   = [np.float32(data[key]) for key in np.sum(list(self.train_features.values()))]
        return data, labels


def make_labels(data, n_classes):
    if   n_classes == 2:
        labels = np.where(np.logical_or(data['p_TruthType']==2, data['p_TruthType']==4), 0, 1)
    elif n_classes == 5:
        truth  = data['p_iffTruth']
        labels = np.where(truth==2, 0, 4     )
        labels = np.where(truth==3, 1, labels)
        labels = np.where(np.logical_or (truth==1, truth==10), 2, labels)
        labels = np.where(np.logical_and(truth>=7, truth<= 9), 3, labels)
    elif n_classes == 9:
        labels = data['p_iffTruth']
        labels = np.where(labels== 9, 4, labels)
        labels = np.where(labels==10, 6, labels)
    else:
        print('\nCLASSIFIER:', n_classes, 'classes not supported -> exiting program\n')
        sys.exit()
    return labels


def class_matrix(train_labels, test_labels, y_prob=[]):
    if y_prob == []: y_pred = test_labels
    else:            y_pred = np.argmax(y_prob, axis=1)
    matrix      = confusion_matrix(test_labels, y_pred)
    matrix      = 100*matrix.T/matrix.sum(axis=1)
    n_classes   = len(matrix)
    test_sizes  = [100*np.sum( test_labels==n)/len( test_labels) for n in np.arange(n_classes)]
    train_sizes = [100*np.sum(train_labels==n)/len(train_labels) for n in np.arange(n_classes)]
    classes     = ['CLASS '+str(n) for n in np.arange(n_classes)]
    if y_prob == []:
        print('\n+--------------------------------------+')
        print(  '| CLASS DISTRIBUTIONS                  |')
        headers = ['CLASS #', 'TRAIN (%)', 'TEST (%)']
        table   = zip(classes, train_sizes, test_sizes)
    else:
        if n_classes > 2:
            headers = ['CLASS #', 'TRAIN', 'TEST'] + classes
            table   = [classes] + [train_sizes] + [test_sizes] + matrix.T.tolist()
            table   = list(map(list, zip(*table)))
            print('\n+'+30*'-'+'+'+35*'-'+12*(n_classes-3)*'-'+'+\n', '\b| CLASS DISTRIBUTIONS (%)',
                  '    ', '| TEST SAMPLE PREDICTIONS (%)       '+12*(n_classes-3)*' '+ '|')
        else:
            headers = ['CLASS #', 'TRAIN (%)', 'TEST (%)', 'ACC. (%)']
            table   = zip(classes, train_sizes, test_sizes, matrix.diagonal())
            print('\n+---------------------------------------------------+')
            print(  '| CLASS DISTRIBUTIONS AND ACCURACIES                |')
    print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".3f"), '\n')


def resize_images(images_array, target_shape=(7,11)):
    if images_array.shape[1:] == target_shape: return images_array
    else: return transform.resize(images_array, ( (len(images_array),) + target_shape))


def check_sample(sample):
    bad_values = [np.sum(np.isfinite(sample[key])==False) for key in sample.keys()]
    print('CLASSIFIER:', sum(bad_values), 'found in sample')


def make_samples(h5_file, output_path, batch_size, sum_e, images, tracks, scalars, int_var, index):
    idx_1, idx_2 = index*batch_size, (index+1)*batch_size
    data         = h5py.File(h5_file, 'r')
    energy       =                data['train']['p_e'][idx_1:idx_2][:, np.newaxis, np.newaxis        ]
    sample_list  = [resize_images(data['train'][ key ][idx_1:idx_2])/energy for key in images        ]
    sample_list += [              data['train'][ key ][idx_1:idx_2]         for key in tracks+scalars]
    sample_dict  = dict(zip(images+tracks+scalars, sample_list))
    sample_dict.update({'em_barrel_Lr1_fine':data['train']['em_barrel_Lr1'][idx_1:idx_2]/energy})
    #sample_list  = [resize_images(data['train'][ key ][idx_1:idx_2])/energy for key in images.keys() ]
    #sample_list += [              data['train'][ key ][idx_1:idx_2] for key in tracks+list(scalars.keys())]
    #sample_dict  = dict(zip(list(images.values())+tracks+list(scalars.values()), sample_list))
    #sample_dict.update({'s1_fine':data['train']['em_barrel_Lr1'][idx_1:idx_2]/energy})
    tracks_list  = [np.expand_dims(get_tracks(sample_dict, e), axis=0) for e in np.arange(batch_size)]
    sample_dict.update({'tracks':np.concatenate(tracks_list), 'true_m':get_truth_m(sample_dict)})
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']: sample_dict[wp] = get_LLH(sample_dict, wp)
    for feature in tracks + ['p_truth_E', 'p_LHValue']: sample_dict.pop(feature)
    data = h5py.File(output_path+'temp_'+'{:=02}'.format(index)+'.h5', 'w' if sum_e==0 else 'a')
    for key in sample_dict.keys():
        shape = (sum_e+batch_size,) + sample_dict[key].shape[1:]
        if sum_e == 0:
            maxshape =  (None,) + sample_dict[key].shape[1:]
            dtype    = 'i4' if key in int_var else 'f4' if key=='tracks' else 'f2'
            data.create_dataset(key, shape, dtype=dtype, maxshape=maxshape, chunks=shape)
        else:
            data[key].resize(shape)
    for key in sample_dict.keys(): data[key][sum_e:sum_e+batch_size,...] = sample_dict[key]


def get_tracks(sample, idx, max_tracks=15):
    tracks_p    = np.cosh(sample['tracks_eta'][idx]) * sample['tracks_pt' ][idx]
    tracks_deta =     abs(sample['tracks_eta'][idx]  - sample['p_eta'     ][idx])
    tracks_dphi =         sample['p_phi'     ][idx]  - sample['tracks_phi'][idx]
    tracks_d0   =         sample['tracks_d0' ][idx]
    tracks      = np.vstack([tracks_p/sample['p_e'][idx], tracks_deta, tracks_dphi, tracks_d0]).T
    numbers     = list(set(np.arange(len(tracks))) - set(np.where(np.isfinite(tracks)==False)[0]))
    tracks      = np.take(tracks, numbers, axis=0)[:max_tracks,:]
    return        np.vstack([tracks, np.zeros((max(0, max_tracks-len(tracks)), 4))])


def get_LLH(sample, wp):
    llh =  np.where( sample[wp] == 0,                         2, 0   )
    return np.where( (llh == 2) & (sample['p_LHValue'] < -1), 1, llh )


def get_truth_m(sample, new=True, m_e=0.511, max_eta=4.9):
    truth_eta = np.vectorize(min)(abs(sample['p_truth_eta']), max_eta)
    truth_s   = sample["p_truth_E"]**2 - (sample['p_truth_pt']*np.cosh(truth_eta))**2
    if new: return np.where(truth_eta == max_eta, -1, np.sqrt(np.vectorize(max)(m_e**2, truth_s)))
    else:   return np.where(truth_eta == max_eta, -1, np.sign(truth_s)*np.sqrt(abs(truth_s)) )


def merge_samples(n_e, n_files, output_path, output_file):
    temp_files = sorted([h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file])
    os.rename(output_path+temp_files[0], output_path+output_file)
    dataset    = h5py.File(output_path+output_file, 'a')
    MB_size    = n_files*sum([np.float16(dataset[key]).nbytes for key in dataset.keys()])/1e6
    print('Merging temporary files (', '\b{:.0f}'.format(MB_size),'MB) into:', end=' ')
    print('output/'+output_file, end=' .', flush=True)
    for key in dataset.keys(): dataset[key].resize((n_e*n_files,) + dataset[key].shape[1:])
    for h5_file in temp_files[1:]:
        data  = h5py.File(output_path+h5_file, 'r')
        index = temp_files.index(h5_file)
        for key in dataset.keys(): dataset[key][index*n_e:(index+1)*n_e] = data[key]
        data.close() ; os.remove(output_path+h5_file)
        print('.', end='', flush=True)
