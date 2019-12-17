import tensorflow as tf, numpy as np, h5py
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


def make_sample(data_file, batch_size, all_features, images=[], denormalize=False, index=0):
    data = h5py.File(data_file, 'r')
    idx_1, idx_2 = index*batch_size, (index+1)*batch_size
    sample_dict  = dict([key, data[key][idx_1:idx_2]] for key in all_features)
    if images != [] and denormalize:
        energy = sample_dict['p_e']
        for key in images: sample_dict[key] = sample_dict[key] * energy[:, np.newaxis, np.newaxis]
        sample_dict['tracks'][:,:,0] = sample_dict['tracks'][:,:,0] * energy[:, np.newaxis]
    return sample_dict


def generator_sample(data_file, features, indices, batch_size=None, index=0):
    data         = h5py.File(data_file, 'r')
    if batch_size != None:
        batch    = np.arange(index*batch_size,(index+1)*batch_size)
        indices  = np.take(indices, batch)
    sample_dict  = dict([key, np.take(data[key], indices, axis=0)] for key in features)
    return sample_dict


def make_labels(data, n_classes):
    if n_classes == 2:
        labels = np.where(np.logical_or(data['p_TruthType']==2, data['p_TruthType']==4), 0, 1)
    if n_classes == 5:
        iff_truth = data['p_iffTruth']
        labels    = np.where(iff_truth==2, 0, 4     )
        labels    = np.where(iff_truth==3, 1, labels)
        labels    = np.where(np.logical_or (iff_truth==1, iff_truth==10), 2, labels)
        labels    = np.where(np.logical_and(iff_truth>=7, iff_truth<= 9), 3, labels)
    return labels


def class_matrix(train_labels, test_labels, y_prob=[]):
    if y_prob == []: y_pred = test_labels
    else:            y_pred = np.argmax(y_prob, axis=1)
    matrix      = confusion_matrix(test_labels, y_pred)
    matrix      = 100*matrix.T/matrix.sum(axis=1)
    n_classes   = len(matrix)
    test_sizes  = [100*np.sum( test_labels==n)/len( test_labels) for n in np.arange(n_classes)]
    train_sizes = [100*np.sum(train_labels==n)/len(train_labels) for n in np.arange(n_classes)]
    classes = ['CLASS '+str(n) for n in np.arange(n_classes)]
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
                  '    ', '| TEST PREDICTIONS (%)              '+12*(n_classes-3)*' '+ '|')
        else:
            headers = ['CLASS #', 'TRAIN (%)', 'TEST (%)', 'ACC. (%)']
            table   = zip(classes, train_sizes, test_sizes, matrix.diagonal())
            print('\n+---------------------------------------------------+')
            print(  '| CLASS DISTRIBUTIONS AND ACCURACY                  |')
    print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f"), '\n')


def check_values(sample):
    bad_values = [np.sum(np.isfinite(sample[key])==False) for key in sample.keys()]
    print('CLASSIFIER:', sum(bad_values), 'found in sample')


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
