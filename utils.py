import tensorflow as tf, numpy as np, h5py
from sklearn.model_selection import train_test_split
from skimage  import transform
from tabulate import tabulate


def make_sample(data_file, images, complete, batch_size, denormalize=False, index=0):
    data = h5py.File(data_file, 'r')
    idx_1, idx_2 = index*batch_size, (index+1)*batch_size
    sample_dict  = dict([key, data[key][idx_1:idx_2]] for key in complete)
    if denormalize:
        energy = sample_dict['p_e']
        for key in images: sample_dict[key] = sample_dict[key] * energy[:, np.newaxis, np.newaxis]
        sample_dict['tracks'][:,:,0] = sample_dict['tracks'][:,:,0] * energy[:, np.newaxis]
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


#def make_samples(data, images, complete, features, batch_size, indices, n_classes):
#    data   = dict([key, np.take(data[key], indices, axis=0)] for key in complete)
#    labels = make_labels(data, n_classes)


def label_sizes(train_labels, test_labels, n_classes):
    test_sizes  = [format(100*np.sum( test_labels==n)/len( test_labels),'4.1f')
                   for n in np.arange(n_classes)]
    train_sizes = [format(100*np.sum(train_labels==n)/len(train_labels),'4.1f')
                   for n in np.arange(n_classes)]
    table = zip(['class '+str(n) for n in np.arange(n_classes)], train_sizes, test_sizes)
    print('\nCLASSIFIER: labels distributions:')
    print(tabulate(table, headers=['CLASS', 'TRAIN (%)', 'TEST (%)'], tablefmt='psql'))


def check_values(sample):
    bad_values = [np.sum(np.isfinite(sample[key])==False) for key in sample.keys()]
    print('CLASSIFIER:', sum(bad_values), 'found in sample')


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_name, n_classes, indices, batch_size, images, complete):
        self.file_name  = file_name  ; self.images    = images
        self.indices    = indices    ; self.complete  = complete
        self.batch_size = batch_size ; self.n_classes = n_classes
    def __len__(self):
        "number of batches per epoch"
        return int(self.indices.size/self.batch_size)
    def __getitem__(self, index):
        data   = make_sample(self.file_name, self.images, self.complete, self.batch_size, index)
        data   = dict([key, np.take(data[key], self.indices, axis=0)] for key in self.complete)
        labels = make_labels(data, self.n_classes)
        data   = [np.float32(data[key]) for key in np.sum(list(features.values()))]

        #data    = load_files(self.file_names, self.indices, self.batch_size, index)
        #images  = resize_images(data, self.images, **self.transforms)
        #tracks  = [data[track]  for track  in self.tracks ]
        #scalars = [data[scalar] for scalar in self.scalars]
        return images + tracks + scalars, data['truthmode']
