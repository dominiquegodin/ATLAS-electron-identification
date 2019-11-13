import tensorflow as tf, numpy as np, h5py
from sklearn.model_selection import train_test_split


def make_indices(h5_files, test_size=0.1, random_state=0):
    train_indices,test_indices = [],[]
    for h5_file in h5_files:
        len_file = len(h5py.File(h5_file,'r')['data'])
        indices  = train_test_split(np.arange(0,len_file), test_size=test_size, random_state=random_state)
        train_indices.append(indices[0]) ; test_indices.append(indices[1])
    return np.array(train_indices), np.array(test_indices)


def load_files(files, indices, batch_size, index):
    batch_size = int(batch_size/len(indices))
    return np.concatenate([ load_tables(f, indices[files.index(f)], batch_size, index) for f in files ])


def load_tables(h5_file, indices, batch_size, index):
    data  = h5py.File(h5_file,'r')['data']
    if index==0:
        batch = np.arange(0,batch_size)
    else:
        batch = np.arange(index*batch_size,(index+1)*batch_size)
    return np.concatenate([ data['table_'+str(indices[i])] for i in batch ])


def pad_images(data, images_list, padding=False):
    images = [data[image] for image in images_list]
    if not padding: return images
    if 's4' in images_list:
        image = np.concatenate([np.zeros((len(data),11,25)), data['s4'], np.zeros((len(data),11,24))], axis=2)
        images[images_list.index('s4')] = image
    if 's5' in images_list:
        image = np.concatenate([np.zeros((len(data),11,25)), data['s5'], np.zeros((len(data),11,24))], axis=2)
        images[images_list.index('s5')] = image
    if 's6' in images_list:
        image = np.concatenate([np.zeros((len(data),11,25)), data['s6'], np.zeros((len(data),11,24))], axis=2)
        images[images_list.index('s6')] = image
    if 'tracks' in images_list:
        image = np.transpose(data['tracks'], (0,2,1) )
        image = np.concatenate([np.zeros((len(data),4,21)), image, np.zeros((len(data),4,20))], axis=2)
        image = np.concatenate([np.zeros((len(data),4,56)), image, np.zeros((len(data),3,56))], axis=1)
        images[images_list.index('tracks')] = image
    return images


def inverse_images(images, tracks, scalars, labels):
    images  = [np.vstack([image, image[:,::-1,:], image[:,:,::-1]]) for image in images]
    tracks  = [np.concatenate(3*[track])  for track  in tracks ]
    scalars = [np.concatenate(3*[scalar]) for scalar in scalars]
    labels  =  np.concatenate(3*[labels])
    return images + tracks + scalars, labels


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_names, indices, batch_size, images, tracks, scalars):
        self.file_names = file_names ; self.images  = images
        self.indices    = indices    ; self.tracks  = tracks
        self.batch_size = batch_size ; self.scalars = scalars
    def __len__(self):
        "number of batches per epoch"
        return int(self.indices.size/self.batch_size)
    def __getitem__(self, index):
        data    = load_files(self.file_names, self.indices, self.batch_size, index)
        images  = pad_images(data, self.images, padding=True)
        tracks  = [data[track]  for track  in self.tracks ]
        scalars = [data[scalar] for scalar in self.scalars]
        return images + tracks + scalars, data['truthmode']


class Get_Batch(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator
    def on_train_batch_begin(self, batch, logs=None):
        #print('\nTraining: batch {}: {}'.format(batch,self.generator[batch][0][0].shape))
        pass
