import tensorflow as tf, numpy as np, h5py
from sklearn.model_selection import train_test_split
from skimage import data, transform


def make_indices(h5_files, test_size=0.1, random_state=0):
    train_indices, test_indices = [],[]
    for h5_file in h5_files:
        len_file = len(h5py.File(h5_file,'r')['data'])
        indices  = train_test_split(np.arange(0,len_file), test_size=test_size, random_state=random_state)
        train_indices.append(indices[0]) ; test_indices.append(indices[1])
    return np.array(train_indices), np.array(test_indices)


def load_files(files, indices, batch_size, index):
    batch_size = int(batch_size/len(indices))
    return np.concatenate([ load_tables(f, indices[files.index(f)], batch_size, index) for f in files ])


def load_tables(h5_file, indices, batch_size, index):
    data  = h5py.File(h5_file,'r')
    batch = np.arange(index*batch_size,(index+1)*batch_size)
    return np.hstack([ data['data/table_'+str(indices[i])][:][0] for i in batch ])


def call_generator(files, indices, batch_size, transforms, features, index):
    return Batch_Generator(files, indices, batch_size, transforms, **features)[index]


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, file_names, indices, batch_size, transforms, images, tracks, scalars):
        self.file_names = file_names ; self.images  = images
        self.indices    = indices    ; self.tracks  = tracks
        self.batch_size = batch_size ; self.scalars = scalars
        self.transforms = transforms
    def __len__(self):
        "number of batches per epoch"
        return int(self.indices.size/self.batch_size)
    def __getitem__(self, index):
        data    = load_files(self.file_names, self.indices, self.batch_size, index)
        images  = resize_images(data, self.images, **self.transforms)
        tracks  = [data[track]  for track  in self.tracks ]
        scalars = [data[scalar] for scalar in self.scalars]
        return images + tracks + scalars, data['truthmode']


def resize_images(data, images_list, target_shape=(11,56), normalize=False):
    images_sample = [ transform.resize(data[array], ((len(data[array]),)+target_shape))
                      if data[array].shape[1:]!=target_shape else data[array] for array in images_list ]
    return normalize_sample(images_sample) if normalize else images_sample


def normalize_sample(images_sample):
    return [ np.vstack([np.expand_dims(normalize_image(image),axis=0)
             for image in array]) for array in images_sample ]


def normalize_image(image):
    old_min, old_max = np.min(image), np.max(image)
    return 0*image if old_min == old_max else (image-old_min)/(old_max-old_min)


def inverse_images(images, tracks, scalars, labels):
    images  = [np.vstack([image, image[:,::-1,:], image[:,:,::-1]]) for image in images]
    tracks  = [np.concatenate(3*[track])  for track  in tracks ]
    scalars = [np.concatenate(3*[scalar]) for scalar in scalars]
    labels  =  np.concatenate(3*[labels])
    return images + tracks + scalars, labels
