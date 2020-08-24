from tensorflow.keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU
from tensorflow.keras.layers import Flatten, Dense, concatenate, Reshape, Dropout
from tensorflow.keras        import regularizers, models
import numpy as np


def multi_CNN(n_classes, NN_type, sample, l2, dropout, CNN, FCN, images, tracks, scalars):
    regularizer = regularizers.l2(l2)
    input_dict  = {key:Input(shape=sample[key].shape[1:], name=key) for key in images+tracks+scalars}
    shape_set   = set([sample[key].shape[1:] for key in images])
    output_list = []
    for shape in shape_set:
        inputs  = [Reshape(shape+(1,))(input_dict[key]) for key in images if sample[key].shape[1:]==shape]
        outputs = concatenate(inputs, axis=3) if len(inputs) > 1 else inputs[0]
        n_maps  = [CNN[shape]['maps'   ][layer] for layer in np.arange(len(CNN[shape]['maps']))]
        kernels = [CNN[shape]['kernels'][layer] for layer in np.arange(len(CNN[shape]['maps']))]
        pools   = [CNN[shape]['pools'  ][layer] for layer in np.arange(len(CNN[shape]['maps']))]
        if np.all(np.array([len(kernel) for kernel in kernels]) >= 3):
            kernels_dim = 3
            outputs     = Reshape(outputs.shape[1:]+(1,))                                          (outputs)
        else: kernels_dim = 2
        kernels = [(kernel+(3-len(kernel))*(1,))[:kernels_dim] for kernel in kernels]
        pools   = [( pool +(3-len( pool ))*(1,))[:kernels_dim] for  pool  in  pools ]
        if NN_type == 'CNN':
            for layer in np.arange(len(CNN[shape]['maps'])):
                if len(kernels[layer]) == 2:
                    outputs = Conv2D(n_maps[layer], kernels[layer], kernel_regularizer=regularizer)(outputs)
                    outputs = MaxPooling2D(pools[layer], padding='same')                           (outputs)
                if len(kernels[layer]) == 3:
                    outputs = Conv3D(n_maps[layer], kernels[layer], kernel_regularizer=regularizer)(outputs)
                    outputs = MaxPooling3D(pools[layer], padding='same')                           (outputs)
                outputs = LeakyReLU(alpha=0)                                                       (outputs)
        output_list += [Flatten()(outputs)]
    for key in tracks+scalars: output_list += [Flatten()(input_dict[key])]
    outputs = concatenate(output_list) if len(output_list)>1 else output_list[0]
    for n_neurons in FCN:
        outputs = Dense(n_neurons, kernel_regularizer=regularizer)                                 (outputs)
        outputs = LeakyReLU(alpha=0)                                                               (outputs)
        outputs = Dropout(dropout)                                                                 (outputs)
    outputs = Dense(n_classes, activation='softmax', dtype='float32')                              (outputs)
    return models.Model(inputs = list(input_dict.values()), outputs = outputs)
