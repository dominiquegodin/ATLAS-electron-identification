from tensorflow.keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU
from tensorflow.keras.layers import Flatten, Dense, concatenate, Reshape, Dropout
from tensorflow.keras        import regularizers, models
import numpy as np


def multi_CNN(n_classes, NN_type, sample, l2, dropout, CNN, FCN, images, tracks, scalars):
    regularizer = regularizers.l2(l2); alpha=0
    input_dict  = {n:Input(shape = sample[n].shape[1:], name = n) for n in images + tracks + scalars}
    shape_set   = set([sample[n].shape[1:] for n in images])
    shape_dict  = {shape:[n for n in images if sample[n].shape[1:] == shape] for shape in shape_set}
    output_list = []
    for shape in shape_dict:
        inputs  = [Reshape(sample[n].shape[1:]+(1,1))(input_dict[n]) for n in shape_dict[shape]]
        outputs = concatenate(inputs, axis=3) if len(inputs) > 1 else inputs[0]
        if NN_type == 'CNN':
            if shape not in CNN: shape = 'tracks'
            for l in np.arange(len(CNN[shape]['maps'])):
                kernel  = CNN[shape]['kernels'][l]; kernel = (kernel+(3-len(kernel))*(1,))[:3]
                pool    = CNN[shape]['pools'  ][l]; pool   = (pool  +(3-len(pool)  )*(1,))[:3]
                outputs = Conv3D(CNN[shape]['maps'][l],kernel,kernel_regularizer=regularizer)(outputs)
                outputs = LeakyReLU(alpha=alpha)                                             (outputs)
                outputs = MaxPooling3D(pool,padding='same')                                  (outputs)
                outputs = Dropout(dropout)                                                   (outputs)
        output_list += [Flatten()(outputs)]
    for track  in tracks : output_list += [Flatten()(input_dict[track ])]
    for scalar in scalars: output_list += [Flatten()(input_dict[scalar])]
    outputs = concatenate(output_list) if len(output_list)>1 else output_list[0]
    for n_neurons in FCN:
        outputs = Dense(n_neurons, kernel_regularizer=regularizer)   (outputs)
        outputs = LeakyReLU(alpha=alpha)                             (outputs)
        outputs = Dropout(dropout)                                   (outputs)
    outputs = Dense(n_classes, activation='softmax', dtype='float32')(outputs)
    return models.Model(inputs = list(input_dict.values()), outputs = outputs)
