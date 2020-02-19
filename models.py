from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ReLU, LeakyReLU
from tensorflow.keras.layers import concatenate, Reshape, Dropout, LSTM, Masking, BatchNormalization
from tensorflow.keras        import regularizers, models


def multi_CNN(n_classes, NN_type, sample, images, tracks, scalars):
    FCL_neurons = [200, 200]; CNN_neurons = [200, 200]
    dropout = 0.2; regularizer = regularizers.l2(1e-6); alpha = 0.
    input_dict  = {n:Input(shape = sample[n].shape[1:], name = n) for n in images+tracks+scalars}
    shape_set   = set([sample[n].shape[1:] for n in images])
    shape_dict  = {shape:[n for n in images if sample[n].shape[1:] == shape] for shape in shape_set}
    output_list = []
    for shape in shape_dict:
        image_inputs  = [Reshape(sample[n].shape[1:]+(1,))(input_dict[n]) for n in shape_dict[shape]]
        image_outputs = concatenate(image_inputs, axis=3) if len(image_inputs)>1 else image_inputs[0]
        if NN_type == 'CNN':
            field = (3, 3) if min(shape) > 5 else (2, 2)
            for n_neurons in CNN_neurons:
                image_outputs = Conv2D(n_neurons, field, kernel_regularizer=regularizer)(image_outputs)
                image_outputs = LeakyReLU(alpha=alpha)                                  (image_outputs)
                if min(shape) > 10: image_outputs = MaxPooling2D(2,2)                   (image_outputs)
                image_outputs = Dropout(dropout)                                        (image_outputs)
        output_list += [Flatten()(image_outputs)]
    for track  in tracks : output_list += [Flatten()(input_dict[track])]
    for scalar in scalars: output_list += [Flatten()(input_dict[scalar])]

    outputs = concatenate(output_list) if len(output_list)>1 else output_list[0]
    for n_neurons in FCL_neurons:
        outputs = Dense(n_neurons, kernel_regularizer=regularizer)   (outputs)
        outputs = LeakyReLU(alpha=alpha)                             (outputs)
        outputs = Dropout(dropout)                                   (outputs)
    outputs = Dense(n_classes, activation='softmax', dtype='float32')(outputs)
    return models.Model(inputs = list(input_dict.values()), outputs = outputs)



