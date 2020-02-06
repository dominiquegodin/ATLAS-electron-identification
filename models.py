from numpy                   import arange
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ReLU, LeakyReLU
from tensorflow.keras.layers import concatenate, Reshape, Dropout, LSTM, Masking, BatchNormalization
from tensorflow.keras        import regularizers, models


def multi_CNNs(n_classes, NN_type, image_shapes, tracks_shape, images, tracks, scalars):
    FC_neurons = [200, 200] ; dropout = 0.2 ; regularizer = regularizers.l2(1e-6) ; alpha=0.
    images_inputs  = [Input(shape = image_shapes[n], name = n) for n in  images]
    tracks_inputs  = [Input(shape = tracks_shape[n], name = n) for n in  tracks]
    scalars_inputs = [Input(shape = ( )            , name = n) for n in scalars]
    images_coarse  = [image for image in images if image_shapes[image] == ( 7, 11)]
    images_fine    = [image for image in images if image_shapes[image] == (56, 11)]
    features_list  = []
    for subset in [subset for subset in [images_coarse, images_fine] if len(subset) != 0]:
        single_images  = [Reshape(image_shapes[n]+(1,))(images_inputs[images.index(n)]) for n in subset]
        images_outputs = concatenate(single_images, axis=3) if len(single_images)>1 else single_images[0]
        if NN_type == 'CNN':
            images_outputs = Conv2D(200, (3,3), kernel_regularizer=regularizer)(images_outputs)
            images_outputs = LeakyReLU(alpha=alpha)                            (images_outputs)
            if subset == images_fine: images_outputs = MaxPooling2D(2,2)       (images_outputs)
            images_outputs = Dropout(dropout)                                  (images_outputs)
            images_outputs = Conv2D(200, (3,3), kernel_regularizer=regularizer)(images_outputs)
            images_outputs = LeakyReLU(alpha=alpha)                            (images_outputs)
            if subset == images_fine: images_outputs = MaxPooling2D(2,2)       (images_outputs)
            images_outputs = Dropout(dropout)                                  (images_outputs)
        images_outputs  = Flatten()(images_outputs)
        features_list  += [images_outputs]
    if len(tracks)  != 0:
        tracks_outputs  = Flatten()(tracks_inputs[0])
        features_list  += [tracks_outputs]
    if len(scalars) != 0:
        single_scalars  = [Reshape((1,))(scalars_inputs[n]) for n in arange(len(scalars))]
        scalars_outputs = concatenate(single_scalars) if len(single_scalars)>1 else single_scalars[0]
        scalars_outputs = Flatten()(scalars_outputs)
        features_list  += [scalars_outputs]
    outputs = concatenate(features_list) if len(features_list)>1 else features_list[0]
    for n_neurons in FC_neurons:
        outputs = Dense(n_neurons, kernel_regularizer=regularizer)   (outputs)
        outputs = LeakyReLU(alpha=alpha)                             (outputs)
        outputs = Dropout(dropout)                                   (outputs)
    outputs = Dense(n_classes, activation='softmax', dtype='float32')(outputs)
    return models.Model(inputs = images_inputs + tracks_inputs + scalars_inputs, outputs = outputs)
