from numpy import arange
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import concatenate, Reshape, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential, Model


def CNN_multichannel(images_shape, tracks_shape, n_classes, images, tracks, scalars):
    images_inputs  = [Input( shape = images_shape ) for i     in images ]
    tracks_inputs  = [Input( shape = tracks_shape ) for track in tracks ]
    scalars_inputs = [Input( shape = ( )          ) for n     in scalars]
    features_list  = []
    if len(images)  != 0:
        single_images   = [Reshape( images_shape+(1,) )( images_inputs[i] ) for i in arange(0,len(images))]
        merged_images   = concatenate( single_images, axis=3 ) if len(single_images)>1 else single_images[0]
        images_outputs  = Conv2D       ( 100, (4,3), activation='relu' )( merged_images  )
        images_outputs  = MaxPooling2D ( 2, 2                          )( images_outputs )
        images_outputs  = Conv2D       ( 50,  (3,3), activation='relu' )( images_outputs )
        images_outputs  = MaxPooling2D ( 2, 2                          )( images_outputs )
        images_outputs  = Flatten      (                               )( images_outputs )
        features_list  += [images_outputs]
    if len(tracks)  != 0:
        tracks_outputs  = Flatten( )( tracks_inputs[0] )
        features_list  += [tracks_outputs]
    if len(scalars) != 0:
        single_scalars  = [Reshape( (1,) )( scalars_inputs[n] ) for n in arange(0,len(scalars))]
        merged_scalars  = concatenate( single_scalars ) if len(single_scalars)>1 else single_scalars[0]
        scalars_outputs = Flatten(  )( merged_scalars )
        features_list  += [scalars_outputs]
    concatenated = concatenate( features_list ) if len(features_list)>1 else features_list[0]
    outputs      = Dense( 100,       activation='relu'    )( concatenated )
    outputs      = Dense( n_classes, activation='softmax' )( outputs      )
    return Model( inputs = images_inputs + tracks_inputs + scalars_inputs, outputs = outputs )
