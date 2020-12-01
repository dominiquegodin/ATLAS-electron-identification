# To convert h5 model to onnx model   : python h5_onnx.py --h5_to_onnx=True --output_dir=outputs
# To perform inference with onnx model: python h5_onnx.py --n_valid=1e4 --output_dir=outputs --eta_region=0.0-2.5


# IMPORT PACKAGES AND FUNCTIONS
import numpy as np
import os, sys, time, h5py, pickle
import onnx, keras2onnx, onnxruntime as ort
from   tensorflow.keras import models
from   argparse         import ArgumentParser
from   utils            import get_dataset, merge_samples


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--h5_to_onnx'  , default =  False,              )
parser.add_argument( '--n_valid'     , default =    1e6, type = float )
parser.add_argument( '--eta_region'  , default = ''                   )
parser.add_argument( '--output_dir'  , default = 'outputs'            )
parser.add_argument( '--scaler_file' , default = 'scaler.pkl'         )
parser.add_argument( '--model_file'  , default = 'model.h5'           )
args = parser.parse_args(); args.n_valid = int(args.n_valid)


# H5PY TO ONNX CONVERSION
if args.h5_to_onnx:
    h5py_model = models.load_model(args.output_dir+'/'+args.model_file)
    onnx_model = keras2onnx.convert_keras(h5py_model, h5py_model.name)
    onnx.save_model(onnx_model, args.output_dir+'/'+h5py_model.name+'.onnx')
    sys.exit()


# ONNX INFERENCE
sess_ort = ort.InferenceSession(args.output_dir+'/'+'model.onnx')
#print(ort.get_device()) #print(sess_ort.get_providers())
#sess_ort.set_providers(['CPUExecutionProvider'])
inputs    = {key.name:key.shape for key in sess_ort.get_inputs()}
n_classes = sess_ort.get_outputs()[0].shape[1]
n_tracks  = inputs['tracks_image'][1]
#for key in sess_ort.get_inputs() : print(key.name, key.shape)
#for key in sess_ort.get_outputs(): print(key.name, key.shape)


# TRAINING VARIABLES
scalars = ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'           ,
           'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2'         ,
           'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge'         ,
           'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_EoverP', 'p_wtots1' , 'p_numberOfInnermostPixelHits']
images  = [ 'em_barrel_Lr0',   'em_barrel_Lr1',   'em_barrel_Lr2',   'em_barrel_Lr3', 'em_barrel_Lr1_fine',
                                'tile_gap_Lr1',
            'em_endcap_Lr0',   'em_endcap_Lr1',   'em_endcap_Lr2',   'em_endcap_Lr3', 'em_endcap_Lr1_fine',
           'lar_endcap_Lr0',  'lar_endcap_Lr1',  'lar_endcap_Lr2',  'lar_endcap_Lr3',
                             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'      ]
others  = ['mcChannelNumber', 'eventNumber', 'p_TruthType', 'p_iffTruth'   , 'p_TruthOrigin', 'p_LHValue' ,
           'p_LHTight'      , 'p_LHMedium' , 'p_LHLoose'  , 'p_ECIDSResult', 'p_eta'        , 'p_et_calo' ,
           'p_vertexIndex'  , 'p_charge'   , 'p_firstEgMotherTruthType'    , 'p_firstEgMotherTruthOrigin' ,
           'correctedAverageMu', 'p_firstEgMotherPdgId'                                                   ]
scalars    = [key for key in scalars if key in inputs.keys()]
images     = [key for key in images  if key in inputs.keys()]
input_data = {'scalars':scalars, 'images':images, 'others':others}


# GENERATING VALIDATION SAMPLE
data_files = get_dataset(eta_region=args.eta_region)
if os.path.isfile(args.output_dir+'/'+args.scaler_file):
    print('\nLoading scalars scaler from', args.output_dir+'/'+args.scaler_file)
    scaler = pickle.load(open(args.output_dir+'/'+args.scaler_file, 'rb'))
else: scaler = None
sample, labels, _ = merge_samples(data_files, (0,args.n_valid), input_data,
                                  n_tracks, n_classes, cuts='', scaler=scaler)


# COMPARING INFERENCE BETWEEN H%PY AND ONNX
sample     = {key:np.float32(sample[key]) for key in images+scalars}
onnx_model = onnx.load(args.output_dir+'/'+'model.onnx')
output     = [key.name for key in onnx_model.graph.output]
onnx_probs = sess_ort.run(output, sample)
h5py_model = models.load_model(args.output_dir+'/'+args.model_file)
h5py_probs = h5py_model.predict(sample, batch_size=1000)


# PREDICTIONS COMPARISON
for n in np.arange(20):
    print('labels:', labels[n], '--> h5:', np.argmax(h5py_probs[n]),
          'onnx:', np.argmax(onnx_probs[0][n]), 'probs:', np.float16(h5py_probs[n]))
# PROBABILITIES COMPARISON
#for n in np.arange(50): print(labels[n], '-->', np.argmax(probs[0][n]), np.float16(probs[0][n]))
