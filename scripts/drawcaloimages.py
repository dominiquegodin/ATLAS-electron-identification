import h5py, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages
from   argparse  import ArgumentParser

matplotlib.use('Agg')


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'         , default = 1        , type=int )
parser.add_argument( '--start_e'     , default = 0        , type=int )
parser.add_argument( '--input_file'  , default = ''                  )
parser.add_argument( '--output_file' , default = 'caloimages.pdf'    )
args = parser.parse_args()

input_path = 'data/2020-04-29/Data_abseta_1.6_2.5_et_0.0_500.0_processes_selection_pid.h5' if args.input_file == '' else args.input_file

f = h5py.File(input_path,'r')
train = f['train']

if 'em_endcap_Lr0' in list(train.keys()):
    images = ['em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3'    ,
            'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3'    ]
else:
    images = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3', #'em_barrel_Lr1_fine',
            'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']

averages = {}

for image in images:
    averages[image] = train[image][0]*0

with PdfPages(args.output_file) as pdf:
    for i_e in range(args.n_e):
        print('drawing electron %s' % i_e)
        fig, axs = plt.subplots(4,2)
        fig.set_size_inches(8.27,11.69)
        i=0
        for image in images:
            row = i if i<len(images)/2 else i-4
            col = 0 if i<len(images)/2 else 1
            #print('drawing row %s and column %s' % (row,col))
            axs[row,col].imshow(train[image][args.start_e + i_e],cmap='hot',interpolation='nearest')
            axs[row,col].title.set_text(image)

            #Update average
            averages[image] = averages[image] + (train[image][args.start_e + i_e] * (1./args.n_e))

            i += 1
        pdf.savefig(fig)
        plt.close()

    print('drawing averages over %s electrons' % args.n_e)
    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(8.27,11.69)

    i=0
    for image in images:
        row = i if i<len(images)/2 else i-4
        col = 0 if i<len(images)/2 else 1
        axs[row,col].imshow(averages[image],cmap='hot',interpolation='nearest')
        axs[row,col].title.set_text('AVG %s' % image)
        i += 1

    pdf.savefig(fig)
    plt.close()