from  argparse import ArgumentParser
from   utils   import print_importances, plot_importances

parser = ArgumentParser()
parser.add_argument( '--importances_in'   , default = 'outputs/2c_10m/bkg_ratio_2d/importance')
parser.add_argument( '--importances_out'  , default = 'outputs/2c_10m/bkg_ratio_2d/prm_imp'   )
parser.add_argument( '--n_reps'           , default = 10, type = int                               )
args = parser.parse_args()

bkg_list = ['global', 'Charge flip', 'Photon conversion', 'b/c hadron decay',
            r'Light flavor (bkg $\gamma$+e)', 'Ligth flavor (hadron)']

file = args.importances_in
path = args.importances_out
n_reps = args.n_reps

for i in range(6):
    if i :
        suf = '_' + str(i)
    else :
        suf = '_bkg'
    title = 'Permutation importance against {} background.\n(averaged over {} repetitions)'.format(bkg_list[i], n_reps)
    results = print_importances(file + suf + '.pkl')
    plot_importances(results, path + suf + 'pdf', title)
