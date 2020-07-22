from  argparse import ArgumentParser
from   utils   import print_importances, plot_importances

parser = ArgumentParser()
parser.add_argument( '--importances_in'   , default = 'outputs/2c_10m/bkg_ratio_2d/importances.pkl')
parser.add_argument( '--importances_out'  , default = 'outputs/2c_10m/bkg_ratio_2d/perm_imp.png'   )
parser.add_argument( '--n_reps'           , default = 10, type = int                               )
args = parser.parse_args()

file = args.importances_in
path = args.importances_out
n_reps = args.n_reps

results = print_importances(file)
plot_importances(results, path, n_reps)
