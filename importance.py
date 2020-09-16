import numpy             as np
import matplotlib.pyplot as plt
import os, sys, h5py, pickle, time
import pandas as pd
from   pandas.plotting import scatter_matrix
from   copy       import deepcopy
from   itertools  import accumulate
from   utils import apply_scaler, valid_results


#################################################################################
#####  FEATURE IMPORTANCE  ######################################################
#################################################################################


def feature_importance(output_dir, n_classes, weight_type, eta_region, n_train, feat, images, scalars,
                       removal, plotting, auto_output_dir=False):
    ###### update LaTeXizer in utils if changes are made to the intput variables. ######
    groups  =  [['em_barrel_Lr1', 'em_barrel_Lr1_fine'], ['em_barrel_Lr0','em_barrel_Lr2', 'em_barrel_Lr3'],            # To compute the feature importance of a group of variables,
                ['em_endcap_Lr0','em_endcap_Lr2','em_endcap_Lr3'], ['em_endcap_Lr1' , 'em_endcap_Lr1_fine'],            # simply add the list of variables into groups.
                ['lar_endcap_Lr0','lar_endcap_Lr1','lar_endcap_Lr2','lar_endcap_Lr3'],
                ['tile_gap_Lr1' ,'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3'],
                ['p_d0' , 'p_d0Sig'], ['p_d0' , 'p_d0Sig' , 'p_qd0Sig'], ['p_f1' , 'p_f3'],
                ['p_nTracks', 'p_sct_weight_charge'], ['p_nTracks', 'p_et_calo'],
                ['em_endcap_Lr2', 'tile_barrel_Lr1', 'p_f1', 'p_qd0Sig', 'p_TRTPID', 'em_endcap_Lr1_fine',
                 'p_sct wt charge', 'p_wstot1', 'p_weta2', 'p_d0', 'p_d0Sig', 'tile_barrel_Lr3', 'em_endcap_Lr0',
                 'em_endcap_Lr3', 'lar_endcap_Lr0', 'p_nTracks', 'tile_gap_Lr1', 'p_EptRatio', 'lar_endcap_Lr1',
                 'p_dPOverP', 'p_numberOfSCTHits', 'lar_endcap_Lr3', 'p_Rphi' , 'p_f3', 'p_ndof', 'p_Eratio']]

    if auto_output_dir == True:
        output_dir = output_dir + '/{}c_{}m/{}/{}'.format(n_classes, round(n_train/1e6),            # Saves the output according to the number of classes, the stats used,
                                                                    weight_type, eta_region)        # the reweighthing and the region
    # FEATURE REMOVAL
    if removal == 'ON':
        images, scalars, feat = feature_removal(feat, images, scalars, groups, images, scalars)
        output_dir = output_dir + '/removal_importance/' + feat                                     # The output directory will be different for each feature.
    create_path(output_dir)                                                                         # That way the model.h5 and their corresponding plots aren't mixed with the other trainings.
    # FEATURE IMPORTANCE PLOTTING
    if plotting in ['prm', 'permutation','rm', 'removal']:
        plot_importance(plotting, output_dir, eta_region, images, scalars, len(groups), n_classes)
        sys.exit()
    return scalars, images, feat, groups




def LaTeXizer(names=[]):
    '''
    Converts variables names to be compatible with LaTeX format.

    If no arguments are given, LaTeXizer returns a dictionary maping each name to its LaTeX conterpart
    and an empty list.
    the converted list of variables names.
    '''
    n_groups = 12
    # Images
    vars  = ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
             'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine',
             'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3', 'tile_gap_Lr1'      ,
             'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'                        ]
    # Scalars
    vars += ['p_Eratio', 'p_Reta'   , 'p_Rhad'     , 'p_Rphi'  , 'p_TRTPID' , 'p_numberOfSCTHits'  ,
             'p_ndof'  , 'p_dPOverP', 'p_deltaEta1', 'p_f1'    , 'p_f3'     , 'p_deltaPhiRescaled2',
             'p_weta2' , 'p_d0'     , 'p_d0Sig'    , 'p_qd0Sig', 'p_nTracks', 'p_sct_weight_charge',
             'p_eta'   , 'p_et_calo', 'p_EptRatio' , 'p_wtots1', 'p_numberOfInnermostPixelHits', 'p_EoverP' ]
    # Groups of variables
    vars += ['group_{}'.format(g) for g in range(n_groups)]

    # LaTeX images
    Lvars =  ['em_barrel_Lr0'  , 'em_barrel_Lr1'  , 'em_barrel_Lr2'  , 'em_barrel_Lr3' , 'em_barrel_Lr1_fine',
              'em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3' , 'em_endcap_Lr1_fine',
              'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3', 'tile_gap_Lr1'      ,
              'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3', 'tracks_image'                        ]
    # LaTeX scalars
    Lvars += [r'$E_{ratio}$', r'$R_{\eta}$', r'$R_{had}$', r'$R_{\phi}$' , r'TRTPID' ,   r'Nb of SCT hits',
              'ndof', r'$\Delta p/p$', r'$\Delta \eta_1$', r'$f_1$'    ,  r'$f_3$' , r'$\Delta \phi _{res}$',
              r'$w_{\eta 2}$',  r'$d_0$', r'$d_0/{\sigma(d_0)}$' , r'qd0Sig'   , r'$n_{Tracks}$',
              r'sct wt charge',r'$\eta$'      , r'$p_t$', r'$E/p_T$'    , r'$w_{stot}$', r'$n_{Blayer}$',r'$E/p$']
    # LaTeX groups of variables
    Lvars += ['em_barrel_Lr1 variables', 'em_barrel variables', 'em_endcap variables', 'em_endcap_Lr1 variables',
              'lar_endcap variables', 'tile variables', r'$d_0$ variables 1', r'$d_0$ variables 2', r'$f_1$ and $f_3$',
              r'$n_{Tracks}$ and sct wt charge',  r'$n_{Tracks}$ and $p_t$', 'detrimental variables']

    # Create a mapping dictionary from the list of variables names to their LaTeX equivalent
    converter = {var : Lvar for var, Lvar in zip(vars, Lvars)}
    # Create a list of the converted variables' names (note that the variables that are not in the LaTeXizer won't be converted)
    Lnames = [converter[name] if name in vars else name for name in names]
    return converter, Lnames

def create_path(dir):
    '''
    Create the path to the given directory if it doesn't extist.
    '''
    for path in list(accumulate([folder+'/' for folder in dir.split('/')])):
        try: os.mkdir(path)
        except OSError: continue
        except FileExistsError: pass

def print_importances(file):
    '''
    Reads the given pickle file, prints its content and returns it.
    '''
    # Reading the file
    with open(file,'rb') as rfp:
        while True:
            try:
                var = pickle.load(rfp)
            except EOFError:
                break
    # Printing the content of the file if it's feature importance data
    try :
        imp = var
        mean, std = np.around(imp[1],3).astype('U5') , np.around(imp[2], 2).astype('U5')
        importance = np.char.add(mean, ' +/- '); importance = np.char.add(importance, std)
        importance = ' | '.join(importance.tolist())
        print('{:<28} : {}'.format(imp[0], importance))
        output = imp
    # Printing the raw variable in the other case
    except : print(var);output = var
    return output

def ranking_plot(results, path, title, images, scalars, groups):
    '''
    Plots a horizontal bar plot ranking of the feature importances from a dictionary.
    '''
    # Maps of the categories for legend purposes
    categories = {'Images'   : (images[:-1], 'indigo'), 'Tracks image': (['tracks_image'], 'lime'),
                  'Scalars'  : (scalars, 'tab:blue'), 'Groups of features': (groups, 'tab:orange')}
    # Data parsing section
    sortedResults = sorted(results.items(), key = lambda lst: lst[1][0], reverse=True) # Sorts the importances in decreasing order
    labels = [tup[0] for tup in sortedResults] # Names of the variables
    newLabels = LaTeXizer(labels)[1] # Converted names
    data = np.array([tup[1][0] for tup in sortedResults])   # Feature importance
    errors = np.array([tup[1][1] for tup in sortedResults]) # Incertitude

    #Plotting section
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.invert_yaxis()
    widths = data
    for cat in categories:
        cat_widths = np.copy(widths)
        cat_err = np.copy(errors)
        category, color = categories[cat]
        indices = np.array([labels.index(feat) for feat in labels if feat not in category])
        if indices.size != 0 :
        # Set the values of the variables that are not in that category to zero (so they won't appear multiple times in the plot)
            cat_widths[indices] = np.zeros(indices.size)
            cat_err[indices] = np.zeros(indices.size)
        ax.barh(newLabels, cat_widths, height=0.75, xerr=cat_err, capsize=5, color=color, label=cat)

    # Red vertical line to highlight the threshold between good and bad variables:
    # Above this line, variables are important; under it, they are detrimental.
    plt.axvline(1, color='r', ls=':')

    # Numerical values of the importance (printed above the bars)
    for width, (index, value)  in zip(np.around(widths,3), enumerate(widths + errors + 0.005*widths[0])):
        plt.text(value, index, str(width), va='center')

    # Plot's finish
    ax.legend(loc='lower right', prop={'size': 14})
    plt.title(title, fontsize=20)
    ax.set_xlabel(r'$\frac{bkg\_rej\_full}{bkg\_rej}$', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    plt.tight_layout()

    # Saving section
    print('Saving plot to {}'.format(path))
    plt.savefig(path)
    return fig, ax

def saving_results(var, fname):
    '''
    Saves the given variable to pickle file and prints its values.
    '''
    fname += '.pkl'
    print('Saving results to {}'.format(fname))
    with open(fname,'wb') as wfp:
        pickle.dump(var, wfp)
    print_importances(fname)


def copy_sample(sample,feats):
    '''
    Initialize a copy of a valid sample that is going to be partially altered.
    '''
    shuffled_sample = {key:value for (key,value) in sample.items() if key not in feats}
    for feat in feats:
        shuffled_sample[feat] = deepcopy(sample[feat])  # Copy of the feature to be shuffled in order to keep the original sample intact
    return shuffled_sample

def shuffling_sample(sample, feats, k=0):
    '''
    Shuffles the specified features in the given sample.
    '''
    print('PERMUTATION #' + str(k+1))
    for feat in feats:
        np.random.shuffle(sample[feat])  # Shuffling of one feature

def pseudo_removal(sample, feats, k=0):
    '''
    Replace the specified features in the given sample by zeros.
    '''
    print('PSEUDO REMOVAL #' + str(k+1))
    for feat in feats:
        sample[feat] = np.zeros_like(sample[feat])  # Pseudo-removal of one feature

def feature_permutation(feats, g, sample, labels, model, bkg_rej_full, train_labels, training, n_classes, n_reps,
                       output_dir):
    '''
    Takes a pretrained model and saves the permutation importance of a feature or a group
    of features to a dictionary in a pickle file.
    '''
    # All the results will be saved in the permutation_importance subdirectory:
    output_dir += '/permutation_importance'
    # The importance of each variable will be saved in a different file:
    name = [feats[0],'group_{}'.format(g)][g>=0]
    fname = output_dir + '/' + name + '_importance'
    create_path(output_dir)
    # Converts the feature into a list to homogenize the format (groups are already given as a list)
    if type(feats) == str :
        feats = [feats]
    # Initialize bkg_rej
    if n_classes == 2 :
        bkg_rej = np.empty((1, n_reps))
    elif n_classes == 6 :
        bkg_rej = np.empty((n_classes, n_reps))
        bkg_rej_full = np.reshape(bkg_rej_full,(n_classes, 1))
    # Permutation of the given features k times
    features = ' + '.join(feats)
    print('\nPERMUTATION OF : ' + features)
    shuffled_sample = copy_sample(sample, feats)
    for k in range(n_reps) :
        shuffling_sample(shuffled_sample,feats, k)
        probs = model.predict(shuffled_sample, batch_size=20000, verbose=2)
        # Background rejection with one feature shuffled
        bkg_rej[:, k] = valid_results(shuffled_sample, labels, probs,
                            train_labels, training, output_dir, 'OFF', False, True)
    # Computation of the importance of the features
    importance = bkg_rej_full / bkg_rej
    imp_mean, imp_std = np.mean(importance, axis=1), np.std(importance, axis=1)
    imp_tup = name, imp_mean, imp_std, bkg_rej
    saving_results(imp_tup, fname)

def plot_importance(mode, output_dir, region, images, scalars, n_groups, n_classes):
    '''
    Opens the importance data files, parses them and then plots a ranking of the features.
    This function works for both permutation and removal importances.
    '''
    # Lists of the types of background in 6 classes
    bkg_list = ['global', 'Charge flip', 'Photon conversion', 'b/c hadron decay',
                r'Light flavor (bkg $\gamma$+e)', 'Ligth flavor (hadron)']
    # Dictionary containing the 3 eta regions in LaTeX format
    eta = {'barrel': r'$0<\eta<1.3$', 'transition': r'$1.3<\eta<1.6$', 'endcap': r'$1.6<\eta<2.5$'}
    groups = ['group_{}'.format(g) for g in range(n_groups)]
    feats = images + scalars + groups
    # Determine the number of bkgs against which the importance is to be computed
    if n_classes == 2: n_bkg = 1
    else: n_bkg = n_classes
    results = [{} for i in range(n_bkg)]

    # Prepare the permutation importance data for plotting
    if mode in ['prm','permutation']:
        mode = 'Permutation'
        # Name of the plot file
        plot = output_dir + '/permutation_importance/prm_imp'
        # Reading the pickle files:
        print('Opening', output_dir + '/permutation_importance/')
        for feat in feats:
            file = output_dir + '/permutation_importance/' + feat + '_importance.pkl'
            try:
                # Extracts the data from the pickle file
                name, imp, err, bkgs = print_importances(file)
                n_reps = 'averaged over {} repetitions, '.format(bkgs[0,:].size)
            except OSError:
                # Notify the user which features couldn't be included in the plot
                print(feat + ' not in directory')
                continue
            # Saves each background results separately
            for i in range(n_bkg):
                results[i].update({feat:(imp[i], err[i])})
        # Extracts the background rejection of the untouched training to give an idea of the scale
        full_bkg_rej = print_importances(output_dir + '/bkg_rej.pkl')

    # Prepare the removal importance data for plotting
    elif mode in ['rm', 'removal']:
        mode = 'Removal'
        imp_dir = '/removal_importance/'
        n_reps = '' # THIS WILL NEED TO BE ADJUSTED IF REMOVAL IMPORTANCE WITH MULTIPLE TRAININGS IS IMPLEMENTED
        feats = ['full'] + feats
        # Name of the plot file
        plot = output_dir + imp_dir + 'rm_imp'
        # Reading the pickle files:
        bkg_rej = {}
        print('Opening:', output_dir + imp_dir)
        for feat in feats:
            file = output_dir + imp_dir + feat + '/importance.pkl'
            try:
                # Extracts the background rejection of the removed features
                feat, bkg_rej[feat] = print_importances(file)
                # Computing the importance of the feature:
                imp = bkg_rej['full']/bkg_rej[feat]
                # Saves each background results separately (except for the full bkg_rej which is saved later)
                for i in range(n_bkg):
                    if feat != 'full': results[i].update({feat:(imp[i], 0.05)})
            except OSError:
                # Notify the user which features couldn't be included in the plot
                print(feat + ' not in directory')
                continue
        # Saves full background rejection to give an idea of the scale
        full_bkg_rej = bkg_rej['full']

    # Plotting
    for i in range(n_bkg):
        if i :
            suf = '_' + str(i)
        else :
            suf = '_bkg'
        title = '{} importance against {} background.\n({} classes, {}region : {} , full background rejection : {})'
        title = title.format(mode, bkg_list[i], n_classes, n_reps, eta[region], full_bkg_rej[i].astype(int))
        ranking_plot(results[i], plot + suf + '.pdf', title, images, scalars, groups)


def feature_removal(arg_feat, images, scalars, groups, arg_im, arg_sc):
    '''
    Removes the specified features from the input variables.
    '''
    i = arg_feat                                        # Image indices
    s = arg_feat - len(images)                          # Scalar indices
    g = arg_feat - len(images + scalars)                # Group of features indices
    print('i : {}, s : {}, g : {}'.format(i,s,g))       # For debugging purposes

    # Fail-safes
    if g > len(groups) :
        print('Argument out of range, aborting...')
        sys.exit()

    if i >= 0 and i < len(images)  :
        if arg_im == 'OFF':
            print('Cannot remove image if images are OFF, aborting...')
            sys.exit()
        # Removal of the specified image
        images, feat = images[:i]+images[i+1:], images[i]

    elif s >= 0 and s < len(scalars) :
        if arg_sc == 'OFF':
            print('Cannot remove scalar if scalars are OFF, aborting...')
            sys.exit()
        # Removal of the specified scalar
        scalars, feat = scalars[:s]+scalars[s+1:], scalars[s]

    elif g >= 0 :
        condition1 = groups[g][0] not in images + scalars
        condition2 = groups[g][0] in images and arg_im == 'OFF'
        condition3 = groups[g][0] in scalars and arg_sc == 'OFF'
        if condition1 or condition2 or condition3 :
            print("Cannot remove features not in the sample, aborting...")
            sys.exit()
        # Removal of the features in the group
        images  = [key for key in images  if key not in groups[g]]
        scalars = [key for key in scalars if key not in groups[g]]
        # Group automatic name:
        feat = 'group_{}'.format(g)

    else : feat = 'full'
    return images, scalars, feat


def correlations(images, scalars, sample, labels, region, output_dir, scaling, scaler_out, arg_im, arg_corr, arg_tracks_means):
    '''
    Separates and prepares the sample for the correlations plots and runs the correlations plots
    '''
    if arg_corr not in ['ON','SCATTER']: return

    # Scalars obtained from the tracks images
    tracks_means = ['p_mean_efrac', 'p_mean_deta'   , 'p_mean_dphi'   , 'p_mean_d0'     ,
                    'p_mean_z0'   , 'p_mean_charge' , 'p_mean_vertex' , 'p_mean_chi2'   ,
                    'p_mean_ndof' , 'p_mean_pixhits', 'p_mean_scthits', 'p_mean_trthits',
                    'p_mean_sigmad0']

    # Adding tracks_means to the scalars for correlation
    if arg_tracks_means == 'ON':
        scalars += tracks_means
        fmode = '_with_tracks'
    elif arg_tracks_means == 'ONLY':
        scalars = tracks_means
        fmode = '_tracks_only'
    else :
        fmode = ''

    output_dir += '/correlations/'
    create_path(output_dir)
    # Applying quantile transform
    if scaling:
        scaler_out = output_dir + scaler_out
        train_sample, sample = apply_scaler(sample, sample, scalars, scaler_out)
        trans = '_QT'
        mode = ' with quantile transform'
    else :
        trans = ''
        mode = ''

    # Adding images means to the scalars
    if arg_im == 'ON':
        for image in images:
            if np.amin(sample[image]) == np.amax(sample[image]) :
                print(image,'is empty')
                continue
            sample[image + '_mean'] = np.mean(sample[image], axis = (1,2))
            scalars += [image + '_mean']
        fmode = '_with_im_means'

    # Separating the sample into signal sample and background sample
    sig_sample = {key : sample[key][np.where(labels == 0)[0]] for key in scalars}
    bkg_sample = {key : sample[key][np.where(labels == 1)[0]] for key in scalars}

    # Evaluating and plotting correlations
    print('CLASSIFIER : evaluating variables correlations')
    plot_correlations(bkg_sample, output_dir, scatter=arg_corr, mode = '\n(Background' + mode + ')',
                 fmode = '_bkg' + trans + fmode, region=region)
    plot_correlations(sig_sample, output_dir, scatter=arg_corr, mode = '\n(Signal' + mode + ')',
                 fmode = '_sig' + trans + fmode, region=region)
    sys.exit() # End the program when correlations are completed

def plot_correlations(sample, dir, scatter=False, LaTeX=True, frmt = '.pdf', mode='', fmode='',region='barrel'):
    '''
    Computes correlation coefficient between the given variables of a sample, then plots
    a matrix of those coefficients.

    OR

    If scatter=True, plots scatter plots between the given variables and their distrubution
    into a matrix.
    '''
    # Mapping of the three eta region for title purposes
    eta = {'barrel': r'$0<\eta<1.3$', 'transition': r'$1.3<\eta<1.6$', 'endcap': r'$1.6<\eta<2.5$'}
    data = pd.DataFrame(sample)

    # Converts the variables' names to be compatible with LaTeX display
    if LaTeX:
        print("LaTeX : ", "ON" if LaTeX else 'OFF')
        data = data.rename(columns = LaTeXizer()[0])
    names = data.columns

    # Computing correlations
    correlations = data.corr()

    # plot scatter plot matrix
    if scatter == 'SCATTER':
        print('Plotting scatter plot matrix')
        scatter_matrix(data, figsize = (18,18))
        plt.suptitle(r'Scatter plot matrix for {}'.format(eta[region]) + mode, fontsize = 20)
        plt.yticks(rotation=-90)
        plt.tight_layout()
        plt.savefig(dir + 'scatter_plot_matrix' + fmode + '.png')

    # plot correlation matrix
    else:
        print('Plotting correlation matrix')
        fig = plt.figure(figsize=(20,18))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        for (i, j), z in np.ndenumerate(correlations):
            ax.text(j, i, '{:0.1f}'.format(z) if abs(z) > 0.15 and z != 1.0 else '', ha='center', va='center', fontsize=8)
        ticks = np.arange(0,len(names),1)
        ax.set_xticks(ticks)#, rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names, fontsize = 14)
        ax.set_yticklabels(names, fontsize = 14)
        plt.title(r'Correlation matrix for {}'.format(eta[region]) + mode, fontsize = 20)
        plt.tight_layout()
        path = dir + 'corr_matrix' + fmode + frmt
        print('Saving matrix to '+ path)
        plt.savefig(path)
