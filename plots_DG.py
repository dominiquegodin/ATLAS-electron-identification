import numpy           as np
import multiprocessing as mp
import sys, os, h5py, pickle, time, itertools, warnings
from   functools         import partial
from   sklearn           import metrics
from   scipy.spatial     import distance
from   matplotlib        import pylab
from   matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FixedLocator
from   matplotlib.lines  import Line2D
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#plt.rcParams['mathtext.fontset'] = 'cm'
#Unicode characters in Linux: CTRL-SHIFT-u ($ϵ$, $ε$, $ϕ$, $φ$, $σ$, $ς$ $η$)


def valid_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return np.sum(y_pred==y_true)/len(y_true)


def LLH_rates(sample, y_true, ECIDS=False, ECIDS_cut=-0.337671):
    LLH_tpr, LLH_fpr = [], []
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        LLH_class0 = sample[wp][y_true==0]
        LLH_class1 = sample[wp][y_true!=0]
        #LLH_tpr.append( np.sum(LLH_class0==0) / len(LLH_class0) )
        #LLH_fpr.append( np.sum(LLH_class1==0) / len(LLH_class1) )
        LLH_tpr.append( np.sum(LLH_class0!=0) / len(LLH_class0) )
        LLH_fpr.append( np.sum(LLH_class1!=0) / len(LLH_class1) )
    if ECIDS:
        ECIDS_class0 = sample['p_ECIDSResult'][y_true==0]
        ECIDS_class1 = sample['p_ECIDSResult'][y_true!=0]
        for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
            LLH_class0 = sample[wp][y_true==0]
            LLH_class1 = sample[wp][y_true!=0]
            #LLH_tpr.append( np.sum((LLH_class0==0) & (ECIDS_class0>=ECIDS_cut)) / len(LLH_class0) )
            #LLH_fpr.append( np.sum((LLH_class1==0) & (ECIDS_class1>=ECIDS_cut)) / len(LLH_class1) )
            LLH_tpr.append( np.sum((LLH_class0!=0) & (ECIDS_class0>=ECIDS_cut)) / len(LLH_class0) )
            LLH_fpr.append( np.sum((LLH_class1!=0) & (ECIDS_class1>=ECIDS_cut)) / len(LLH_class1) )
    return LLH_fpr, LLH_tpr


def plot_history(history, output_dir, key='accuracy'):
    if history == None or len(history.epoch) < 2: return
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    val = plt.plot(np.array(history.epoch)+1, 100*np.array(history.history[key]), label='Training')
    plt.plot(np.array(history.epoch)+1, 100*np.array(history.history['val_'+key]), '--',
             color=val[0].get_color(), label='Testing')
    min_acc = np.floor(100*min( history.history[key]+history.history['val_'+key] ))
    max_acc = np.ceil (100*max( history.history[key]+history.history['val_'+key] ))
    plt.xlim([1, max(history.epoch)+1])
    plt.xticks( np.append(1, np.arange(5, max(history.epoch)+2, step=5)) )
    plt.xlabel('Epochs',fontsize=25)
    plt.ylim( max(70,min_acc),max_acc )
    plt.yticks( np.arange(max(80,min_acc), max_acc+1, step=1) )
    plt.ylabel(key.title()+' (%)',fontsize=25)
    plt.legend(loc='lower right', fontsize=20, numpoints=3)
    file_name = output_dir+'/'+'train_history.png'
    print('Saving training accuracy history to:', file_name, '\n'); plt.savefig(file_name)


def plot_heatmaps(sample, labels, output_dir):
    n_classes = max(labels)+1
    label_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion'   ,     3:'Heavy Flavour',
                  4:'Light Flavour e$/\gamma$'        , 5:'Light Flavour Hadron', 'bkg':'Background'}
    pt  =     sample['pt']  ;  pt_bins = np.arange(0,81,1)
    eta = abs(sample['eta']); eta_bins = np.arange(0,2.55,0.05)
    extent = [eta_bins[0], eta_bins[-1], pt_bins[0], pt_bins[-1]]
    fig = plt.figure(figsize=(20,10)); axes = plt.gca()
    for n in np.arange(n_classes):
        plt.subplot(2, 3, n+1)
        heatmap = np.histogram2d(eta[labels==n], pt[labels==n], bins=[eta_bins,pt_bins], density=False)[0]
        plt.imshow(heatmap.T, origin='lower', extent=extent, cmap='Blues', interpolation='bilinear', aspect="auto")
        plt.title(label_dict[n]+' ('+format(100*np.sum(labels==n)/len(labels),'.1f')+'%)', fontsize=25)
        if n//3 == 1: plt.xlabel('$|\eta|$', fontsize=25)
        if n %3 == 0: plt.ylabel('$E_\mathrm{T}$ (GeV)', fontsize=25)
    fig.subplots_adjust(left=0.05, top=0.95, bottom=0.1, right=0.95, wspace=0.15, hspace=0.25)
    file_name = output_dir+'/'+'heatmap.png'
    print('Saving heatmap plots to:', file_name)
    plt.savefig(file_name); sys.exit()


def var_histogram(sample, labels, n_etypes, weights, bins, output_dir, prefix, var,
                  density=True, separate_norm=False, log=True):
    n_classes = max(labels)+1
    plt.figure(figsize=(12,8)); pylab.grid(False); axes = plt.gca()
    # Axes parameters
    axes.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(which='major', direction='in', length=10, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(axis="both", pad=8, labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(1.5)
        axes.spines[axis].set_color('black')
    if var == 'pt':
        variable = sample[var]
        plt.xlabel('$E_\mathrm{T}$ (GeV)', fontsize=30, loc='right')
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
            bins = np.arange(np.min(variable), 102)
        else:
            bins = bins[var]
        if log:
            separate_norm=False
            if n_classes == 2: plt.xlim(4.5, 5000); plt.ylim(1e-5,1e2); plt.xscale('log'); plt.yscale('log')
            else             : plt.xlim(4.5, 5000); plt.ylim(1e-8,1e0); plt.xscale('log'); plt.yscale('log')
            plt.xticks( [4.5,10,100,1000,5000], [4.5,'$10^1$','$10^2$','$10^3$',r'$5\!\times\!10^3$'] )
            #plt.yticks(np.logspace(-8,0,9))
        else:
            separate_norm=True
            axes.xaxis.set_minor_locator(FixedLocator(bins))
            plt.xticks( np.append([4.5],np.arange(0, 101, 20)), [4.5,0,20,40,60,80,100] )
            #xticks = np.append([4.5], plt.xticks()[0])
            plt.xlim(0, 100); plt.ylim(0,25)
            axes.xaxis.set_minor_locator(FixedLocator(np.arange(0,101,2)))
            axes.yaxis.set_minor_locator(AutoMinorLocator(5))
    if var == 'eta':
        variable = abs(sample[var])
        plt.xlabel('$|\eta|$', fontsize=30, loc='right')
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
            step = 0.05; bins = np.arange(0, 2.5+step, step)
        else:
            bins = bins[var]
        axes.xaxis.set_minor_locator(FixedLocator(bins))
        plt.xticks(np.arange(0, 2.6, 0.5))
        #plt.xticks(bins, [str(n) for n in bins]); plt.xticks(bins, [format(n,'.1f') for n in bins])
        pylab.xlim(0, 2.5)
        if separate_norm:
            pylab.ylim(0, 90)
            #plt.yticks(np.arange(0, 100, 10))
            axes.yaxis.set_minor_locator(AutoMinorLocator(5))
        else:
            pylab.ylim(1e-3, 2e0)
            plt.yscale('log')
    #bins[-1] = max(bins[-1], max(variable)+1e-3)
    if n_etypes <= 5:
        label_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion'  ,     3:'Heavy Flavour',
                      4:'Light Flavour', 'bkg':'Background'}
    if n_etypes == 6:
        label_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion'   ,     3:'Heavy Flavour',
                      4:'Light Flavour e$\;\!/\;\!\gamma$', 5:'Light Flavour Hadron', 'bkg':'Background'}
    color_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple', 5:'tab:brown'}
    #if prefix == 'train':
    #    color_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:orange', 3:'tab:orange', 4:'tab:purple', 5:'tab:orange'}
    if n_classes == 2: label_dict[1] = 'Background'
    if np.all(weights) == None: weights = np.ones(len(variable))
    h = np.zeros((len(bins)-1,n_classes))
    for n in np.arange(n_classes):
        class_values  =  variable[labels==n]
        class_weights =   weights[labels==n]
        class_weights =  class_weights/(np.sum(class_weights) if separate_norm else len(variable))
        #class_weights =* 100
        if density:
            indices        = np.searchsorted(bins, class_values, side='right')
            class_weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
        #pylab.hist(class_values, bins, histtype='step', weights=class_weights, color=color_dict[n],
        #           label=label_dict[n]+' ('+format(100*len(class_values)/len(variable),'.1f')+'%)', lw=2)[0]
        pylab.hist(class_values, bins, histtype='step', weights=class_weights,
                   color=color_dict[n], label=label_dict[n], lw=3)[0]
    #plt.ylabel('Distribution density (%)' if density else 'Distribution (%)', fontsize=25)
    plt.ylabel('Fraction of Candidates (GeV$^{-1}$)', fontsize=25, loc='top')
    loc, ncol = ('upper right',2) if var=='pt' else ('upper right',2)
    # Create new legend handles with existing colors
    handles, labels = axes.get_legend_handles_labels()
    new_handles = [Line2D([], [], lw=3, c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, loc=loc, fontsize=16.5, columnspacing=1.,
               frameon=True, handlelength=2, ncol=ncol, facecolor=None, framealpha=1.).set_zorder(10)
    # Adding ATLAS messaging
    xpos, ha = (0.02, 'left') if var=='pt' else (0.02, 'left')
    plt.text(xpos, 0.95, r'$\bf ATLAS$ Simulation Internal', {'color':'black', 'fontsize':18},
             va='center', ha=ha, transform=axes.transAxes, zorder=20)
    plt.text(xpos, 0.91, r'$\sqrt{s}=$13$\,$TeV', {'color':'black', 'fontsize':18},
             va='center', ha=ha, transform=axes.transAxes, zorder=20)
    plt.subplots_adjust(left=0.1, top=0.97, bottom=0.12, right=0.95)
    file_name = output_dir+'/'+str(var)+'_'+prefix+'.png'
    print('Saving', prefix, 'sample', format(var,'3s'), 'distributions to:', file_name)
    plt.savefig(file_name)


def plot_discriminant(sample, y_true, y_prob, n_etypes, output_dir, separation=False, bkg='bkg'):
    if n_etypes <= 5:
        label_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion'  ,     3:'Heavy Flavour',
                      4:'Light Flavour', 'bkg':'Background'}
    if n_etypes == 6:
        label_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion'   ,     3:'Heavy Flavour',
                      4:'Light Flavour e$\;\!/\;\!\gamma$', 5:'Light Flavour Hadron', 'bkg':'Background'}
        #label_dict = {0:'Electron & charge flip'         , 1:'Photon conversion'    ,   2  :'Heavy flavor',
        #              3:'Light flavor (e$^\pm$/$\gamma$)', 4:'Light flavor (hadron)', 'bkg':'Background'}
    color_dict = {0:'tab:blue'    , 1:'tab:orange'   , 2:'tab:green'            ,   3  :'tab:red'   ,
                  4:'tab:purple'                     , 5:'tab:brown'            , 'bkg':'tab:orange'}
    if separation:
        label_dict.pop('bkg')
    else:
        label_dict={0:'Prompt Electron', 1:label_dict[bkg]}
        color_dict={0:'tab:blue'       , 1:color_dict[bkg]}
    n_classes = len(label_dict)
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def inverse_logit(x):
        return 1/(1+10**(-x))
    def print_JSD(P, Q, idx, color, text):
        plt.text(0.945, 1.01-3*idx/100, 'JSD$_{0,\!'+text+'}$:',
                 {'color':'black', 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
        plt.text(0.990, 1.01-3*idx/100, format(distance.jensenshannon(P, Q), '.3f'),
                 {'color':color  , 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
    def class_histo(y_true, y_prob, bins, colors):
        h = np.full((len(bins)-1,n_classes), 0.)
        from utils import make_labels
        class_labels = make_labels(sample, n_classes)
        for n in np.unique(class_labels):
            class_probs   = y_prob[class_labels==n]
            #class_weights = len(class_probs)*[100/len(y_true)]
            class_weights = len(class_probs)*[1/len(y_true)]
            #class_weights = len(class_probs)*[100/len(class_probs)]
            h[:,n] = pylab.hist(class_probs, bins=bins, label=label_dict[n], histtype='step',
                                weights=class_weights, log=True, color=colors[n], lw=3)[0]
        if n_classes == 2:
            colors = len(colors)*['black']
        if False:
            for n in set(label_dict)-set([0]):
                new_y_true = y_true[np.logical_or(y_true==0, class_labels==n)]
                new_y_prob = y_prob[np.logical_or(y_true==0, class_labels==n)]
                fpr, tpr, threshold = metrics.roc_curve(new_y_true, new_y_prob, pos_label=0)
                sig_ratio = np.sum(y_true==0)/len(new_y_true)
                max_index = np.argmax(sig_ratio*tpr + (1-fpr)*(1-sig_ratio))
                axes.axvline(threshold[max_index], ymin=0, ymax=1, ls='--', lw=1, color=colors[n])
            for n in set(np.unique(class_labels))-set([0]):
                print_JSD(h[:,0], h[:,n], n, colors[n], str(n))
            if n_classes > 2:
                print_JSD(h[:,0], np.sum(h[:,1:], axis=1), n_classes, 'black', '\mathrm{bkg}')
        axes.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                         bottom=True, top=True, left=True, right=True)
        axes.tick_params(which='major', direction='in', length=10, width=1.5, colors='black',
                         bottom=True, top=True, left=True, right=True)
        axes.tick_params(axis="both", pad=8, labelsize=18)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes.spines[axis].set_linewidth(1.5)
            axes.spines[axis].set_color('black')
        #plt.xlabel('$p_{\operatorname{sig}}$ (%)', fontsize=30)
        plt.xlabel('$\mathcal{D}$ (%)'     , fontsize=30, loc='right')
        plt.ylabel('Fraction of Candidates', fontsize=25, loc='top')
        # Create new legend handles with existing colors
        handles, labels = axes.get_legend_handles_labels()
        new_handles = [Line2D([], [], lw=3, c=h.get_edgecolor()) for h in handles]
        plt.legend(handles=new_handles, labels=labels, loc='upper right', fontsize=16.5, columnspacing=1.,
                   frameon=True, handlelength=2, ncol=2, facecolor=None, framealpha=1.).set_zorder(10)
        # Adding ATLAS messaging
        plt.text(0.02, 0.95, r'$\bf ATLAS$ Simulation Internal',
                 {'color':'black', 'fontsize':18},  va='center', ha='left', transform=axes.transAxes, zorder=20)
        #plt.text(0.02, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}\:\leqslant\:$15 GeV',
        plt.text(0.02, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}>$15$\,$GeV',
        #plt.text(0.02, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}$ inclusive',
                 {'color':'black', 'fontsize':18},  va='center', ha='left', transform=axes.transAxes, zorder=20)
    plt.figure(figsize=(12,16))
    plt.subplot(2, 1, 1); pylab.grid(False); axes = plt.gca()
    pylab.xlim(0,100); pylab.ylim(1e-7 if n_classes>2 else 1e-6, 1e0)
    plt.xticks(np.arange(0,101,step=10))
    #pylab.xlim(0,10); pylab.ylim(1e-2 if n_classes>2 else 1e-2, 1e2)
    #plt.xticks(np.arange(0,11,step=1))
    bin_step = 0.5; bins = np.arange(0, 100+bin_step, bin_step)
    class_histo(y_true, 100*y_prob, bins, color_dict)
    plt.subplot(2, 1, 2); pylab.grid(False); axes = plt.gca()
    x_min=-10; x_max=5; pylab.xlim(x_min, x_max); pylab.ylim(1e-6 if n_classes>2 else 1e-5, 3e-1)
    pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    pos += [0.5]           + [1-10**float(n)      for n in np.arange(-1,-x_max-1,-1)]
    lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ]
    lab += [1,10,50,90,99] + ['99.'+n*'9'         for n in np.arange(1,x_max-1)     ]
    #x_min=-10; x_max=-1; pylab.xlim(x_min, x_max); pylab.ylim(1e-2 if n_classes>2 else 1e-4, 1e2)
    #pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    #lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ] + [1,10]
    #lab += ['0.50   '] + ['$1\!-\!10^{'+str(n)+'}$' for n in np.arange(-1,-x_max-1,-1)]
    plt.xticks(logit(np.array(pos)), lab, rotation=20)
    bin_step = 0.1; bins = np.arange(x_min-1, x_max+1, bin_step)
    y_prob = logit(y_prob)
    class_histo(y_true, y_prob, bins, color_dict)
    plt.subplots_adjust(top=0.99, bottom=0.07, left=0.1, right=0.95, hspace=0.2)
    file_name = output_dir+'/discriminant.png'
    print('Saving test sample discriminant   to:', file_name); plt.savefig(file_name)


def plot_ROC_curves(sample, y_true, y_prob, output_dir, ROC_type, ECIDS,
                    first_cuts=None, ROC_values=None, multiplots=False):
    LLH_fpr, LLH_tpr = LLH_rates(sample, y_true, ECIDS)
    if ROC_values != None:
        fpr, tpr = ROC_values[0]
    #if ROC_values != None:
    #    index = output_dir.split('_')[-1]
    #    index = ROC_values[0].shape[1]-1 if index == 'bkg' else int(index)
    #    fpr_full, tpr_full = ROC_values[0][:,index], ROC_values[0][:,0]
    #    fpr     , tpr      = ROC_values[1][:,index], ROC_values[1][:,0]
    else:
        if first_cuts is None: first_cuts = np.full_like(y_true, True, dtype=bool)
        pos_rates = {key:np.sum(first_cuts[y_true==n])/np.sum(y_true==n) for key,n in zip(['tpr','fpr'],[0,1])}
        y_true, y_prob = y_true[first_cuts], y_prob[first_cuts]
        #fpr, tpr, thresholds = metrics.roc_curve(y_true, sample['p_LHValue'], pos_label=0)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=0)
        #fpr, tpr, thresholds = fpr[::-2][::-1], tpr[::-2][::-1], thresholds[::-2][::-1] #for linear interpolation
        fpr, tpr, thresholds = pos_rates['fpr']*fpr[fpr!=0], pos_rates['tpr']*tpr[fpr!=0], thresholds[fpr!=0]
        pickle.dump({'fpr':fpr, 'tpr':tpr}, open(output_dir+'/'+'pos_rates.pkl','wb'), protocol=4)
    signal_ratio       = np.sum(y_true==0)/len(y_true)
    accuracy           = tpr*signal_ratio + (1-fpr)*(1-signal_ratio)
    best_tpr, best_fpr = tpr[np.argmax(accuracy)], fpr[np.argmax(accuracy)]
    #colors  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green']
    colors = 3*['black'] + 3*['none']
    labels = ['tight', 'medium', 'loose','tight+ECIDS', 'medium+ECIDS', 'loose+ECIDS']
    #markers = 3*['o'] + 3*['D']
    markers = 2*['o','D','^']
    epsilon = 'ϵ' #'\epsilon'
    sig_eff, bkg_eff = '$'+epsilon+'_{\operatorname{sig}}$', '$'+epsilon+'_{\operatorname{bkg}}$'
    fig = plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    axes.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(which='major', direction='in', length=10, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(axis="both", pad=8, labelsize=18)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(1.5)
        axes.spines[axis].set_color('black')
    if ROC_type == 1:
        pylab.grid(False, which="both")
        x_min = min(80, 10*np.floor(10*max(min(LLH_tpr),np.min(tpr))))
        if fpr[np.argwhere(tpr >= x_min/100)[0]] != 0:
            y_max = 10*np.ceil( 1/min(np.append(fpr[tpr >= x_min/100], min(LLH_fpr)))/10 )
            if y_max > 200: y_max = 100*(np.ceil(y_max/100))
            y_max *= 1.2
        else:
            y_max = 1000*np.ceil(max(1/fpr)/1000)
        plt.xlim([x_min, 100]); plt.ylim([1, y_max])
        def interpolation(tpr, fpr, value):
            try:
                #idx1 = np.argmax(tpr[tpr<value])
                idx1 = np.where(tpr<=value)[0][-1]
                idx2 = np.where(tpr>=value)[0][ 0]
                M = (1/fpr[idx2]-1/fpr[idx1])/(tpr[idx2]-tpr[idx1]) if tpr[idx2]!=tpr[idx1] else 0
                return 1/fpr[idx1] + M*(value-tpr[idx1])
            except:
                return None
        #LLH_scores = [1/fpr[np.argmin(abs(tpr-value))] for value in LLH_tpr]
        LLH_scores = [interpolation(tpr, fpr, value) for value in LLH_tpr]
        for n in np.arange(len(LLH_scores[:3])):
            if LLH_scores[n] is None: continue
            axes.axhline(LLH_scores[n], xmin=(LLH_tpr[n]-x_min/100)/(1-x_min/100), xmax=1,
            ls='--', linewidth=0.5, color='tab:gray', zorder=10)
            axes.axvline(100*LLH_tpr[n], ymin=abs(1/LLH_fpr[n]-1)/(plt.yticks()[0][-1]-1),
            ymax=abs(LLH_scores[n]-1)/(plt.yticks()[0][-1]-1), ls='--', linewidth=0.5, color='tab:gray', zorder=5)
            if LLH_scores[n] < 1e5: score_text = format(LLH_scores[n],'.0f')
            else                  : score_text = format(LLH_scores[n],'.1e').replace('e+0','e')
            plt.text(100+(100.3-100)/(100.3-50)*(100.3-x_min), LLH_scores[n], score_text,
                     {'color':colors[n], 'fontsize':15}, va="center", ha="left")
        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        yticks = plt.yticks()[0]
        axes.yaxis.set_ticks( np.append([1], yticks[1:]) )
        axes.yaxis.set_minor_locator(FixedLocator(np.arange(yticks[1]/5, yticks[-1], yticks[1]/5 )))
        plt.xlabel(sig_eff+' (%)', fontsize=30, loc='right')
        plt.ylabel('1/'+bkg_eff  , fontsize=30, loc='top')
        P, = plt.plot(100*tpr, 1/fpr, color='tab:blue', lw=3, zorder=10)
        n_sig, n_bkg = np.sum(y_true==0), np.sum(y_true==1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lim_inf = 1/( fpr + np.sqrt(fpr/n_bkg) )
            lim_sup = 1/( fpr - np.sqrt(fpr/n_bkg) )
            lim_sup = np.nan_to_num(lim_sup)
        plt.fill_between(100*tpr, lim_inf, lim_sup, alpha=0.2)
        if multiplots:
            def get_legends(n_zip, output_dir):
                file_path, color, ls = n_zip
                file_path += '/' + output_dir.split('/')[-1] + '/pos_rates.pkl'
                fpr, tpr = pickle.load(open(file_path, 'rb')).values()
                fpr, tpr = fpr[fpr!=0], tpr[fpr!=0]
                P, = plt.plot(100*tpr, 1/fpr, color=color, lw=2 if color=='dimgray' else 3, ls=ls, zorder=10)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    lim_inf = 1/( fpr + np.sqrt(fpr/n_bkg) )
                    lim_sup = 1/( fpr - np.sqrt(fpr/n_bkg) )
                    lim_sup = np.nan_to_num(lim_sup)
                plt.fill_between(100*tpr, lim_inf, lim_sup, color=color, alpha=0.2, edgecolor=None)
                return P
            color_list = ['tab:orange', 'tab:green', 'tab:red', 'dimgray']
            #linestyles = ['--', '-.', ':']
            linestyles = [(0,(5, 1)), (0,(3, 1, 1, 1,)), (0,(1, 1)), '-']
            file_tag   = '0-5000GeV'
            file_paths = ['outputs/6c_180m/HLV+tracks', 'outputs/6c_180m/HLV+images',
                          'outputs/6c_180m/HLV', 'outputs/6c_180m/LLH_results']
            #file_paths = ['outputs/5c_180m/HLV+tracks', 'outputs/5c_180m/HLV+images',
            #              'outputs/5c_180m/HLV', 'outputs/5c_180m/images']
            leg_labels = ['HLV$\:\!+\:\!$tracks$\:\!+\:\!$images', 'HLV$\:\!+\:\!$tracks',
                          'HLV$\:\!+\:\!$images', 'HLV', 'LLH']#, 'Tracks$\:\!+\:\!$images']
            #file_paths = ['outputs/6c_180m/scalars+tracks+images/pred_ratios',
            #              'outputs/6c_180m/scalars+tracks+images/none_ratios']
            #leg_labels = ['Truth Ratios', 'Predicted Ratios', 'Agnostic Ratios']
            #file_paths = ['outputs/5c_180m/HLV+tracks+images_no-charge']
            #leg_labels = ['With Charge', 'Without Charge']
            #file_paths = ['outputs/6c_180m/HLV+tracks+images_no-coarse-Lr1']
            #leg_labels = ['HLV$\:\!+\:\!$tracks$\:\!+\:\!$images', 'HLV$\:\!+\:\!$tracks$\:\!+\:\!$images (no coarse Lr1\'s)']
            file_paths = [path+'/'+file_tag for path in file_paths]
            Ps = [P] + [get_legends(n_zip, output_dir) for n_zip in zip(file_paths,color_list,linestyles)]
            anchor = (1.,0.55) if ECIDS else (1.,0.77)
            L = plt.legend(Ps, leg_labels, loc='upper right', bbox_to_anchor=anchor, fontsize=17,
                           ncol=1, frameon=False, facecolor=None, framealpha=1); L.set_zorder(10)
        if ROC_values != None:
            tag = 'combined bkg'
            extra_labels = {1:'{$w_{\operatorname{bkg}}$} for best AUC',
                            2:'{$w_{\operatorname{bkg}}$} for best TNR @ 70% $'+epsilon+'_{\operatorname{sig}}$'}
            extra_colors = {1:'tab:orange', 2:'tab:green'   , 3:'tab:red'     , 4:'tab:brown'}
            for n in [1,2]:#range(1,len(ROC_values)):
                extra_fpr, extra_tpr = ROC_values[n]
                extra_len_0 = np.sum(extra_fpr==0)
                plt.plot(100*extra_tpr[extra_len_0:], 1/extra_fpr[extra_len_0:],
                         label=extra_labels[n], color=extra_colors[n], lw=2, zorder=10)
            plt.rcParams['agg.path.chunksize'] = 1000
            plt.scatter(100*tpr_full[len_0:], 1/fpr_full[len_0:], color='silver', marker='.')
        if best_fpr != 0 and False:
            plt.scatter( 100*best_tpr, 1/best_fpr, s=40, marker='D', c='tab:blue',
                         label="{0:<15s} {1:>3.2f}%".format('Best accuracy:',100*max(accuracy)), zorder=10 )
        for n in np.arange(len(LLH_scores)):
            if LLH_scores[n] is None: continue
            plt.scatter( 100*LLH_tpr[n], 1/LLH_fpr[n], s=70, marker=markers[n], zorder=20,
                         edgecolors='black', c=colors[n], linewidth=2,
                         label='$'+epsilon+'_{\operatorname{sig}}^{\operatorname{LH}}$'
                         +'='+format(100*LLH_tpr[n],'.1f') +'%' +r'$\rightarrow$'
                         +'$'+epsilon+'_{\operatorname{bkg}}^{\operatorname{LH}}$/'
                         +'$'+epsilon+'_{\operatorname{bkg}}^{\operatorname{NN}}$='
                         +format(LLH_fpr[n]*LLH_scores[n], '>.1f' if LLH_fpr[n]*LLH_scores[n]<1e2 else '>.0f')
                         +' ('+labels[n]+')' )
            xerr_inf = np.sqrt(LLH_tpr[n]/n_sig)
            xerr_sup = np.sqrt(LLH_tpr[n]/n_sig)
            yerr_inf =  1/LLH_fpr[n] - 1/( LLH_fpr[n] + np.sqrt(LLH_fpr[n]/n_bkg) )
            yerr_sup = -1/LLH_fpr[n] + 1/( LLH_fpr[n] - np.sqrt(LLH_fpr[n]/n_bkg) )
            plt.errorbar( 100*LLH_tpr[n], 1/LLH_fpr[n], ecolor='black', linewidth=2,
                          xerr=[[xerr_inf],[xerr_sup]], yerr=[[yerr_inf],[yerr_sup]] )
        #plt.text(0.07, 0.95, r'$\bf ATLAS$ Simulation Internal',
        #         {'color':'black', 'fontsize':18},  va='center', ha='left', transform=axes.transAxes, zorder=20)
        #plt.text(0.07, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}\:\leqslant\:$15 GeV',
        #plt.text(0.07, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}>$15$\,$GeV',
        #plt.text(0.07, 0.91, r'$\sqrt{s}=$13$\,$TeV $\:;\: E_\mathrm{T}$ inclusive',
        #         {'color':'black', 'fontsize':18},  va='center', ha='left', transform=axes.transAxes, zorder=20)
        plt.legend(loc='upper right', fontsize=17 if ECIDS else 17, frameon=False,
                   handletextpad=0., facecolor=None, framealpha=1).set_zorder(10)
        if multiplots: plt.gca().add_artist(L)
    if ROC_type == 2:
        plt.xlim([0.6, 1]); plt.ylim([0.9, 1-1e-4])
        plt.xticks([0.6, 0.7, 0.8, 0.9, 1], [60, 70, 80, 90, 100])
        plt.yscale('logit')
        plt.yticks([0.9, 0.99, 0.999, 0.9999], [90, 99, 99.9, 99.99])
        axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        axes.xaxis.set_minor_formatter(plt.NullFormatter())
        axes.yaxis.set_minor_formatter(plt.NullFormatter())
        plt.xlabel('Signal Efficiency '+sig_eff+' (%)', fontsize=25)
        plt.ylabel('Background Rejection $1\!-\!$'+bkg_eff+' (%)', fontsize=25)
        plt.text(0.8, 0.67, 'AUC: '+format(metrics.auc(fpr,tpr),'.4f'),
                 {'color':'black', 'fontsize':22},  va='center', ha='center', transform=axes.transAxes)
        val = plt.plot(tpr, (1-fpr), label='Signal vs Background', color='#1f77b4', lw=2)
        plt.scatter( best_tpr, (1-best_fpr), s=40, marker='o', c=val[0].get_color(),
                     label="{0:<16s} {1:>3.2f}%".format('Best accuracy:', 100*max(accuracy)) )
        for LLH in zip(LLH_tpr, LLH_fpr, colors, labels, markers):
            plt.scatter(LLH[0], 1-LLH[1], s=40, marker=LLH[4], c=LLH[2], label='('+format(100*LLH[0],'.1f')
                        +'%, '+format(100*(1-LLH[1]),'.1f')+')'+r'$\rightarrow$'+LLH[3])
        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='upper right', fontsize=15, numpoints=3)
    if ROC_type == 3:
        def make_plot(location):
            plt.xlabel('Signal probability threshold (%)', fontsize=25); plt.ylabel('(%)',fontsize=25)
            val_1 = plt.plot(thresholds[1:],   tpr[1:], color='tab:blue'  , label='Signal Efficiency'   , lw=2)
            val_2 = plt.plot(thresholds[1:], 1-fpr[1:], color='tab:orange', label='Background Rejection', lw=2)
            val_3 = plt.plot(thresholds[1:], accuracy[1:], color='black'  , label='Accuracy', zorder=10 , lw=2)
            for LLH in zip(LLH_tpr, LLH_fpr):
                p1 = plt.scatter(thresholds[np.argwhere(tpr>=LLH[0])[0]], LLH[0],
                                 s=40, marker='o', c=val_1[0].get_color())
                p2 = plt.scatter(thresholds[np.argwhere(tpr>=LLH[0])[0]], 1-LLH[1],
                                 s=40, marker='o', c=val_2[0].get_color())
            l1 = plt.legend([p1, p2], ['LLH '+sig_eff, 'LLH $1\!-\!$'+bkg_eff], loc='lower left', fontsize=13)
            plt.scatter( best_threshold, max(accuracy), s=40, marker='o', c=val_3[0].get_color(),
                         label='{0:<10s} {1:>5.2f}%'.format('Best accuracy:',100*max(accuracy)), zorder=10 )
            plt.legend(loc=location, fontsize=15, numpoints=3); plt.gca().add_artist(l1)
        best_threshold = threshold[np.argmax(accuracy)]
        plt.figure(figsize=(12,16))
        plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
        plt.xlim([0, 1]);   plt.xticks(np.arange(0,1.01,0.1)   , np.arange(0,101,10))
        plt.ylim([0.6, 1]); plt.yticks(np.arange(0.6,1.01,0.05), np.arange(60,101,5))
        make_plot('lower center')
        plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
        x_min=-2; x_max=3; y_min=0.1; y_max=1-1e-4;
        pylab.ylim(y_min, y_max); pylab.xlim(10**x_min, 1-10**(-x_max))
        pos  =                       [  10**float(n)  for n in np.arange(x_min,0)           ]
        pos += [0.5]               + [1-10**float(n)  for n in np.arange(-1,-x_max-1,-1)    ]
        lab  =                       [ '0.'+n*'0'+'1' for n in np.arange(abs(x_min)-3,-1,-1)]
        lab += [1, 10, 50, 90, 99] + ['99.'+n*'9'     for n in np.arange(1,x_max-1)         ]
        plt.xscale('logit'); plt.xticks(pos, lab)
        plt.yscale('logit'); plt.yticks([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999], [10, 50, 90, 99, 99.9, 99.99])
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.xaxis.set_minor_formatter(plt.NullFormatter())
        axes.yaxis.set_minor_formatter(plt.NullFormatter())
        make_plot('upper center')
    if ROC_type == 4:
        best_tpr = tpr[np.argmax(accuracy)]
        plt.xlim([60, 100.0])
        plt.ylim([80, 100.0])
        plt.xticks(np.arange(60,101,step=5))
        plt.yticks(np.arange(80,101,step=5))
        plt.xlabel('Signal Efficiency (%)',fontsize=25)
        plt.ylabel('(%)',fontsize=25)
        plt.plot(100*tpr[1:], 100*(1-fpr[1:]), label='Background rejection', color='darkorange', lw=2)
        val = plt.plot(100*tpr[1:], 100*accuracy[1:], label='Accuracy', color='black', lw=2, zorder=10)
        plt.scatter( 100*best_tpr, 100*max(accuracy), s=40, marker='o', c=val[0].get_color(),
                     label="{0:<10s} {1:>5.2f}%".format('Best accuracy:',100*max(accuracy)), zorder=10 )
        plt.legend(loc='lower center', fontsize=15, numpoints=3)
    file_name = output_dir+'/ROC_'+output_dir.split('class_')[-1]+('_'+str(ROC_type) if ROC_type!=1 else '')+'.png'
    fig.subplots_adjust(left=0.12, top=0.97, bottom=0.12, right=0.94)
    print('Saving ROC curve   plot to:', file_name); plt.savefig(file_name)
    if multiplots:
        for pkl_file in [name for name in os.listdir(output_dir) if '.pkl' in name]:
            os.remove(output_dir+'/'+pkl_file)
        try: os.mkdir(output_dir+'/../ROC_curves')
        except FileExistsError: pass
        plt.savefig(output_dir+'/../ROC_curves/ROC_'+output_dir.split('class_')[-1]+'.png')


def CNN_fpr(tpr, fpr, LLH_tpr):
    """ Getting CNN bkg_eff for a given LLH sig_eff """
    idx1 = np.where(tpr <= LLH_tpr)[0][-1]
    idx2 = np.where(tpr >= LLH_tpr)[0][ 0]
    M = (fpr[idx2]-fpr[idx1])/(tpr[idx2]-tpr[idx1]) if tpr[idx2]!=tpr[idx1] else 0
    return fpr[idx1] + M*(LLH_tpr-tpr[idx1])


def performance_plots(sample, y_true, y_prob, output_dir, ECIDS=False):
    output_dir += '/'+'performance_plots'
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    iter_tuples = itertools.product(['eta', 'pt', 'mu'], ['loose', 'tight'])
    arguments = [(sample, y_true, y_prob, output_dir, *iter_tuple, ECIDS) for iter_tuple in iter_tuples]
    processes = [mp.Process(target=CNNvsLLH, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
def get_eff(sample, y_true, y_prob, threshold, var, wp_idx, bin, ECIDS, return_dict):
    var_array = np.abs(sample[var]) if var=='eta' else sample[var]
    cuts = np.logical_and(var_array>=bin[0], var_array<bin[1])
    data, y_true, y_prob = {key:sample[key][cuts] for key in sample}, y_true[cuts], y_prob[cuts]
    LLH_fpr, LLH_tpr = LLH_rates(data, y_true, ECIDS)
    LLH_fpr, LLH_tpr = LLH_fpr[-3:][wp_idx], LLH_tpr[-3:][wp_idx]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=0)
    idx = np.argmin(np.abs(threshold-thresholds))
    CNN_eff =                     LLH_tpr  if var in ['eta','pt'] else   tpr[idx]
    CNN_rej = 1/CNN_fpr(tpr, fpr, LLH_tpr) if var in ['eta','pt'] else 1/fpr[idx]
    return_dict[bin] = {'n_sig':np.sum(y_true==0), 'LLH_eff':  LLH_tpr, 'CNN_eff':CNN_eff,
                        'n_bkg':np.sum(y_true==1), 'LLH_rej':1/LLH_fpr, 'CNN_rej':CNN_rej}
def CNNvsLLH(sample, y_true, y_prob, output_dir, var, wp, ECIDS, isolation=False):
    eta_bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
    pt_bins  = [4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]
    mu_bins  = [0, 20, 30, 40, 50, 60, 80]
    var_bins = {'eta':eta_bins, 'pt':pt_bins, 'mu':mu_bins}[var]
    bin_tuples = list(zip(var_bins[:-1], var_bins[1:]))
    wp_idx = {'tight':0, 'medium':1, 'loose':2}[wp]
    LLH_fpr, LLH_tpr = LLH_rates(sample, y_true, ECIDS)
    LLH_fpr, LLH_tpr = LLH_fpr[-3:], LLH_tpr[-3:][wp_idx]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=0)
    threshold = thresholds[np.argmin(np.abs(tpr-LLH_tpr))]
    manager   = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample, y_true, y_prob, threshold, var, wp_idx, bin, ECIDS, return_dict) for bin in bin_tuples]
    processes = [mp.Process(target=get_eff, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    n_sig, LLH_eff, CNN_eff = [np.array([return_dict[bin][key] for bin in bin_tuples])
                               for key in ['n_sig', 'LLH_eff', 'CNN_eff']]
    n_bkg, LLH_rej, CNN_rej = [np.array([return_dict[bin][key] for bin in bin_tuples])
                               for key in ['n_bkg', 'LLH_rej', 'CNN_rej']]
    mean, bin_err = [np.mean(bin) for bin in bin_tuples], np.diff(var_bins)/2
    LLH_eff_err = np.sqrt(LLH_eff/n_sig)
    CNN_eff_err = np.sqrt(CNN_eff/n_sig)
    LLH_rej_err = [LLH_rej - 1/(1/LLH_rej + np.sqrt(1/LLH_rej/n_bkg)),
                  -LLH_rej + 1/(1/LLH_rej - np.sqrt(1/LLH_rej/n_bkg))]
    CNN_rej_err = [CNN_rej - 1/(1/CNN_rej + np.sqrt(1/CNN_rej/n_bkg)),
                  -CNN_rej + 1/(1/CNN_rej - np.sqrt(1/CNN_rej/n_bkg))]
    sig_eff, bkg_rej = '$ϵ_{\operatorname{sig}}$', '1/$ϵ_{\operatorname{bkg}}$'
    fig1, (ax11,ax12) = plt.subplots(figsize=(8,6), ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    fig2, (ax21,ax22) = plt.subplots(figsize=(8,6), ncols=1, nrows=2, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax11.errorbar(mean, LLH_rej        , xerr=bin_err, yerr=LLH_rej_err        , marker='o',
                  elinewidth=2, linestyle='', ecolor='k',color='k', label='LLH')
    ax11.errorbar(mean, CNN_rej        , xerr=bin_err, yerr=CNN_rej_err        , marker='o',
                  elinewidth=2, linestyle='', ecolor='#3d84bf', color='#3d84bf', label='CNN')
    ax12.errorbar(mean, CNN_rej/LLH_rej, xerr=bin_err, yerr=CNN_rej_err/LLH_rej, marker='o',
                  elinewidth=2, linestyle='', ecolor='#3d84bf', color='#3d84bf'             )
    ax21.errorbar(mean, LLH_eff        , xerr=bin_err, yerr=LLH_eff_err        , marker='o',
                  elinewidth=2, linestyle='', ecolor='k'      , color='k'      , label='LLH')
    ax21.errorbar(mean, CNN_eff        , xerr=bin_err, yerr=CNN_eff_err        , marker='o',
                  elinewidth=2, linestyle='', ecolor='#3d84bf', color='#3d84bf', label='CNN')
    ax22.errorbar(mean, CNN_eff/LLH_eff, xerr=bin_err, yerr=CNN_eff_err/LLH_eff, marker='o',
                  elinewidth=2, linestyle='', ecolor='#3d84bf', color='#3d84bf'             )
    for AX in [ax11, ax12, ax21, ax22]:
        AX.tick_params(which='minor', direction='in', length=4, width=1.5, colors='black',
                         bottom=True, top=True, left=True, right=True)
        AX.tick_params(which='major', direction='in', length=8, width=1.5, colors='black',
                         bottom=True, top=True, left=True, right=True)
        AX.tick_params(axis="x", pad=6, labelsize=14)
        AX.tick_params(axis="y", pad=5, labelsize=14)
        for axis in ['top', 'bottom', 'left', 'right']: AX.spines[axis].set_linewidth(1.5)
        if AX == ax11 or AX == ax21:
            legend_title = r'$\bf ATLAS$ Simulation Internal'+'\n'+r'$\sqrt{s} = 13\,$TeV'+'\n' \
                         + 'HLV$\:\!+\:\!$tracks$\:\!+\:\!$images'+'\n' + wp.title()            \
                         + (' + isolation' if isolation else '')                                \
                         + (' ('+sig_eff+'='+format(LLH_tpr,'.3f')+')' if var=='mu' else '')
            AX.annotate(legend_title, xy=(0.03, 0.95), xycoords='axes fraction', fontsize=15,
                        horizontalalignment='left', verticalalignment='top')
            AX.legend(loc='upper right', frameon=False, fontsize=15, handletextpad=0)
    label_dict = {'eta':'$|\eta|$', 'pt':'$E_\mathrm{T}$ (GeV)', 'mu':r'$\langle \mu \rangle$'}
    if isolation: y11_min, y11_max = (0, 100) if wp=='loose' else (0, 3000)
    else        : y11_min, y11_max = (0, 600) if wp=='loose' else (0, 3000)
    if var == 'eta':
        x_ticks   = np.arange(0, 2.6, 0.5)
        locators  = np.arange(0, 2.6, 0.1)
        n_locators = 2
        y12_ticks = np.arange(2, 9  , 2  )
        if isolation: y21_ticks = np.arange(0.80, 1.10, 0.05) if wp=='loose' else np.arange(0.60, 1.05, 0.05)
        else        : y21_ticks = np.arange(0.84, 1.03, 0.02) if wp=='loose' else np.arange(0.60, 1.05, 0.05)
    elif var == 'pt':
        x_ticks   = [4.5] + list(np.arange(10,110,10))
        locators  = np.arange(0, 101, 2  )
        n_locators = 2
        y12_ticks = np.arange(0, 13 , 4  )
        y21_ticks = np.arange(0.60, 1.20, 0.05) if wp=='loose' else np.arange(0.10, 1.30, 0.1)
        if isolation: y21_ticks = np.arange(0.80, 1.06, 0.05) if wp=='loose' else np.arange(0.10, 1.30, 0.1)
        else        : y21_ticks = np.arange(0.60, 1.20, 0.05) if wp=='loose' else np.arange(0.10, 1.30, 0.1)
    elif var == 'mu':
        x_ticks   = np.arange(0, 90, 10)
        locators  = np.arange(0, 90, 2 )
        n_locators = 5
        y12_ticks = np.arange(4, 8  , 1  )
        if isolation: y21_ticks = np.arange(0.92, 0.99, 0.01) if wp=='loose' else np.arange(0.70, 0.95, 0.05)
        else        : y21_ticks = np.arange(0.90, 0.98, 0.01) if wp=='loose' else np.arange(0.70, 0.95, 0.05)
    y22_ticks = np.arange(0.99, 1.01, 0.01)
    ax11.set_ylabel(bkg_rej, loc='top', fontsize=18, labelpad=0)
    ax11.axis(ymin=y11_min, ymax=y11_max)
    ax11.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax12.set_xlabel(label_dict[var], fontsize=18, loc='right', labelpad=2)
    ax12.set_xlim(x_ticks[0], x_ticks[-1])
    ax12.set_xticks(x_ticks) ; ax12.set_xticklabels(x_ticks)
    ax12.set_ylabel('CNN/LLH', fontsize=15, loc='top')
    ax12.set_ylim(ymin=y12_ticks[0], ymax=y12_ticks[-1])
    ax12.set_yticks(y12_ticks)
    ax12.xaxis.set_minor_locator(FixedLocator(locators))
    ax12.yaxis.set_minor_locator(AutoMinorLocator(n_locators))
    #if var == 'eta' and wp == 'loose':
    #    ax12.arrow(2.42, 4.5, 0, 1.0, head_width=0.05, width=0.01, head_length=0.4,
    #               edgecolor='#3d84bf', facecolor='#3d84bf')
    ax21.set_ylabel(sig_eff, loc='top', fontsize=18, labelpad=0)
    ax21.set_ylim(y21_ticks[0], y21_ticks[-1])
    ax21.set_yticks(y21_ticks)
    ax21.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax22.set_ylabel('CNN/LLH', fontsize=14, loc='top', labelpad=5 )
    ax22.set_xlabel(label_dict[var], fontsize=18, loc='right', labelpad=2)
    ax22.set_xlim(x_ticks[0], x_ticks[-1])
    ax22.set_xticks(x_ticks) ; ax22.set_xticklabels(x_ticks)
    ax22.axis(ymin=y22_ticks[0], ymax=y22_ticks[-1])
    ax22.set_yticks(y22_ticks)
    ax22.xaxis.set_minor_locator(FixedLocator(locators))
    ax22.yaxis.set_minor_locator(AutoMinorLocator(n_locators))
    for Type, FIG in zip(['bkg-rej','sig-eff'],[fig1,fig2]):
        FIG.align_labels()
        file_name = output_dir+'/'+Type+'_'+var+'_'+wp+'.png'
        FIG.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.98, hspace=0.1)
        print('Saving performance    plot to:', file_name); FIG.savefig(file_name)


def performance_ratio(sample, y_true, y_prob, bkg, output_dir, eta_inclusive, output_tag='CNN2LLH'):
    eta_bins = [0, 0.6, 1.15, 1.37, 1.52, 2.01, 2.47] if not eta_inclusive else [0, 5]
    pt_bins  = [4.5, 10, 20, 30, 40, 60, 80, 5000]
    ratios = {wp:ratio_meshgrid(sample, y_true, y_prob, wp, output_dir, eta_bins, pt_bins)
              for wp in ['loose', 'medium', 'tight']}
    pickle_file = 'CNN2LLH_inclusive.pkl' if eta_inclusive else 'CNN2LLH.pkl'
    if output_tag == 'CNN2CNN':
        input_dir = '6c_180m/HLV/0-5000GeV'
        input_dir = output_dir.split('outputs')[0]+'outputs'+'/'+input_dir
        try: input_ratios = pickle.load(open(input_dir+'/'+'class_0vs'+str(bkg).title()+'/'+pickle_file,'rb'))
        except FileNotFoundError: return
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for wp in ratios:
                if len(ratios[wp]) != len(input_ratios[wp]): return
                ratios[wp] = np.nan_to_num(ratios[wp]/input_ratios[wp], nan=0, posinf=0, neginf=0)
    if output_tag == 'CNN2LLH':
        pickle.dump(ratios, open(output_dir+'/'+pickle_file,'wb'))
    vmin = min([np.min(n[n!=0]) for n in ratios.values()])
    vmax = max([np.max(n[n!=0]) for n in ratios.values()])
    if eta_inclusive:
        file_name = output_dir+'/'+output_tag+'_inclusive.png'
        inclusive_meshgrid(eta_bins, pt_bins, ratios, file_name, output_tag, vmin, vmax)
    else:
        for wp in ratios:
            file_name = output_dir+'/'+output_tag+'_'+wp+'.png'
            binned_meshgrid(eta_bins, pt_bins, ratios[wp], file_name, vmin=None, vmax=None)
def ratio_meshgrid(sample, y_true, y_prob, wp, output_dir, eta_bins, pt_bins, ECIDS=False):
    bin_tuples = list(itertools.product(zip(eta_bins[:-1], eta_bins[1:]), zip(pt_bins[:-1], pt_bins[1:])))
    manager   = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample, y_true, y_prob, wp, bin, ECIDS, return_dict) for bin in bin_tuples]
    processes = [mp.Process(target=bkg_eff_ratio, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return np.array([return_dict[bin] for bin in bin_tuples]).reshape(-1,len(pt_bins)-1).T
def bkg_eff_ratio(sample, y_true, y_prob, wp, bin, ECIDS, return_dict):
    wp = {'tight':0, 'medium':1, 'loose':2}[wp]
    eta, pt = np.abs(sample['eta']), sample['pt']
    cut_list = [eta>=bin[0][0], eta<bin[0][1], pt>=bin[1][0], pt<bin[1][1]]
    cuts = np.logical_and.reduce(cut_list)
    data, y_true, y_prob = {key:sample[key][cuts] for key in sample}, y_true[cuts], y_prob[cuts]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=0)
            fpr, tpr    = fpr[::-2][::-1], tpr[::-2][::-1] #for linear interpolation
            LLH_fpr, LLH_tpr = LLH_rates(data, y_true, ECIDS)
            LLH_fpr, LLH_tpr = LLH_fpr[-3:], LLH_tpr[-3:]
            bkg_rej_ratio = LLH_fpr[wp]/CNN_fpr(tpr, fpr, LLH_tpr[wp])
            #print( bin, np.sum(y_true==0), np.sum(y_true==1), thresholds[np.argmin(abs(tpr-LLH_tpr[wp]))],
            #       LLH_tpr[wp], 1/LLH_fpr[wp], 1/CNN_fpr(tpr, fpr, LLH_tpr[wp]) )
        except:
            bkg_rej_ratio = 0
    return_dict[bin] = np.nan_to_num(bkg_rej_ratio, nan=0, posinf=0, neginf=0)
def inclusive_meshgrid(eta_bins, pt_bins, ratios, file_name, tag, vmin=None, vmax=None, color='black'):
    eta = np.arange(0, len(eta_bins))
    pt  = np.arange(0, len( pt_bins))
    #plt.figure(figsize=(6 if tag=='CNN2LLH' else 5.2,7.5))
    plt.figure(figsize=(6 if tag=='CNN2LLH' else 5.4,7.5))
    def make_plot(wp, ratios, idx=None):
        if idx is None: idx = list(ratios.keys()).index(wp) + 1
        ratios = ratios[wp]
        plt.subplot(1,3,idx); ax = plt.gca()
        plt.pcolormesh(eta, pt, ratios, cmap="Blues", vmin=vmin, vmax=vmax)
        plt.xlabel(wp.title(), fontsize=20)
        if idx == 1 and tag == 'CNN2LLH':
            plt.yticks(np.arange(len( pt_bins)), (pt_bins[:-1]+[r'$\infty$']) if pt_bins[-1]>=1e3 else pt_bins)
            plt.ylabel('$E_\mathrm{T}$ (GeV)', fontsize=30, loc='top')
        else:
            plt.yticks(np.arange(len( pt_bins)), len(pt_bins)*[''])
        for x in range(len(eta_bins)-1):
            for y in range(len( pt_bins)-1):
                text = 'Ind' if ratios[y,x]==0 else format(ratios[y,x],'.1f')
                plt.text(x+0.5, y+0.5, text, {'color':color, 'fontsize':20}, ha='center', va='center')
        plt.grid(True, color="grey", lw=1, alpha=1)
        left_axis = True if (idx==1 and tag=='CNN2LLH') else False
        plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False,
                        labelleft=left_axis, left=left_axis, labelsize=18, direction='in', length=5, width=1)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        if pt_bins[-1] >= 1e3 and idx ==1:
            font_sizes = (len(pt_bins)-1)*[18]+[30]
            for tick, size in zip(plt.yticks()[-1], font_sizes): tick.set_fontsize(size)
        if idx ==2:
            plt.title('CNN$\,/\,$LLH' if tag=='CNN2LLH' else 'CNN$\,/\,$FCN', fontsize=25)
    for wp in ratios: make_plot(wp, ratios)
    cbar = plt.colorbar(fraction=0.05, pad=0.06, aspect=100, shrink=1)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.1f') for n in ticks])
    cbar.ax.tick_params(labelsize=15, length=5, width=1.5)
    cbar.outline.set_linewidth(1.5)
    if tag=='CNN2CNN':
        cbar.set_label('Ratio of Background Rejections', fontsize=22, labelpad=8, rotation=90, loc='top')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.subplots_adjust(left=0.03 if tag=='CNN2CNN' else 0.17, right=0.84 if tag=='CNN2CNN' else None,
                        top=0.95, bottom=0.05, wspace=0.06)
    print('Saving ratio meshgrid plot to:', file_name); plt.savefig(file_name)
def binned_meshgrid(eta_bins, pt_bins, ratios, file_name, vmin=None, vmax=None, color='black'):
    eta = np.arange(0, len(eta_bins))
    pt  = np.arange(0, len( pt_bins))
    plt.figure(figsize=(11,7.5)); ax = plt.gca()
    if vmin is None: vmin = np.min(ratios[ratios!=0])
    if vmax is None: vmax = np.max(ratios[ratios!=0])
    plt.pcolormesh(eta, pt, ratios, cmap="Blues", vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(eta_bins)), eta_bins)
    plt.yticks(np.arange(len( pt_bins)), (pt_bins[:-1]+[r'$\infty$']) if pt_bins[-1]>=1e3 else pt_bins)
    for x in range(len(eta_bins)-1):
        for y in range(len(pt_bins)-1):
            text = 'Ind' if ratios[y,x]==0 else format(ratios[y,x],'.1f')
            plt.text(x+0.5, y+0.5, text, {'color':color, 'fontsize':20}, ha='center', va='center')
    plt.grid(True, color="grey", lw=1, alpha=1)
    plt.tick_params(axis='both', which='major', labelsize=18)
    if pt_bins[-1] >= 1e3:
        font_sizes = (len(pt_bins)-1)*[18]+[30]
        for tick, size in zip(plt.yticks()[-1], font_sizes):
            tick.set_fontsize(size)
    plt.xlabel('$|\eta|$'            , fontsize=30, loc='right')
    plt.ylabel('$E_\mathrm{T}$ (GeV)', fontsize=30, loc='top'  )
    ax.tick_params(which='major', direction='in', length=5, width=1, colors='black',
                   bottom=True, top=True, left=True, right=True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    cbar = plt.colorbar(fraction=0.04, pad=0.02)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.1f') for n in ticks])
    cbar.ax.tick_params(labelsize=15, length=5, width=1.5)
    cbar.outline.set_linewidth(1.5)
    cbar.set_label('Ratio of Background Rejections', fontsize=22, labelpad=10, rotation=90, loc='top')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.tight_layout()
    print('Saving ratio meshgrid plot to:', file_name); plt.savefig(file_name)


def ratio_plots(valid_sample, valid_labels, valid_probs, n_etypes, output_dir):
    from utils import make_discriminant, compo_matrix
    channels_dict = {'Zee' :[361106], 'ttbar':[410470], 'Ztautau':[361108], 'Wtaunu':[361102,361105],
                     'JF17':[423300], 'JF35' :[423302], 'JF50'   :[423303], 'Wenu'  :[361100,361103]}
    manager = mp.Manager(); return_dict = manager.dict()
    processes = [mp.Process(target=get_ratios, args=(valid_sample, valid_labels, valid_probs, n_etypes,
                            channels_dict[process], process, return_dict)) for process in channels_dict]
    for task in processes: task.start()
    for task in processes: task.join()
    channels_list = ['Zee', 'Wenu', 'Ztautau', 'Wtaunu', 'ttbar', 'JF17', 'JF35', 'JF50']
    ratios  = {process:return_dict[process][0] for process in channels_list}
    bkg_rej = {process:return_dict[process][1] for process in channels_list}
    ratios = np.vstack([ratios[process]['truth'] for process in ratios] ).T
    #ratios = np.vstack([ratios[process]['pred']-ratios[process]['truth'] for process in ratios] ).T
    #ratios = np.vstack([(ratios[process]['pred']-ratios[process]['truth'])/ratios[process]['truth']
    #                    for process in ratios] ).T
    #bkg_rej = np.vstack([100*(bkg_rej[process]['pred' ]-bkg_rej[process]['none'])/bkg_rej[process]['none']
    #                     for process in bkg_rej]).T
    #bkg_rej = np.vstack([100*(bkg_rej[process]['truth']-bkg_rej[process]['pred'])/bkg_rej[process]['pred']
    #                     for process in bkg_rej]).T
    X_val = [r'Z$\rightarrow$ee'     , r'W$\rightarrow$e$\nu$'     , r'Z$\rightarrow \tau\tau$',
             r'W$\rightarrow \tau\nu$', r't$\bar{\operatorname{t}}$', r'JF17', r'JF35', r'JF50' ]
    #Y_val = [' Combined \nBackground', 'Charge Flip', '   Photon   \nConversion', ' Heavy \nFlavour',
    #         'Light Flavour\n        e$/\gamma$        ', 'Light Flavour\n    Hadron    '       ]
    Y_val = [' Prompt \nElectron', 'Charge Flip', '   Photon   \nConversion', ' Heavy \nFlavour',
             'Light Flavour\n        e$/\gamma$        ', 'Light Flavour\n    Hadron    '       ]
    #plot_meshgrid(X_val, Y_val, bkg_rej, output_dir, vmin=0, vmax=200, prec=0)
    plot_meshgrid(X_val, Y_val, ratios, output_dir, prec=1, vmax=50); sys.exit()
def plot_meshgrid(X_val, Y_val, Z_val, output_dir, prec=2, vmin=None, vmax=None, color='black'):
    X_idx = np.arange(0, len(X_val)+1) - 0.5
    Y_idx = np.arange(0, len(Y_val)+1) - 0.5
    plt.figure(figsize=(12.5,7.5)); ax = plt.gca()
    if vmin is None: vmin = np.min(Z_val[Z_val!=-1])
    else           : vmin = max(vmin, np.min(Z_val[Z_val!=-1]))
    if vmax is None: vmax = np.max(Z_val[Z_val!=-1])
    else           : vmax = min(vmax, np.max(Z_val[Z_val!=-1]))
    plt.pcolormesh(X_idx, Y_idx, Z_val, cmap="Blues"  , edgecolors='grey', vmin=vmin, vmax=vmax)
    #plt.pcolormesh(X_idx, Y_idx, Z_val, cmap="Blues_r", edgecolors='black', vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(X_val)), X_val, rotation=0)
    plt.yticks(np.arange(len(Y_val)), Y_val, rotation=35)
    for x in range(len(X_val)):
        for y in range(len(Y_val)):
            text = 'Ind' if not np.isfinite(Z_val[y,x]) else format(Z_val[y,x],'.'+str(prec)+'f')
            plt.text(x, y, text, {'color':color, 'fontsize':20}, ha='center', va='center')
    plt.grid(True, color='grey', lw=1, alpha=1, which='minor')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Process', fontsize=30, loc='right')
    #plt.ylabel('Background Class', fontsize=30, loc='top'  )
    ax.tick_params(which='major', direction='in', length=5, width=1.5, colors='black',
                   bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="x", pad=5, labelsize=20)
    ax.tick_params(axis="y", pad=0, labelsize=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    cbar = plt.colorbar(fraction=0.04, pad=0.02)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.0f') for n in ticks])
    cbar.ax.tick_params(labelsize=15, length=5, width=1.5)
    cbar.outline.set_linewidth(1.5)
    cbar.set_label('Truth Ratio (%)', fontsize=22, labelpad=10, rotation=90, loc='top')
    #cbar.set_label('Predicted Ratio$-$Truth Ratio (%)', fontsize=22, labelpad=0, rotation=90, loc='top')
    #cbar.set_label('(Predicted Ratio$-$Truth Ratio)/Truth Ratio', fontsize=22, labelpad=10, rotation=90, loc='top')
    #plt.title('Relative Improvement in Background Rejection\n from Agnostic to Predicted Ratios (%)', fontsize=25)
    #plt.title('Relative Improvement in Background Rejection\n from Predicted to Truth Ratios (%)', fontsize=25)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.tight_layout()
    file_name = output_dir+'/'+'class_ratios.png'
    print('Saving meshgrid to:', file_name); plt.savefig(file_name)
def get_bkg_rej(sample, labels, probs, n_etypes, ratios, ratio_type, bkg_rej_dict, sig_list=[0], val=90):
    def bkg_rej(sample, labels, probs, n_etypes, ratios, sig_list, bkg, return_dict):
        from utils import make_discriminant
        _, labels, probs = make_discriminant(sample, labels, probs, n_etypes, sig_list, bkg, ratios)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fpr, tpr, _ = metrics.roc_curve(labels, probs[:,0], pos_label=0)
            fpr, tpr    = fpr[::-2][::-1], tpr[::-2][::-1] #for linear interpolation
            #return_dict[bkg] = np.nan_to_num(1/fpr[np.argwhere(tpr>=val/100)[0]][0], nan=1.)
            return_dict[bkg] = 1/fpr[np.argwhere(tpr>=val/100)[0]][0]
    bkg_list  = ['bkg'] + list(set(np.arange(n_etypes))-set(sig_list))
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample, labels, probs, n_etypes, ratios, sig_list, bkg, return_dict) for bkg in bkg_list]
    processes = [mp.Process(target=bkg_rej, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    bkg_rej_dict[ratio_type] = np.array([return_dict[bkg] for bkg in bkg_list])
def get_ratios(sample, labels, probs, n_etypes, channels, process, return_dict):
    cuts = np.logical_or.reduce([sample['mcChannelNumber']==n for n in channels])
    labels, probs, sample = labels[cuts], probs[cuts], {key:sample[key][cuts] for key in sample}
    from utils import compo_matrix
    class_ratios = compo_matrix(labels, labels, probs, n_etypes)
    manager = mp.Manager(); bkg_rej_dict = manager.dict()
    arguments = [(sample, labels, probs, n_etypes, class_ratios[ratio_type], ratio_type, bkg_rej_dict)
                 for ratio_type in ['none', 'pred', 'truth']]
    processes = [mp.Process(target=get_bkg_rej, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    bkg_rej = {ratio_type:bkg_rej_dict[ratio_type] for ratio_type in ['none', 'pred', 'truth']}
    #Removing class 0 and renormalizing ratio vectors
    #class_ratios = {key:100*np.array(val[1:]/np.sum(val[1:])) for key,val in class_ratios.items()}
    return_dict[process] = class_ratios, bkg_rej


def combine_ROC_curves(output_dir, cuts=''):
    import multiprocessing as mp, pickle
    from scipy.interpolate import make_interp_spline
    from utils import NN_weights
    def mp_roc(idx, output_dir, return_dict):
        result_file = output_dir+'/'+'results_'+str(idx)+'.pkl'
        sample, labels, probs = pickle.load(open(result_file, 'rb'))
        #cuts = (sample["p_et_calo"] >= 0) & (sample["p_et_calo"] <= 500)
        if cuts == '': cuts = len(labels)*[True]
        sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]
        fpr, tpr, threshold = metrics.roc_curve(labels, probs[:,0], pos_label=0)
        LLH_fpr, LLH_tpr = LLH_rates(sample, labels)
        print('LOADING VALIDATION RESULTS FROM', result_file)
        return_dict[idx] = fpr, tpr, threshold, LLH_fpr, LLH_tpr
    manager  = mp.Manager(); return_dict = manager.dict()
    idx_list = [1, 2, 3, 4]
    names    = ['no weight', 'flattening', 'match2s', 'match2max']
    colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
    lines    = ['-', '-', '-', '-', '--', '--']
    processes = [mp.Process(target=mp_roc, args=(idx, output_dir, return_dict)) for idx in idx_list]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC_curve.png'
    x_min = []; y_max = []
    for n in np.arange(len(idx_list)):
        fpr, tpr, threshold, LLH_fpr, LLH_tpr = return_dict[idx_list[n]]; len_0  = np.sum(fpr==0)
        x_min += [min(60, 10*np.floor(10*LLH_tpr[0]))]
        y_max += [100*np.ceil(max(1/fpr[np.argwhere(tpr >= x_min[-1]/100)[0]], 1/LLH_fpr[0])/100)]
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], color=colors[n], label=names[n], linestyle=lines[n], lw=2)
        #label = str(bkg_class) if bkg_class != 0 else 'others'
        #val   = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='class 0 vs '+label, lw=2)
        #for LLH in zip(LLH_tpr, LLH_fpr): plt.scatter(100*LLH[0], 1/LLH[1], s=40, marker='o', c=val[0].get_color())
    plt.xlim([min(x_min), 100]); plt.ylim([1, 250])  #plt.ylim([1, max(y_max)])
    axes.xaxis.set_major_locator(MultipleLocator(10))
    axes.yaxis.set_ticks( np.append([1], np.arange(50,300,50)) )
    plt.xlabel('Signal Efficiency (%)',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25); #plt.yscale("log")
    plt.legend(loc='upper right', fontsize=15, numpoints=3)
    plt.savefig(file_name); sys.exit()
    '''
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC1_curve.png'
    for LLH_tpr in [0.7, 0.8, 0.9]:
        bkg_rej  = [1/return_dict[idx][0][np.argwhere(return_dict[idx][1] >= LLH_tpr)[0]][0]
                    for idx in idx_list]
        #bkg_rej /= np.mean(bkg_rej)
        #n_weights = [NN_weights((5,13), CNN_dict, [200, 200], 2) for idx in idx_list]
        #bkg_rej   = [1e5*bkg_rej[n]/n_weights[n] for n in np.arange(len(idx_list))]
        plt.scatter(idx_list, bkg_rej, s=40, marker='o')
        idx_array    = np.linspace(min(idx_list), max(idx_list), 1000)
        spline       = make_interp_spline(idx_list, bkg_rej, k=2)
        plt.plot(idx_array, spline(idx_array), label=format(100*LLH_tpr,'.0f')+'% sig. eff.', lw=2)
    plt.xlim([min(idx_list)-1, max(idx_list)+1])
    axes.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim([0, 1500])
    axes.yaxis.set_major_locator(MultipleLocator(100))
    plt.xlabel('Maximum Number of Tracks',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25)
    plt.legend(loc='lower center', fontsize=15, numpoints=3)
    plt.savefig(file_name)
    '''


def cal_images(sample, labels, layers, output_dir, mode='random', scale='free', soft=False):
    import multiprocessing as mp
    def get_image(sample, labels, e_class, key, mode, image_dict):
        start_time = time.time()
        if mode == 'random':
            for counter in np.arange(10000):
                index = np.random.choice(np.where(labels==e_class)[0])
                image = abs(sample[key][index])
                if np.max(image) !=0: break
            #print( e_class, key, index )
        if mode == 'mean': image = np.mean(sample[key][labels==e_class], axis=0)
        if mode == 'std' : image = np.std (sample[key][labels==e_class], axis=0)
        print('plotting layer '+format(key,length+'s')+' for class '+str(e_class), end='', flush=True)
        print(' (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        image_dict[(e_class,key)] = np.float32(image)
    layers    = [layer for layer in layers if layer in sample.keys()]
    classes   = list(np.unique(labels))
    length    = str(max(len(n) for n in layers))
    manager   =  mp.Manager(); image_dict = manager.dict()
    processes = [mp.Process(target=get_image, args=(sample, labels, e_class, key, mode, image_dict))
                 for e_class in classes for key in layers]
    print('PLOTTING CALORIMETER IMAGES (mode='+mode+', scale='+str(scale)+')')
    for job in processes: job.start()
    for job in processes: job.join()
    file_name = output_dir+'/cal_images.png'
    print('SAVING IMAGES TO:', file_name, '\n')
    if   len(layers) == 7: figsize = (18,13)
    elif len(layers) == 8: figsize = (18,15)
    elif len(layers) == 9: figsize = (18,15.5)
    else                 : figsize = (18,21)
    if len(classes) == 2:  figsize = (7,14)
    fig = plt.figure(figsize=figsize)
    for e_class in classes:
        for key in layers:
            #image_dict[(e_class,key)] -= min(0,np.min(image_dict[(e_class,key)]))
            #image_dict[(e_class,key)]  = np.abs(image_dict[(e_class,key)])
            image_dict[(e_class,key)] = np.maximum(1e-8,image_dict[(e_class,key)])
        if scale == 'class':
            vmin = min([np.min(image_dict[(e_class,key)]) for key in layers])
            vmax = max([np.max(image_dict[(e_class,key)]) for key in layers])
        for key in layers:
            if scale == 'layer':
                vmin = min([np.min(image_dict[(e_class,key)]) for e_class in classes])
                vmax = max([np.max(image_dict[(e_class,key)]) for e_class in classes])
            if scale == 'free':
                vmin = np.min(image_dict[(e_class,key)])
                vmax = np.max(image_dict[(e_class,key)])
            plot_image(image_dict[(e_class,key)], classes, e_class, layers, key, vmin, vmax, soft)
    wspace = -0.1; hspace = 0.4
    #fig.subplots_adjust(left=0.02, top=0.97, bottom=0.05, right=0.98, hspace=hspace, wspace=wspace)
    fig.subplots_adjust(left=0.02, top=0.96, bottom=0.05, right=0.99, hspace=hspace, wspace=wspace)
    fig.savefig(file_name); sys.exit()


def plot_image(image, classes, e_class, layers, key, vmin, vmax, soft, log=False):
    image, vmin, vmax = 100*image, 100*vmin, 100*vmax
    if len(classes) <=5:
        class_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion',
                      3:'Heavy Flavour'  , 4:'Light Flavour'}
    else:
        class_dict = {0:'Prompt Electron', 1:'Charge Flip', 2:'Photon Conversion', 3:'Heavy Flavour',
                      4:'Light Flavour e/$\gamma$', 5:'Light Flavour Hadron', 6:'KnownUnknown'}
    layer_dict = {'em_barrel_Lr0'  :'EM Presampler' ,
                  'em_barrel_Lr1'  :'EM Barrel L1'  , 'em_barrel_Lr1_fine':'EM Barrel L1'  ,
                  'em_barrel_Lr2'  :'EM Barrel L2'  , 'em_barrel_Lr3'     :'EM Barrel L3'  ,
                  'tile_gap_Lr1'   :'Tile Gap'      ,
                  'em_endcap_Lr0'  :'EM Presampler' ,
                  'em_endcap_Lr1'  :'EM Endcap L1'  , 'em_endcap_Lr1_fine':'EM Endcap L1'  ,
                  'em_endcap_Lr2'  :'EM Endcap L2'  , 'em_endcap_Lr3'     :'EM Endcap L3'  ,
                  'lar_endcap_Lr0' :'LAr Endcap L0' , 'lar_endcap_Lr1'    :'LAr Endcap L1' ,
                  'lar_endcap_Lr2' :'LAr Endcap L2' , 'lar_endcap_Lr3'    :'LAr Endcap L3' ,
                  'tile_barrel_Lr1':'Tile Barrel L1', 'tile_barrel_Lr2'   :'Tile Barrel L2',
                  'tile_barrel_Lr3':'Tile Barrel L3'}
    n_classes = len(classes)
    class_idx = classes.index(e_class)
    if n_classes == 2: class_dict[1] = 'background'
    e_layer  = layers.index(key)
    n_layers = len(layers)
    plot_idx = n_classes*e_layer + class_idx + 1
    plt.subplot(n_layers, n_classes, plot_idx)
    #title   = class_dict[e_class]+'\n('+layer_dict[key]+')'
    #title   = layer_dict[key]+'\n('+class_dict[e_class]+')'
    if plot_idx-1 < n_classes: title = class_dict[e_class] + '\n' + layer_dict[key]
    else                     : title = layer_dict[key]
    #title = layer_dict[key]
    limits  = [-0.1375, 0.1375, -0.0875, 0.0875]
    x_label = '$\phi$'                             if e_layer   == n_layers-1 else ''
    x_ticks = [limits[0],-0.05,0.05,limits[1]]     if e_layer   == n_layers-1 else []
    y_label = '$\eta$'                             if class_idx == 0          else ''
    y_ticks = [limits[2],-0.05,0.0,0.05,limits[3]] if class_idx == 0          else []
    plt.title(title, fontweight='normal', fontsize=12)
    plt.xlabel(x_label,fontsize=17); plt.xticks(x_ticks)
    plt.ylabel(y_label,fontsize=17); plt.yticks(y_ticks)
    if log:
        plt.imshow(image, cmap='viridis', interpolation='bilinear' if soft else None,
                   extent=limits, norm=colors.LogNorm(1e-3,1e2))
    else:
        plt.imshow(np.float32(image), cmap='viridis', interpolation='bilinear' if soft else None,
                   extent=limits, vmax=1 if np.max(image)==0 else vmax)
    cbar = plt.colorbar(pad=0.02)
    if log:
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-2, 1e-1, 1, 10, 100])


def plot_vertex(sample):
    bins = np.arange(0,50,1)
    fig = plt.figure(figsize=(12,8))
    pylab.xlim(-0.5,10.5)
    plt.xticks (np.arange(0,11,1))
    pylab.ylim(0,100)
    plt.xlabel('Track vertex value', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    weights = len(sample)*[100/len(sample)]
    pylab.hist(sample, bins=bins, weights=weights, histtype='bar', align='left', rwidth=0.5, lw=2)
    file_name = 'outputs/tracks_vertex.png'
    print('Printing:', file_name)
    plt.savefig(file_name)



def plot_scalars(sample, sample_trans, variable):
    bins = np.arange(-1,1,0.01)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    pylab.xlim(-1,1)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    #pylab.hist(sample_trans[variable], bins=bins, histtype='step', density=True)
    pylab.hist(variable, bins=bins, histtype='step', density=False)
    plt.subplot(1,2,2)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    pylab.hist(sample_trans[variable], bins=bins)
    file_name = 'outputs/plots/scalars/'+variable+'.png'
    print('Printing:', file_name)
    plt.savefig(file_name)


def plot_tracks(tracks, labels, variable):
    tracks_var = {'efrac':{'idx':0, 'mean_lim':( 0,      3), 'max_lim':(0,    2), 'diff_lim':(0,    1)},
                  'deta' :{'idx':1, 'mean_lim':( 0, 0.0005), 'max_lim':(0, 0.03), 'diff_lim':(0, 0.04)},
                  'dphi' :{'idx':2, 'mean_lim':( 0,  0.001), 'max_lim':(0,  0.1), 'diff_lim':(0, 0.05)},
                  'd0'   :{'idx':3, 'mean_lim':( 0,    0.2), 'max_lim':(0,  0.1), 'diff_lim':(0,  0.3)},
                  'z0'   :{'idx':4, 'mean_lim':( 0,    0.5), 'max_lim':(0,  0.3), 'diff_lim':(0,   10)}}
    classes    = np.arange(max(labels)+1)
    n_e        = np.arange(len(labels)  )
    n_tracks   = np.sum(abs(tracks), axis=2)
    n_tracks   = np.array([len(np.where(n_tracks[n,:]!=0)[0]) for n in n_e])
    var        = tracks[..., tracks_var[variable]['idx']]
    var_mean   = np.array([np.mean(    var[n,:n_tracks[n]])  if n_tracks[n]!=0 else None for n in n_e])
    var_max    = np.array([np.max (abs(var[n,:n_tracks[n]])) if n_tracks[n]!=0 else None for n in n_e])
    var_diff   = np.array([np.mean(np.diff(np.sort(var[n,:n_tracks[n]])))
                           if n_tracks[n]>=2 else None for n in n_e])
    var_diff   = np.array([(np.max(var[n,:n_tracks[n]]) - np.min(var[n,:n_tracks[n]]))/(n_tracks[n]-1)
                           if n_tracks[n]>=2 else None for n in n_e])
    var_mean   = [var_mean[np.logical_and(labels==n, var_mean!=None)] for n in classes]
    var_max    = [var_max [np.logical_and(labels==n, var_max !=None)] for n in classes]
    var_diff   = [var_diff[np.logical_and(labels==n, var_diff!=None)] for n in classes]
    n_tracks   = [n_tracks[labels==n                                ] for n in classes]
    trk_mean   = [np.mean(n_tracks[n])                                for n in classes]
    fig  = plt.figure(figsize=(18,7))
    xlim = (0, 15)
    bins = np.arange(xlim[0], xlim[1]+2, 1)
    for n in [1,2]:
        plt.subplot(1,2,n); axes = plt.gca()
        plt.xlim(xlim)
        plt.xlabel('Number of tracks'      , fontsize=20)
        plt.xticks( np.arange(xlim[0],xlim[1]+1,1) )
        plt.ylabel('Normalized entries (%)', fontsize=20)
        title = 'Track number distribution (' + str(len(classes)) + '-class)'
        if n == 1: title += '\n(individually normalized)'
        weights = [len(n_tracks[n]) for n in classes] if n==1 else len(classes)*[len(labels)]
        weights = [len(n_tracks[n])*[100/weights[n]] for n in classes]
        plt.title(title, fontsize=20)
        label  =  ['class '+str(n)+' (mean: '+format(trk_mean[n],'3.1f')+')' for n in classes]
        plt.hist([n_tracks[n] for n in classes][::-1], bins=bins, lw=2, align='left',
                 weights=weights[::-1], label=label[::-1], histtype='step')
        plt.text(0.99, 0.05, '(sample: '+str(len(n_e))+' e)', {'color': 'black', 'fontsize': 12},
                 ha='right', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_number.png'; print('Printing:', file_name)
    plt.savefig(file_name)
    fig     = plt.figure(figsize=(22,6)); n = 1
    metrics = {'mean':(var_mean, 'Average'), 'max':(var_max, 'Maximum absolute'),
               'diff':(var_diff, 'Average difference')}
    #metrics = {'mean':(var_mean, 'Average'), 'max':(var_mean, 'Average'),
    #           'diff':(var_mean, 'Average')}
    for metric in metrics:
        plt.subplot(1, 3, n); axes = plt.gca(); n+=1
        n_e    = sum([len(metrics[metric][0][n]) for n in classes])
        x1, x2 = tracks_var[variable][metric+'_lim']
        bins   = np.arange(0.9*x1, 1.1*x2, (x2-x1)/100)
        plt.xlim([x1, x2])
        plt.title (metrics[metric][1] + ' value of ' + str(variable) + '\'s', fontsize=20)
        plt.xlabel(metrics[metric][1] + ' value'                            , fontsize=20)
        plt.ylabel('Normalized entries (%)'                                 , fontsize=20)
        #weights = [len(metrics[metric][0][n])*[100/len(metrics[metric][0][n])] for n in classes]
        weights = [len(metrics[metric][0][n])*[100/n_e] for n in classes]
        plt.hist([metrics[metric][0][n] for n in classes][::-1], weights=weights[::-1], stacked=False,
                 histtype='step', label=['class '+str(n) for n in classes][::-1], bins=bins, lw=2)
        plt.text(0.01, 0.97, '(sample: '+str(n_e)+' e)', {'color': 'black', 'fontsize': 12},
                 ha='left', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_'+str(variable)+'.png'; print('Printing:', file_name)
    plt.savefig(file_name)
