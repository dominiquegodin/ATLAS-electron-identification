import numpy as np, h5py, sys, time
import matplotlib; matplotlib.use('Agg')
#import matplotlib; matplotlib.use('pdf')
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib import pylab
from   sklearn    import metrics
import os, math, pickle

def valid_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return sum(y_pred==y_true)/len(y_true)


def get_LLH(data, y_true):
    eff_class0, eff_class1 = [],[]
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        y_class0 = data[wp][y_true == 0]
        y_class1 = data[wp][y_true == 1]
        eff_class0.append( np.sum(y_class0 == 0)/len(y_class0) )
        eff_class1.append( np.sum(y_class1 == 0)/len(y_class1) )
    return eff_class0, eff_class1


def plot_history(history, key='accuracy', file_name='outputs/history.png'):
    if len(history.epoch) < 2: return
    print('CLASSIFIER: saving training accuracy history in:', file_name)
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    val = plt.plot(np.array(history.epoch)+1, 100*np.array(history.history[key]), label='Training')
    plt.plot(np.array(history.epoch)+1, 100*np.array(history.history['val_'+key]), '--',
             color=val[0].get_color(), label='Testing')
    min_acc = np.floor(100*min( history.history[key]+history.history['val_'+key] ))
    max_acc = np.ceil (100*max( history.history[key]+history.history['val_'+key] ))
    plt.xlim([1, max(history.epoch)+1])
    plt.xticks( np.append(1,np.arange(5,max(history.epoch)+2,step=5)) )
    plt.xlabel('Epochs',fontsize=20)
    plt.ylim( max(80,min_acc),max_acc )
    plt.yticks( np.arange(max(80,min_acc),max_acc+1,step=1) )
    plt.ylabel(key.title()+' (%)',fontsize=20)
    plt.legend(loc='lower right', fontsize=20, numpoints=3)
    plt.savefig(file_name)


def plot_distributions(y_true, y_prob, var_name='',output_dir='outputs/',postfix=''):
    if var_name=='': var_name='distributions'
    file_name=output_dir+var_name+postfix+'.png'

    print('CLASSIFIER: saving test sample distributions in:', file_name)
    
    if var_name=='distributions':
        probs_class0   = 100*y_prob[:,0][ y_true==0 ]
        probs_class1   = 100*y_prob[:,0][ y_true==1 ]
        bins     = np.arange(0, 100, 0.1)
        pylab.xlim(-0.5,100.5)
        plt.xticks(np.arange(0,101,step=10))
        xstring= 'Signal Probability (%)'
        ystring='Distribution (% per '+str(100/len(bins))+'% bin)'
        legloc='upper center'
    else:
        probs_class0   = y_prob[ y_true==0 ]
        probs_class1   = y_prob[ y_true==1 ]
        bins     = np.arange(-2.5, 2.5, 0.1)
        ystring='arbitrary unit'
        legloc='upper right'
        pass

    if var_name=="eta":
        bins     = np.arange(-2.5, 2.5, 0.1)
        pylab.xlim(-2.5,2.5)
        plt.xticks(np.arange(-2.5,2.5,step=0.1))
        xstring='Eta'
        ystring='arbitrary unit'
    elif var_name=="pt":
        bins     = np.arange(0, 100, 1)
        pylab.xlim(-2.5,2.5)
        plt.xticks(np.arange(0,100,step=5))
        xstring='Pt [GeV]'
        ystring='arbitrary unit'
        pass

    weights_class0 = len(probs_class0)*[100/len(probs_class0)]
    weights_class1 = len(probs_class1)*[100/len(probs_class1)]

    histtype ='step'
    plt.figure(figsize=(12,8))
    pylab.grid(True)

    pylab.hist( probs_class0, bins=bins, label='Signal',
                facecolor='blue', histtype=histtype, weights=weights_class0 )
    pylab.hist( probs_class1, bins=bins, label='Background',
                facecolor='red',  histtype=histtype, weights=weights_class1 )

    plt.xlabel(xstring,fontsize=20)
    plt.ylabel(ystring,fontsize=20)
    plt.legend(title=postfix[1:],loc=legloc, fontsize=15, numpoints=3)

    plt.savefig(file_name)
    plt.close()

def plot_ROC_curves(test_sample, y_true, y_prob, ROC_type, postfix='',output_dir='outputs/'):
    file_name = output_dir
    if postfix!='':file_name+='differential/'
    if not os.path.isdir(file_name): os.mkdir(file_name)
    file_name+= 'ROC'+str(ROC_type)+'_curve'+postfix+'.png'
    print('CLASSIFIER: saving test sample ROC'+str(ROC_type)+' curve in:   ', file_name)
    eff_class0, eff_class1 = get_LLH(test_sample, y_true)

    #y_prob = y_prob[np.logical_or(y_true==0, y_true==1)]
    #y_true = y_true[np.logical_or(y_true==0, y_true==1)]

    #FalsePositveRate, TruePositveRate
    fpr, tpr, threshold    = metrics.roc_curve(y_true, y_prob[:,0], pos_label=0)
    signal_ratio           = len(y_true[y_true==0])/len(y_true)
    accuracy               = tpr*signal_ratio + (1-fpr)*(1-signal_ratio)
    best_tpr, best_fpr     = tpr[np.argmax(accuracy)], fpr[np.argmax(accuracy)]
    colors = [ 'red', 'blue', 'green' ]
    labels = [ 'LLH tight', 'LLH medium', 'LLH loose' ]
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0, 101, 10))
    plt.xlabel('Signal Efficiency (%)',fontsize=20)
    if ROC_type == 1:
        plt.xlim([0, 100])
        plt.ylim([0, 100.5])
        axes.yaxis.set_ticks(np.arange(0, 101, 10))
        plt.ylabel('Background Rejection (%)',fontsize=20)
        plt.text(22, 34, 'AUC: '+str(format(metrics.auc(fpr,tpr),'.4f')),
                {'color': 'black', 'fontsize': 22}, va="center", ha="center")
        val = plt.plot(100*tpr, 100*(1-fpr), label='Signal vs Bkg', color='#1f77b4')
        plt.scatter( 100*best_tpr, 100*(1-best_fpr), s=30, marker='D', c=val[0].get_color(),
                     label="{0:<16s} {1:>3.2f}%".format('Best Accuracy:',100*max(accuracy)) )
        for LLH in zip( eff_class0, eff_class1, colors, labels ):
            plt.scatter( 100*LLH[0], 100*(1-LLH[1]), s=40, marker='o', c=LLH[2], label='('+\
                         str( format(100*LLH[0],'.1f'))+'%, '+str( format(100*(1-LLH[1]),'.1f') )+\
                         ')'+r'$\rightarrow$'+LLH[3] )
        plt.legend(loc='lower left', fontsize=15, numpoints=3)
        plt.savefig(file_name)
    if ROC_type == 2:
        pylab.grid(False)
        len_0 = len(fpr[fpr==0])
        x_min = min(60, 10*np.floor(10*eff_class0[0]))
        y_max = 10000
        if fpr[np.argwhere(np.diff(np.sign(tpr-x_min/100)))[0][0]]>0 and eff_class1[0]>0:
            y_max = 100*np.ceil(max(1/fpr[np.argwhere(np.diff(np.sign(tpr-x_min/100)))[0][0]], 1/eff_class1[0])/100)
        elif fpr[np.argwhere(np.diff(np.sign(tpr-x_min/100)))[0][0]]>0:
            y_max = 100*np.ceil(1/fpr[np.argwhere(np.diff(np.sign(tpr-x_min/100)))[0][0]]/100)
        elif eff_class1[0]>0:
            y_max = 100*np.ceil(1/eff_class1[0]/100)
            pass
        plt.xlim([x_min, 100])
        plt.ylim([1,   y_max])
        LLH_scores = [1/fpr[np.argwhere(tpr >= value)[0]] for value in eff_class0]
        for n in np.arange(len(LLH_scores)):
            axes.axhline(LLH_scores[n], xmin=(eff_class0[n]-x_min/100)/(1-x_min/100), xmax=1,
                         ls='--', linewidth=0.5, color='#1f77b4')
            if eff_class0[n]>0 and eff_class1[n]>0:
                axes.axvline(100*eff_class0[n], ymin=abs(1/eff_class1[n]-1)/(plt.yticks()[0][-1]-1),
                             ymax=abs(LLH_scores[n]-1)/(plt.yticks()[0][-1]-1), ls='--', linewidth=0.5, color='#1f77b4')
        for val in LLH_scores:
            plt.text(100.2, val, str(int(val)), {'color': '#1f77b4', 'fontsize': 10}, va="center", ha="left")
        axes.yaxis.set_ticks( np.append([1],plt.yticks()[0][1:]) )
        plt.ylabel('1/(Background Efficiency)',fontsize=20)
        val = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='Signal vs Bkg ' + postfix[1:]+(" %d e"%len(y_true)), color='#1f77b4',)
        plt.scatter( 100*best_tpr, 1/best_fpr, s=30, marker='D', c=val[0].get_color(),
                     label="{0:<15s} {1:>3.2f}%".format('Best Accuracy:',100*max(accuracy)) )
        for LLH in zip( eff_class0, eff_class1, colors, labels ):
            if LLH[0]>0 and LLH[1]>0:
                plt.scatter( 100*LLH[0], 1/LLH[1], s=40, marker='o', c=LLH[2], label='('+\
                             str(format(100*LLH[0],'.1f'))+'%, '+str(format(1/LLH[1],'.0f'))+\
                             ')'+r'$\rightarrow$'+LLH[3] )
        plt.legend(loc='upper right', fontsize=15, numpoints=3)
        plt.savefig(file_name)
    if ROC_type == 3:
        best_threshold = threshold[np.argmax(accuracy)]
        plt.xlim([0, 100])
        plt.ylim([max(50, 10*np.round(min(10*accuracy))),10*np.ceil(10*max(accuracy))])
        plt.xlabel('Discrimination Threshold (%)',fontsize=30)
        plt.ylabel('Accuracy (%)',fontsize=30)
        val = plt.plot(100*threshold[1:], 100*accuracy[1:], color='#1f77b4')
        #plt.plot( 100*threshold[1:], 100*tpr[1:], color='r')
        #plt.plot( 100*threshold[1:], 100*(1-fpr[1:]), color='g')
        std_accuracy  = test_accuracy(y_true, y_prob)
        std_threshold = np.argwhere(np.diff(np.sign(accuracy-std_accuracy))).flatten()
        #std_threshold = np.argwhere(accuracy >= std_accuracy)[0].flatten()
        plt.scatter( [ 50], #100*threshold[std_threshold[-1]] ],
                     [ 100*accuracy [std_threshold[-1]] ],
                     s=40, marker='o', c=val[0].get_color(),
                     label="{0:<17s} {1:>2.2f}%".format('Accuracy at 50%:',100*accuracy[std_threshold[-1]]) )
        plt.scatter( 100*best_threshold, 100*max(accuracy), s=40, marker='D', c=val[0].get_color(),
                     label="{0:<16s} {1:>8.2f}%".format('Best Accuracy:',100*max(accuracy)) )
        plt.legend(loc='lower center', fontsize=15, numpoints=3)
        plt.savefig(file_name)
        pass
    plt.close()
        
def differential_plots(test_LLH, y_true, y_prob, boundaries, bin_indices,varname,output_dir='outputs/'):

    tmp_idx=0

    x_centers = list()
    x_errs    = list()

    y_prob_sig = y_prob[y_true==0]

    sigEffs    = [0.7,0.8,0.9]
    globalCuts = list()
    bkg_rejs_fEff  = {}
    bkg_errs_fEff  = {}
    bkg_rejs_gEff  = {}
    bkg_errs_gEff  = {}
    sig_effs_gEff  = {}
    sig_errs_gEff  = {}
    for sigEff in sigEffs:
        bkg_rejs_fEff.update({sigEff:[]})
        bkg_errs_fEff.update({sigEff:[]})
        bkg_rejs_gEff.update({sigEff:[]})
        bkg_errs_gEff.update({sigEff:[]})
        sig_effs_gEff.update({sigEff:[]})
        sig_errs_gEff.update({sigEff:[]})
        pCut = np.percentile(y_prob_sig,(1-sigEff)*100,axis=0) [0] #global cut
        globalCuts.append(pCut)
        pass

    for bin_idx in bin_indices:
        if bin_idx.size==0: 
            tmp_idx+=1
            continue
            pass


        fill_rej=False
        pfix ="_"+varname+"%d" % tmp_idx
        if tmp_idx!=0:                  pfix+="_Lo%.2f" % boundaries[tmp_idx-1]         #lo
        if tmp_idx!=len(bin_indices)-1: pfix+="_Hi%.2f" % boundaries[tmp_idx]           #hi
        
        if tmp_idx!=0 and tmp_idx!=len(bin_indices)-1:
            x_center = (boundaries[tmp_idx-1] + boundaries[tmp_idx])/2
            x_centers.append(x_center)
            x_err      = boundaries[tmp_idx]-x_center
            x_errs.append(x_err)
            fill_rej=True
            pass

        new_test_labels=y_true.take(bin_idx)
        new_y_prob     =y_prob.take(bin_idx,axis=0)

        new_test_LLH=dict()
        for llh in test_LLH:
            #print(llh)
            new_test_LLH[llh]=test_LLH[llh][bin_idx]
            pass

        if not (len(new_y_prob)==len(new_test_labels) and len(new_test_labels)==len(new_test_LLH['p_LHTight'])):
            print("data size for data, label, llh= ",len(new_y_prob),len(new_test_labels),len(new_test_LLH['p_LHTight']))

        if not(~np.isnan(new_y_prob).any() and ~np.isinf(new_y_prob).any()): print("Nan or Inf detected")

        plot_ROC_curves(new_test_LLH, new_test_labels, new_y_prob, ROC_type=2, postfix=pfix,output_dir=output_dir)
        #plot_distributions (new_test_labels,new_y_prob,output_dir=output_dir+'differential/',postfix=pfix)

        if fill_rej: 
            fill_bkg_rejs_f(bkg_rejs_fEff,bkg_errs_fEff,
                            new_y_prob,new_test_labels,sigEffs)
            fill_info_g    (bkg_rejs_gEff,bkg_errs_gEff,
                            sig_effs_gEff,sig_errs_gEff,
                            new_y_prob,new_test_labels,sigEffs,globalCuts)
            
        tmp_idx+=1
        pass

#    print(x_centers)
#    print(x_errs)
#    print(bkg_rejs_fEff[.7])
#    print(bkg_errs_fEff[.7])
#    print()
#    print(bkg_rejs_fEff[.8])
#    print(bkg_errs_fEff[.8])
#    print()
#    print(bkg_rejs_fEff[.9])
#    print(bkg_errs_fEff[.9])
#
    plot_rej_vsX_curves(x_centers,x_errs, bkg_rejs_fEff,bkg_errs_fEff, sigEffs,varname,output_dir,boundaries[-1],cType="Flat" ,makeOutput=True)
    plot_rej_vsX_curves(x_centers,x_errs, bkg_rejs_gEff,bkg_errs_gEff, sigEffs,varname,output_dir,boundaries[-1],cType="GlobB",makeOutput=True)
    plot_rej_vsX_curves(x_centers,x_errs, sig_effs_gEff,sig_errs_gEff, sigEffs,varname,output_dir,boundaries[-1],cType="GlobS",makeOutput=True)

    return

def plot_rej_vsX_curves(x_centers,x_errs, 
                        bkg_rejs,bkg_errs, 
                        sigEffs,varname,output_dir,x_up,
                        cType='Flat',makeOutput=False):
    plt.close()
    style.use('classic')
    #ax = plt.figure().add_subplot(111)

    errGraphs = dict()

    for sigEff in sigEffs: 
        plt.errorbar(np.asarray(x_centers), np.asarray(bkg_rejs[sigEff]), 
                     xerr=np.asarray(x_errs), yerr= np.asarray(bkg_errs[sigEff]), 
                     marker='o',capsize=2, linestyle='None',fillstyle='none')
        data_points = [np.asarray(x_centers), 
                       np.asarray(bkg_rejs[sigEff]), 
                       np.asarray(x_errs), 
                       np.asarray(bkg_errs[sigEff])]
        errGraphs[cType+"_%d"%(sigEff*100)]=data_points
        pass

    if   varname.find("eta")>=0:
        pylab.xlim(-2.5,2.5)
        plt.xticks(np.arange(-2.5,2.6,step=0.5))
    elif varname.find("pt")>=0: 
        #pylab.ylim(0,1800)
        pylab.xlim(0,x_up+20)
        plt.xticks(np.arange(0,x_up,step=50))
        pass

        
    ystring='Bkg-rejection'
    if cType.find('GlobS')!=-1: ystring='Sig-efficiency'
    plt.ylabel(ystring,fontsize=15)
    if varname.find('pt')!=-1: plt.xlabel(varname+' [GeV]',fontsize=15)
    else:                      plt.xlabel(varname,         fontsize=15)
    plt.title(cType+" efficiency plot. sig-eff: ",fontweight='bold')

    #KM: some sort of legend
    plt.text(0.75, 1.02, '70%', transform=plt.gca().transAxes, color='b', fontsize=15)
    plt.text(0.85, 1.02, '80%', transform=plt.gca().transAxes, color='g', fontsize=15)
    plt.text(0.95, 1.02, '90%', transform=plt.gca().transAxes, color='r', fontsize=15)

    output_name=output_dir+"rej_vs_"    
    if cType.find('GlobS')!=-1: output_name=output_dir+"eff_vs_"
    output_name+=varname+"_"+cType+".png"
    print(output_name)
    plt.savefig(output_name)
    #plt.close()

    if makeOutput: #pickle.dump(errGraphs, open(output_dir+cType+'_graphs.pickle', 'wb'))
        outfilename=output_dir+cType+'_graphs.pkl'
        print('Writing pickle file:', outfilename)
        pickle.dump(errGraphs,open(outfilename, 'wb'))
        pass

    return

def fill_bkg_rejs_f(bkg_rejs_fEff,bkg_errs_fEff,new_y_prob,new_test_labels,sigEffs):

    for sigEff in sigEffs:
        new_y_prob_sig = new_y_prob[new_test_labels==0]
        new_y_prob_bkg = new_y_prob[new_test_labels==1]
        #KM: get percentile cut value
        pCut = np.percentile(new_y_prob_sig,(1-sigEff)*100,axis=0) [0] #70% from right side --> 1 - sig_eff
        binaryClassified_bkg = new_y_prob_bkg[:,0] > pCut

        bkgRej    = 0
        bkgRejErr = 0
        if binaryClassified_bkg.sum()>0:
            bkgRej = binaryClassified_bkg.size / binaryClassified_bkg.sum()    #inverted bkg-eff as rejection
            bkgRejErr = math.sqrt(bkgRej * (bkgRej-1) / binaryClassified_bkg.sum())
            pass

        bkg_rejs_fEff[sigEff].append(bkgRej)
        bkg_errs_fEff[sigEff].append(bkgRejErr)
        pass
    return

def fill_info_g(bkg_rejs_gEff,bkg_errs_gEff,
                sig_effs_gEff,sig_errs_gEff,
                new_y_prob,new_test_labels,sigEffs,globalCuts):

    for sigEff_target,globCut in zip(sigEffs,globalCuts):
        new_y_prob_sig = new_y_prob[new_test_labels==0]
        new_y_prob_bkg = new_y_prob[new_test_labels==1]
        binaryClassified_bkg = new_y_prob_bkg[:,0] > globCut
        binaryClassified_sig = new_y_prob_sig[:,0] > globCut

        bkgRej    = 0
        bkgRejErr = 0
        sigEff    = 0
        sigEffErr = 0
        if binaryClassified_bkg.sum()>0:
            bkgRej = binaryClassified_bkg.size / binaryClassified_bkg.sum()    #inverted bkg-eff as rejection
            sigEff = binaryClassified_sig.sum()/ binaryClassified_sig.size     #sig-eff
            bkgRejErr = math.sqrt(bkgRej * (bkgRej-1) / binaryClassified_bkg.sum())
            sigEffErr = math.sqrt(sigEff * (1-sigEff) / binaryClassified_sig.size )
            pass

        bkg_rejs_gEff[sigEff_target].append(bkgRej)
        bkg_errs_gEff[sigEff_target].append(bkgRejErr)
        sig_effs_gEff[sigEff_target].append(sigEff)
        sig_errs_gEff[sigEff_target].append(sigEffErr)
        pass
    return


def plot_image(cal_image, n_classes, e_class, images, image):
    #norm_type = None
    norm_type = colors.LogNorm(0.0001,1)
    limits = [-0.13499031, 0.1349903, -0.088, 0.088]
    e_image  = images.index(image)
    n_images = len(images)
    plot_number = n_classes*( e_image ) + e_class + 1
    plt.subplot(n_images, n_classes, plot_number)
    title='Class '+str(e_class)+' - Layer '+ image
    x_label, y_label = '' ,''
    x_ticks, y_ticks = [], []
    if e_image == n_images-1:
        x_label = '$\phi$'
        x_ticks = [limits[0],-0.05,0.05,limits[1]]
    if e_class == 0:
        y_label = '$\eta$'
        y_ticks = [limits[2],-0.05,0.0,0.05,limits[3]]
    plt.title(title,fontweight='bold')
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.imshow(cal_image.transpose(), cmap='Reds', extent=limits, norm=norm_type)
    plt.colorbar(pad=0.02)
    return


def cal_images(files, images, file_name='outputs/cal_images.png'):
    print('\nCLASSIFIER: saving calorimeter images in:', file_name,'\n')
    fig = plt.figure(figsize=(8,12))
    for e_class in np.arange( 0, len(files) ):
        input_file = h5py.File( files[e_class], 'r' )
        e_number   = np.random.randint( 0, len(input_file['data']), size=1 )[0]
        for image in images: plot_image( input_file['data/table_'+str(e_number)][image][0],
                                         len(files), e_class, images, image )
    hspace, wspace = 0.4, -0.6
    fig.subplots_adjust(left=-0.4, top=0.95, bottom=0.05, right=0.95, hspace=hspace, wspace=wspace)
    fig.savefig(file_name)
    plt.show() ; sys.exit()


def plot_scalars(sample, sample_trans, variable):
    bins = np.arange(100)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    pylab.hist(sample_trans[variable], bins=bins, histtype='step', density=True)
    pylab.hist(sample      [variable], bins=bins, histtype='step', density=True)
    plt.subplot(1,2,2)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    pylab.hist(sample_trans[variable], bins=bins)
    file_name = 'outputs/scalars/'+variable+'.png'
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
    file_name = 'outputs/tracks_number.png'; print('Printing:', file_name)
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
    file_name = 'outputs/tracks_'+str(variable)+'.png'; print('Printing:', file_name)
    plt.savefig(file_name)
