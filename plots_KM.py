import numpy as np, h5py, sys, time
import matplotlib; matplotlib.use('Agg')
#import matplotlib; matplotlib.use('pdf')
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib import pylab
from   sklearn    import metrics
import os, math, pickle


def get_LLH(data, y_true):
    eff_class0, eff_class1 = [],[]
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        y_class0 = data[wp][y_true == 0]
        y_class1 = data[wp][y_true == 1]
        eff_class0.append( np.sum(y_class0 == 0)/len(y_class0) )
        eff_class1.append( np.sum(y_class1 == 0)/len(y_class1) )
    return eff_class0, eff_class1


def plot_distributions_KM(y_true, y_prob, var_name='',output_dir='outputs/',postfix=''):
    if var_name=='': var_name='distributions'
    file_name=output_dir+'/'+var_name+postfix+'.png'

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

    print('CLASSIFIER:', output_dir)

    file_name = output_dir
    #if postfix!='':file_name+='differential/'
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
        LLH_scores = [1/fpr[np.argwhere(tpr>=value)[0]] for value in eff_class0
                      if fpr[np.argwhere(tpr>=value)[0]]!=0]
        #LLH_scores = [10,10,10]#[1/fpr[np.argwhere(tpr >= value)[0]] for value in eff_class0]
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

def differential_plots(test_LLH, y_true, y_prob, boundaries, bin_indices,varname='pt',output_dir='outputs/',evalLLH=False):

    plot_ROC_curves(test_LLH, y_true, y_prob, ROC_type=2,output_dir=output_dir+"/differential/")

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
        if evalLLH: pfix+='LLH'


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
        #for llh in test_LLH:
        for llh in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
            #print(llh)
            new_test_LLH[llh]=test_LLH[llh][bin_idx]
            pass

        if not (len(new_y_prob)==len(new_test_labels) and len(new_test_labels)==len(new_test_LLH['p_LHTight'])):
            print("data size for data, label, llh= ",len(new_y_prob),len(new_test_labels),len(new_test_LLH['p_LHTight']))

        if not(~np.isnan(new_y_prob).any() and ~np.isinf(new_y_prob).any()): print("Nan or Inf detected")

        plot_ROC_curves(new_test_LLH, new_test_labels, new_y_prob, ROC_type=2, postfix=pfix,output_dir=output_dir+'/differential/')
        #plot_distributions_KM(new_test_labels,new_y_prob,output_dir=output_dir+'differential/',postfix=pfix)

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
    plot_rej_vsX_curves(x_centers,x_errs, bkg_rejs_fEff,bkg_errs_fEff, sigEffs,varname,output_dir,boundaries[-1],cType="Flat" ,makeOutput=True,evalLLH=evalLLH)
    plot_rej_vsX_curves(x_centers,x_errs, bkg_rejs_gEff,bkg_errs_gEff, sigEffs,varname,output_dir,boundaries[-1],cType="GlobB",makeOutput=True,evalLLH=evalLLH)
    plot_rej_vsX_curves(x_centers,x_errs, sig_effs_gEff,sig_errs_gEff, sigEffs,varname,output_dir,boundaries[-1],cType="GlobS",makeOutput=True,evalLLH=evalLLH)

    return

def plot_rej_vsX_curves(x_centers,x_errs,
                        bkg_rejs,bkg_errs,
                        sigEffs,varname,output_dir,x_up,
                        cType='Flat',makeOutput=False,evalLLH=False):
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
    
    title_str = cType
    if evalLLH: title_str+= "LLH"
    title_str += " efficiency plot. sig-eff: "
    plt.title(title_str,fontweight='bold') #plt.title(cType+" efficiency plot. sig-eff: ",fontweight='bold')


    #KM: some sort of legend
    plt.text(0.75, 1.02, '70%', transform=plt.gca().transAxes, color='b', fontsize=15)
    plt.text(0.85, 1.02, '80%', transform=plt.gca().transAxes, color='g', fontsize=15)
    plt.text(0.95, 1.02, '90%', transform=plt.gca().transAxes, color='r', fontsize=15)

    output_name=output_dir+'/'+"rej_vs_"
    if cType.find('GlobS')!=-1: output_name=output_dir+'/'+"eff_vs_"
    output_name+=varname+"_"+cType
    if evalLLH: output_name+="LLH"
    output_name+=".png"    #output_name+=varname+"_"+cType+".png"
    print(output_name)
    plt.savefig(output_name)
    #plt.close()

    if makeOutput: #pickle.dump(errGraphs, open(output_dir+cType+'_graphs.pickle', 'wb'))
        outfilename=output_dir+'/'+cType
        if evalLLH: outfilename+="LLH"
        outfilename+='_graphs.pkl'    #outfilename=output_dir+cType+'_graphs.pkl'
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
