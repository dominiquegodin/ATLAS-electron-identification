import numpy as np, h5py, sys, time
import matplotlib ; matplotlib.use('pdf') #matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   matplotlib import pylab
from   sklearn    import metrics


def val_accuracy( y_true, y_prob ):
    y_pred = np.argmax(y_prob, axis=1)
    return sum(y_pred==y_true)/len(y_true)


def get_LLH(test_LLH, y_true):
    eff_class0, eff_class1 = [],[]
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        y_class0 = test_LLH[wp][y_true == 0]
        y_class1 = test_LLH[wp][y_true == 1]
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
    plt.xlim([1,max(history.epoch)+1])
    plt.xticks( np.append(1,np.arange(5,max(history.epoch)+2,step=5)) )
    plt.xlabel('Epochs',fontsize=20)
    plt.ylim( max(80,min_acc),max_acc )
    plt.yticks( np.arange(max(80,min_acc),max_acc+1,step=1) )
    plt.ylabel(key.title()+' (%)',fontsize=20)
    plt.legend(loc='lower right', fontsize=20, numpoints=3)
    plt.savefig(file_name)


def plot_distributions(y_true, y_prob, file_name='outputs/distributions.png'):
    print('CLASSIFIER: saving test sample distributions in:', file_name)
    probs_class0   = 100*y_prob[:,0][ y_true==0 ]
    probs_class1   = 100*y_prob[:,0][ y_true==1 ]
    weights_class0 = len(probs_class0)*[100/len(probs_class0)]
    weights_class1 = len(probs_class1)*[100/len(probs_class1)]
    bins     = np.arange(0, 100, 0.1)
    histtype ='step'
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    pylab.xlim(-0.5,100.5)
    plt.xticks(np.arange(0,101,step=10))
    pylab.hist( probs_class0, bins=bins, label='Signal',
                facecolor='blue', histtype=histtype, weights=weights_class0 )
    pylab.hist( probs_class1, bins=bins, label='Background',
                facecolor='red',  histtype=histtype, weights=weights_class1 )
    plt.xlabel('Signal Probability (%)',fontsize=20)
    plt.ylabel('Distribution (% per '+str(100/len(bins))+'% bin)',fontsize=20)
    plt.legend(loc='upper center', fontsize=20, numpoints=3)
    plt.savefig(file_name)


def plot_ROC_curves(test_sample, y_true, y_prob, ROC_type):
    file_name = 'outputs/ROC'+str(ROC_type)+'_curve.png'
    print('CLASSIFIER: saving test sample ROC'+str(ROC_type)+' curve in:   ', file_name)
    eff_class0, eff_class1 = get_LLH(test_sample, y_true)

    #y_prob = y_prob[np.logical_or(y_true==0, y_true==1)]
    #y_true = y_true[np.logical_or(y_true==0, y_true==1)]

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
        y_max = 100*np.ceil(max(1/fpr[np.argwhere(tpr >= x_min/100)[0]], 1/eff_class1[0])/100)
        plt.xlim([x_min, 100])
        plt.ylim([1,   y_max])
        LLH_scores = [1/fpr[np.argwhere(tpr >= value)[0]] for value in eff_class0]
        for n in np.arange(len(LLH_scores)):
            axes.axhline(LLH_scores[n], xmin=(eff_class0[n]-x_min/100)/(1-x_min/100), xmax=1,
            ls='--', linewidth=0.5, color='#1f77b4')
            axes.axvline(100*eff_class0[n], ymin=abs(1/eff_class1[n]-1)/(plt.yticks()[0][-1]-1),
            ymax=abs(LLH_scores[n]-1)/(plt.yticks()[0][-1]-1), ls='--', linewidth=0.5, color='#1f77b4')
        for val in LLH_scores:
            plt.text(100.2, val, str(int(val)), {'color': '#1f77b4', 'fontsize': 10}, va="center", ha="left")
        axes.yaxis.set_ticks( np.append([1],plt.yticks()[0][1:]) )
        plt.ylabel('1/(Background Efficiency)',fontsize=20)
        val = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='Signal vs Bkg', color='#1f77b4')
        plt.scatter( 100*best_tpr, 1/best_fpr, s=30, marker='D', c=val[0].get_color(),
                     label="{0:<15s} {1:>3.2f}%".format('Best Accuracy:',100*max(accuracy)) )
        for LLH in zip( eff_class0, eff_class1, colors, labels ):
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
        std_accuracy  = val_accuracy(y_true, y_prob)
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
