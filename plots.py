import numpy as np, h5py, sys
import matplotlib.pyplot as plt
from   matplotlib import pylab


def accuracy( y_true, y_prob ):
    y_pred = np.argmax(y_prob, axis=1)
    return sum(y_pred==y_true)/len(y_true)


def plot_accuracy(history, key='accuracy', file_name='outputs/accuracy.png'):
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
    plt.ylim(min_acc,max_acc)
    plt.yticks(np.arange(min_acc,max_acc+1,step=1))
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
    pylab.xlim(0,100)
    plt.xticks(np.arange(0,101,step=10))
    pylab.hist( probs_class0, bins=bins, label='Signal',
                facecolor='blue', histtype=histtype, weights=weights_class0 )
    pylab.hist( probs_class1, bins=bins, label='Background',
                facecolor='red',  histtype=histtype, weights=weights_class1 )
    plt.xlabel('Signal Probability (%)',fontsize=20)
    plt.ylabel('Distribution (% per '+str(100/len(bins))+'% bin)',fontsize=20)
    plt.legend(loc='upper center', fontsize=20, numpoints=3)
    plt.savefig(file_name)


def get_LLH(files, indices):
    from utils import load_files
    sample  = load_files(files, indices, indices.size, index=0)
    y_true  = sample['truthmode']
    eff_class0, rej_class1 = [],[]
    for wp in ['llh_tight', 'llh_medium', 'llh_loose']:
        y_LLH   = sample[wp]
        y_class0 = y_LLH[y_true == 0]
        y_class1 = y_LLH[y_true == 1]
        eff_class0.append( len(y_class0[y_class0==0])/len(y_class0) )
        rej_class1.append( len(y_class1[y_class1!=0])/len(y_class1) )
    return eff_class0, rej_class1


def plot_ROC1_curve(files, indices, y_true, y_prob, file_name='outputs/ROC1_curve.png'):
    from sklearn.metrics import roc_curve, auc
    print('CLASSIFIER: saving test sample ROC curve in:    ', file_name)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:,0], pos_label=0)
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    plt.xlim([0, 100.5])
    plt.ylim([0, 100.5])
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0, 101, 10))
    axes.yaxis.set_ticks(np.arange(0, 101, 10))
    plt.xlabel('Signal Efficiency (%)',fontsize=20)
    plt.ylabel('Background Rejection (%)',fontsize=20)
    plt.text(24, 34, 'AUC: '+str(format(auc(fpr,tpr),'.4f')),
             {'color': 'black', 'fontsize': 24}, va="center", ha="center")
    plt.plot(100*tpr, 100*(1-fpr), label='Signal vs Fake+Bkg', color='#1f77b4')
    eff_class0, rej_class1 = get_LLH( files, indices )
    colors = [ 'green', 'blue', 'red' ]
    labels = [ 'LLH tight:       ', 'LLH medium: ', 'LLH loose:      ' ]
    for LLH in zip( eff_class0, rej_class1, colors, labels ):
        plt.scatter( 100*LLH[0], 100*LLH[1], s=40, marker='o', c=LLH[2],
        label=LLH[3]+'('+str(format(LLH[0],'.3f'))+', '+str(format(LLH[1],'.3f'))+')' )
    plt.legend(loc='lower left', fontsize=17, numpoints=3)
    plt.savefig(file_name)


def plot_ROC2_curve(files, indices, y_true, y_prob, file_name='outputs/ROC2_curve.png'):
    from sklearn.metrics import roc_curve, auc
    print('CLASSIFIER: saving test sample ROC curve in:    ', file_name,'\n')
    fpr, tpr, _ = roc_curve(y_true, y_prob[:,0], pos_label=0)#, drop_intermediate=False)
    fpr[0:len(fpr[fpr==0])] = fpr[len(fpr[fpr==0])+1]
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    plt.xlim([0, 100])
    plt.ylim([0, 1.1*max(1/fpr)])
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0, 101, 10))
    plt.xlabel('Signal Efficiency (%)',fontsize=20)
    plt.ylabel('1/(Background Efficiency)',fontsize=20)
    plt.text(15, 750, 'AUC: '+str(format(auc(tpr,1/fpr),'.0f')),
             {'color': 'black', 'fontsize': 24}, va="center", ha="center")
    plt.plot(100*tpr, 1/fpr, label='Signal vs Fake+Bkg', color='#1f77b4')
    eff_class0, rej_class1 = get_LLH( files, indices )
    colors = [ 'green', 'blue', 'red' ]
    labels = [ 'LLH tight:       ', 'LLH medium: ', 'LLH loose:      ' ]
    for LLH in zip( eff_class0, rej_class1, colors, labels ):
        plt.scatter( 100*LLH[0], 1/(1-LLH[1]), s=40, marker='o', c=LLH[2],
        label=LLH[3]+'('+str(format(LLH[0],'.3f'))+', '+str(format(1/(1-LLH[1]),'.0f'))+')' )
    plt.legend(loc='upper right', fontsize=17, numpoints=3)
    plt.savefig(file_name)


def plot_image(cal_image, n_classes, e_class, images, image):
    norm_type = None #norm_type = LogNorm(0.0001, 1)
    limits = [-0.13499031, 0.1349903, -0.088, 0.088]
    e_image  = images.index(image)
    n_images = len(images)
    plot_number = n_classes*( e_image ) + e_class + 1
    plt.subplot(n_images, n_classes, plot_number)
    title='Class '+str(e_class)+' - Layer '+ image
    x_label, y_label = '' ,'' #None, None
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
    plt.colorbar()
    return


def cal_images(files, images, file_name='outputs/cal_images.png'):
    print('\nCLASSIFIER: saving calorimeter images in:', file_name,'\n')
    fig = plt.figure(figsize=(12,12))
    for e_class in np.arange( 0, len(files) ):
        input_file = h5py.File( files[e_class], 'r' )
        e_number   = np.random.randint( 0, len(input_file['data']), size=1 )[0]
        for image in images: plot_image( input_file['data/table_'+str(e_number)][image][0],
                                         len(files), e_class, images, image )
    hspace, wspace = 0.4, -0.4
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    fig.savefig(file_name)
    plt.show() ; sys.exit()
