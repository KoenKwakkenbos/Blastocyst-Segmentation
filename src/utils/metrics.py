import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def save_loss_curve(output_path, results, i_fold):
    epochs = range(len(results.history["val_loss"]))
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(epochs, results.history["binary_io_u"], label = 'training')
    ax1.plot(epochs, results.history["val_binary_io_u"], label = 'validation')
    ax1.set(xlabel = 'Epochs', ylabel ='Jaccard Index')
    ax1.legend()

    ax2.plot(epochs, results.history["loss"], label = 'training')
    ax2.plot(epochs, results.history["val_loss"], label = 'validation')
    ax2.set(xlabel = 'Epochs',ylabel = 'Loss')
    ax2.set_ylim(0, 0.45)
    ax2.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_path, 'val_acc_epochs' + str(i_fold+1) + '.png'))
    plt.close()
