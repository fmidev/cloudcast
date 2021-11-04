import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def plot_convlstm(ground_truth, predictions, mnwc):
    fig, axes = plt.subplots(3, ground_truth.shape[0], figsize=(16, 7), constrained_layout=True)

    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(ground_truth[idx]), cmap='gray_r')
        ax.set_title(f'ground truth frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(predictions[idx]), cmap='gray_r')
        ax.set_title(f'prediction frame {idx}')
        ax.axis('off')

    for idx, ax in enumerate(axes[2]):
        ax.imshow(np.squeeze(mnwc[idx]), cmap='gray_r')
        ax.set_title(f'mnwc frame {idx}')
        ax.axis('off')

    plt.show()


def plot_mae(data, labels, step=timedelta(minutes=15), title=None):
    print(data)
    print(labels)
    assert(len(data) == len(labels))
    fig = plt.figure()
    ax = plt.axes()

    x = list(map(lambda x: step * x, range(1, 1+len(data[0]))))
    x = list(map(lambda x: '{}m'.format(int(x.total_seconds() / 60)), x))

    for i,mae in enumerate(data):
        ax.plot(x, mae, label=labels[i])

    plt.legend()
    plt.title(title)
    plt.show()


def plot_timeseries(datas, labels, title=None):
    assert(len(datas) == len(labels))
    nrows = len(datas)
    ncols = datas[0].shape[0]
    print(f'nrows={nrows},ncols={ncols}')
    #fig = plt.figure(figsize=((ncols*1.5),nrows*1.5), constrained_layout=True)
    fig, bigaxes = plt.subplots(nrows=nrows, ncols=1, figsize=((ncols*2),nrows*2), constrained_layout=False, squeeze=False)
    fig.suptitle(title)
    for i, bigax in enumerate(bigaxes.flatten(), start=0):
        bigax.set_title(labels[i])
        bigax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        bigax._frameon = False
        bigax.axis('off')

    num=1
    for i in range(len(datas)):
        for j in range(datas[i].shape[0]):
            ax = fig.add_subplot(nrows, ncols, num)
            num += 1
            ax.imshow(np.squeeze(datas[i][j]), cmap='gray_r')
            ax.axis('off')
            if i == (nrows - 1):
                ax.set_title(f'{(j+1)*15}m', y=0, pad=-25)

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.show()


def plot_hist(hist, model_dir):
    print(hist.history)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/accuracy.png'.format(model_dir))

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}/loss.png'.format(model_dir))

