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


def plot_mae(data, labels, step=timedelta(minutes=15), title=None, xvalues=None):
    assert(len(data) == len(labels))
    fig = plt.figure(figsize=(12,7))
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    xreal = np.asarray(range(len(data[0])))

    if xvalues is None:
        xlabels = list(map(lambda x: step * x, range(1, 1+len(data[0]))))
        xlabels = list(map(lambda x: '{}m'.format(int(x.total_seconds() / 60)), xlabels))
    else:
        xlabels = list(map(lambda x: '{}m'.format(int(x * 15)), xvalues))

    labels = list(map(lambda x: x.replace('True','T')
                                 .replace('False','F')
                                 .replace('binary_crossentropy', 'bc')
                                 .replace('MeanSquaredError', 'MSE'), labels))

    for i,mae in enumerate(data):
        mae = np.asarray(mae)
        x = xreal[np.isfinite(mae)]
        y = mae[np.isfinite(mae)]
        ax.plot(x, y, label=labels[i], linestyle='-', marker='o')

    ax.set_xticks(xreal)
    ax.set_xticklabels(xlabels)
    plt.title(title)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show(block=False)


def plot_timeseries(datas, labels, title=None, initial_data=None, start_from_zero=False):
    assert(len(datas) == len(labels))
    nrows = len(datas)
    ncols = datas[0].shape[0]
    if initial_data is not None:
        nrows += 1
        labels = ['initial'] + labels

    print(f'nrows={nrows},ncols={ncols}')
    fig, bigaxes = plt.subplots(nrows=nrows, ncols=1, figsize=((ncols*2),nrows*2), constrained_layout=False, squeeze=False)
    fig.suptitle(title)
    for i, bigax in enumerate(bigaxes.flatten(), start=0):
        bigax.set_title(labels[i])
        bigax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        bigax._frameon = False
        bigax.axis('off')

    num=1
    write_time=nrows-1

    if initial_data is not None:
        ax = fig.add_subplot(nrows, ncols, num)
        ax.imshow(np.squeeze(initial_data), cmap='gray_r')
        ax.axis('off')
        num=ncols+1
        write_time=nrows-2

    offset = 0 if start_from_zero else 1

    for i in range(len(datas)):
        for j in range(datas[i].shape[0]):
            ax = fig.add_subplot(nrows, ncols, num)
            num += 1
            ax.imshow(np.squeeze(datas[i][j]), cmap='gray_r')
            ax.axis('off')
            if i == write_time:
                ax.set_title(f'{(j+offset)*15}m', y=0, pad=-25)

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.show(block=False)


def plot_hist(hist, model_dir = None, show=False):
    print(hist)
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if show:
        plt.show()
    else:
        plt.savefig('{}/accuracy.png'.format(model_dir))

    plt.close()
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if show:
        plt.show()
    else:
        plt.savefig('{}/loss.png'.format(model_dir))

