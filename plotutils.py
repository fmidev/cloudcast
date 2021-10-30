import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def plot_convlstm(ground_truth, predictions, mnwc):
    fig, axes = plt.subplots(4, ground_truth.shape[0], figsize=(16, 8), constrained_layout=True)

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

    for idx, ax in enumerate(axes[3]):
        r = ax.imshow(np.squeeze(ground_truth[idx] - predictions[idx]), cmap='bwr')

        if idx == 0:
            plt.colorbar(r, ax=axes[3])
        ax.set_title(f'diff frame {idx}')
        ax.axis('off')

    plt.show()


def plot_mae(data, labels, step=timedelta(minutes=15)):
    print(data)
    print(labels)
    assert(len(data) == len(labels))
    fig = plt.figure()
    ax = plt.axes()

    x = list(map(lambda x: step * x, range(len(data[0]))))
    x = list(map(lambda x: '{}m'.format(int(x.total_seconds() / 60)), x))

    for i,mae in enumerate(data):
        ax.plot(x, mae, label=labels[i])

    plt.legend()
    plt.title(f'mae over {len(data[0])} predictions')
    plt.show()


def plot_timeseries(datas, labels, title=None):
    assert(len(datas) == len(labels))
    nrows = len(datas)
    ncols = datas[0].shape[0]
    fig = plt.figure(figsize=((ncols*1.5),3), constrained_layout=True)
    fig.suptitle(title)
    for i in range(len(datas)):
        for j in range(datas[i].shape[0]):
            ax = fig.add_subplot(nrows, ncols, 1+i+j)
            ax.imshow(np.squeeze(datas[i][j]), cmap='gray_r')
            ax.axis('off')
            ax.set_title(f'{j*15}m')

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

