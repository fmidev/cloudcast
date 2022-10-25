from glob import glob 
from base.plotutils import *
from base.verifutils import *

import sys

def get_label(label):
    lbl = label.split('/')[-1].split('_')
    lbl.pop()
    lbl.pop()
    return reduce_label('_'.join(lbl))
 

def _plot_histogram(directory):
    datas=[]
    labels=[]
    for i in glob(f"{directory}/*_histogram.npy"):
        print(f"Reading {i}")
        labels.append(get_label(i))
        datas.append(np.load(i, allow_pickle=True))
#        print(datas[-1][0])
        print(datas[-1][1])


    assert(len(datas)>0)
    plot_bargraph(datas, labels)


def plot_mae(directory):
    datas=[]
    labels=[]
    for i in glob(f"{directory}/*_mae.npy"):
        print(f"Reading {i}")

        labels.append(get_label(i))
        datas.append(np.load(i))

    assert(len(datas)>0)

    plot_linegraph(datas, labels, title='Mean Absolute Error'.format(datas[0].shape[0]), ylabel='mae', plot_dir= None, start_from_zero=True, add_mean_value_to_label=True)

def _plot_mae2d(directory):

    datas={}
    for i in glob(f"{directory}/*_mae2d.npy"):
        print(f"Reading {i}")
        label = get_label(i)
        data = np.load(i)

        plot_on_map(np.squeeze(data), title=label, plot_dir=None)

def plot(directory, plot_type):
    if plot_type == "histogram":
        _plot_histogram(directory)
    if plot_type == "mae":
        plot_mae(directory)
    if plot_type == "mae2d":
        _plot_mae2d(directory)
#    plot_categorical_scores(directory)
#    plot_ssim(directory)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: {} directory type_of_graph".format(sys.argv[0]))
        print("type: histogram, mae, mae2d")
        sys.exit(1)

    plot(sys.argv[1], sys.argv[2])
