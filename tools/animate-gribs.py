import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import glob
import sys
import subprocess
from base.gributils import *
from base.plotutils import reduce_label

from matplotlib import rcParams

rcParams['animation.convert_path'] = r'/usr/bin/convert'

def extract_labels(directories):
    labels = []
    for d in directories:
        labels.append(reduce_label(d.split('/')[-1]))

    return labels


def read_gribs(directories):
    alldatas = []

    print("Directories: {}".format(directories))
    for directory in directories:
        gribs = glob.glob('{}/*.grib2'.format(directory))

        datas = []
        for gribfile in gribs:
            print("Reading {}".format(gribfile))
            datas.append(np.squeeze(read_grib(gribfile, img_size=(128,128))))

        alldatas.append(datas)

    for x in alldatas[1:]:
        assert(len(x) == len(alldatas[0]))

    print("Number of datas: {}".format(len(alldatas)))
    assert(len(alldatas) > 0)

    return alldatas


def save_anim(filename, datas, labels):
    rows = 1
    cols = len(datas)
    n_frames = len(datas[0])

    if cols > 2:
      rows = 2
      cols = int(0.5 + cols / rows)
 
    print(rows, cols)
    figsize = (9, 6) # (18, 14)
    fig, ax = plt.subplots(rows, cols,figsize=figsize)
    ax = ax.ravel()

    def update(j):
        for i in range(len(datas)):
            ax[i].imshow(datas[i][j], cmap=cm.Greys_r)
            ax[i].set_title("{} {}/{}".format(labels[i], j, n_frames), fontsize=16)
            ax[i].set_axis_off()

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), interval=350, repeat_delay=1000)
    anim.save(filename, dpi=80, writer='imagemagick')
    plt.show()
    plt.close()

    print(f"Saved {n_frames} frame animation to file '{filename}'")


if len(sys.argv) == 1:
    print("Usage: {} anim_file_name.gif path_to_gribs path_to_gribs ...".format(sys.argv[0]))
    sys.exit(1)


filename = sys.argv[1]
labels = extract_labels(sys.argv[2:])
datas = read_gribs(sys.argv[2:])
save_anim(filename, datas, labels)
