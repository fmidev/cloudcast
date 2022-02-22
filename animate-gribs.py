import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import glob
import sys
import subprocess
from gributils import *

def read_gribs(directories):
    alldatas = []

    for directory in directories:
        gribs = glob.glob('{}/*.grib2'.format(directory))

        datas = []
        for gribfile in gribs:
            datas.append(np.squeeze(read_grib(gribfile, img_size=(128,128))))

        alldatas.append(datas)

    for x in alldatas[1:]:
        assert(len(x) == len(alldatas[0]))

    return alldatas


def save_anim(filename, datas):
    rows = 1
    cols = len(datas)
    n_frames = len(datas[0])

    if cols > 2:
      rows = 2
      cols = int(0.5 + cols / 2)
 
    print(rows, cols)
    fig, ax = plt.subplots(rows, cols,figsize=(16, 12))
    models = ['unet bc','unet ssim','unet fss', 'ref']
    def update(j):
        for i in range(len(datas)):
            ax.flat[i].imshow(datas[i][j], cmap=cm.Greys_r)
            ax.flat[i].set_title("{} {}/{}".format(models[i], j, n_frames), fontsize=20)
            ax.flat[i].set_axis_off()

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), interval=200)
    anim.save(filename, dpi=80, writer='imagemagick')
    plt.close()

    print(f"Saved {len(datas[0])} frame animation to file '{filename}'")


filename = sys.argv.pop()

datas = read_gribs(sys.argv[1:])
save_anim(filename, datas)
