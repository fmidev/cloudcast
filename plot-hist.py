from plotutils import *
import sys
import ast
import os
import glob

path = sys.argv[1]

if os.path.isdir(path):
    fname = glob.glob(f'{path}/hist*.txt')

    if len(fname) == 0:
        print(f"No files matching 'hist*.txt' found from {path}")
        sys.exit(1)
    elif len(fname) > 1:
        print(f"Found more than one 'hist*.txt' file from {path}, using first one")
    
    fname = fname[0]
else:
    fname = path

model_dir = os.path.realpath(fname).split('/')[-2]
with open(fname) as fp:
    data = ast.literal_eval(fp.read())   
    plot_hist(data, model_dir=model_dir, show=True)

