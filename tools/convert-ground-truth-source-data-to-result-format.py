import numpy as np
import sys
from datetime import datetime
import pickle

outfile = sys.argv[2]
data = np.load(sys.argv[1], allow_pickle=True)

times = data["arr_1"]
times = [datetime.strptime(time, "%Y%m%dT%H%M%S") for time in times]
datas = data["arr_0"]

data_dict = {"gt": {"time": times, "data": datas}}

pickled_data = pickle.dumps(data_dict, protocol=4)

# Convert the pickled data into a NumPy array of dtype object
#obj_array = np.array(pickled_data, dtype=object)

with open(outfile, "wb") as f:
    pickle.dump(data_dict, f, protocol=4)

# np.savez_compressed(outfile, {"gt": {"time": times, "data": datas}})
