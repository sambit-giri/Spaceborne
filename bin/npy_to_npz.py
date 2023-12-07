import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg')
import common_lib.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

folder_path = '/Users/davide/Documents/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/covmat/PyCCL'
npy_dict = dict(mm.get_kv_pairs(folder_path, extension='npy'))

for key in npy_dict.keys():
    np.savez_compressed(f'{folder_path}/{key}.npz', npy_dict[key])
