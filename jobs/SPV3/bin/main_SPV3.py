import sys
import time
from pathlib import Path

import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(str(project_path.parent / 'common_data/common_lib'))
import my_module as mm
import cosmo_lib as csmlib

# general config
sys.path.append(str(project_path.parent / 'common_data/common_config'))
import mpl_cfg

# job configuration
sys.path.append(str(project_path / 'jobs'))
import SPV3.configs.config_SPV3 as cfg

# project libraries
sys.path.append(str(project_path / 'bin'))
import ell_values_running as ell_utils
import Cl_preprocessing_running as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import utils_running as utils
import unit_test

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()

# TODO ind will be different for the different number of z bins
# TODO update consistency_checks
# TODO finish exploring the cls
# TODO check that the number of ell bins is the same as in the files
# TODO make sure you changed fsky
# TODO change sigma_eps2?
# TODO double check the delta values


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_config = cfg.general_config
covariance_config = cfg.covariance_config
Sijkl_config = cfg.Sijkl_config
FM_config = cfg.FM_config
cosmo_params_dict = csmlib.cosmo_par_dict_classy

which_probe_response = covariance_config['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')

zbins_SPV3 = (7, 9, 10, 11, 13, 15)

for general_config['zbins'] in zbins_SPV3:
    # for (general_config['ell_max_WL'], general_config['ell_max_GC']) in ((5000, 3000), (1500, 750)):

    # utils.consistency_checks(general_config, covariance_config)

    zbins = general_config['zbins']

    ind = np.genfromtxt(f'{project_path}/config/common_data/ind_files/variable_zbins/indici_vincenzo_like_zbins{zbins}.dat', dtype=int)
    covariance_config['ind'] = ind

    # Sijkl_dav = np.load(f"{project_path}/config/common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA



    # some variables used for I/O naming
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    ell_max_XC = ell_max_GC
    nbl_WL = general_config['nbl_WL']

    # compute ell and delta ell values
    ell_WL, delta_l_WL = ell_utils.ISTF_ells(general_config['nbl_WL'], general_config['ell_min'],
                                             general_config['ell_max_WL'])
    ell_WL = np.log10(ell_WL)

    ell_dict = {}
    ell_dict['ell_WL'] = ell_WL
    ell_dict['ell_GC'] = np.copy(ell_WL[10 ** ell_WL < ell_max_GC])
    ell_dict['ell_WA'] = np.copy(ell_WL[10 ** ell_WL > ell_max_GC])
    ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])

    nbl_GC = ell_dict['ell_GC'].shape[0]
    nbl_WA = ell_dict['ell_WA'].shape[0]
    nbl_XC = nbl_GC

    # ! not super sure about these deltas
    delta_dict = {}
    delta_dict['delta_l_WL'] = delta_l_WL
    delta_dict['delta_l_GC'] = np.copy(delta_l_WL[:nbl_GC])
    delta_dict['delta_l_WA'] = np.copy(delta_l_WL[nbl_GC:])

    cl_ll_3d = cl_utils.get_spv3_cls_3d(probe='WL', nbl=nbl_WL, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='cls')
    cl_gg_3d = cl_utils.get_spv3_cls_3d(probe='GC', nbl=nbl_GC, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='cls')
    cl_wa_3d = cl_utils.get_spv3_cls_3d(probe='WA', nbl=nbl_WA, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='cls')
    cl_3x2pt_5d = cl_utils.get_spv3_cls_3d(probe='3x2pt', nbl=nbl_XC, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='cls')

    rl_ll_3d = cl_utils.get_spv3_cls_3d(probe='WL', nbl=nbl_WL, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='responses')
    rl_gg_3d = cl_utils.get_spv3_cls_3d(probe='GC', nbl=nbl_GC, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='responses')
    rl_wa_3d = cl_utils.get_spv3_cls_3d(probe='WA', nbl=nbl_WA, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='responses')
    rl_3x2pt_5d = cl_utils.get_spv3_cls_3d(probe='3x2pt', nbl=nbl_XC, zbins=zbins, ell_max_WL=ell_max_WL, cls_or_responses='responses')

    cl_dict_3D = {
        'C_LL_WLonly_3D': cl_ll_3d,
        'C_GG_3D': cl_gg_3d,
        'C_WA_3D': cl_wa_3d,
        'D_3x2pt': cl_3x2pt_5d}

    Rl_dict_3D = {
        'R_LL_WLonly_3D': rl_ll_3d,
        'R_GG_3D': rl_gg_3d,
        'R_WA_3D': rl_wa_3d,
        'R_3x2pt': rl_3x2pt_5d}

    # Sijkl from PySSC
    # Sijkl = Sijkl_utils.compute_Sijkl(cosmo_params_dict, Sijkl_config, zbins=general_config['zbins'])

    # if cfg.Sijkl_config['save_Sijkl']:
    #     np.save(f"Sijkl_WF{Sijkl_config['input_WF']}_nz{z_points}_{has_IA}.npy")


assert 1 == 0, 'this is a test'

mm.matshow(Sijkl[0, 0, :, :])
mm.matshow(Sijkl_dav[0, 0, :, :])

if cfg.Sijkl_config['save_Sijkl']:
    np.save(project_path / f'output/sijkl/sijkl_wf{cfg.Sijkl_config["input_WF"]}.npy')

assert 1 > 2

# compute covariance matrix
cov_dict = covmat_utils.compute_cov(general_config, covariance_config,
                                    ell_dict, delta_dict, cl_dict_3D, Rl_dict_3D, Sijkl)
# compute Fisher Matrix
FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)

# save:
if covariance_config['save_covariance']:
    np.save(job_path / f'output/covmat/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl_WL}_2D.npy', cov_dict['cov_WL_GO_2D'])
np.save(job_path / f'output/covmat/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl_WL}_2D.npy', cov_dict['cov_GC_GO_2D'])
np.save(job_path / f'output/covmat/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl_WL}_2D.npy',
        cov_dict['cov_3x2pt_GO_2D'])
np.save(job_path / f'output/covmat/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl_WL}_2D.npy', cov_dict['cov_WA_GO_2D'])

np.save(job_path / f'output/covmat/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl_WL}_Rl{which_probe_response_str}_2D.npy',
        cov_dict['cov_WL_GS_2D'])
np.save(job_path / f'output/covmat/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl_WL}_Rl{which_probe_response_str}_2D.npy',
        cov_dict['cov_GC_GS_2D'])
np.save(
    job_path / f'output/covmat/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl_WL}_Rl{which_probe_response_str}_2D.npy',
    cov_dict['cov_3x2pt_GS_2D'])
np.save(job_path / f'output/covmat/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl_WL}_Rl{which_probe_response_str}_2D.npy',
        cov_dict['cov_WA_GS_2D'])

if FM_config['save_FM']:
    np.savetxt(job_path / f"output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl_WL}.txt", FM_dict['FM_WL_GO'])
np.savetxt(job_path / f"output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl_WL}.txt", FM_dict['FM_GC_GO'])
np.savetxt(job_path / f"output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl_WL}.txt", FM_dict['FM_3x2pt_GO'])

np.savetxt(job_path / f"output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
           FM_dict['FM_WL_GS'])
np.savetxt(job_path / f"output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
           FM_dict['FM_GC_GS'])
np.savetxt(job_path / f"output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
           FM_dict['FM_3x2pt_GS'])

if FM_config['save_FM_as_dict']:
    sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

if general_config['save_cls']:
    for key in cl_dict_3D.keys():
        np.save(job_path / f"output/cl_3D/{key}.npy", cl_dict_3D[f'{key}'])

print('done')
