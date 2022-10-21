import pickle
import sys
import time
from pathlib import Path
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(f'{project_path}/lib')
import my_module as mm

import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path}/config')
import mpl_cfg

# job configuration
sys.path.append(f'{job_path}/config')
import config_SPV3 as cfg

# project libraries
sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils
import Cl_preprocessing_running as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import utils_running as utils
import unit_test

matplotlib.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()

# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_config
covariance_cfg = cfg.covariance_config
Sijkl_cfg = cfg.Sijkl_config
FM_cfg = cfg.FM_config

which_probe_response = covariance_cfg['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')

for general_cfg['zbins'] in general_cfg['zbins_list']:
    # for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000), (1500, 750)):
    for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000),):
        for (general_cfg['EP_or_ED']) in ('ED',):

            # utils.consistency_checks(general_cfg, covariance_cfg)

            # some variables used for I/O naming, just to make things shorter
            zbins = general_cfg['zbins']
            EP_or_ED = general_cfg['EP_or_ED']
            ell_max_WL = general_cfg['ell_max_WL']
            ell_max_GC = general_cfg['ell_max_GC']
            ell_max_XC = ell_max_GC
            nbl_WL_32 = general_cfg['nbl_WL_32']

            ind = np.genfromtxt(f'{project_path}/input/ind_files/variable_zbins/{covariance_cfg["ind_ordering"]}_like/'
                                f'indici_{covariance_cfg["ind_ordering"]}_like_zbins{zbins}.dat', dtype=int)
            covariance_cfg['ind'] = ind

            assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
                'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

            # compute ell and delta ell values in the reference (optimistic) case
            ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_32'], general_cfg['ell_min'],
                                                                    general_cfg['ell_max_WL_opt'], recipe='ISTF')

            ell_WL_nbl32 = np.log10(ell_WL_nbl32)

            # perform the cuts
            ell_dict = {}
            ell_dict['ell_WL'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_WL])
            ell_dict['ell_GC'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_GC])
            ell_dict['ell_WA'] = np.copy(
                ell_WL_nbl32[(10 ** ell_WL_nbl32 > ell_max_GC) & (10 ** ell_WL_nbl32 < ell_max_WL)])
            ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])

            # set corresponding # of ell bins
            nbl_WL = ell_dict['ell_WL'].shape[0]
            nbl_GC = ell_dict['ell_GC'].shape[0]
            nbl_WA = ell_dict['ell_WA'].shape[0]
            nbl_3x2pt = nbl_GC
            general_cfg['nbl_WL'] = nbl_WL

            delta_dict = {}
            delta_dict['delta_l_WL'] = np.copy(delta_l_WL_nbl32[:nbl_WL])
            delta_dict['delta_l_GC'] = np.copy(delta_l_WL_nbl32[:nbl_GC])
            delta_dict['delta_l_WA'] = np.copy(delta_l_WL_nbl32[nbl_GC:])

            # set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
            nbl_WL_opt = 32
            nbl_GC_opt = 29
            nbl_WA_opt = 3
            nbl_3x2pt_opt = 29

            if ell_max_WL == general_cfg['ell_max_WL_opt']:
                assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
                    'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

            # ! import and reshape Cl and Rl
            cl_ll_3d = cl_utils.get_spv3_cls_3d('WL', nbl_WL_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='cl', EP_or_ED=EP_or_ED)
            cl_gg_3d = cl_utils.get_spv3_cls_3d('GC', nbl_GC_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='cl', EP_or_ED=EP_or_ED)
            cl_wa_3d = cl_utils.get_spv3_cls_3d('WA', nbl_WA_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='cl', EP_or_ED=EP_or_ED)
            cl_3x2pt_5d = cl_utils.get_spv3_cls_3d('3x2pt', nbl_3x2pt_opt, general_cfg, zbins,
                                                   general_cfg['ell_max_WL_opt'],
                                                   cl_or_rl='cl', EP_or_ED=EP_or_ED)

            rl_ll_3d = cl_utils.get_spv3_cls_3d('WL', nbl_WL_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='rl', EP_or_ED=EP_or_ED)
            rl_gg_3d = cl_utils.get_spv3_cls_3d('GC', nbl_GC_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='rl', EP_or_ED=EP_or_ED)
            rl_wa_3d = cl_utils.get_spv3_cls_3d('WA', nbl_WA_opt, general_cfg, zbins, general_cfg['ell_max_WL_opt'],
                                                cl_or_rl='rl', EP_or_ED=EP_or_ED)
            rl_3x2pt_5d = cl_utils.get_spv3_cls_3d('3x2pt', nbl_3x2pt_opt, general_cfg, zbins,
                                                   general_cfg['ell_max_WL_opt'],
                                                   cl_or_rl='rl', EP_or_ED=EP_or_ED)

            if general_cfg['cl_BNT_transform']:
                BNT_matrix = np.genfromtxt(f'{general_cfg["BNT_matrix_path"]}/{general_cfg["BNT_matrix_filename"]}')
                cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix)
                cl_gg_3d = cl_utils.cl_BNT_transform(cl_gg_3d, BNT_matrix)
                cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix)
                cl_3x2pt_5d = cl_utils.cl_BNT_transform(cl_3x2pt_5d, BNT_matrix)
                print('you shuld BNT transform the responses do this with the responses too!')

            if ell_max_WL == general_cfg['ell_max_WL_opt']:
                if not np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :]):
                    rtol = 1e-10
                    assert (np.allclose(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :], rtol=rtol, atol=0)), \
                        'cl_wa_3d should be obtainable from cl_ll_3d!'
                    print(f'cl_wa_3d and cl_ll_3d[nbl_GC:nbl_WL, :, :] are not exactly equal, but have a relative '
                          f'difference of less than {rtol}')

            # cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
            if ell_max_WL == 1500:
                cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
                cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
                cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
                cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

                rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
                rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
                rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
                rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

            cl_dict_3D = {
                'C_LL_WLonly_3D': cl_ll_3d,
                'C_GG_3D': cl_gg_3d,
                'C_WA_3D': cl_wa_3d,
                'C_3x2pt_5D': cl_3x2pt_5d}

            rl_dict_3D = {
                'R_LL_WLonly_3D': rl_ll_3d,
                'R_GG_3D': rl_gg_3d,
                'R_WA_3D': rl_wa_3d,
                'R_3x2pt_5D': rl_3x2pt_5d}

            # ! compute or load Sijkl

            # get number of z points in nz to name the sijkl file
            z_arr, _ = Sijkl_utils.load_WF(Sijkl_cfg, zbins, EP_or_ED)
            nz = z_arr.shape[0]

            sijkl_folder = Sijkl_cfg['sijkl_folder']
            sijkl_filename = f'sijkl_WF{Sijkl_cfg["WF_suffix"]}_nz{nz}_zbins{zbins:02}_{EP_or_ED}_hasIA{Sijkl_cfg["has_IA"]}.npy'

            if Sijkl_cfg['use_precomputed_sijkl']:
                sijkl = np.load(f'{sijkl_folder}/{sijkl_filename}')
            else:
                sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, Sijkl_cfg, zbins=zbins, EP_or_ED=EP_or_ED)

                if Sijkl_cfg['save_Sijkl']:
                    np.save(f'{sijkl_folder}/{sijkl_filename}', sijkl)

            # ! compute covariance matrix
            if covariance_cfg['compute_covmat']:
                cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                    ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

            # ! compute Fisher Matrix
            if FM_cfg['compute_FM']:
                FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict)

            # ! save:
            # this is just to set the correct probe names
            probe_dav_dict = {
                'WL': 'LL_WLonly_3D',
                'GC': 'GG_3D',
                'WA': 'WA_3D',
                '3x2pt': '3x2pt_5D'}

            # cl_input_folder = general_cfg['cl_input_folder']
            # if general_cfg['save_cls_3d']:
            #     for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
            #         np.save(f'{cl_input_folder}/3D_reshaped/{probe_vinc}/'
            #                 f'dv-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
            #                 cl_dict_3D[f'C_{probe_dav_dict[probe_dav]}'])
            #
            #         if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
            #             np.savetxt(
            #                 f'{cl_input_folder}/3D_reshaped/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
            #                 10 ** ell_dict[f'ell_{probe_dav}'])
            #             np.savetxt(
            #                 f'{cl_input_folder}/3D_reshaped/{probe_vinc}/delta_ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
            #                 delta_dict[f'delta_l_{probe_dav}'])
            #
            # rl_input_folder = general_cfg['rl_input_folder']
            # if general_cfg['save_rls_3d']:
            #     for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
            #         np.save(f'{rl_input_folder}/3D_reshaped/{probe_vinc}/'
            #                 f'rf-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
            #                 rl_dict_3D[f'R_{probe_dav_dict[probe_dav]}'])
            #
            #         if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
            #             np.savetxt(
            #                 f'{rl_input_folder}/3D_reshaped/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
            #                 10 ** ell_dict[f'ell_{probe_dav}'])

            # ! new code
            clrl_dict = {
                'cl_inputname': 'dv',
                'rl_inputname': 'rf',
                'cl_dict_3D': cl_dict_3D,
                'rl_dict_3D': rl_dict_3D,
                'cl_dict_key': 'C',
                'rl_dict_key': 'R',
            }
            for cl_or_rl in ['cl', 'rl']:
                folder = general_cfg[f'{cl_or_rl}_folder']
                if general_cfg[f'save_{cl_or_rl}s_3d']:
                    for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
                        np.save(f'{folder}/3D_reshaped/{probe_vinc}/'
                                f'{clrl_dict[f"{cl_or_rl}_inputname"]}-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
                                clrl_dict[f"{cl_or_rl}_dict_3D"][f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}'])

                        if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                            np.savetxt(f'{folder}/3D_reshaped/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                                10 ** ell_dict[f'ell_{probe_dav}'])
                            np.savetxt(f'{folder}/3D_reshaped/{probe_vinc}/delta_ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                                delta_dict[f'delta_l{probe_dav}'])
            # ! end new code

            covmat_path = f'{covariance_cfg["cov_output_folder"]}/zbins{zbins:02}'
            for ndim in (2, 4, 6):
                if covariance_cfg[f'save_cov_{ndim}D']:

                    # save GO, GS or GO, GS and SS
                    which_cov_list = ['GO', 'GS']
                    # which_cov_list = ['GO']
                    Rl_str_list = ['', f'_Rl{which_probe_response_str}']
                    # Rl_str_list = ['', ]
                    if covariance_cfg[f'save_cov_SS']:
                        which_cov_list.append('SS')
                        Rl_str_list.append(f'_Rl{which_probe_response_str}')

                    # set probes to save; the ndim == 6 is different
                    probe_list = ['WL', 'GC', '3x2pt', 'WA']
                    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
                    # in this case, 3x2pt is saved in 10D as a dictionary
                    if ndim == 6:
                        probe_list = ['WL', 'GC', 'WA']
                        ellmax_list = [ell_max_WL, ell_max_GC, ell_max_WL]
                        nbl_list = [nbl_WL, nbl_GC, nbl_WA]

                    # save all covmats in the optimistic case
                    if ell_max_WL == 5000:

                        for which_cov, Rl_str in zip(which_cov_list, Rl_str_list):
                            for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                                np.save(f'{covmat_path}/'
                                        f'covmat_{which_cov}_{probe}_lmax{ell_max}_nbl{nbl}_zbins{zbins:02}_{EP_or_ED}{Rl_str}_{ndim}D.npy',
                                        cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])

                        # in this case, 3x2pt is saved in 10D as a dictionary
                        if ndim == 6:
                            filename = f'{covmat_path}/covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_nbl{nbl_3x2pt}_zbins{zbins:02}_{EP_or_ED}{Rl_str}_10D.pickle'
                            with open(filename, 'wb') as handle:
                                pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle,
                                            protocol=pickle.HIGHEST_PROTOCOL)

                    # in the pessimistic case, save only WA
                    elif ell_max_WL == 1500:
                        for which_cov, Rl_str in zip(['GO', 'GS'], ['', f'_Rl{which_probe_response_str}']):
                            np.save(
                                f'{covmat_path}/covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_zbins{zbins:02}_{EP_or_ED}{Rl_str}_{ndim}D.npy',
                                cov_dict[f'cov_WA_{which_cov}_{ndim}D'])

            # save in .dat for Vincenzo, only in the optimistic case and in 2D
            if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
                path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
                for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
                    for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                        np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                                   f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                                   cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

        # check for Stefano
        print('GHOST CODE BELOW')
        npairs = (zbins * (zbins + 1)) // 2
        cov_WL_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GO_6D'], nbl_WL, npairs, ind[:npairs, :])
        cov_GC_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GO_6D'], nbl_GC, npairs, ind[:npairs, :])
        cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GS_6D'], nbl_WL, npairs, ind[:npairs, :])
        cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GS_6D'], nbl_GC, npairs, ind[:npairs, :])
        assert (np.array_equal(cov_WL_GO_4D, cov_dict[f'cov_WL_GO_4D']))
        assert (np.array_equal(cov_GC_GO_4D, cov_dict[f'cov_GC_GO_4D']))
        assert (np.array_equal(cov_WL_GS_4D, cov_dict[f'cov_WL_GS_4D']))
        assert (np.array_equal(cov_GC_GS_4D, cov_dict[f'cov_GC_GS_4D']))

"""
if FM_cfg['save_FM']:
    np.savetxt(f"{job_path}/output/FM/FM_WL_GO_lmax{ell_max_WL}_nbl{nbl_WL}.txt", FM_dict['FM_WL_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_GC_GO_lmax{ell_max_GC}_nbl{nbl_WL}.txt", FM_dict['FM_GC_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GO_lmax{ell_max_XC}_nbl{nbl_WL}.txt", FM_dict['FM_3x2pt_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_WL_GS_lmax{ell_max_WL}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_WL_GS'])
    np.savetxt(f"{job_path}/output/FM/FM_GC_GS_lmax{ell_max_GC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_GC_GS'])
    np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GS_lmax{ell_max_XC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_3x2pt_GS'])

if FM_cfg['save_FM_as_dict']:
    sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)
"""
print('done')
