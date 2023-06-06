import sys
import time
import warnings
from pathlib import Path
import numpy as np
import scipy
import pdb
from matplotlib import pyplot as plt

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm

import cl_preprocessing as cl_utils

script_name = sys.argv[0]


# def get_idxs_to_delete(ell_values, ell_cuts, is_auto_spectrum):
#     """ ell_values can be the bin center or the bin lower edge; Francis suggests the second option is better"""
#     warnings.warn('delete this function from here')
#     zbins = 13
#     if is_auto_spectrum:
#         idxs_to_delete = []
#         count = 0
#         for ell_idx, ell_val in enumerate(ell_values):
#             for zi in range(zbins):
#                 for zj in range(zi, zbins):
#                     if ell_val > ell_cuts[zi, zj]:
#                         idxs_to_delete.append(count)
#                     count += 1
#
#     elif not is_auto_spectrum:
#         idxs_to_delete = []
#         count = 0
#         for ell_idx, ell_val in enumerate(ell_values):
#             for zi in range(zbins):
#                 for zj in range(zbins):
#                     if ell_val > ell_cuts[zi, zj]:
#                         idxs_to_delete.append(count)
#                     count += 1
#     else:
#         raise ValueError('is_auto_spectrum must be True or False')
#
#     return idxs_to_delete
#
#
###############################################################################
################## CODE TO COMPUTE THE FISHER MATRIX ##########################
###############################################################################


###########################################

# XXX attention! the dC_LL matrix should have (nParams - 10) as dimension,
# since WL has no bias. This would complicate the structure of the datacector 
# and taking nParams instead seems to have ho impact on the final result.

def dC_4D_to_3D(dC_4D, nbl, zpairs, nparams_tot, ind):
    """expand the zpair indices into zi, zj, according to the ind ordering as usual"""

    dC_3D = np.zeros((nbl, zpairs, nparams_tot))
    for ell in range(nbl):
        for alf in range(nparams_tot):
            dC_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_4D[ell, :, :, alf], zpairs, ind)
    return dC_3D


def dC_dict_to_4D_array(dC_dict_3D, param_names, nbl, zbins, derivatives_prefix, is_3x2pt=False, n_probes=2):
    """
    :param param_names: filename of the parameter, e.g. 'Om'; dCldOm = d(C(l))/d(Om)
    :param dC_dict_3D:
    :param nbl:
    :param zbins:
    :param obs_name: filename of the observable, e.g. 'Cl'; dCldOm = d(C(l))/d(Om)
    :param is_3x2pt: whether to will the 5D derivatives vector
    :param n_probes:
    :return:
    """
    # param_names should be params_tot in all cases, because when the derivative dows not exist
    # in dC_dict_3D the output array will remain null
    if is_3x2pt:
        dC_4D = np.zeros((nbl, n_probes, n_probes, zbins, zbins, len(param_names)))
    else:
        dC_4D = np.zeros((nbl, zbins, zbins, len(param_names)))

    if not dC_dict_3D:
        warnings.warn('The input dictionary is empty')

    no_derivative_counter = 0
    for idx, param_name in enumerate(param_names):
        for key, value in dC_dict_3D.items():
            if f'{derivatives_prefix}{param_name}' in key:
                dC_4D[..., idx] = value

        # a check, if the derivative wrt the param is not in the folder at all
        if not any(f'{derivatives_prefix}{param_name}' in key for key in dC_dict_3D.keys()):
            print(f'Derivative {derivatives_prefix}{param_name} not found; setting the corresponding entry to zero')
            no_derivative_counter += 1
        if no_derivative_counter == len(param_names):
            raise ImportError('No derivative found for any of the parameters in the input dictionary')
    return dC_4D


def invert_matrix_LU(covariance_matrix):
    # Perform LU decomposition
    P, L, U = scipy.linalg.lu(covariance_matrix)
    # Invert the matrix using the decomposition
    return np.linalg.inv(L) @ np.linalg.inv(U) @ P


def ell_cuts_derivatives(FM_cfg, ell_dict, dC_LL_4D, dC_WA_4D, dC_GG_4D, dC_3x2pt_6D):
    raise Exception('this function works, but you need to cut the covariance matrix using the corresponsing indices, '
                    'ie using the "1-dimensional cutting" approach by Vincenzo')

    if not FM_cfg['deriv_ell_cuts']:
        return dC_LL_4D, dC_WA_4D, dC_GG_4D, dC_3x2pt_6D

    print('Performing the ell cuts on the derivatives...')

    ell_cuts_dict = ell_dict['ell_cuts_dict']
    ell_cuts_LL = ell_cuts_dict['LL']
    ell_cuts_GG = ell_cuts_dict['GG']
    param_names_3x2pt = FM_cfg['param_names_3x2pt']

    cl_cut = cl_utils.cl_ell_cut  # just to abbreviate the name to fit in one line
    for param_idx in range(len(param_names_3x2pt)):
        dC_LL_4D[:, :, :, param_idx] = cl_cut(dC_LL_4D[:, :, :, param_idx], ell_cuts_LL, ell_dict['ell_WL'])
        dC_WA_4D[:, :, :, param_idx] = cl_cut(dC_WA_4D[:, :, :, param_idx], ell_cuts_LL, ell_dict['ell_WA'])
        dC_GG_4D[:, :, :, param_idx] = cl_cut(dC_GG_4D[:, :, :, param_idx], ell_cuts_GG, ell_dict['ell_GC'])
        dC_3x2pt_6D[:, :, :, :, :, param_idx] = cl_utils.cl_ell_cut_3x2pt(
            dC_3x2pt_6D[:, :, :, :, :, param_idx], ell_cuts_dict, ell_dict['ell_3x2pt'])

    return dC_LL_4D, dC_WA_4D, dC_GG_4D, dC_3x2pt_6D


def compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict, BNT_matrix=None):
    # shorten names
    zbins = general_cfg['zbins']
    use_WA = general_cfg['use_WA']
    GL_or_LG = covariance_cfg['GL_or_LG']
    ind = covariance_cfg['ind']
    block_index = covariance_cfg['block_index']
    nparams_tot = FM_cfg['nparams_tot']
    param_names_3x2pt = FM_cfg['param_names_3x2pt']

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_XC, nbl_3x2pt = ell_GC, nbl_GC
    nbl_WA = ell_WA.shape[0]

    # set the flattening convention for the derivatives vector, based on the setting used to reduce the covariance
    # matrix' dimensions
    # TODO review this
    if block_index in ['ell', 'vincenzo', 'C-style']:
        which_flattening = 'C'
    elif block_index in ['ij', 'sylvain', 'F-style']:
        which_flattening = 'F'
    else:
        raise ValueError("block_index should be either 'ell', 'vincenzo', 'C-style', 'ij', 'sylvain' or 'F-style'")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[zpairs_auto:(zpairs_auto + zpairs_cross), [2, 3]] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), [3, 2]]

    # ===================================================================================================================
    # import and invert covariance matrices
    # ===================================================================================================================

    if cov_dict['cov_WA_GO_2D'].shape == (0, 0):
        warnings.warn('cov_WA_GO_2D is empty, setting use_WA to False and the covariance matrix to the identity')
        general_cfg['use_WA'] = False
        cov_dict['cov_WA_GO_2D'] = np.eye(nbl_WA * zpairs_auto)
        cov_dict['cov_WA_GS_2D'] = np.eye(nbl_WA * zpairs_auto)

    # invert GO covmats
    print('Starting covariance matrix inversion...')
    start_time = time.perf_counter()
    # TODO try to use scipy.sparse.linalg.inv
    cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_GO_2D'])
    cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_GO_2D'])
    cov_WA_GO_2D_inv = np.linalg.inv(cov_dict['cov_WA_GO_2D'])
    cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GO_2D'])
    print(f'GO covariance matrices inverted in {(time.perf_counter() - start_time):.2f} s')

    # invert GS covmats
    start_time = time.perf_counter()
    cov_WL_GS_2D_inv = np.linalg.inv(cov_dict['cov_WL_GS_2D'])
    cov_GC_GS_2D_inv = np.linalg.inv(cov_dict['cov_GC_GS_2D'])
    cov_WA_GS_2D_inv = np.linalg.inv(cov_dict['cov_WA_GS_2D'])
    cov_3x2pt_GS_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GS_2D'])
    print(f'GS covariance matrices inverted in {(time.perf_counter() - start_time):.2f} s')

    # load reshaped derivatives, with shape (nbl, zbins, zbins, nparams)
    dC_LL_4D = deriv_dict['dC_LL_4D']
    dC_GG_4D = deriv_dict['dC_GG_4D']
    dC_WA_4D = deriv_dict['dC_WA_4D']
    dC_3x2pt_6D = deriv_dict['dC_3x2pt_6D']

    assert dC_LL_4D.shape == (nbl_WL, zbins, zbins, nparams_tot), f'dC_LL_4D has incorrect shape: {dC_LL_4D.shape}'
    assert dC_GG_4D.shape == (nbl_GC, zbins, zbins, nparams_tot), f'dC_GG_4D has incorrect shape: {dC_GG_4D.shape}'
    assert dC_WA_4D.shape == (nbl_WA, zbins, zbins, nparams_tot), f'dC_WA_4D has incorrect shape: {dC_WA_4D.shape}'
    assert dC_3x2pt_6D.shape == (2, 2, nbl_3x2pt, zbins, zbins, nparams_tot), \
        f'dC_3x2pt_6D has incorrect shape: {dC_3x2pt_6D.shape}'

    #
    # dC_LLfor3x2pt_2D = np.zeros((nbl_3x2pt * zpairs_auto, nparams_tot))
    # dC_GLfor3x2pt_2D = np.zeros((nbl_3x2pt * zpairs_cross, nparams_tot))
    # dC_GGfor3x2pt_2D = np.zeros((nbl_3x2pt * zpairs_auto, nparams_tot))
    # count = 0
    # for ell_idx, ell_val in enumerate(ell_dict['ell_3x2pt']):
    #     for zi in range(zbins):
    #         for zj in range(zi, zbins):
    #             dC_LLfor3x2pt_2D[count, :] = dC_3x2pt_6D[ell_idx, 0, 0, zi, zj, :]
    #             dC_GGfor3x2pt_2D[count, :] = dC_3x2pt_6D[ell_idx, 1, 1, zi, zj, :]
    #             count += 1
    #
    # count = 0
    # for ell_idx, ell_val in enumerate(ell_dict['ell_3x2pt']):
    #     for zi in range(zbins):
    #         for zj in range(zi, zbins):
    #             dC_GLfor3x2pt_2D[count, :] = dC_3x2pt_6D[ell_idx, 1, 0, zi, zj, :]
    #             count += 1
    #
    # dC_3x2pt_2D = np.vstack((dC_LLfor3x2pt_2D, dC_GLfor3x2pt_2D, dC_GGfor3x2pt_2D))
    # dC_3x2pt_2D_v2 = np.concatenate((dC_LLfor3x2pt_2D, dC_GLfor3x2pt_2D, dC_GGfor3x2pt_2D), axis=1)
    # assert np.array_equal(dC_3x2pt_2D, dC_3x2pt_2D_v2), 'something is wrong with the 3x2pt derivatives'

    # plt.figure()
    # plt.plot(dC_3x2pt_2D[:, 0], label='dC_3x2pt_2D_flattened by hand')

    if FM_cfg['derivatives_BNT_transform']:

        assert covariance_cfg['cov_BNT_transform'], 'you should BNT transform the covariance as well'
        assert BNT_matrix is not None, 'you should provide a BNT matrix'

        for param_idx in range(len(param_names_3x2pt)):
            dC_LL_4D[:, :, :, param_idx] = cl_utils.cl_BNT_transform(dC_LL_4D[:, :, :, param_idx], BNT_matrix, 'L', 'L')
            dC_WA_4D[:, :, :, param_idx] = cl_utils.cl_BNT_transform(dC_WA_4D[:, :, :, param_idx], BNT_matrix, 'L', 'L')
            dC_3x2pt_6D[:, :, :, :, :, param_idx] = cl_utils.cl_BNT_transform_3x2pt(
                dC_3x2pt_6D[:, :, :, :, :, param_idx], BNT_matrix)

    # ! ell-cut the derivatives in 3d
    # dC_LL_4D_v1, dC_WA_4D_v1, dC_GG_4D_v1, dC_3x2pt_6D_v1 = ell_cuts_derivatives(FM_cfg, ell_dict,
    #                                                                              dC_LL_4D, dC_WA_4D,
    #                                                                              dC_GG_4D, dC_3x2pt_6D)

    # separate the different 3x2pt contributions
    # ! delicate point, double check
    if GL_or_LG == 'GL':
        probe_A, probe_B = 1, 0
    elif GL_or_LG == 'LG':
        probe_A, probe_B = 0, 1
    else:
        raise ValueError('GL_or_LG must be "GL" or "LG"')

    dC_LLfor3x2pt_4D = dC_3x2pt_6D[0, 0, :, :, :, :]
    dC_XCfor3x2pt_4D = dC_3x2pt_6D[probe_A, probe_B, :, :, :, :]
    dC_GGfor3x2pt_4D = dC_3x2pt_6D[1, 1, :, :, :, :]

    assert np.array_equal(dC_GGfor3x2pt_4D, dC_GG_4D), "dC_GGfor3x2pt_4D and dC_GG_4D are not equal"
    assert nbl_3x2pt == nbl_GC, 'nbl_3x2pt and nbl_GC are not equal'

    # flatten z indices, obviously following the ordering given in ind
    # separate the ind for the different probes
    ind_auto = ind[:zpairs_auto, :]
    ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]  # ! watch out for the ind switch!!

    dC_LL_3D = dC_4D_to_3D(dC_LL_4D, nbl_WL, zpairs_auto, nparams_tot, ind_auto)
    dC_GG_3D = dC_4D_to_3D(dC_GG_4D, nbl_GC, zpairs_auto, nparams_tot, ind_auto)
    dC_WA_3D = dC_4D_to_3D(dC_WA_4D, nbl_WA, zpairs_auto, nparams_tot, ind_auto)
    dC_LLfor3x2pt_3D = dC_4D_to_3D(dC_LLfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto)
    dC_XCfor3x2pt_3D = dC_4D_to_3D(dC_XCfor3x2pt_4D, nbl_3x2pt, zpairs_cross, nparams_tot, ind_cross)
    dC_GGfor3x2pt_3D = dC_4D_to_3D(dC_GGfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto)

    # concatenate the flattened components of the 3x2pt datavector
    dC_3x2pt_3D = np.concatenate((dC_LLfor3x2pt_3D, dC_XCfor3x2pt_3D, dC_GGfor3x2pt_3D), axis=1)

    # collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
    # in the same way not with the old datavector, which was ordered in a different way...
    dC_LL_2D = np.reshape(dC_LL_3D, (nbl_WL * zpairs_auto, nparams_tot), order=which_flattening)
    dC_GG_2D = np.reshape(dC_GG_3D, (nbl_GC * zpairs_auto, nparams_tot), order=which_flattening)
    dC_WA_2D = np.reshape(dC_WA_3D, (nbl_WA * zpairs_auto, nparams_tot), order=which_flattening)
    dC_3x2pt_2D = np.reshape(dC_3x2pt_3D, (nbl_3x2pt * zpairs_3x2pt, nparams_tot), order=which_flattening)

    # ! cut the *flattened* derivatives vector
    if FM_cfg['deriv_ell_cuts']:
        dC_LL_2D = np.delete(dC_LL_2D, ell_dict['idxs_to_delete_dict']['LL'], axis=0)
        dC_GG_2D = np.delete(dC_GG_2D, ell_dict['idxs_to_delete_dict']['GG'], axis=0)
        dC_WA_2D = np.delete(dC_WA_2D, ell_dict['idxs_to_delete_dict']['WA'], axis=0)
        dC_3x2pt_2D = np.delete(dC_3x2pt_2D, ell_dict['idxs_to_delete_dict']['3x2pt'], axis=0)

    # if the ell cuts removed all WA bins (which is in fact the case)
    if dC_WA_2D.shape[0] == 0:
        dC_WA_2D = np.ones((nbl_WA * zpairs_auto, nparams_tot))

    ######################### COMPUTE FM #####################################

    start = time.perf_counter()
    FM_WL_GO = np.einsum('ia,ik,kb->ab', dC_LL_2D, cov_WL_GO_2D_inv, dC_LL_2D, optimize='optimal')
    FM_GC_GO = np.einsum('ia,ik,kb->ab', dC_GG_2D, cov_GC_GO_2D_inv, dC_GG_2D, optimize='optimal')
    FM_WA_GO = np.einsum('ia,ik,kb->ab', dC_WA_2D, cov_WA_GO_2D_inv, dC_WA_2D, optimize='optimal')
    FM_3x2pt_GO = np.einsum('ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D, optimize='optimal')
    print(f'GO FM done in {(time.perf_counter() - start):.2f} s')

    start = time.perf_counter()
    FM_WL_GS = np.einsum('ia,ik,kb->ab', dC_LL_2D, cov_WL_GS_2D_inv, dC_LL_2D, optimize='optimal')
    FM_GC_GS = np.einsum('ia,ik,kb->ab', dC_GG_2D, cov_GC_GS_2D_inv, dC_GG_2D, optimize='optimal')
    FM_WA_GS = np.einsum('ia,ik,kb->ab', dC_WA_2D, cov_WA_GS_2D_inv, dC_WA_2D, optimize='optimal')
    FM_3x2pt_GS = np.einsum('ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GS_2D_inv, dC_3x2pt_2D, optimize='optimal')
    print(f'GS FM done in {(time.perf_counter() - start):.2f} s')

    # sum WA, this is the actual FM_3x2pt
    if use_WA:
        FM_3x2pt_GO += FM_WA_GO
        FM_3x2pt_GS += FM_WA_GS

    # store the matrices in the dictionary
    probe_names = ['WL', 'GC', 'WA', '3x2pt']
    FMs_GO = [FM_WL_GO, FM_GC_GO, FM_WA_GO, FM_3x2pt_GO]
    FMs_GS = [FM_WL_GS, FM_GC_GS, FM_WA_GS, FM_3x2pt_GS]

    FM_dict = {}
    for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
        FM_dict[f'FM_{probe_name}_GO'] = FM_GO
        FM_dict[f'FM_{probe_name}_GS'] = FM_GS

    print("FMs computed in %.2f seconds" % (time.perf_counter() - start))

    return FM_dict


def save_FM(fm_folder, FM_dict, FM_cfg, save_txt=False, save_dict=True, **save_specs):
    """saves the FM in .txt and .pickle formats
    :param fm_folder:
    """

    ell_max_WL = save_specs['ell_max_WL']
    ell_max_GC = save_specs['ell_max_GC']
    ell_max_XC = save_specs['ell_max_XC']
    nbl_WL = save_specs['nbl_WL']
    nbl_GC = save_specs['nbl_GC']
    nbl_WA = save_specs['nbl_WA']
    nbl_3x2pt = save_specs['nbl_3x2pt']

    probe_list = ['WL', 'GC', '3x2pt', 'WA']
    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]

    if save_txt:
        for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
            for which_cov in ['GO', 'GS']:
                FM_txt_filename = FM_cfg['FM_txt_filename'].format(probe=probe, which_cov=which_cov, ell_max=ell_max,
                                                                   nbl=nbl, **save_specs)
                np.savetxt(f'{fm_folder}/{FM_txt_filename}.txt', FM_dict[f'FM_{probe}_{which_cov}'])

    if save_dict:
        FM_dict_filename = FM_cfg['FM_dict_filename'].format(**save_specs)
        mm.save_pickle(f'{fm_folder}/{FM_dict_filename}.pickle', FM_dict)

    else:
        print('No Fisher matrix saved')
        pass

# old way to compute the FM, slow
# # COMPUTE FM GO
# FM_WL_GO = mm.compute_FM_2D(nbl_WL, zpairs_auto, nparams_tot, cov_WL_GO_2D_inv, dC_LL_2D)
# FM_GC_GO = mm.compute_FM_2D(nbl_GC, zpairs_auto, nparams_tot, cov_GC_GO_2D_inv, dC_GG_2D)
# FM_WA_GO = mm.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GO_2D_inv, dC_WA_2D)
# FM_3x2pt_GO = mm.compute_FM_2D(nbl_3x2pt, zpairs_3x2pt, nparams_tot, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D)
#
# # COMPUTE FM GS
# FM_WL_GS = mm.compute_FM_2D(nbl_WL, zpairs_auto, nparams_tot, cov_WL_GS_2D_inv, dC_LL_2D)
# FM_GC_GS = mm.compute_FM_2D(nbl_GC, zpairs_auto, nparams_tot, cov_GC_GS_2D_inv, dC_GG_2D)
# FM_WA_GS = mm.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GS_2D_inv, dC_WA_2D)
# FM_3x2pt_GS = mm.compute_FM_2D(nbl_3x2pt, zpairs_3x2pt, nparams_tot, cov_3x2pt_GS_2D_inv, dC_3x2pt_2D)
