import gc
import time
import warnings
import numpy as np
import scipy
from spaceborne import sb_lib as sl
from spaceborne import cl_utils as cl_utils


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
            dC_3D[ell, :, alf] = sl.array_2D_to_1D_ind(dC_4D[ell, :, :, alf], zpairs, ind)
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
        dC_4D = np.zeros((n_probes, n_probes, nbl, zbins, zbins, len(param_names)))
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
            print(f'Derivative {derivatives_prefix}{param_name} not found; setting the corresponding FM entry to zero')
            no_derivative_counter += 1
        if no_derivative_counter == len(param_names):
            raise ImportError('No derivative found for any of the parameters in the input dictionary')
    return dC_4D


def invert_matrix_LU(covariance_matrix):
    # Perform LU decomposition
    P, L, U = scipy.linalg.lu(covariance_matrix)
    # Invert the matrix using the decomposition
    return np.linalg.inv(L) @ np.linalg.inv(U) @ P


def ell_cuts_derivatives(FM_cfg, ell_dict, dC_LL_4D, dC_GG_4D, dC_3x2pt_6D):
    raise Exception('this function works, but you need to cut the covariance matrix using the corresponsing indices, '
                    'ie using the "1-dimensional cutting" approach by Vincenzo')

    if not FM_cfg['deriv_ell_cuts']:
        return dC_LL_4D, dC_GG_4D, dC_3x2pt_6D

    print('Performing the ell cuts on the derivatives...')

    ell_cuts_dict = ell_dict['ell_cuts_dict']
    ell_cuts_LL = ell_cuts_dict['LL']
    ell_cuts_GG = ell_cuts_dict['GG']
    param_names_3x2pt = FM_cfg['param_names_3x2pt']

    cl_cut = cl_utils.cl_ell_cut  # just to abbreviate the name to fit in one line
    for param_idx in range(len(param_names_3x2pt)):
        dC_LL_4D[:, :, :, param_idx] = cl_cut(dC_LL_4D[:, :, :, param_idx], ell_cuts_LL, ell_dict['ell_WL'])
        dC_GG_4D[:, :, :, param_idx] = cl_cut(dC_GG_4D[:, :, :, param_idx], ell_cuts_GG, ell_dict['ell_GC'])
        dC_3x2pt_6D[:, :, :, :, :, param_idx] = cl_utils.cl_ell_cut_3x2pt(
            dC_3x2pt_6D[:, :, :, :, :, param_idx], ell_cuts_dict, ell_dict['ell_3x2pt'])

    return dC_LL_4D, dC_GG_4D, dC_3x2pt_6D


def compute_FM(covariance_cfg, fm_cfg, ell_dict, cov_dict, deriv_dict, BNT_matrix=None):
    
    # shorten names
    GL_or_LG = fm_cfg['GL_or_LG']
    zbins = fm_cfg['zbins']
    ind = fm_cfg['ind']
    block_index = fm_cfg['block_index']
    param_names_3x2pt = [param for param in fm_cfg['FM_ordered_params'].keys() if param != 'ODE']
    nparams_tot = len(param_names_3x2pt)

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_XC, nbl_3x2pt = ell_GC, nbl_GC
    
    # set the flattening convention for the derivatives vector, based on the setting used to reduce the covariance
    # matrix' dimensions
    # TODO review this
    if block_index in ['ell', 'vincenzo', 'C-style']:
        which_flattening = 'C'
    elif block_index in ['ij', 'sylvain', 'F-style']:
        which_flattening = 'F'
    else:
        raise ValueError("block_index should be either 'ell', 'vincenzo', 'C-style', 'ij', 'sylvain' or 'F-style'")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)

    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[zpairs_auto:(zpairs_auto + zpairs_cross), [2, 3]] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), [3, 2]]

    # ! import and invert covariance matrices
    print('Starting covariance matrix inversion...')
    start_time = time.perf_counter()
    cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_g_2D'])
    cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_g_2D'])
    cov_XC_GO_2D_inv = np.linalg.inv(cov_dict['cov_XC_g_2D'])
    cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_g_2D'])
    # cov_2x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_2x2pt_g_2D'])
    print(f'Gaussian covariance matrices inverted in {(time.perf_counter() - start_time):.2f} s')

    if fm_cfg['compute_SSC']:
        start_time = time.perf_counter()
        cov_WL_GS_2D_inv = np.linalg.inv(cov_dict['cov_WL_g_2D'] + cov_dict['cov_WL_ssc_2D'] +  cov_dict['cov_WL_cng_2D'])
        cov_GC_GS_2D_inv = np.linalg.inv(cov_dict['cov_GC_g_2D'] + cov_dict['cov_GC_ssc_2D'] +  cov_dict['cov_GC_cng_2D'])
        cov_XC_GS_2D_inv = np.linalg.inv(cov_dict['cov_XC_g_2D'] + cov_dict['cov_XC_ssc_2D'] +  cov_dict['cov_XC_cng_2D'])
        cov_3x2pt_GS_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_g_2D'] + cov_dict['cov_3x2pt_ssc_2D'] +  cov_dict['cov_3x2pt_cng_2D'])
        # cov_2x2pt_GS_2D_inv = np.linalg.inv(cov_dict['cov_2x2pt_tot_2D'])
        print(f'Total covariance matrices inverted in {(time.perf_counter() - start_time):.2f} s')
    else:
        cov_WL_GS_2D_inv = np.eye(cov_dict['cov_WL_tot_2D'].shape[0])
        cov_GC_GS_2D_inv = np.eye(cov_dict['cov_GC_tot_2D'].shape[0])
        cov_XC_GS_2D_inv = np.eye(cov_dict['cov_XC_tot_2D'].shape[0])
        cov_3x2pt_GS_2D_inv = np.eye(cov_dict['cov_3x2pt_tot_2D'].shape[0])
        # cov_2x2pt_GS_2D_inv = np.eye(cov_dict['cov_2x2pt_tot_2D'].shape[0])
        warnings.warn('Skipping computing g+ng (tot) constraints, setting the inverse covmats to identity')
        
    del cov_dict
    gc.collect()

    # load reshaped derivatives, with shape (nbl, zbins, zbins, nparams)
    dC_LL_4D = deriv_dict['dC_LL_4D']
    dC_GG_4D = deriv_dict['dC_GG_4D']
    dC_3x2pt_6D = deriv_dict['dC_3x2pt_6D']
    
    assert dC_LL_4D.shape == (nbl_WL, zbins, zbins, nparams_tot), f'dC_LL_4D has incorrect shape: {dC_LL_4D.shape}'
    assert dC_GG_4D.shape == (nbl_GC, zbins, zbins, nparams_tot), f'dC_GG_4D has incorrect shape: {dC_GG_4D.shape}'
    assert dC_3x2pt_6D.shape == (2, 2, nbl_3x2pt, zbins, zbins, nparams_tot), \
        f'dC_3x2pt_6D has incorrect shape: {dC_3x2pt_6D.shape}'

    if fm_cfg['derivatives_BNT_transform']:

        # assert covariance_cfg['cov_BNT_transform'], 'you should BNT transform the covariance as well'
        assert BNT_matrix is not None, 'you should provide a BNT matrix'

        print('BNT-transforming the derivatives..')
        for param_idx in range(len(param_names_3x2pt)):
            dC_LL_4D[:, :, :, param_idx] = cl_utils.cl_BNT_transform(dC_LL_4D[:, :, :, param_idx], BNT_matrix, 'L', 'L')
            dC_3x2pt_6D[:, :, :, :, :, param_idx] = cl_utils.cl_BNT_transform_3x2pt(
                dC_3x2pt_6D[:, :, :, :, :, param_idx], BNT_matrix)

    # ! ell-cut the derivatives in 3d
    # dC_LL_4D_v1, dC_GG_4D_v1, dC_3x2pt_6D_v1 = ell_cuts_derivatives(FM_cfg, ell_dict,
    #                                                                              dC_LL_4D,
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

    np.testing.assert_allclose(dC_GGfor3x2pt_4D, dC_GG_4D, atol=0, rtol=1e-5,
                               err_msg="dC_GGfor3x2pt_4D and dC_GG_4D are not equal")
    assert nbl_3x2pt == nbl_GC, 'nbl_3x2pt and nbl_GC are not equal'

    # flatten z indices, obviously following the ordering given in ind
    # separate the ind for the different probes
    ind_auto = ind[:zpairs_auto, :]
    ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]  # ! watch out for the ind switch!!

    dC_LL_3D = dC_4D_to_3D(dC_LL_4D, nbl_WL, zpairs_auto, nparams_tot, ind_auto)
    dC_GG_3D = dC_4D_to_3D(dC_GG_4D, nbl_GC, zpairs_auto, nparams_tot, ind_auto)
    dC_LLfor3x2pt_3D = dC_4D_to_3D(dC_LLfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto)
    dC_XCfor3x2pt_3D = dC_4D_to_3D(dC_XCfor3x2pt_4D, nbl_3x2pt, zpairs_cross, nparams_tot, ind_cross)
    dC_GGfor3x2pt_3D = dC_4D_to_3D(dC_GGfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto)

    # concatenate the flattened components of the 3x2pt datavector
    dC_3x2pt_3D = np.concatenate((dC_LLfor3x2pt_3D, dC_XCfor3x2pt_3D, dC_GGfor3x2pt_3D), axis=1)
    # dC_2x2pt_3D = np.concatenate((dC_XCfor3x2pt_3D, dC_GGfor3x2pt_3D), axis=1)

    # collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
    # in the same way not with the old datavector, which was ordered in a different way...
    dC_LL_2D = np.reshape(dC_LL_3D, (nbl_WL * zpairs_auto, nparams_tot), order=which_flattening)
    dC_GG_2D = np.reshape(dC_GG_3D, (nbl_GC * zpairs_auto, nparams_tot), order=which_flattening)
    dC_XC_2D = np.reshape(dC_XCfor3x2pt_3D, (nbl_3x2pt * zpairs_cross, nparams_tot), order=which_flattening)
    dC_3x2pt_2D = np.reshape(dC_3x2pt_3D, (nbl_3x2pt * zpairs_3x2pt, nparams_tot), order=which_flattening)
    # dC_2x2pt_2D = np.reshape(dC_2x2pt_3D, (nbl_3x2pt * (zpairs_3x2pt - zpairs_auto), nparams_tot), order=which_flattening)

    # ! cut the *flattened* derivatives vector
    if fm_cfg['deriv_ell_cuts']:
        print('Performing the ell cuts on the derivatives...')
        dC_LL_2D = np.delete(dC_LL_2D, ell_dict['idxs_to_delete_dict']['LL'], axis=0)
        dC_GG_2D = np.delete(dC_GG_2D, ell_dict['idxs_to_delete_dict']['GG'], axis=0)
        dC_XC_2D = np.delete(dC_XC_2D, ell_dict['idxs_to_delete_dict'][GL_or_LG], axis=0)
        dC_3x2pt_2D = np.delete(dC_3x2pt_2D, ell_dict['idxs_to_delete_dict']['3x2pt'], axis=0)
        # raise ValueError('the above cuts are correct, but I should be careful when defining the 2x2pt datavector/covmat,\
            # as n_elem_ll will be lower because of the cuts...')


    ######################### COMPUTE FM #####################################

    start = time.perf_counter()
    FM_WL_GO = np.einsum('ia,ik,kb->ab', dC_LL_2D, cov_WL_GO_2D_inv, dC_LL_2D, optimize='optimal')
    FM_GC_GO = np.einsum('ia,ik,kb->ab', dC_GG_2D, cov_GC_GO_2D_inv, dC_GG_2D, optimize='optimal')
    FM_XC_GO = np.einsum('ia,ik,kb->ab', dC_XC_2D, cov_XC_GO_2D_inv, dC_XC_2D, optimize='optimal')
    FM_3x2pt_GO = np.einsum('ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D, optimize='optimal')
    # FM_2x2pt_GO = np.einsum('ia,ik,kb->ab', dC_2x2pt_2D, cov_2x2pt_GO_2D_inv, dC_2x2pt_2D, optimize='optimal')
    print(f'GO FM done in {(time.perf_counter() - start):.2f} s')

    start = time.perf_counter()
    FM_WL_GS = np.einsum('ia,ik,kb->ab', dC_LL_2D, cov_WL_GS_2D_inv, dC_LL_2D, optimize='optimal')
    FM_GC_GS = np.einsum('ia,ik,kb->ab', dC_GG_2D, cov_GC_GS_2D_inv, dC_GG_2D, optimize='optimal')
    FM_XC_GS = np.einsum('ia,ik,kb->ab', dC_XC_2D, cov_XC_GS_2D_inv, dC_XC_2D, optimize='optimal')
    FM_3x2pt_GS = np.einsum('ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GS_2D_inv, dC_3x2pt_2D, optimize='optimal')
    # FM_2x2pt_GS = np.einsum('ia,ik,kb->ab', dC_2x2pt_2D, cov_2x2pt_GS_2D_inv, dC_2x2pt_2D, optimize='optimal')
    print(f'GS FM done in {(time.perf_counter() - start):.2f} s')

    # store the matrices in the dictionary
    # probe_names = ['WL', 'GC', 'XC', '3x2pt', '2x2pt']
    # FMs_GO = [FM_WL_GO, FM_GC_GO, FM_XC_GO, FM_3x2pt_GO, FM_2x2pt_GO]
    # FMs_GS = [FM_WL_GS, FM_GC_GS, FM_XC_GS, FM_3x2pt_GS, FM_2x2pt_GS]
    probe_names = ['WL', 'GC', 'XC', '3x2pt']
    FMs_GO = [FM_WL_GO, FM_GC_GO, FM_XC_GO, FM_3x2pt_GO]
    FMs_GS = [FM_WL_GS, FM_GC_GS, FM_XC_GS, FM_3x2pt_GS]

    which_ng_cov_suffix = 'tot'

    FM_dict = {}
    if fm_cfg['compute_SSC']:
        for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
            FM_dict[f'FM_{probe_name}_G'] = FM_GO
            FM_dict[f'FM_{probe_name}_G{which_ng_cov_suffix}'] = FM_GS
    else:
        for probe_name, FM_GO in zip(probe_names, FMs_GO):
            FM_dict[f'FM_{probe_name}_GO'] = FM_GO

    print("FMs computed in %.2f seconds" % (time.perf_counter() - start))

    return FM_dict


def save_FM(fm_folder, FM_dict, FM_cfg, cases_tosave, save_txt=False, save_dict=True, **save_specs):
    """saves the FM in .txt and .pickle formats
    :param fm_folder:
    """
    raise DeprecationWarning(
        'this function is too convoluted, no need to save individual txt files? maybe it makes sense for git...')

    ell_max_WL = save_specs['ell_max_WL']
    ell_max_GC = save_specs['ell_max_GC']
    ell_max_3x2pt = save_specs['ell_max_3x2pt']
    nbl_WL = save_specs['nbl_WL']
    nbl_GC = save_specs['nbl_GC']
    nbl_WA = save_specs['nbl_WA']
    nbl_3x2pt = save_specs['nbl_3x2pt']

    probe_list = ['WL', 'GC', '3x2pt', 'WA']
    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_3x2pt, ell_max_WL]
    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]

    # TODO deprecate cases_tosave
    # TODO deprecate this, do I really need to save the different FM in txt format?
    # if save_txt:
    #     # there is no SSC-only Fisher!
    #     if 'SS' in cases_tosave:
    #         cases_tosave.remove('SS')
    #     for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
    #         for which_cov in cases_tosave:
    #             FM_txt_filename = FM_cfg['FM_txt_filename'].format(probe=probe, which_cov=which_cov, ell_max=ell_max,
    #                                                                nbl=nbl, **save_specs)
    #             np.savetxt(f'{fm_folder}/{FM_txt_filename}.txt', FM_dict[f'FM_{probe}_{which_cov}'])

    if save_dict:
        FM_dict_filename = FM_cfg['FM_dict_filename'].format(**save_specs)
        sl.save_pickle(f'{fm_folder}/{FM_dict_filename}.pickle', FM_dict)

    else:
        print('No Fisher matrix saved')
        pass

# old way to compute the FM, slow - legacy code
# # COMPUTE FM GO
# FM_WL_GO = sl.compute_FM_2D(nbl_WL, zpairs_auto, nparams_tot, cov_WL_GO_2D_inv, dC_LL_2D)
# FM_GC_GO = sl.compute_FM_2D(nbl_GC, zpairs_auto, nparams_tot, cov_GC_GO_2D_inv, dC_GG_2D)
# FM_WA_GO = sl.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GO_2D_inv, dC_WA_2D)
# FM_3x2pt_GO = sl.compute_FM_2D(nbl_3x2pt, zpairs_3x2pt, nparams_tot, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D)
#
# # COMPUTE FM GS
# FM_WL_GS = sl.compute_FM_2D(nbl_WL, zpairs_auto, nparams_tot, cov_WL_GS_2D_inv, dC_LL_2D)
# FM_GC_GS = sl.compute_FM_2D(nbl_GC, zpairs_auto, nparams_tot, cov_GC_GS_2D_inv, dC_GG_2D)
# FM_WA_GS = sl.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GS_2D_inv, dC_WA_2D)
# FM_3x2pt_GS = sl.compute_FM_2D(nbl_3x2pt, zpairs_3x2pt, nparams_tot, cov_3x2pt_GS_2D_inv, dC_3x2pt_2D)
