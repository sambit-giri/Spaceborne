import numpy as np
import os

ROOT = os.getenv('ROOT')
DATA_ROOT = f'{ROOT}/common_data/Spaceborne/jobs/SPV3'
SPV3_folder = f'{ROOT}/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3'

# TODO to be updated for DR1:
fsky_dr1 = 0.06841711056057495
area_deg2_dr1 = 2822

# * SETTINGS TO CHANGE (IN GENERAL) FOR CLUSTER RUNS
# * 'SSC_code'
# * 'load_precomputed_cov': False,
# * 'save_cov': True,
# * use_CLOE_cls: False, or True, remember I don't have the derivatives
# * fm_last_folder
# * fm_and_cov_suffix
# * which_sigma2_B
# * ell_max_GC: 5000,
# * ell_max_3x2pt: 5000,
# * mpl.use('Agg') in the main
 


which_forecast = 'SPV3'
fsky = 0.3563380664078408
GL_or_LG = 'GL'


fm_last_folder = '/jan_2024'
fm_and_cov_suffix = '_cNG_splinedefault'

# ! choose the flagship version and whether you want to use the BNT transform
flagship_version = 2

cl_BNT_transform = False
cov_BNT_transform = False
deriv_BNT_transform = False

cl_ell_cuts = False
cov_ell_cuts = False
deriv_ell_cuts = False

if cl_BNT_transform or cov_BNT_transform or deriv_BNT_transform:
    BNT_transform = True
else:
    BNT_transform = False

if cl_ell_cuts or cov_ell_cuts or deriv_ell_cuts:
    ell_cuts = True
else:
    ell_cuts = False

if cl_ell_cuts:
    assert cov_ell_cuts is False, 'if you want to apply ell cuts to the cls, you cannot apply them to the cov'
    assert deriv_ell_cuts, 'if you want to apply ell cuts to the cls, you hould also apply them to the derivatives'

if cov_ell_cuts:
    assert cl_ell_cuts is False, 'if you want to apply ell cuts to the cov, you cannot apply them to the cls'
    assert deriv_ell_cuts, 'if you want to apply ell cuts to the cov, you hould also apply them to the derivatives'

assert flagship_version == 2, 'the files for the multicut case are only available for Flagship_2'

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'

general_cfg = {
    'fid_yaml_filename': ROOT + '/Spaceborne/common_cfg/SPV3_fiducial_params_magcut245_zbins{zbins:02d}.yml',
    'ell_min': 10,
    'ell_max_WL': 3000,
    'ell_max_GC': 3000,
    'ell_max_3x2pt': 3000,
    'zbins': 13,
    'zbins_list': None,
    'EP_or_ED': 'EP',
    'n_probes': 2,
    'which_forecast': which_forecast,
    'use_WA': False,
    'save_cls_3d': False,
    'save_rls_3d': False,

    'flat_or_nonflat': 'Flat',  # Flat or NonFlat

    # the case with the largest range is nbl_WL_opt.. This is the reference ell binning from which the cuts are applied;
    # in principle, the other binning should be consistent with this one and should not be hardcoded, as long as
    # lmax=5000, 3000 holds
    'nbl_WL_opt': 32,  # this is the value from which the various bin cuts are applied, do not change it
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied, do not change it

    'ell_cuts': ell_cuts,
    'which_cuts': 'Vincenzo',
    'center_or_min': 'center',  # cut if the bin *center* or the bin *lower edge* is larger than ell_max[zi, zj]
    'cl_ell_cuts': cl_ell_cuts,
    'ell_cuts_folder': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/ell_cuts',
    'ell_cuts_filename': 'lmax_cut_{probe:s}_{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                         'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'kmax_h_over_Mpc_ref': 1.0,  # this is used when ell_cuts is False, also...?
    # 'kmax_list_1_over_Mpc': np.array((0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 3.00, 5.00, #10.00)),
    # 'kmax_h_over_Mpc_list': np.array([0.37108505, 0.74217011, 1.11325516, 1.48434021, 1.85542526,
    #                                   2.22651032, 2.59759537, 2.96868042, 4.45302063, 7.42170105, 14.84340211]),
    'kmax_h_over_Mpc_list': np.array([0.1, 0.16681005, 0.27825594, 0.46415888, 0.77426368,
                                      1.29154967, 2.15443469, 3.59381366, 5.9948425, 10.]),

    'BNT_transform': BNT_transform,  # ! to be deprecated?
    'cl_BNT_transform': cl_BNT_transform,

    'idIA': 2,
    'idB': 3,
    'idM': 3,
    'idR': 1,
    'idBM': 1,  # XXX ! what is this?

    'which_pk': 'HMCodeBar',
    'which_pk_list': ('HMCodeBar', 'TakaBird', 'HMCode2020', 'Bacco', 'EE2'),

    'use_CLOE_cls': False,
    'cloe_bench_folder': f'{ROOT}/my_cloe_data',
    'cl_folder': SPV3_folder + '/OutputFiles/DataVectors/Noiseless/{which_pk:s}',
    'rl_folder': f'{SPV3_folder}' + '/OutputFiles/ResFun/{which_pk:s}',
    'cl_filename': 'dv-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-MS{magcut_source:03d}-idIA{idIA:d}-idB{idB:d}-idM{idM:d}-idR{idR:d}.dat',
    'rl_filename': 'resfun-idBM{idBM:02d}.dat',  # XXX it's idBM... anyway, not using the responses at the moment

    'zmax': 2.5,
    'magcut_source': 245,
    'magcut_lens': 245,
    'zcut_source': 2,
    'zcut_lens': 2,
    'flagship_version': flagship_version,

    'has_rsd': False,
    'has_magnification_bias': True,

    'CLOE_pk_filename': SPV3_folder+ '/InputFiles/InputPS/{which_pk:s}/InFiles/{flat_or_nonflat:s}/h/PddVsZedLogK-h_6.700e-01.dat'
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {

    # ordering-related stuff
    'triu_tril': 'triu',
    'row_col_major': 'row-major',
    'block_index': 'ell',
    'GL_or_LG': GL_or_LG,

    'which_probe_response': 'variable',
    'response_const_value': None,  # it used to be 4 for a constant probe response, which is quite wrong

    # n_gal, sigma_eps, fsky, all entering the covariance matrix
    'fsky': fsky,  # ! new
    'sigma_eps2': (0.26 * np.sqrt(2)) ** 2,  # ! new
    'ng': None,  # ! the new value is 28.73 (for Flagship_1), but I'm taking the value from the ngbTab files

    'nofz_folder': f'{SPV3_folder}/InputFiles/InputNz/NzFid',
    'nuisance_folder': f'{SPV3_folder}/InputFiles/InputNz/NzPar',
    'nofz_filename': 'nzTab-{EP_or_ED:s}{zbins:02d}-zedMin{zcut_source:02d}-zedMax{zmax:02d}-mag{magcut_source:03d}.dat',
    'nuisance_filename': 'ngbsTab-{EP_or_ED:s}{zbins:02d}-zedMin{zcut_source:02d}-zedMax{zmax:02d}-mag{magcut_source:03d}.dat',

    # sources (and lenses) redshift distributions
    # 'nofz_folder': f'{ROOT}/likelihood-mcmc-generator/input_files/SPV3',
    # 'nuisance_folder': f'{ROOT}/likelihood-mcmc-generator/input_files/SPV3',
    # 'nofz_filename': 'nzTabSPV3.dat',
    # 'nuisance_filename': 'nuiTabSPV3.dat',


    'shift_nz': True,  # ! are vincenzo's kernels shifted?? it looks like they are not
    'compute_bnt_with_shifted_nz_for_zcuts': False,  # ! let's test this
    'shift_nz_interpolation_kind': 'linear',
    'normalize_shifted_nz': True,
    'nz_gaussian_smoothing': False,  # does not seem to have a large effect...
    'nz_gaussian_smoothing_sigma': 2,
    'include_ia_in_bnt_kernel_for_zcuts': False,


    'plot_nz_tocheck': True,

    'cov_BNT_transform': cov_BNT_transform,
    'cov_ell_cuts': cov_ell_cuts,

    'compute_covmat': True,
    'compute_SSC': True,

    'save_cov': False,
    'cov_file_format': 'npz',  # or npy
    'save_cov_dat': False,  # this is the format used by Vincenzo

    # in cov_dict
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GO': False,
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_2DCLOE': False,  # outermost loop is on the probes

    'test_against_benchmarks': False,
    'test_against_CLOE_benchmarks': False,
    'test_against_vincenzo': False,
    'compute_GSSC_condition_number': False,

    # ! no folders for ell_cut_center or min
    'cov_folder': f'{DATA_ROOT}/output/Flagship_{flagship_version}/covmat/BNT_{BNT_transform}' + '/ell_cuts_{cov_ell_cuts:s}',
    'cov_filename': 'covmat_{which_cov:s}_{ng_cov_code:s}_{probe:s}_zbins{EP_or_ED:s}{zbins:02d}_'
                    'ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}_'
                    'idIA{idIA:1d}_idB{idB:1d}_idM{idM:1d}_idR{idR:1d}_pk{which_pk:s}_{ndim:d}D' + fm_and_cov_suffix,
    'cov_vinc_folder': f'{SPV3_folder}/OutputFiles/CovMats/GaussOnly/Full',
    'cov_vinc_filename': 'cmfull-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-MS{magcut_source:03d}-'
                         'idIA{idIA:d}-idB{idB:d}-idM{idM:d}-idR{idR:d}.npz',

    'SSC_code': 'OneCovariance',  # ! 'PySSC' or 'PyCCL' or 'Spaceborne' or 'OneCovariance'
    'check_if_recently_created': False,

    'PyCCL_cfg': {
        'probe': '3x2pt',
        # 'cNG' or 'SSC'. Which non-Gaussian covariance terms to compute. Must be a tuple
        'which_ng_cov': ('SSC', 'cNG', ),
        'test_GLGL': False,  # must be set to False for actual 3x2pt runs
        
        'which_pk_for_pyccl': 'PyCCL',  # 'PyCCL' (the one stored in cosmo obj) or 'CLOE' (from input files - to be implemented)

        'load_precomputed_cov': True,
        'save_cov': True,
        
        'load_precomputed_tkka': False,
        'save_hm_responses': True,
        'save_tkka': True,

        'cov_path': f'{DATA_ROOT}/output/Flagship_{flagship_version}/covmat/PyCCL' + fm_last_folder,
        'cov_filename': 'cov_{which_ng_cov:s}_pyccl_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_'
                        'nbl{nbl:d}_ellmax{lmax:d}_zbins{EP_or_ED:s}{zbins:02d}' + fm_and_cov_suffix + '.npz',


        'which_sigma2_B': 'mask',  # 'mask' or 'spaceborne' (with mask) or None
        'area_deg2_mask': 14700,
        'nside_mask': 2048,
        'ell_mask_filename': ROOT + '/common_data/mask/ell_circular_1pole_{area_deg2:d}deg2_nside{nside:d}.npy',
        'cl_mask_filename': ROOT + '/common_data/mask/Cell_circular_1pole_{area_deg2:d}deg2_nside{nside:d}.npy',
        # 'z_grid_sigma2_B_filename': ROOT + '/exact_SSC/output/sigma2/z_grid_sigma2_zsteps3000_ISTF.npy',
        # 'sigma2_B_filename': ROOT + '/exact_SSC/output/sigma2/sigma2_zsteps3000_ISTF.npy',
        'sigma2_suffix': 'mask',  # this is the filename suffix for the sigma2_B file saved directly from cov_SSC in CCL

        # z_grid min and max should probably coincide. play around with steps to find the minimum number
        'z_grid_tkka_min': 0.,
        'z_grid_tkka_max': 6,
        'k_grid_tkka_min': 1e-5,
        'k_grid_tkka_max': 1e2,
        'z_grid_tkka_steps_SSC': 200,
        'k_grid_tkka_steps_SSC': 1024,
        'z_grid_tkka_steps_cNG': 100,
        'k_grid_tkka_steps_cNG': 512,
        
        'n_samples_wf': 1000,
        'bias_model': 'polynomial',  # TODO this is not used at the moment (for SPV3)
    },

    'Spaceborne_cfg': {
        'probe': '3x2pt',
        'which_ng_cov': ('SSC', ),  # only 'SSC' available in this case
        'load_precomputed_cov': True,  # always True for the moment, I have to compute the integral with Julia

        'cov_path': f'{DATA_ROOT}/output/Flagship_{flagship_version}/covmat/Spaceborne/separate_universe' + fm_last_folder,
        'cov_filename': 'cov_{which_ng_cov:s}_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_nbl{nbl:d}_ellmax{lmax:d}'
                        '_zbins{EP_or_ED:s}{zbins:02d}_zsteps{z_steps_sigma2:d}_k{k_txt_label:s}'
                        '_convention{cl_integral_convention:s}.npy',

        # settings for sigma2
        'cl_integral_convention': 'PySSC',  # or Euclid, but gives same results as it should!!! TODO remove this
        'k_txt_label': '1overMpc',
        'use_precomputed_sigma2': True,  # still need to understand exactly where to call/save this
        'z_min_sigma2': 0.001,
        'z_max_sigma2': 3,
        'z_steps_sigma2': 2899,
        'log10_k_min_sigma2': -4,
        'log10_k_max_sigma2': 1,
        'k_steps_sigma2': 20_000,
    },

    'OneCovariance_cfg': {
        'which_ng_cov': ('cNG', ),
        'load_precomputed_cov': True,  # this must be True for OneCovariance
        'use_OneCovariance_Gaussian': False,

        'cov_path': f'{DATA_ROOT}/output/Flagship_2/covmat/OneCovariance/output_SPV3_v3_dense_LiFECls',
        # 'cov_path': f'{ROOT}/common_data/OneCovariance/output_SPV3_std',
        'cov_filename': 'cov_{which_ng_cov:s}_onecovariance_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_'
                        'nbl{nbl:d}_ellmax{lmax:d}_zbins{EP_or_ED:s}{zbins:02d}.npz',
    }

}

if ell_cuts:
    covariance_cfg['cov_filename'] = covariance_cfg['cov_filename'].replace('_{ndim:d}D',
                                                                            '_kmaxhoverMpc{kmax_h_over_Mpc:.03f}_{ndim:d}D')

Sijkl_cfg = {
    'wf_input_folder': f'{SPV3_folder}/InputFiles/InputSSC/Windows',
    'wf_filename': 'wi{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:02d}-MS{magcut_source:02d}-idIA{idIA:d}-idB{idB:d}-idM{idM:d}-idR{idR:d}.dat',
    'wf_normalization': 'IST',
    'nz': None,  # ! is this used?
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl

    'Sijkl_folder': f'{DATA_ROOT}/output/Flagship_{flagship_version}/sijkl',
    'Sijkl_filename': 'sijkl_WF-FS{flagship_version:01d}_nz{nz:d}_zbins{EP_or_ED:s}{zbins:02}_IA{IA_flag:}'
                      '_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

# param_names_dict = {
#     'cosmo': ["Om", "Ob", "wz", "wa", "h", "ns", "s8", 'logT_AGN'],
#     'IA': ["Aia", "eIA"],
#     'shear_bias': [f'm{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
#     'dzWL': [f'dzWL{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
#     'galaxy_bias': [f'bG{zbin_idx:02d}' for zbin_idx in range(1, 5)],
#     'magnification_bias': [f'bM{zbin_idx:02d}' for zbin_idx in range(1, 5)],
#     # 'dzGC': [f'dzGC{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)]
# }
# # declare the set of parameters under study
# param_names_3x2pt = list(np.concatenate([param_names_dict[key] for key in param_names_dict.keys()]))

# I cannot define the fiducial values here because I need to import the files for the galaxy bias


FM_txt_filename = covariance_cfg['cov_filename'].replace('covmat_', 'FM_').replace('_{ndim:d}D', '')
FM_dict_filename = covariance_cfg['cov_filename'].replace('covmat_', 'FM_')
FM_dict_filename = FM_dict_filename.replace('_{ndim:d}D', '')
FM_dict_filename = FM_dict_filename.replace('_{probe:s}', '')
FM_dict_filename = FM_dict_filename.replace('_{which_cov:s}', '_{which_ng_cov:s}')
deriv_filename = covariance_cfg['cov_filename'].replace('covmat_', 'dDVd')
FM_cfg = {
    'compute_FM': True,

    # 'param_names_dict': param_names_dict,
    # 'fiducials_dict': None,  # this needs to be set in the main, since it depends on the n_gal file
    # 'param_names_3x2pt': param_names_3x2pt,
    # 'nparams_tot': len(param_names_3x2pt),  # total (cosmo + nuisance) number of parameters

    'save_FM_txt': True,
    'save_FM_dict': True,

    'load_preprocess_derivatives': False,
    'derivatives_folder': SPV3_folder + '/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}/{EP_or_ED:s}{zbins:02d}',

    'derivatives_filename': '{derivatives_prefix:s}{param_name:s}-{probe:s}-ML{magcut_lens:03d}-MS{magcut_source:03d}-{EP_or_ED:s}{zbins:02d}.dat',
    'derivatives_prefix': 'dDVd',

    'derivatives_BNT_transform': deriv_BNT_transform,
    'deriv_ell_cuts': deriv_ell_cuts,

    'fm_folder': f'{DATA_ROOT}/output/Flagship_{flagship_version}/FM/' +
                 'BNT_{BNT_transform:s}/ell_cuts_{ell_cuts:s}/{which_cuts:s}/ell_{center_or_min:s}{fm_last_folder}',
    'fm_last_folder': fm_last_folder,
    'FM_txt_filename': FM_txt_filename,
    'FM_dict_filename': FM_dict_filename,

    'test_against_benchmarks': False,

    'test_against_vincenzo': False,
    'fm_vinc_folder': SPV3_folder + '/OutputFiles/FishMat/{go_gs_vinc:s}/{flat_or_nonflat:s}/{which_pk:s}',
    'fm_vinc_filename': 'fm-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-MS{magcut_source:03d}-idIA{idIA:d}-idB{idB:d}-idM{idM:d}-idR{idR:d}.dat',

}
