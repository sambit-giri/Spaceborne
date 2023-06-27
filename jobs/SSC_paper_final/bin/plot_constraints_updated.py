import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append('../../bin/plot_FM_running')
import plots_FM_running as plot_utils

sys.path.append('../../../common_lib_and_cfg/common_config')
import mpl_cfg

sys.path.append('../../../common_lib_and_cfg/common_lib')
import my_module as mm

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
mpl.use('Qt5Agg')

# ! options
specs_str = 'idMag0-idRSD0-idFS0-idSysWL3-idSysGC4'
nbl_WL_opt = 32
ep_or_ed = 'EP'
zbins = 10
num_params_tokeep = 7
fix_curvature = True
fix_galaxy_bias = False
fix_shear_bias = False
fix_dz = True
include_fom = True
fid_shear_bias_prior = 1e-4
shear_bias_priors = [.5e-4, 5e-4, 50e-4]
shear_bias_prior = .5e-4
galaxy_bias_perc_priors = [0.001, 1, 1000]  # = sigma_b / b, with sigma_b the prior and b the fiducial value
add_shear_bias_prior = True
add_galaxy_bias_prior = True
string_columns = ['probe', 'go_or_gs', 'fix_shear_bias', 'add_shear_bias_prior', 'shear_bias_prior',
                  'galaxy_bias_perc_prior']
probe_vinc_toplot = 'GCO'
# ! options


go_or_gs_folder_dict = {
    'GO': 'GaussOnly',
    'GS': 'GaussSSC',
}
probes_vinc = ('WLO', 'GCO', '3x2pt')
probes_vinc = ('WLO',)

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', 'GS']:
    for fix_galaxy_bias in [False, True]:
        for probe_vinc in probes_vinc:
            # for shear_bias_prior in shear_bias_priors:
            for galaxy_bias_perc_prior in galaxy_bias_perc_priors:

                names_params_to_fix = []

                fm_path = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1_restored/' \
                          f'FishMat_restored/{go_or_gs_folder_dict[go_or_gs]}/{probe_vinc}/FS1NoCuts'
                fm_name = f'fm-{probe_vinc}-{nbl_WL_opt}-wzwaCDM-NonFlat-GR-TB-{specs_str}-{ep_or_ed}{zbins}'
                fm = np.genfromtxt(f'{fm_path}/{fm_name}.dat')

                with open('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/'
                          'fiducial_params_dict_for_FM.yml') as f:
                    fiducials_dict = yaml.safe_load(f)

                assert len(fiducials_dict) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'

                # fix some of the parameters (i.e., which columns to remove)
                if fix_curvature:
                    print('fixing curvature')
                    names_params_to_fix += ['Om_Lambda0']

                if fix_shear_bias:
                    print('fixing shear bias parameters')
                    names_params_to_fix += [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]

                if fix_galaxy_bias:
                    print('fixing galaxy bias parameters')
                    names_params_to_fix += [f'b{(zi + 1):02d}_photo' for zi in range(zbins)]

                if fix_dz:
                    print('fixing dz parameters')
                    names_params_to_fix += [f'dz{(zi + 1):02d}_photo' for zi in range(zbins)]

                fm, fiducials_dict = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix,
                                                   remove_null_rows_cols=True)

                param_names = list(fiducials_dict.keys())

                # add prior on shear bias
                if add_shear_bias_prior and not fix_shear_bias and probe_vinc != 'GCO':
                    shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                    shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
                    fm = mm.add_prior_to_fm(fm, fiducials_dict, shear_bias_param_names, shear_bias_prior_values)

                # add prior on galaxy bias
                if add_galaxy_bias_prior and not fix_galaxy_bias and probe_vinc != 'WLO':
                    galaxy_bias_param_names = [f'b{(zi + 1):02d}_photo' for zi in range(zbins)]

                    # go from sigma_b / b_fid to sigma_b
                    galaxy_bias_idxs = [param_names.index(galaxy_bias_param_name)
                                        for galaxy_bias_param_name in galaxy_bias_param_names]
                    # ! update the fiducial values to FS1!!!!

                    galaxy_bias_fid_values = np.array(list(fiducials_dict.values()))[galaxy_bias_idxs]
                    galaxy_bias_prior_values = galaxy_bias_perc_prior * galaxy_bias_fid_values
                    fm = mm.add_prior_to_fm(fm, fiducials_dict, galaxy_bias_param_names, galaxy_bias_prior_values)

                uncert_fm = mm.uncertainties_fm_v2(fm, fiducials_dict, which_uncertainty='marginal',
                                                   normalize=True,
                                                   percent_units=True)[:num_params_tokeep]

                # compute the FoM
                w0wa_idxs = param_names.index('w_0'), param_names.index('w_a')
                fom = mm.compute_FoM(fm, w0wa_idxs)

                df_columns_names = string_columns + [param_name for param_name in fiducials_dict.keys()][
                                                    :num_params_tokeep] + ['FoM']

                # this is a list of lists just to have a 'row list' instead of a 'column list',
                # I still haven't figured out the problem...
                df_columns_values = [[probe_vinc, go_or_gs, fix_shear_bias,
                                      add_shear_bias_prior, shear_bias_prior, galaxy_bias_perc_prior] +
                                     uncert_fm.tolist() + [fom]]

                # assert False
                assert len(df_columns_names) == len(df_columns_values[0]), 'Wrong number of columns!'

                fm_uncert_df_to_concat = pd.DataFrame(df_columns_values, columns=df_columns_names)
                fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)

# try again the percent diff thing
df_gs = fm_uncert_df[fm_uncert_df["go_or_gs"] == "GS"]
df_go = fm_uncert_df[fm_uncert_df["go_or_gs"] == "GO"]
arr_gs = df_gs.iloc[:, len(string_columns):].select_dtypes('number').values
arr_go = df_go.iloc[:, len(string_columns):].select_dtypes('number').values
perc_diff_df = df_gs
perc_diff_df.iloc[:, len(string_columns):] = (arr_gs / arr_go - 1) * 100
perc_diff_df['go_or_gs'] = 'perc_diff'
perc_diff_df.iloc[:, -1] = (arr_go[:, -1] / arr_gs[:, -1] - 1) * 100
fm_uncert_df = pd.concat([fm_uncert_df, perc_diff_df], axis=0, ignore_index=True)

ylabel = r'$(\sigma_{\rm GS}/\sigma_{\rm G} - 1) \times 100$ [%]'

# shorten the dataframe to only one probe and one type of uncertainty (perc_diff)
fm_uncert_df_toplot = fm_uncert_df[(fm_uncert_df['go_or_gs'] == 'GO') &
                                   (fm_uncert_df['probe'] == probe_vinc_toplot) &
                                   (fm_uncert_df['fix_shear_bias'] == False)
                                   ]

data = fm_uncert_df_toplot.iloc[:, len(string_columns):].values
label_list = [f'fix_shear_bias={fix_shear_bias}; shear_bias_prior={shear_bias:02f}' for fix_shear_bias, shear_bias in
              zip(fm_uncert_df_toplot['fix_shear_bias'].values,
                  fm_uncert_df_toplot['shear_bias_prior'].values)]
label_list = [f'galaxy_bias_perc_prior={galaxy_bias_perc_prior:02f}' for galaxy_bias_perc_prior in
              fm_uncert_df_toplot['galaxy_bias_perc_prior'].values]
title = None

if include_fom:
    num_params_tokeep += 1
plot_utils.bar_plot(data, title, label_list, bar_width=0.2, nparams=num_params_tokeep, param_names_label=None,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom, figsize=(10, 8))

plt.savefig('../output/plots/WL_vs_GC_vs_3x2pt_GOGS_perc_uncert_increase.pdf', bbox_inches='tight', dpi=600)
