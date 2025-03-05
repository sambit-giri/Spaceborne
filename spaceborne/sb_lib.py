from copy import deepcopy
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import numpy as np
import yaml
import pickle
import itertools
import os
import inspect
import datetime
import scipy
from scipy.integrate import simpson as simps
from scipy.special import jv
from scipy.interpolate import interp1d, CubicSpline, RectBivariateSpline
import subprocess


symmetrize_output_dict = {
    ('L', 'L'): True,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): True,
}


mpl_rcParams_dict = {
    'lines.linewidth': 1.5,
    'font.size': 17,
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',
    #  'mathtext.fontset': 'stix',
    #  'font.family': 'STIXGeneral',
    'figure.figsize': (15, 10),
    'lines.markersize': 8,
    # 'axes.grid': True,
    # 'figure.constrained_layout.use': False,
    # 'axes.axisbelow': True
}

mpl_other_dict = {
    'cosmo_labels_TeX': ["$\\Omega_{{\\rm m},0}$", "$\\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                         "$\\sigma_8$", "${\\rm log}_{10}(T_{\\rm AGN}/{\\rm K})$"],
    'IA_labels_TeX': ['$A_{\\rm IA}$', '$\\eta_{\\rm IA}$', '$\\beta_{\\rm IA}$'],
    # 'galaxy_bias_labels_TeX': build_labels_TeX(zbins)[0],
    # 'shear_bias_labels_TeX': build_labels_TeX(zbins)[1],
    # 'zmean_shift_labels_TeX': build_labels_TeX(zbins)[2],

    'cosmo_labels': ['Om', 'Ob', 'wz', 'wa', 'h', 'ns', 's8', 'logT'],
    'IA_labels': ['AIA', 'etaIA', 'betaIA'],
    # 'galaxy_bias_labels': build_labels(zbins)[0],
    # 'shear_bias_labels': build_labels(zbins)[1],
    # 'zmean_shift_labels': build_labels(zbins)[2],

    'ylabel_perc_diff_wrt_mean': "$ \\bar{\\sigma}_\\alpha^i / \\bar{\\sigma}^{\\; m}_\\alpha -1 $ [%]",
    'ylabel_sigma_relative_fid': '$ \\sigma_\\alpha/ \\theta^{fid}_\\alpha $ [%]',
    'dpi': 500,

    'pic_format': 'pdf',
    'h_over_mpc_tex': '$h\\,{\\rm Mpc}^{-1}$',
    'kmax_tex': '$k_{\\rm max}$',
    'kmax_star_tex': '$k_{\\rm max}^\\star$',
}

def matshow_vcenter(matrix, vcenter=0):
    """Plots a matrix with a 0-centered, asymmetric colorbar."""
    from matplotlib.colors import TwoSlopeNorm

    plt.matshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=vcenter))
    plt.colorbar()
    plt.show()
    

def j0(x):
    return jv(0, x)


def j1(x):
    return jv(1, x)


def j2(x):
    return jv(2, x)



def import_cls(cl_tab_in: np.ndarray):

    assert cl_tab_in.shape[1] == 4, 'input cls should have 4 columns'
    assert np.min(cl_tab_in[:, 1]) == 0, 'tomographic redshift indices should start from 0'
    assert np.min(cl_tab_in[:, 2]) == 0, 'tomographic redshift indices should start from 0'
    assert np.max(cl_tab_in[:, 1]) == np.max(cl_tab_in[:, 2]), 'tomographic redshift indices should be \
        the same for both z_i and z_j'

    zbins = int(np.max(cl_tab_in[:, 1]) + 1)
    ell_values = np.unique(cl_tab_in[:, 0])

    cl_3d = np.zeros((len(ell_values), zbins, zbins))

    for row in range(cl_tab_in.shape[0]):
        ell_val, zi, zj = cl_tab_in[row, 0], int(cl_tab_in[row, 1]), int(cl_tab_in[row, 2])
        ell_ix = np.where(ell_values == ell_val)[0][0]
        cl_3d[ell_ix, zi, zj] = cl_tab_in[row, 3]

    return ell_values, cl_3d


def savetxt_aligned(filename, array_2d, header_list, col_width=25, decimals=8):

    header = ''
    for i in range(len(header_list)):
        offset = 2 if i == 0 else 0
        string = f"{header_list[i]:<{col_width - offset}}"
        header += string

    # header = ''.join([f"{header_list[i]:<{col_width - 2}}" for i in range(len(header_list))])
    fmt = [f'%-{col_width}.{decimals}f'] * len(array_2d[0])
    np.savetxt(filename, array_2d, header=header, fmt=fmt, delimiter='')


def nz_fits_to_txt(fits_filename):
    """
    Converts the official SGS-like fits file to the usual (z, nz) format.

    Parameters
    ----------
    fits_filename : str
        The full path and filename of the fits file to be converted.
    """

    import euclidlib as el

    z, nz = el.photo.redshift_distributions(fits_filename)

    nz_arr = np.zeros((len(z), len(nz) + 1))
    nz_arr[:, 0] = z

    plt.figure()
    for bin in nz:
        plt.plot(z, nz[bin])
        nz_arr[:, bin] = nz[bin]

    return nz_arr


def compare_funcs(x, y_tuple: dict, logscale_y=[False, False], logscale_x=False,
                  title=None, ylim_diff=None):

    names = list(y_tuple.keys())
    y_tuple = list(y_tuple.values())
    colors = plt.get_cmap("tab10").colors  # Get tab colors

    if x is None:
        x = np.arange(len(y_tuple[0]))

    fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[2, 1], )
    fig.subplots_adjust(hspace=0)

    for i, y in enumerate(y_tuple):
        ls = '--' if i > 0 else '-'
        # alpha = 0.8 if i > 0 else 1
        ax[0].plot(x, y, label=names[i], c=colors[i], ls=ls)
    ax[0].legend()

    for i in range(1, len(y_tuple)):
        ax[1].plot(x, percent_diff(y_tuple[i], y_tuple[0]), c=colors[i], ls='-')
    ax[1].set_ylabel('A/B - 1 [%]')
    ax[1].axhspan(-10, 10, alpha=0.2, color='gray')

    for i in range(2):
        if logscale_y[i]:
            ax[i].set_yscale('log')

    if logscale_x:
        for i in range(2):
            ax[i].set_xscale('log')

    if ylim_diff is not None:
        ax[1].set_ylim(ylim_diff)

    if title is not None:
        fig.suptitle(title)

    plt.show()


def get_git_info():
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).strip().decode('utf-8')

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).strip().decode('utf-8')

        return branch, commit
    except subprocess.CalledProcessError:
        return None, None


def mirror_upper_to_lower_vectorized(A):
    # Check if A is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix")

    # Create a copy of the original matrix
    result = A.copy()

    # Use numpy's triu_indices to get the indices of the upper triangle
    triu_indices = np.triu_indices_from(A, k=1)

    # Mirror the upper triangular elements to the lower triangle
    result[(triu_indices[1], triu_indices[0])] = A[triu_indices]

    return result


def check_interpolate_input_tab(input_tab: np.ndarray, z_grid_out: np.ndarray, zbins: int) -> tuple:
    """
    Interpolates the input table over the 0th dimension using a cubic spline 
    and returns the interpolated values on the specified grid.

    Parameters:
    - input_tab (numpy.ndarray): The input table with shape (z_points, zbins + 1).
    - z_grid_out (numpy.ndarray): The output grid for interpolation.
    - zbins (int): The number of redshift bins.

    Returns:
    - output_tab (numpy.ndarray): The interpolated table with shape (len(z_grid_out), zbins).
    """
    assert input_tab.shape[1] == zbins + 1, 'The input table should have shape (z_points, zbins + 1)'

    # Perform cubic spline interpolation
    spline = CubicSpline(x=input_tab[:, 0], y=input_tab[:, 1:], axis=0)
    output_tab = spline(z_grid_out)

    return output_tab, spline

# @deprecated(reason="ep_or_ed option has been deprecated")


def get_ngal(ngal_in, ep_or_ed, zbins, ep_check_tol):

    if isinstance(ngal_in, (int, float)):
        assert ep_or_ed == 'EP', 'n_gal must be a scalar in the equipopulated (EP) case'
        ngal_out = ngal_in

    elif type(ngal_in) is list:
        assert len(ngal_in) == zbins, 'n_gal must be a vector of length zbins'
        ngal_out = ngal_in

    elif type(ngal_in) is str:
        nofz = np.genfromtxt(ngal_in)
        assert nofz.shape[1] == zbins + 1, 'nz must be an array of shape (n_z_points, zbins + 1)'
        z_nofz = nofz[:, 0]
        nofz = nofz[:, 1:]
        ngal_out = simps(y=nofz, x=z_nofz, axis=0)

    if ep_or_ed == 'EP' and not isinstance(ngal_out, (int, float)):
        for zi in range(zbins):
            assert np.allclose(ngal_out[0], ngal_out[zi], atol=0, rtol=ep_check_tol), \
                'n_gal must be the same for all zbins in the equipopulated (EP) case'

    return ngal_out


def interp_2d_arr(x_in, y_in, z2d_in, x_out, y_out, output_masks):
    """
    Interpolate a 2D array onto a new grid using bicubic spline interpolation.

    Parameters:
    - x_in (numpy.ndarray): The x-coordinates of the input 2D array.
    - y_in (numpy.ndarray): The y-coordinates of the input 2D array.
    - z2d_in (numpy.ndarray): The 2D input array to be interpolated.
    - x_out (numpy.ndarray): The x-coordinates of the output grid.
    - y_out (numpy.ndarray): The y-coordinates of the output grid.
    - output_masks (bool): A boolean flag indicating whether to mask the output array.

    Returns:
    - x_out_masked (numpy.ndarray): The x-coordinates of the output grid, clipped to avoid interpolation errors.
    - y_out_masked (numpy.ndarray): The y-coordinates of the output grid, clipped to avoid interpolation errors.
    - z2d_interp (numpy.ndarray): The interpolated 2D array.
    - x_mask (numpy.ndarray): A boolean mask indicating which elements of the original x_out array were used.
    - y_mask (numpy.ndarray): A boolean mask indicating which elements of the original y_out array were used.
    """

    z2d_func = RectBivariateSpline(x=x_in, y=y_in, z=z2d_in)

    # clip x and y grids to avoid interpolation errors
    x_mask = np.logical_and(x_in.min() <= x_out, x_out < x_in.max())
    y_mask = np.logical_and(y_in.min() <= y_out, y_out < y_in.max())
    x_out_masked = x_out[x_mask]
    y_out_masked = y_out[y_mask]

    if len(x_out_masked) < len(x_out):
        print(f"x array trimmed: old range [{x_out.min():.2e}, {x_out.max():.2e}], "
              f"new range [{x_out_masked.min():.2e}, {x_out_masked.max():.2e}]")
    if len(y_out_masked) < len(y_out):
        print(f"y array trimmed: old range [{y_out.min():.2e}, {y_out.max():.2e}], "
              f'new range [{y_out_masked.min():.2e}, {y_out_masked.max():.2e}]')

    # with RegularGridInterpolator:
    # TODO untested
    # z2d_func = RegularGridInterpolator((x_in, y_in), z2d_in, method='linear')
    # xx, yy = np.meshgrid(x_out_masked, y_out_masked)
    # z2d_interp = z2d_interp((xx, yy)).T

    z2d_interp = z2d_func(x_out_masked, y_out_masked)

    if output_masks:
        return x_out_masked, y_out_masked, z2d_interp, x_mask, y_mask
    else:
        return x_out_masked, y_out_masked, z2d_interp


def test_cov_FM(output_path, benchmarks_path, extension):
    """tests that the outputs do not change between the old and the new version"""
    old_dict = dict(get_kv_pairs(benchmarks_path, extension))
    new_dict = dict(get_kv_pairs(output_path, extension))

    # check if the dictionaries are empty
    assert len(old_dict) > 0, 'No files in the benchmarks path ❌'
    assert len(new_dict) > 0, 'No files in the output path ❌'

    assert old_dict.keys() == new_dict.keys(), 'The number of files or their names has changed ❌'

    if extension == 'npz':
        for key in old_dict.keys():
            try:
                np.array_equal(old_dict[key]['arr_0'], new_dict[key]['arr_0'])
            except AssertionError:
                f'The file {benchmarks_path}/{key}.{extension} is different ❌'
    else:
        for key in old_dict.keys():
            try:
                np.array_equal(old_dict[key], new_dict[key])
            except AssertionError:
                f'The file {benchmarks_path}/{key}.{extension} is different ❌'

    print('tests passed successfully: the outputs are the same as the benchmarks ✅')


def regularize_covariance(cov_matrix, lambda_reg=1e-5):
    """
    Regularizes the covariance matrix by adding lambda * I.

    Parameters:
    - cov_matrix: Original covariance matrix (numpy.ndarray)
    - lambda_reg: Regularization parameter

    Returns:
    - Regularized covariance matrix
    """
    n = cov_matrix.shape[0]
    identity_matrix = np.eye(n)
    cov_matrix_reg = cov_matrix + lambda_reg * identity_matrix
    return cov_matrix_reg


def get_simpson_weights(n):
    """
    Function written by Marco Bonici
    """
    number_intervals = (n - 1) // 2
    weight_array = np.zeros(n)
    if n == number_intervals * 2 + 1:
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
    else:
        weight_array[0] += 0.5
        weight_array[1] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i + 1] += 1 / 3
            weight_array[2 * i + 2] += 4 / 3
            weight_array[2 * i + 3] += 1 / 3
        weight_array[-1] += 0.5
        weight_array[-2] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
        weight_array /= 2
    return weight_array


def zpair_from_zidx(zidx, ind):
    """ Return the zpair corresponding to the zidx for a given ind array.
    To be thoroughly tested, but quite straightforward"""
    assert ind.shape[1] == 2, 'ind array must have shape (n, 2), maybe you are passing the full ind file instead of ind_auto/ind_cross'
    return np.where((ind == [zidx, zidx]).all(axis=1))[0][0]


def plot_dominant_array_element(arrays_dict, tab_colors, elements_auto, elements_cross, elements_3x2pt):
    """
    Plot 2D arrays from a dictionary, highlighting the dominant component in each element.
    Colors are assigned based on the array with the dominant component at each position.
    If no component is dominant (all are zero), the color will be white.
    """

    centers = [elements_auto // 2, elements_auto + elements_cross //
               2, elements_auto + elements_cross + elements_auto // 2]
    labels = ['WL', 'GGL', 'GCph']

    # Stack arrays along a new dimension and calculate the absolute values
    stacked_abs_arrays = np.abs(np.stack(list(arrays_dict.values()), axis=-1))

    # Find indices of the dominant array at each position
    dominant_indices = np.argmax(stacked_abs_arrays, axis=-1)

    # Add an extra category for non-dominant cases (where all arrays are zero)
    non_dominant_value = -1  # Choose a value that doesn't conflict with existing indices
    dominant_indices[np.all(stacked_abs_arrays == 0, axis=-1)] = non_dominant_value

    # Prepare the colormap, including an extra color for non-dominant cases
    selected_colors = ['white'] + tab_colors[:len(arrays_dict)]  # 'white' is for non-dominant cases
    cmap = ListedColormap(selected_colors)

    # Plot the dominant indices with the custom colormap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dominant_indices, cmap=cmap, vmin=non_dominant_value, vmax=len(arrays_dict) - 1)

    # Create a colorbar with labels
    # Set the ticks so they are at the center of each color segment
    cbar_ticks = np.linspace(non_dominant_value, len(arrays_dict) - 1, len(selected_colors))
    cbar_labels = ['0'] + list(arrays_dict.keys())  # 'None' corresponds to the non-dominant case
    cbar = plt.colorbar(im, ticks=cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    lw = 2
    plt.axvline(elements_auto, c='k', lw=lw)
    plt.axvline(elements_auto + elements_cross, c='k', lw=lw)
    plt.axhline(elements_auto, c='k', lw=lw)
    plt.axhline(elements_auto + elements_cross, c='k', lw=lw)
    plt.xticks([])
    plt.yticks([])

    for idx, label in enumerate(labels):
        x = centers[idx]
        plt.text(x, -1.5, label, va='bottom', ha='center')
        plt.text(-1.5, x, label, va='center', ha='right', rotation='vertical')

    plt.show()


def cov_3x2pt_dict_8d_to_10d(cov_3x2pt_dict_8D, nbl, zbins, ind_dict, probe_ordering,
                             symmetrize_output_dict: bool = symmetrize_output_dict):
    cov_3x2pt_dict_10D = {}
    for probe_A, probe_B in probe_ordering:
        for probe_C, probe_D in probe_ordering:
            cov_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D] = cov_4D_to_6D_blocks(
                cov_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D],
                nbl, zbins,
                ind_dict[probe_A, probe_B],
                ind_dict[probe_C, probe_D],
                symmetrize_output_dict[probe_A, probe_B],
                symmetrize_output_dict[probe_C, probe_D])
    return cov_3x2pt_dict_10D


def write_cl_ascii(ascii_folder, ascii_filename, cl_3d, ells, zbins):

    with open(f'{ascii_folder}/{ascii_filename}.ascii', 'w') as file:
        # Write header
        file.write(f'#ell\ttomo_i\ttomo_j\t{ascii_filename}\n')

        # Iterate over the array and write the data
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    # Format the line with appropriate spacing
                    file.write(f"{ell_val:.3f}\t{zi + 1}\t{zj + 1}\t{value:.10e}\n")


def write_cl_tab(ascii_folder, ascii_filename, cl_3d, ells, zbins):

    with open(f'{ascii_folder}/{ascii_filename}', 'w') as file:
        file.write(f'#ell\t\tzi\tzj\t{ascii_filename}\n')
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    file.write(f"{ell_val:.3f}\t\t{zi}\t{zj}\t{value:.10e}\n")


def compare_fm_constraints(*fm_dict_list, labels, keys_toplot_in, normalize_by_gauss, which_uncertainty,
                           reference, colors, abs_FoM, nparams_toplot_in=8, save_fig=False, fig_path=None):

    masked_fm_dict_list = []
    masked_fid_pars_dict_list = []
    uncertainties_dict = {}
    fom_dict = {}
    legend_x_anchor = 1.4

    assert keys_toplot_in == 'all' or type(keys_toplot_in) is list, 'keys_toplot must be a list or "all"'
    assert colors is None or type(colors) is list, 'colors must be a list or "all"'

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors

    # maks fm and fid pars dict
    for fm_dict in fm_dict_list:
        masked_fm_dict, masked_fid_pars_dict = {}, {}

        # define keys and remove unused ones
        keys_toplot = list(fm_dict.keys())
        if 'fiducial_values_dict' in keys_toplot:
            keys_toplot.remove('fiducial_values_dict')
        keys_toplot = [key for key in keys_toplot if not key.startswith('FM_WA_')]
        keys_toplot = [key for key in keys_toplot if not key.startswith('FM_2x2pt_')]

        for key in keys_toplot:
            masked_fm_dict[key], masked_fid_pars_dict[key] = mask_fm_v2(fm_dict[key],
                                                                        fm_dict['fiducial_values_dict'],
                                                                        names_params_to_fix=[],
                                                                        remove_null_rows_cols=True)
        masked_fm_dict_list.append(masked_fm_dict)
        masked_fid_pars_dict_list.append(masked_fid_pars_dict)

    # compute reference uncertainties
    print(key, masked_fid_pars_dict_list[0].keys())
    for key in keys_toplot:
        nparams_toplot = nparams_toplot_in
        param_names = list(masked_fid_pars_dict_list[0][key].keys())[:nparams_toplot]
        uncertainties_dict[key] = np.array([uncertainties_fm_v2(masked_fm_dict[key], fiducials_dict=masked_fid_pars_dict[key],
                                                                which_uncertainty=which_uncertainty, normalize=True)[:nparams_toplot]
                                            for masked_fm_dict, masked_fid_pars_dict in zip(masked_fm_dict_list, masked_fid_pars_dict_list)])
        w0wa_idxs = (param_names.index('wz'), param_names.index('wa'))
        fom_dict[key] = np.array([compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)
                                 for masked_fm_dict in masked_fm_dict_list])
        uncertainties_dict[key] = np.column_stack((uncertainties_dict[key], fom_dict[key]))
    param_names.append('FoM')

    keys_toplot = keys_toplot if keys_toplot_in == 'all' else keys_toplot_in

    # plot, and if necessary normalize by the G-only uncertainty
    for key in keys_toplot:
        probe = key.split('_')[1]

        ylabel = 'rel. unc. [%]'
        if normalize_by_gauss and not key.endswith('_G'):
            probe = key.split('_')[1]
            ng_cov = key.split('_')[2]
            uncertainties_dict[key] = (uncertainties_dict[key] / uncertainties_dict[f'FM_{probe}_G'] - 1) * 100
            ylabel = f'{ng_cov}/G - 1 [%]'

            if abs_FoM:
                uncertainties_dict[key][:, -1] = np.fabs(uncertainties_dict[key][:, -1])

        n_rows = 2 if len(fm_dict_list) > 1 else 1
        fig, ax = plt.subplots(n_rows, 1, figsize=(10, 5), sharex=True)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        ax[0].set_title(f'{which_uncertainty} uncertainties, {key}')
        for i, uncert in enumerate(uncertainties_dict[key]):
            ax[0].scatter(param_names, uncert, label=f'{labels[i]}', marker='o', c=colors[i], alpha=0.6)
        ax[0].axhline(0, c='k', ls='--')
        ax[0].set_ylabel(ylabel)
        ax[0].legend(ncol=1, loc='center right', bbox_to_anchor=(legend_x_anchor, 0.))
        ax[0].grid()

        start_idx = 0
        title_str = reference
        if reference == 'first_key':
            ref = uncertainties_dict[key][0]
            start_idx = 1
            title_str = labels[0]
        elif reference == 'median':
            ref = np.median(uncertainties_dict[key], axis=0)
        elif reference == 'mean':
            ref = np.mean(uncertainties_dict[key], axis=0)
        else:
            raise ValueError('reference must be one of "first_key", "median", or "mean"')

        if len(uncertainties_dict[key]) > 1:
            diffs = [percent_diff(uncert, ref) for uncert in uncertainties_dict[key][start_idx:]]

            for i, diff in enumerate(diffs):
                ax[1].scatter(param_names, diff, marker='o', c=colors[i + start_idx], alpha=0.6)
            ax[1].fill_between((0, nparams_toplot), -10, 10, color='k', alpha=0.1, label='$\\pm 10\\%$')

        ax[1].set_ylabel(f'% diff wrt\n{title_str}\n')
        ax[1].legend(ncol=1, loc='center right', bbox_to_anchor=(legend_x_anchor, 0.5))
        ax[1].grid()

        if save_fig:
            plt.savefig(f'{fig_path}/{key}.png', dpi=400, bbox_inches='tight')


def compare_param_cov_from_fm_pickles(fm_pickle_path_a, fm_pickle_path_b, which_uncertainty, compare_fms=True, compare_param_covs=True,
                                      plot=True, n_params_toplot=10):

    fm_dict_a = load_pickle(fm_pickle_path_a)
    fm_dict_b = load_pickle(fm_pickle_path_b)
    masked_fm_dict_a, masked_fid_pars_dict_a = {}, {}
    masked_fm_dict_b, masked_fid_pars_dict_b = {}, {}

    # check that the keys match
    assert fm_dict_a.keys() == fm_dict_b.keys()

    # check if the dictionaries contained in the key 'fiducial_values_dict' match
    assert fm_dict_a['fiducial_values_dict'] == fm_dict_b['fiducial_values_dict'], 'fiducial values do not match!'

    # check that the values match
    for key in fm_dict_a.keys():
        if key != 'fiducial_values_dict' and 'WA' not in key:
            print('Comparing ', key)

            masked_fm_dict_a[key], masked_fid_pars_dict_a[key] = mask_fm_v2(fm_dict_a[key],
                                                                            fm_dict_a['fiducial_values_dict'],
                                                                            names_params_to_fix=[],
                                                                            remove_null_rows_cols=True)
            masked_fm_dict_b[key], masked_fid_pars_dict_b[key] = mask_fm_v2(fm_dict_b[key],
                                                                            fm_dict_b['fiducial_values_dict'],
                                                                            names_params_to_fix=[],
                                                                            remove_null_rows_cols=True)

            cov_a = np.linalg.inv(masked_fm_dict_a[key])
            cov_b = np.linalg.inv(masked_fm_dict_b[key])

            if compare_fms:
                compare_arrays(masked_fm_dict_a[key], masked_fm_dict_b[key], 'FM_A', 'FM_B', plot_diff_threshold=5)

            if compare_param_covs:

                compare_arrays(cov_a, cov_b, 'cov_A', 'cov_B', plot_diff_threshold=5)

            if plot:
                param_names = list(masked_fid_pars_dict_a[key].keys())[:n_params_toplot]
                uncert_a = uncertainties_fm_v2(masked_fm_dict_a[key], fiducials_dict=masked_fid_pars_dict_a[key],
                                               which_uncertainty=which_uncertainty, normalize=True)[:n_params_toplot]
                uncert_b = uncertainties_fm_v2(masked_fm_dict_b[key], fiducials_dict=masked_fid_pars_dict_b[key],
                                               which_uncertainty=which_uncertainty, normalize=True)[:n_params_toplot]
                diff = percent_diff(uncert_a, uncert_b)

                plt.figure()
                plt.title(f'Marginalised uncertainties, {key}')
                plt.plot(param_names, uncert_a, label='FM_A')
                plt.plot(param_names, uncert_b, ls='--', label='FM_B')
                plt.plot(param_names, diff, label='percent diff')
                plt.legend()


def is_file_created_in_last_x_hours(file_path, hours):
    """
    Check if the specified file was created in the last 24 hours.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    bool: True if the file was created in the last 24 hours, False otherwise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Get the current time
    now = datetime.datetime.now()

    # Get the file creation time
    creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))

    # Calculate the time difference
    time_diff = now - creation_time

    # Check if the difference is less than or equal to 24 hours
    return time_diff <= datetime.timedelta(hours=hours)


def block_diag(array_3d):
    """
    Useful for visualizing nbl, zbins, zbins arrays at a glance
    """
    nbl = array_3d.shape[0]
    return scipy.linalg.block_diag(*[array_3d[ell, :, :] for ell in range(nbl)])


def compare_df_keys(dataframe, key_to_compare, value_a, value_b, num_string_columns):
    """
    This function compares two rows of a dataframe and returns a new row with the percentage difference between the two
    :param dataframe:
    :param key_to_compare:
    :param value_a:
    :param value_b:
    :param num_string_columns: number of columns containing only strings or various options, such as whether to fix a certain prior or not...
    :return:
    """
    import pandas as pd
    
    df_A = dataframe[dataframe[key_to_compare] == value_a]
    df_B = dataframe[dataframe[key_to_compare] == value_b]
    arr_A = df_A.iloc[:, num_string_columns:].select_dtypes('number').values
    arr_B = df_B.iloc[:, num_string_columns:].select_dtypes('number').values

    if arr_A.shape[0] != arr_B.shape[0]:
        raise ValueError(f"Cannot compare groups with different sizes: {arr_A.shape[0]} vs {arr_B.shape[0]}")

    perc_diff_df = df_A.copy()
    # ! the reference is G, this might change to G + SSC + cNG
    perc_diff_df.iloc[:, num_string_columns:] = percent_diff(arr_B, arr_A)
    perc_diff_df[key_to_compare] = f'perc_diff_{value_b}'
    perc_diff_df['FoM'] = -perc_diff_df['FoM']  # ! abs? minus??
    dataframe = pd.concat([dataframe, perc_diff_df], axis=0, ignore_index=True)

    # dataframe = dataframe.drop_duplicates()
    columns_to_consider = [col for col in dataframe.columns if col not in ['fm', 'fiducials_dict']]
    dataframe = dataframe.drop_duplicates(subset=columns_to_consider)

    return dataframe


def contour_FoM_calculator(sample, param1, param2, sigma_level=1):
    """ This function has been written by Santiago Casas.
    Computes  the FoM from getDist samples.
    add()sample is a getDist sample object, you need as well the shapely 
    package to compute polygons. The function returns the 1sigma FoM, 
    but in principle you could compute 2-, or 3-sigma "FoMs"
    """
    from shapely.geometry import Polygon
    contour_coords = {}
    density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
    contour_levels = density.contours
    contours = plt.contour(density.x, density.y, density.P, sorted(contour_levels))
    for ii, contour in enumerate(contours.collections):
        paths = contour.get_paths()
        for path in paths:
            xy = path.vertices
            x = xy[:, 0]
            y = xy[:, 1]
            contour_coords[ii] = list(zip(x, y))
    sigma_lvls = {3: 0, 2: 1, 1: 2}
    poly = Polygon(contour_coords[sigma_lvls[sigma_level]])  # 0:3sigma, 1:2sigma, 2:1sigma
    area = poly.area
    FoM_area = (2.3 * np.pi) / area
    return FoM_area, density


def can_be_pickled(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False


def load_cov_from_probe_blocks(path, filename, probe_ordering):
    """
    Load the covariance matrix from the probe blocks in 4D. The blocks are stored in a dictionary with keys
    corresponding to the probes in the order specified in probe_ordering. The symmetrization of the blocks is done
    while loading, so only 6 blocks need to be actually stored.
    :param path: Path to the folder containing the covariance blocks.
    :param filename: Filename of the covariance blocks. The filename should contain the placeholders {probe_A},
    {probe_B}, {probe_C}, {probe_D} which will be replaced with the actual probe names.
    :param probe_ordering: Probe ordering tuple
    :return:

    YOU SHOULD USE deepcopy, otherwise the different blocks become correlated (and you e.g. divide twice by fsky)
    """
    cov_ssc_dict_8D = {}
    for row, (probe_a, probe_b) in enumerate(probe_ordering):
        for col, (probe_c, probe_d) in enumerate(probe_ordering):
            if col >= row:  # Upper triangle and diagonal
                formatted_filename = filename.format(probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d)
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = np.load(f"{path}/{formatted_filename}")

                if formatted_filename.endswith('.npz'):
                    cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = cov_ssc_dict_8D[
                        probe_a, probe_b, probe_c, probe_d]['arr_0']

            else:  # Lower triangle, set using symmetry
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = deepcopy(cov_ssc_dict_8D[
                    probe_c, probe_d, probe_a, probe_b].transpose(1, 0, 3, 2))

    for key in cov_ssc_dict_8D.keys():
        assert cov_ssc_dict_8D[key].ndim == 4, (f'covariance matrix {key} has ndim={cov_ssc_dict_8D[key].ndim} instead '
                                                f'of 4')

    return cov_ssc_dict_8D


def save_dict_to_file(dict_data, file_path, indent=4):
    """
    Save a dictionary to a text file in a nicely formatted way.

    Parameters:
    dict_data (dict): The dictionary to be saved.
    file_path (str): The path of the file where the dictionary will be saved.
    indent (int, optional): The number of spaces for indentation. Default is 4.
    """
    with open(file_path, 'w') as f:
        json.dump(dict_data, f, indent=indent)


def figure_of_correlation(correl_matrix):
    """
    Compute the Figure of Correlation (FoC) from the correlation matrix correl_matrix.

    Parameters:
    - correl_matrix (2D numpy array): The correlation matrix.

    Returns:
    - FoC (float): The Figure of Correlation.
    """
    # Invert the correlation matrix
    correl_matrix_inv = np.linalg.inv(correl_matrix)
    # Compute the FoC
    foc = np.sqrt(np.linalg.det(correl_matrix_inv))

    return foc


def plot_correlation_matrix(correlation_matrix, labels, title):
    """
    Plots a heatmap of the given correlation matrix with provided labels.

    Parameters:
    - correlation_matrix (2D numpy array): The correlation matrix to be plotted.
    - labels (list): List of parameter names for labeling the axes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Using the RdBu_r colormap for the heatmap
    # cax = ax.matshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    # cax = ax.matshow(correlation_matrix, cmap='RdBu_r')
    # cax = ax.matshow(correlation_matrix, cmap='viridis')

    # Display color bar
    # cbar = fig.colorbar(cax)

    # Set labels
    # ax.set_xticks(np.arange(len(labels)))
    # ax.set_yticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)

    # Rotate x-axis labels for better clarity
    # plt.xticks()

    # Set the title
    ax.set_title(title, loc='center', rotation=0)

    # Display the plot
    plt.show()


def find_inverse_from_array(input_x, input_y, desired_y, interpolation_kind='linear'):
    from pynverse import inversefunc
    input_y_func = interp1d(input_x, input_y, kind=interpolation_kind)
    desired_y = inversefunc(input_y_func, y_values=desired_y, domain=(input_x[0], input_x[-1]))
    return desired_y


def add_ls_legend(ls_dict):
    """
    Add a legend for line styles.

    Parameters
    ----------
    ls_dict : dict
        A dictionary mapping line styles to labels.
        E.g. {'-': 'delta', '--': 'gamma'}
    """
    handles = []
    for ls, label in ls_dict.items():
        handles.append(mlines.Line2D([], [], color='black', linestyle=ls, label=label))
    plt.legend(handles=handles, loc='best')


def save_correlation_matrix_plot(matrix1, matrix2, labels):
    """Yet to be fully tested. Plot the positive and negative values with different colorbars
    TODO make it only for one matrix!"""

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 5, width_ratios=[6, 0.3, 6, 0.3, 0.3])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    cbar_ax_pos = plt.subplot(gs[3])
    cbar_ax_neg = plt.subplot(gs[4])

    # Define the colormap
    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues_r')

    # Divide positive and negative parts for each matrix
    matrix1_pos = np.ma.masked_less_equal(matrix1, 0)
    matrix1_neg = -np.ma.masked_greater_equal(matrix1, 0)

    matrix2_pos = np.ma.masked_less_equal(matrix2, 0)
    matrix2_neg = -np.ma.masked_greater_equal(matrix2, 0)

    # Plot Gaussian matrix
    ax1.imshow(matrix1_pos, cmap=cmap_pos, norm=LogNorm())
    ax1.imshow(matrix1_neg, cmap=cmap_neg, norm=LogNorm())

    ax1.set_title("Gaussian")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    # Plot Gaussian + SSC matrix
    im2a = ax2.imshow(matrix2_pos, cmap=cmap_pos, norm=LogNorm())
    im2b = ax2.imshow(matrix2_neg, cmap=cmap_neg, norm=LogNorm())

    ax2.set_title("Gaussian + SSC")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)

    # Add colorbars
    cbar_pos = fig.colorbar(im2a, cax=cbar_ax_pos, orientation='vertical')
    cbar_neg = fig.colorbar(im2b, cax=cbar_ax_neg, orientation='vertical')

    plt.tight_layout()
    plt.show()


def table_to_3d_array(file_path, is_auto_spectrum):
    """convert CLOE .dat files format to 3d arrays, following the indexing in the header
    """

    # Read the header and data from the file
    with open(file_path, 'r') as f:
        header = f.readline().strip().split('\t')

    cl_2d = np.genfromtxt(file_path)
    ells = cl_2d[:, 0]
    cl_2d = cl_2d[:, 1:]
    nbl = len(ells)

    ind = zpairs_header_to_ind_array(header[1:]) - 1  # -1 to revert to 0-based counting
    zbins = np.max(ind) + 1

    cl_3d = np.zeros((nbl, zbins, zbins))
    for ell_idx in range(nbl):
        for zpair_idx, (zi, zj) in enumerate(zip(ind[:, 0], ind[:, 1])):
            cl_3d[ell_idx, zi, zj] = cl_2d[ell_idx, zpair_idx]

    if is_auto_spectrum:
        for ell in range(nbl):
            cl_3d[ell, ...] = symmetrize_2d_array(cl_3d[ell, ...])

    return ells, cl_3d


def zpairs_header_to_ind_array(header):
    # Initialize an empty list to store the pairs of integers
    ind = []

    # Loop through each element in the list
    for zpairs_str_list in header:
        # Split the element by the '-' character
        zpairs_str_list = zpairs_str_list.split('-')

        # Remove the 'E'mand 'P' from each part and convert to integer
        zpairs = [int(zpair_str.replace('E', '').replace('P', '')) for zpair_str in zpairs_str_list]

        # Append the pair of integers to the list
        ind.append(zpairs)

    # Convert the list of pairs into a numpy array
    ind = np.array(ind)

    return ind


def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx


def flatten_dict(nested_dict):
    """Flatten a nested dictionary."""
    flattened = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def get_filenames_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames


def test_folder_content(output_path, benchmarks_path, extension, verbose=False, rtol=1e-10):
    """Test if the files in the output folder are equal to the benchmark files and list the discrepancies.

    Parameters:
    output_path (str): The path to the folder containing the output files.
    benchmarks_path (str): The path to the folder containing the benchmark files.
    extension (str): The extension of the files to be tested.

    Returns:
    dict: A dictionary containing the comparison results and discrepancies.
    """
    old_files = os.listdir(benchmarks_path)
    new_files = os.listdir(output_path)

    # Ignore hidden files and filter by extension
    old_files = [file for file in old_files if not file.startswith('.') and file.endswith(extension)]
    new_files = [file for file in new_files if not file.startswith('.') and file.endswith(extension)]

    discrepancies = {
        'missing_in_benchmark': set(new_files) - set(old_files),
        'missing_in_output': set(old_files) - set(new_files),
        'comparison_results': []
    }

    max_length_benchmark = max((len(file) for file in discrepancies['missing_in_benchmark']), default=0)
    max_length_output = max((len(file) for file in discrepancies['missing_in_output']), default=0)

    # Use the maximum length from either set for alignment
    max_length = max(max_length_benchmark, max_length_output)

    for file in discrepancies['missing_in_benchmark']:
        print(f"{file:<{max_length}} \t output ✅ \t benchmark ❌")

    for file in discrepancies['missing_in_output']:
        print(f"{file:<{max_length}} \t output ❌ \t benchmark ✅")
        print(f"{file} \t output ❌ \t benchmark ✅")

    for file_name in set(old_files).intersection(new_files):
        old_file_path = os.path.join(benchmarks_path, file_name)
        new_file_path = os.path.join(output_path, file_name)

        try:
            if extension == 'npz':
                np.testing.assert_allclose(np.load(old_file_path)['arr_0'], np.load(new_file_path)['arr_0'],
                                           verbose=verbose, rtol=rtol, atol=0)
            elif extension == 'npy':
                np.testing.assert_allclose(np.load(old_file_path), np.load(new_file_path), verbose=verbose,
                                           rtol=rtol, atol=0)
            elif extension == 'txt' or extension == 'dat':
                np.testing.assert_allclose(np.genfromtxt(old_file_path), np.genfromtxt(new_file_path),
                                           verbose=verbose, rtol=rtol, atol=0)
            else:
                raise ValueError(f"Unknown extension: {extension}")
        except AssertionError as exc:
            discrepancies['comparison_results'].append((file_name, str(exc)))
            print(f'\nFile {file_name} does not match: {exc}')
        else:
            discrepancies['comparison_results'].append((file_name, 'Match'))
            print(f"{file_name:<{max_length}} \t matches to within {rtol * 100}% ✅")

    # Provide a summary of the results
    num_comparisons = len(discrepancies['comparison_results'])
    num_matches = sum(1 for _, result in discrepancies['comparison_results'] if result == 'Match')
    num_discrepancies = num_comparisons - num_matches

    print(f"\nSummary: {num_comparisons} files compared, {num_matches} matches, {num_discrepancies} discrepancies.\n")

    return discrepancies


def import_files(folder_path, extension):
    """
    Imports all files with a specific extension from a folder and stores their contents in a dictionary.

    Parameters:
    folder_path (str): The path to the folder from which to import files.
    extension (str): The file extension to look for.

    Returns:
    dict: A dictionary with filenames as keys and file contents as values.
    """
    files_dict = {}
    # Get the list of relevant file names using the get_filenames_in_folder function
    filenames = get_filenames_in_folder(folder_path)

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        # Determine how to load the file based on the extension
        if extension in ['.txt', '.csv']:
            file_content = np.genfromtxt(file_path, delimiter=',')  # Adjust delimiter if necessary
        elif extension == '.npy':
            file_content = np.load(file_path)
        elif extension == '.npz':
            file_content = np.load(file_path)['arr_0']
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        # Use the file name without an extension as the key
        key = os.path.splitext(filename)[0]
        files_dict[key] = file_content

    return files_dict


def is_increasing(arr):
    return np.all(np.diff(arr) > 0)


def save_pickle(filename, obj):
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    with open(f'{filename}', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_compressed_pickle(title, data):
    import bz2
    with bz2.BZ2File(title + '.pbz2', 'wb') as handle:
        pickle.dump(data, handle)


def load_compressed_pickle(file):
    import bz2
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def read_yaml(filename):
    """ A function to read YAML file. filename must include the path and the extension"""
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


# @njit
def percent_diff(array_1, array_2, abs_value=False):

    array_1 = np.atleast_1d(array_1)  # Ensure array-like behavior
    array_2 = np.atleast_1d(array_2)

    diff = (array_1 / array_2 - 1) * 100

    # avoid nans
    both_zeros = np.logical_and(array_1 == 0, array_2 == 0)

    diff[both_zeros] = 0

    if abs_value:
        return np.abs(diff)
    else:
        # Convert back to scalar if necessary
        return diff.item() if diff.size == 1 else diff  


# @njit
def percent_diff_mean(array_1, array_2):
    """
    result is in "percent" units
    """
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


# @njit
def _percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    if eraseNaN:
        diff = np.where(array_1 == array_2, 0, percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)
    if log:
        diff = np.log10(diff)
    if abs_val:
        diff = np.abs(diff)
    return diff


def percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    """
    Calculate the percent difference between two arrays, handling NaN values.
    """
    # Handle NaN values
    if eraseNaN:
        # Mask where NaN values are present
        diff = np.ma.masked_where(np.isnan(array_1) | np.isnan(array_2),
                                  percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)

    # Handle log transformation
    if log:
        # Mask zero differences before taking the log
        diff = np.ma.masked_where(diff == 0, diff)
        diff = np.log10(np.ma.abs(diff))  # Masked values will be ignored in the log

    # Handle absolute values
    if abs_val:
        diff = np.ma.abs(diff)

    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f"has any element of the arrays a disagreement > {threshold}%? ", boolean)


def compute_smape(vec_true, vec_test, cov_mat=None):
    """
    Computes the SMAPE (Symmetric Mean Absolute Percentage Error) for a given 1D array with weighted elements

    Args:
        vec_true (np.array): array of true values
        vec_test (np.array): array of predicted/approximated values
        cov_mat (np.array): covariance matrix for vec_true

    Returns:
        float: SMAPE value
    """
    if type(vec_true) == np.ndarray and type(vec_test) == np.ndarray:
        assert len(vec_true) == len(vec_test), "Arrays must have the same length"
        assert vec_true.ndim == 1 and vec_test.ndim == 1, 'arrays must be 1D'

    if cov_mat is not None:
        assert cov_mat.shape[0] == cov_mat.shape[1] == len(vec_true), 'cov_mat must be a square matrix with the same ' \
                                                                      'length as the input vectors'
        weights = vec_true / np.sqrt(np.diag(cov_mat))
    else:
        weights = np.ones_like(vec_true)  # uniform weights

    numerator = weights * np.abs(vec_true - vec_test)
    denominator = np.abs(vec_true) + np.abs(vec_test)

    return 100 * np.mean(numerator / denominator)  # the output is already a precentage


def compute_diff_sigma(vec_true, vec_test, sigma):
    """
    Compute the element-wise difference between two vectors vec_true and vec_test,
    and divide it by sigma.

    Args:
    - vec_true (numpy.ndarray): A numpy array representing the first vector.
    - vec_test (numpy.ndarray): A numpy array representing the second vector.
    - sigma (numpy.ndarray): A numpy array representing the vector of standard deviations.

    Returns:
    - A numpy array representing the element-wise difference between vec_true and vec_test,
      divided by sigma.
    """
    diff = np.abs(vec_true - vec_test)
    return 100 * diff / sigma


def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def compare_arrays(A, B, name_A='A', name_B='B',
                   plot_diff=True, plot_array=True,
                   log_array=True, log_diff=False,
                   abs_val=False, plot_diff_threshold=None,
                   white_where_zero=True, plot_diff_hist=False):

    if np.array_equal(A, B):
        print(f'{name_A} and {name_B} are equal ✅')
        return

    for rtol in [1e-3, 1e-2, 5e-2]:  # these are NOT percent units
        if np.allclose(A, B, rtol=rtol, atol=0):
            print(f'{name_A} and {name_B} are close within relative tolerance of {rtol * 100}%) ✅')
            return

    diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=abs_val)
    higher_rtol = plot_diff_threshold or 5.0
    max_diff = np.max(diff_AB)
    result_emoji = '❌' if max_diff > higher_rtol or np.isnan(max_diff) else '✅'
    no_outliers = np.sum(diff_AB > higher_rtol)
    additional_info = (f'\nMax discrepancy: {max_diff:.2f}%;'
                       f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}'
                       f'\nFraction of elements with discrepancy > {higher_rtol}%: {no_outliers / diff_AB.size:.5f}')
    print(f'Are {name_A} and {name_B} different by less than {higher_rtol}%? {result_emoji} {additional_info}')

    # Check that arrays are 2D if any plotting is requested.
    if (plot_diff or plot_array):
        assert A.ndim == 2 and B.ndim == 2, 'Plotting is only implemented for 2D arrays'

    # Determine number of rows:
    nrows = (1 if plot_array else 0) + (1 if plot_diff else 0)
    ncols = 2  # Always show 2 panels per row

    fig, ax = plt.subplots(nrows, ncols, figsize=(17, 7 * nrows), constrained_layout=True)

    # Ensure ax is always 2D
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)  # Convert row array to 2D
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)  # Convert column array to 2D

    # If plotting arrays, prepare data and plot in first row.
    if plot_array:
        if abs_val:
            A_toplot, B_toplot = np.abs(A), np.abs(B)
        if log_array:
            A_toplot, B_toplot = np.log10(A_toplot), np.log10(B_toplot)

        im = ax[0, 0].matshow(A_toplot)
        ax[0, 0].set_title(f'{name_A}')
        fig.colorbar(im, ax=ax[0, 0])

        im = ax[0, 1].matshow(B_toplot)
        ax[0, 1].set_title(f'{name_B}')
        fig.colorbar(im, ax=ax[0, 1])

    # If plotting differences, prepare diff data and plot in next row.
    if plot_diff:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=abs_val)
        diff_BA = percent_diff_nan(B, A, eraseNaN=True, log=False, abs_val=abs_val)

        if plot_diff_threshold is not None:
            # Mask out small differences (set them to white via the colormap's "bad" color)
            diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, diff_AB)
            diff_BA = np.ma.masked_where(np.abs(diff_BA) < plot_diff_threshold, diff_BA)

        if log_diff:
            # Replace nonpositive with nan to avoid -inf
            diff_AB = np.log10(np.abs(diff_AB))
            diff_BA = np.log10(np.abs(diff_BA))

        im = ax[1, 0].matshow(diff_AB)
        ax[1, 0].set_title('(A/B - 1) * 100')
        fig.colorbar(im, ax=ax[1, 0])

        im = ax[1, 1].matshow(diff_BA)
        ax[1, 1].set_title('(B/A - 1) * 100')
        fig.colorbar(im, ax=ax[1, 1])

    fig.suptitle(f'log_array={log_array}, abs_val={abs_val}, log_diff={log_diff}')
    plt.show()

    if plot_diff_hist:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=False)
        plt.figure()
        plt.hist(diff_AB.flatten(), bins=30, log=True, density=True)
        plt.xlabel('% difference')
        plt.ylabel('frequency')
        plt.show()


def compare_folder_content(path_A: str, path_B: str, filetype: str):
    """
    Compare the content of 2 folders. The files in folder A should be a subset of the files in folder B.
    """
    dict_A = dict(get_kv_pairs(path_A, filetype))
    dict_B = dict(get_kv_pairs(path_B, filetype))

    for key in dict_A.keys():
        if np.array_equal(dict_A[key], dict_B[key]):
            result_emoji = '✅'
        else:
            result_emoji = '❌'
        print(f'is {key} equal in both folders? {result_emoji}')


def namestr(obj, namespace):
    """ does not work with slices!!! (why?)"""
    return [name for name in namespace if namespace[name] is obj][0]


def plot_FM(array, style=".-"):
    name = namestr(array, globals())
    plt.plot(range(7), array, style, label=name)


def find_null_rows_cols_2D(array):
    """
    :param array:

    :return null_rows_idxs: list
        array of null rows/columns indices
    """
    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    null_rows_idxs = np.where(np.all(array == 0, axis=0))[0]
    null_cols_idxs = np.where(np.all(array == 0, axis=1))[0]

    assert np.array_equal(null_rows_idxs, null_cols_idxs), \
        'null rows and columns indices should be the same for Fisher matrices'

    if null_rows_idxs.shape[0] == 0:
        # print('The input array has no null rows/columns')
        return None
    else:
        # print(f'The input array had null rows and columns at indices {null_rows_idxs}')
        return null_rows_idxs


def remove_rows_cols_array2D(array, rows_idxs_to_remove):
    """
    Removes the *same* rows and columns from an array
    :param array: numpy.ndarray. input 2D array
    :param rows_idxs_to_remove: list. rows (and columns) to delete
    :return: array without null rows and columns
    """
    if rows_idxs_to_remove is None:
        warnings.warn('null_rows_idxs is None, returning the input array')
        return array

    if len(rows_idxs_to_remove) == 0:
        warnings.warn('null_rows_idxs is empty, returning the input array')
        return array

    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array = np.delete(array, rows_idxs_to_remove, axis=0)
    array = np.delete(array, rows_idxs_to_remove, axis=1)
    return array


def remove_null_rows_cols_2D(array_2d: np.ndarray):
    """
    Remove null rows and columns from a 2D numpy array.

    Args:
        array_2d (numpy.ndarray): The 2D numpy array to remove null rows and columns from.

    Returns:
        numpy.ndarray: The 2D numpy array with null rows and columns removed.
    """

    assert array_2d.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array_2d = array_2d[~np.all(array_2d == 0, axis=1)]
    array_2d = array_2d[:, ~np.all(array_2d == 0, axis=0)]
    return array_2d


def mask_FM_null_rowscols(FM, params, fid):
    """
    Mask the Fisher matrix, fiducial values and parameter list deleting the null rows and columns
    :param FM: Fisher Matrix, 2D numpy array
    :return: masked FM, fiducial values and parameter list
    """
    null_idx = find_null_rows_cols_2D(FM)

    if null_idx is None:
        return FM, params, fid

    FM = remove_rows_cols_array2D(FM, null_idx)
    params = np.delete(params, obj=null_idx, axis=0)
    fid = np.delete(fid, obj=null_idx, axis=0)
    assert len(fid) == len(params), 'the fiducial values and parameter lists should have the same length'
    return FM, list(params), list(fid)


def mask_fm_null_rowscols_v2(fm, fiducials_dict):
    """
    Mask the Fisher matrix, fiducial values and parameter list deleting the null rows and columns
    :param FM: Fisher Matrix, 2D numpy array
    :return: masked FM, fiducial values and parameter list
    """
    param_names = list(fiducials_dict.keys())
    param_values = list(fiducials_dict.values())

    null_idxs = find_null_rows_cols_2D(fm)

    if null_idxs is None:
        return fm, fiducials_dict

    # remove null rows and columns
    fm = remove_rows_cols_array2D(fm, null_idxs)
    # update fiducials dict
    fiducials_dict = {param_names[i]: param_values[i] for i in range(len(param_names)) if i not in null_idxs}

    return fm, fiducials_dict


def mask_FM(FM, param_names_dict, fiducials_dict, params_tofix_dict, remove_null_rows_cols=True):
    """
    Trim the Fisher matrix to remove null rows/columns and/or fix nuisance parameters
    :param FM:
    :param remaining_param_names_list:
    :param fid:
    :param n_cosmo_params:
    :param kwargs:
    :return:
    """

    if type(param_names_dict) == list or type(param_names_dict) == np.ndarray:
        param_names_dict = {'_': param_names_dict}
    if type(fiducials_dict) == list or type(fiducials_dict) == np.ndarray:
        fiducials_dict = {'_': fiducials_dict}
    if type(params_tofix_dict) == list or type(params_tofix_dict) == np.ndarray:
        params_tofix_dict = {'_': params_tofix_dict}

    # join param_names_dict.values() into single list
    all_param_names_list = list(itertools.chain(*list(param_names_dict.values())))
    all_fiducials_list = list(itertools.chain(*list(fiducials_dict.values())))

    # TODO - add option to fix specific parameter
    # TODO  - test this!!
    idx_todelete = []
    for key in params_tofix_dict.keys():
        if params_tofix_dict[key]:
            _param_names_list = param_names_dict[key]
            param_idxs = [all_param_names_list.index(param_name) for param_name in _param_names_list]
            idx_todelete.append(param_idxs)

    # make a continuous list
    # idx_todelete = np.flatten(idx_todelete.flatten())
    idx_todelete = list(itertools.chain(*idx_todelete))

    if idx_todelete:
        FM = np.delete(FM, idx_todelete, axis=0)
        FM = np.delete(FM, idx_todelete, axis=1)
        remaining_param_names_list = np.delete(all_param_names_list, idx_todelete)
        remaining_fiducials_list = np.delete(all_fiducials_list, idx_todelete)
    else:
        remaining_param_names_list = all_param_names_list
        remaining_fiducials_list = all_fiducials_list

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        FM, remaining_param_names_list, remaining_fiducials_list = mask_FM_null_rowscols(FM, remaining_param_names_list,
                                                                                         remaining_fiducials_list)

    return FM, list(remaining_param_names_list), list(remaining_fiducials_list)


def mask_fm_v2(fm: np.ndarray, fiducials_dict: dict, names_params_to_fix: list, remove_null_rows_cols: bool):
    """
    Trim the Fisher matrix to remove null rows/columns and/or fix nuisance parameters
    """
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    assert len(list(fiducials_dict.keys())) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'

    if names_params_to_fix is not None:
        fm = fix_params_in_fm(fm, names_params_to_fix, fiducials_dict)  # cut fm entries
        # update fiducials_dict
        fiducials_dict = {key: fiducials_dict[key] for key in fiducials_dict.keys() if key not in names_params_to_fix}

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        fm, fiducials_dict = mask_fm_null_rowscols_v2(fm, fiducials_dict)

    return fm, fiducials_dict


def fix_params_in_fm(fm, names_params_to_fix, fiducials_dict):
    param_names = list(fiducials_dict.keys())
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    # check the correctness of the parameters' names
    for param_to_fix in names_params_to_fix:
        assert param_to_fix in param_names, f'Parameter {param_to_fix} not found in param_names!'

    rows_idxs_to_remove = [param_names.index(param_to_fix) for param_to_fix in names_params_to_fix]
    fm = remove_rows_cols_array2D(fm, rows_idxs_to_remove)
    # print(f'Removing rows and columns from FM:\n{rows_idxs_to_remove}')

    return fm


def add_prior_to_fm(fm, fiducials_dict, prior_param_names, prior_param_values):
    """ adds a FM of priors (with elements 1/sigma in the correct positions) to the input FM"""
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    assert len(list(fiducials_dict.keys())) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'
    fid_param_names = list(fiducials_dict.keys())

    for prior_param_name in prior_param_names:
        if prior_param_name not in fid_param_names:
            warnings.warn(f'Prior parameter {prior_param_names} not found in fiducial parameters dict!')
            return fm

    prior_param_idxs = [fid_param_names.index(prior_param_name) for prior_param_name in prior_param_names]

    prior_fm = np.zeros(fm.shape)
    prior_fm[prior_param_idxs, prior_param_idxs] = 1 / np.array(prior_param_values)
    return fm + prior_fm



def uncertainties_fm_v2(fm, fiducials_dict, which_uncertainty='marginal', normalize=True, percent_units=True):
    """
    returns relative 1-sigma error
    """

    param_names = list(fiducials_dict.keys())
    param_values = np.array(list(fiducials_dict.values()))

    assert len(param_names) == param_values.shape[0] == fm.shape[0] == fm.shape[1], \
        'param_names and param_values must have the same length and be equal to the number of rows and columns of fm'

    if which_uncertainty == 'marginal':
        fm_inv = np.linalg.inv(fm)
        sigma_fm = np.sqrt(np.diag(fm_inv))
    elif which_uncertainty == 'conditional':
        sigma_fm = np.sqrt(1 / np.diag(fm))
    else:
        raise ValueError('which_uncertainty must be either "marginal" or "conditional"')

    if normalize:
        # if the fiducial for is 0, substitute with 1 to avoid division by zero; if it's -1, take the absolute value
        param_values = np.where(param_values == 0, 1, param_values)
        param_values = np.where(param_values < 0, np.abs(param_values), param_values)
        # normalize to get the relative uncertainties
        sigma_fm /= param_values

    if percent_units:
        return sigma_fm * 100

    return sigma_fm


def build_labels_TeX(zbins):
    galaxy_bias_label = ['$b_{%i}$' % (i + 1) for i in range(zbins)]
    shear_bias_label = ['$m_{%i}$' % (i + 1) for i in range(zbins)]
    zmean_shift_label = ['$dz_{%i}$' % (i + 1) for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def build_labels(zbins):
    galaxy_bias_label = [f'b{(i + 1):02d}' for i in range(zbins)]
    shear_bias_label = [f'm{(i + 1):02d}' for i in range(zbins)]
    zmean_shift_label = [f'dz{(i + 1):02d}' for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def matshow(array, title="title", log=True, abs_val=False, threshold=None, 
            only_show_nans=False, matshow_kwargs: dict=None):
    """
    :param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise, 
    keep only the elements above the threshold (i.e., mask the ones below the threshold)
    :return:
    """

    if matshow_kwargs is None:
        matshow_kwargs = {}
    if only_show_nans:
        warnings.warn('only_show_nans is True, better switch off log and abs_val'
                      ' for the moment', stacklevel=2)
        # Set non-NaN elements to 0 and NaN elements to 1
        array = np.where(np.isnan(array), 1, 0)
        title += ' (only NaNs shown)'

    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        with np.errstate(divide='ignore', invalid='ignore'):
            array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f" \n(masked below {threshold} \\%)"

    plt.matshow(array, **matshow_kwargs)
    plt.colorbar()
    plt.title(title)
    plt.show()


def get_kv_pairs(path_import, extension='npy'):
    """
    Load txt or dat files in dictionary.
    To use it, wrap it in "dict(), e.g.:
        loaded_dict = dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension == 'npy' or extension == 'npz':
        load_function = np.load
    elif extension == 'txt' or extension == 'dat':
        load_function = np.genfromtxt
    else:
        raise NotImplementedError("extension must be either 'npy', 'npz', 'txt' or 'dat'")

    for path in Path(path_import).glob(f"*.{extension}"):
        yield path.stem, load_function(str(path))


def get_kv_pairs_v2(path_import, extension='npy'):
    """
    Load npy, npz, txt, or dat files in dictionary.
    To use it, wrap it in "dict(), e.g.:
        loaded_dict = dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension == 'npy' or extension == 'npz':
        load_function = np.load
    elif extension == 'txt' or extension == 'dat':
        def load_function(p): return np.genfromtxt(p, encoding='latin1')  # Handle non-UTF-8 encoding
    else:
        raise NotImplementedError("extension must be either 'npy', 'npz', 'txt' or 'dat'")

    for path in Path(path_import).glob(f"*.{extension}"):
        if path.is_file():  # Ensure it's a file, not a directory
            try:
                yield path.stem, load_function(str(path))
            except UnicodeDecodeError as e:
                print(f"Error decoding file {path}: {e}")
            except Exception as e:
                print(f"Error loading file {path}: {e}")


# to display the names (keys) more tidily
def show_keys(arrays_dict):
    for key in arrays_dict:
        print(key)


# @njit
def symmetrize_Cl(Cl, nbl, zbins):
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                Cl[ell, j, i] = Cl[ell, i, j]
    return Cl


def generate_ind(triu_tril_square, row_col_major, size):
    """
    Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str. if 'triu', returns the indices for the upper triangular part of the matrix.
    If 'tril', returns the indices for the lower triangular part of the matrix
    If 'full_square', returns the indices for the whole matrix
    :param row_col_major: str. if True, the indices are returned in row-major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'
    assert triu_tril_square in ['triu', 'tril', 'full_square'], 'triu_tril_square must be either "triu", "tril" or ' \
                                                                '"full_square"'

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """
    Builds the good old ind file
    """

    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack((LL_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)
    GL_columns = np.hstack((GL_columns, generate_ind('full_square', row_col_major, size))).astype(int)
    GG_columns = np.hstack((GG_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


def build_ind_dict(triu_tril, row_col_major, size, GL_OR_LG):

    ind = build_full_ind(triu_tril, row_col_major, size)
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    ind_dict = {}
    ind_dict['L', 'L'] = ind[:zpairs_auto, :]
    ind_dict['G', 'G'] = ind[(zpairs_auto + zpairs_cross):, :]

    if GL_OR_LG == 'LG':
        ind_dict['L', 'G'] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_OR_LG == 'GL':
        ind_dict['G', 'L'] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    return ind_dict


# CHECK FOR DUPLICATES
def cl_2D_to_3D_symmetric(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    assert Cl_2D.shape == (nbl, zpairs), f'cl_2d must have shape (nbl, zpairs) = ({nbl}, {zpairs})'

    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_asymmetric(Cl_2D, nbl, zbins, order):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert order in ['row-major', 'col-major', 'C', 'F'], 'order must be either "row-major", "C" (equivalently), or' \
                                                          '"col-major", "F" (equivalently)'
    if order == 'row-major':
        order = 'C'
    elif order == 'col-major':
        order = 'F'

    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape, order=order)
    return Cl_3D


def Cl_3D_to_2D_symmetric(Cl_3D, nbl, npairs, zbins=10):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs)  according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_2D = np.zeros((nbl, npairs))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_2D[ell, i] = Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]]
    return Cl_2D


def Cl_3D_to_2D_asymmetric(Cl_3D):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs), rows first 
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert Cl_3D.ndim == 3, 'Cl_3D must be a 3D array'

    nbl = Cl_3D.shape[0]
    zbins = Cl_3D.shape[1]
    zpairs_cross = zbins ** 2

    Cl_2D = np.reshape(Cl_3D, (nbl, zpairs_cross))

    # Cl_2D = np.zeros((nbl, zpairs_cross))
    # for ell in range(nbl):
    #     Cl_2D[ell, :] = Cl_3D[ell, :].flatten(order='C')
    return Cl_2D


def cl_3D_to_2D_or_1D(cl_3D, ind, is_auto_spectrum, use_triu_row_major, convert_to_2D, block_index):
    """ reshape from (nbl, zbins, zbins) to (nbl, zpairs), according to the ordering given in the ind file
    (valid for asymmetric Cij, i.e. C_XC)
    """

    # warnings.warn('finish this function!! (old warning, I dont remember exactly what is missing...)')
    assert cl_3D.ndim == 3, 'cl_3D must be a 3D array'
    assert cl_3D.shape[1] == cl_3D.shape[2], 'cl_3D must be a square array of shape (nbl, zbins, zbins)'

    zbins = cl_3D.shape[1]
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    if use_triu_row_major:
        ind = build_full_ind('triu', 'row-major', zbins)

    # Select appropriate indices based on whether the spectrum is auto or cross
    zpairs = zpairs_auto if is_auto_spectrum else zpairs_cross
    selected_ind = ind[:zpairs, 2:] if is_auto_spectrum else ind[zpairs_auto:zpairs_auto + zpairs_cross, 2:]

    # Vectorize the selection of elements based on the indices
    cl_2D = cl_3D[:, selected_ind[:, 0], selected_ind[:, 1]]

    if convert_to_2D:
        return cl_2D

    # Flatten the array based on the 'block_index' parameter
    order = 'C' if block_index == 'ell' else 'F' if block_index == 'zpair' else None
    if order is None:
        raise ValueError('block_index must be either "ell" or "zpair"')

    return cl_2D.flatten(order=order)


# @njit
def cl_1D_to_3D(cl_1d, nbl: int, zbins: int, is_symmetric: bool):
    """ This is used to unpack Vincenzo's files for SPV3
    Still to be thoroughly checked."""

    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    if is_symmetric:
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(iz, zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1

    else:  # take all elements, not just the upper triangle
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1
    return cl_3d


# @njit
def symmetrize_2d_array(array_2d):
    """ mirror the lower/upper triangle """

    # if already symmetric, do nothing
    if check_symmetric(array_2d, exact=True):
        return array_2d

    # there is an implicit "else" here, since the function returns array_2d if the array is symmetric
    assert array_2d.ndim == 2, 'array must be square'
    size = array_2d.shape[0]

    # check that either the upper or lower triangle (not including the diagonal) is null
    triu_elements = array_2d[np.triu_indices(size, k=+1)]
    tril_elements = array_2d[np.tril_indices(size, k=-1)]
    assert np.all(triu_elements) == 0 or np.all(tril_elements) == 0, 'neither the upper nor the lower triangle ' \
                                                                     '(excluding the diagonal) are null'

    if np.any(np.diag(array_2d)) != 0:
        warnings.warn('the diagonal elements are all null')

    # symmetrize
    array_2d = np.where(array_2d, array_2d, array_2d.T)
    # check
    if not check_symmetric(array_2d, exact=False):
        warnings.warn('check failed: the array is not symmetric')

    return array_2d


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """ mirror the lower/upper triangle """
    assert array_3D.shape == (nbl, zbins, zbins), 'shape of input array must be (nbl, zbins, zbins)'

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D
    return array_3D


def array_2D_to_1D_ind(array_2D, zpairs, ind):
    """ unpack according to "ind" ordering, same as for the Cls """
    assert ind.shape[0] == zpairs, 'ind must have lenght zpairs'

    array_1D = np.zeros(zpairs)
    for p in range(zpairs):
        i, j = ind[p, 2], ind[p, 3]
        array_1D[p] = array_2D[i, j]
    return array_1D


# @njit
def compute_FM_3D(nbl, npairs, nParams, cov_inv, D_3D):
    """ Compute FM using 3D datavector - 2D + the cosmological parameter axis - and 3D covariance matrix (working but
    deprecated in favor of compute_FM_2D)"""
    b = np.zeros((nbl, npairs, nParams))
    FM = np.zeros((nParams, nParams))
    for alf in range(nParams):
        for bet in range(nParams):
            for elle in range(nbl):
                b[elle, :, bet] = cov_inv[elle, :, :] @ D_3D[elle, :, bet]
                FM[alf, bet] = FM[alf, bet] + (D_3D[elle, :, alf] @ b[elle, :, bet])
    return FM


# @njit
def compute_FM_2D(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]
    return FM


def compute_FM_2D_optimized(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    warnings.warn('deprecate this?')
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]

    # do it with np.einsum in one line
    # FM = np.einsum('ij,ik,jk->ij', D_2D, b, cov_2D_inv)
    b = np.einsum('ij,jk->ik', cov_2D_inv, D_2D)
    FM = np.einsum('ij,jk->ik', D_2D, b)
    return FM


def compute_FoM(FM, w0wa_idxs):
    cov_param = np.linalg.inv(FM)
    # cov_param_reduced = cov_param[start:stop, start:stop]
    cov_param_reduced = cov_param[np.ix_(w0wa_idxs, w0wa_idxs)]

    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins ** 2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


def covariance_einsum(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    example code to compute auto probe data and spectra, if needed
    cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
    noise_LL_5D = noise_3x2pt_5D[0, 0, ...][np.newaxis, np.newaxis, ...]
    cov_WL_6D = sl.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
    """

    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    cov_9d = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d

    n_probes = cov_9d.shape[0]
    zbins = cov_9d.shape[-1]
    cov_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d[:, :, :, :, np.arange(nbl), ...]

    return cov_10d


def covariance_einsum_split(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    example code to compute auto probe data and spectra, if needed
    cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
    noise_LL_5D = noise_3x2pt_5D[0, 0, ...][np.newaxis, np.newaxis, ...]
    cov_WL_6D = sl.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
    """

    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d, cl_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d, cl_5d)
    cov_9d_sva = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', noise_5d, noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', noise_5d, noise_5d)
    cov_9d_sn = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d, noise_5d)
    term_2 = np.einsum('ACLik, BDLjl -> ABCDLijkl', noise_5d, cl_5d)
    term_3 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d, noise_5d)
    term_4 = np.einsum('ADLil, BCLjk -> ABCDLijkl', noise_5d, cl_5d)
    cov_9d_mix = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2 + term_3 + term_4, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d_sva, cov_9d_sn, cov_9d_mix

    n_probes = cov_9d_sva.shape[0]
    zbins = cov_9d_sva.shape[-1]

    cov_10d_sva = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d_sn = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d_mix = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))

    cov_10d_sva[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d_sva[:, :, :, :, np.arange(nbl), ...]
    cov_10d_sn[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d_sn[:, :, :, :, np.arange(nbl), ...]
    cov_10d_mix[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d_mix[:, :, :, :, np.arange(nbl), ...]

    return cov_10d_sva, cov_10d_sn, cov_10d_mix


def covariance_SSC_einsum(cl_5d, rl_5d, s_ABCD_ijkl, fsky, optimize='greedy'):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as rl_5d

    :param cl_5d:
    :param rl_5d:
    :param noise_5d:
    :param fsky:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.
    """

    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape == rl_5d.shape, 'cl_5d and rl_5d must have the same shape'

    zbins = cl_5d.shape[-1]

    assert s_ABCD_ijkl.shape == (
        cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins), \
        's_ABCD_ijkl must have shape (cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins) = ' \
        f'{(cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins)}'

    cov_SSC_10d = np.einsum('ABLij, ABLij, CDMkl, CDMkl, ABCDijkl -> ABCDLMijkl', rl_5d, cl_5d, rl_5d, cl_5d,
                            s_ABCD_ijkl, optimize=optimize)

    return cov_SSC_10d / fsky


def cov_10D_dict_to_array(cov_10D_dict, nbl, zbins, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    assert n_probes == 2, 'if more than 2 probes are used, the probe_dict must be changed ' \
                          '(promote it to an argument of the function!)'
    cov_10D_array = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    probe_dict = {'L': 0, 'G': 1}
    for A, B, C, D in cov_10D_dict.keys():
        cov_10D_array[probe_dict[A], probe_dict[B], probe_dict[C], probe_dict[D], ...] = cov_10D_dict[A, B, C, D]

    return cov_10D_array


def cov_10D_array_to_dict(cov_10D_array, probe_ordering):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    cov_10D_dict = {}
    probe_dict = {'L': 0, 'G': 1}
    for A_str, B_str in probe_ordering:
        for C_str, D_str in probe_ordering:
            A_idx, B_idx, C_idx, D_idx = probe_dict[A_str], probe_dict[B_str], probe_dict[C_str], probe_dict[D_str]
            cov_10D_dict[A_str, B_str, C_str, D_str] = cov_10D_array[A_idx, B_idx, C_idx, D_idx, ...]

    return cov_10D_dict


# @njit
def build_3x2pt_dict(array_3x2pt):
    dict_3x2pt = {}
    if array_3x2pt.ndim == 5:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :, :]
    elif array_3x2pt.ndim == 4:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :]
    return dict_3x2pt


def build_3x2pt_array(cl_LL_3D, cl_GG_3D, cl_GL_3D, n_probes, nbl, zbins):
    warnings.warn("shape is (n_probes, n_probes, nbl, zbins, zbins), NOT (nbl, n_probes, n_probes, zbins, zbins)")
    assert cl_LL_3D.shape == (nbl, zbins, zbins), f'cl_LL_3D.shape = {cl_LL_3D.shape}, should be (nbl, zbins, zbins)'
    assert cl_GL_3D.shape == (nbl, zbins, zbins), f'cl_GL_3D.shape = {cl_GL_3D.shape}, should be (nbl, zbins, zbins)'
    assert cl_GG_3D.shape == (nbl, zbins, zbins), f'cl_GG_3D.shape = {cl_GG_3D.shape}, should be (nbl, zbins, zbins)'
    cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
    cl_3x2pt_5D[0, 1, :, :, :] = cl_GL_3D.transpose(0, 2, 1)
    cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
    cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
    return cl_3x2pt_5D


def cov_3x2pt_10D_to_4D(cov_3x2pt_10D, probe_ordering, nbl, zbins, ind_copy, GL_OR_LG):
    """
    Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block separately
    in 4D, then stacks the blocks in the right order to output cov_3x2pt_4D
    (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """

    # if it's an array, convert to dictionary for the function to work
    if type(cov_3x2pt_10D) == np.ndarray:
        cov_3x2pt_dict_10D = cov_10D_array_to_dict(cov_3x2pt_10D, probe_ordering)
    elif type(cov_3x2pt_10D) == dict:
        cov_3x2pt_dict_10D = cov_3x2pt_10D
    else:
        raise ValueError('cov_3x2pt_10D must be either a dictionary or an array')

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second
    # element of the datavector
    if GL_OR_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', \
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
    elif GL_OR_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', \
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_zpairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind_copy[:npairs_auto, :]
    ind_dict['G', 'G'] = ind_copy[(npairs_auto + npairs_cross):, :]
    if GL_OR_LG == 'LG':
        ind_dict['L', 'G'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_OR_LG == 'GL':
        ind_dict['G', 'L'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(cov_3x2pt_dict_10D[A, B, C, D], nbl, npairs_dict[A, B],
                                                                npairs_dict[C, D], ind_dict[A, B], ind_dict[C, D])

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = cov_3x2pt_8D_dict_to_4D(cov_3x2pt_dict_4D, probe_ordering, combinations)

    return cov_3x2pt_4D


def cov_3x2pt_8D_dict_to_4D(cov_3x2pt_8D_dict, probe_ordering, combinations=None):
    """
    Convert a dictionary of 4D blocks into a single 4D array. This is the same code as in the last part of the function above.
    :param cov_3x2pt_8D_dict: Dictionary of 4D covariance blocks
    :param probe_ordering: tuple of tuple probes, e.g., (('L', 'L'), ('G', 'L'), ('G', 'G'))
    :combinations: list of combinations to use, e.g., [['L', 'L', 'L', 'L'], ['L', 'L', 'L', 'G'], ...]
    :return: 4D covariance array
    """

    # if combinations is not provided, construct it
    if combinations is None:
        combinations = []
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                combinations.append([A, B, C, D])

    for key in cov_3x2pt_8D_dict.keys():
        assert cov_3x2pt_8D_dict[key].ndim == 4, (
            f'covariance matrix {key} has ndim={cov_3x2pt_8D_dict[key].ndim} instead '
            f'of 4')

    # check that the number of combinations is correct
    assert len(combinations) == len(list(cov_3x2pt_8D_dict.keys())), \
        f'number of combinations ({len(combinations)}) does not match the number of blocks in the input dictionary ' \
        f'({len(list(cov_3x2pt_8D_dict.keys()))})'

    # check that the combinations are correct
    for i, combination in enumerate(combinations):
        assert tuple(combination) in list(
            cov_3x2pt_8D_dict.keys()), f'combination {combination} not found in the input dictionary'

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


def cov_3x2pt_4d_to_10d_dict(cov_3x2pt_4d, zbins, probe_ordering, nbl, ind_copy, optimize=False):

    zpairs_auto, zpairs_cross, _ = get_zpairs(zbins)

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    ind_auto = ind_copy[:zpairs_auto, :]
    ind_cross = ind_copy[zpairs_auto:zpairs_cross + zpairs_auto, :]
    ind_dict = {('L', 'L'): ind_auto,
                ('G', 'L'): ind_cross,
                ('G', 'G'): ind_auto}

    assert probe_ordering == (('L', 'L'), ('G', 'L'), ('G', 'G')), 'more elaborate probe_ordering not implemented yet'

    # slice the 4d cov to be able to use cov_4D_to_6D_blocks on the nine separate blocks
    zpairs_sum = zpairs_auto + zpairs_cross
    cov_3x2pt_8d_dict = {}
    cov_3x2pt_8d_dict['L', 'L', 'L', 'L'] = cov_3x2pt_4d[:, :, :zpairs_auto, :zpairs_auto]
    cov_3x2pt_8d_dict['L', 'L', 'G', 'L'] = cov_3x2pt_4d[:, :, :zpairs_auto, zpairs_auto:zpairs_sum]
    cov_3x2pt_8d_dict['L', 'L', 'G', 'G'] = cov_3x2pt_4d[:, :, :zpairs_auto, zpairs_sum:]

    cov_3x2pt_8d_dict['G', 'L', 'L', 'L'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, :zpairs_auto]
    cov_3x2pt_8d_dict['G', 'L', 'G', 'L'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, zpairs_auto:zpairs_sum]
    cov_3x2pt_8d_dict['G', 'L', 'G', 'G'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, zpairs_sum:]

    cov_3x2pt_8d_dict['G', 'G', 'L', 'L'] = cov_3x2pt_4d[:, :, zpairs_sum:, :zpairs_auto]
    cov_3x2pt_8d_dict['G', 'G', 'G', 'L'] = cov_3x2pt_4d[:, :, zpairs_sum:, zpairs_auto:zpairs_sum]
    cov_3x2pt_8d_dict['G', 'G', 'G', 'G'] = cov_3x2pt_4d[:, :, zpairs_sum:, zpairs_sum:]

    if optimize:
        # this version is only marginally faster, it seems
        cov_4D_to_6D_blocks_func = cov_4D_to_6D_blocks_opt
    else:
        # safer, default value
        cov_4D_to_6D_blocks_func = cov_4D_to_6D_blocks

    cov_3x2pt_10d_dict = {}
    for key in cov_3x2pt_8d_dict.keys():
        cov_3x2pt_10d_dict[key] = cov_4D_to_6D_blocks_func(
            cov_3x2pt_8d_dict[key], nbl, zbins,
            ind_dict[key[0], key[1]], ind_dict[key[2], key[3]],
            symmetrize_output_dict[key[0], key[1]],
            symmetrize_output_dict[key[2], key[3]])

    return cov_3x2pt_10d_dict


# @njit
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs) 
    to (nbl, nbl, zbins, zbins, zbins, zbins). Not valid for 3x2pt, the total
    shape of the matrix is (nbl, nbl, zbins, zbins, zbins, zbins), not big 
    enough to store 3 probes. Use cov_4D functions or cov_6D as a dictionary
    instead,
    """
    # TODO deprecate this in favor of cov_4D_to_6D_blocks

    npairs_auto, npairs_cross, npairs_tot = get_zpairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross
    else:
        raise ValueError('probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported')

    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs: maybe you\'re passing the whole ind file ' \
                                   'instead of ind[:npairs, :] - or similia'

    # TODO use jit
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    # ! this part makes this function very slow
    if probe in ['LL', 'GG']:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


# @njit
def cov_6D_to_4D(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins)
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"
    cov_4D = np.zeros((nbl, nbl, zpairs, zpairs))

    for ell2 in range(nbl):  # added this loop in the latest version, before it was vectorized
        for ij in range(zpairs):
            for kl in range(zpairs):
                # rename for better readability
                i, j, k, l = ind[ij, -2], ind[ij, -1], ind[kl, -2], ind[kl, -1]
                cov_4D[:, ell2, ij, kl] = cov_6D[:, ell2, i, j, k, l]

    return cov_4D


# TODO finish this
def cov_6D_to_4D_optim(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins)
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"

    i_indices = ind[:, -2]
    j_indices = ind[:, -1]
    k_indices = ind[:, -2]
    l_indices = ind[:, -1]

    cov_4D = cov_6D[:, :, i_indices, j_indices, k_indices, l_indices]
    cov_4D = cov_4D.reshape((nbl, nbl, zpairs, zpairs))

    return cov_4D


# @njit
def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine (valid for auto-covariance 
    LL-LL, GG-GG, GL-GL and LG-LG). n_columns is used to determine whether the ind array has 2 or 4 columns
    (if it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'
    assert cov_6D.shape[0] == cov_6D.shape[1] == nbl, 'number of angular bins does not match first two cov axes'

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays (dictionary)
    # the penultimante element is the first index, the last one the second index (see s - 1, s - 2 below)
    n_columns_AB = ind_AB.shape[1]  # of columns: this is to understand the format of the file
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, 'ind_AB and ind_CD must have the same number of columns'
    nc = n_columns_AB  # make the name shorter

    cov_4D = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for ij in range(npairs_AB):
                for kl in range(npairs_CD):
                    i, j, k, l = ind_AB[ij, nc - 2], ind_AB[ij, nc - 1], ind_CD[kl, nc - 2], ind_CD[kl, nc - 1]
                    cov_4D[ell1, ell2, ij, kl] = cov_6D[ell1, ell2, i, j, k, l]
    return cov_4D


# @njit
def cov_4D_to_6D_blocks(cov_4D, nbl, zbins, ind_ab, ind_cd,
                        symmetrize_output_ab: bool, symmetrize_output_cd: bool):
    """
    Reshapes the 4D covariance matrix to a 6D covariance matrix, even for the cross-probe (non-square) blocks needed
    to build the 3x2pt covariance.

    This function can be used for the normal routine (valid for auto-covariance, i.e., LL-LL, GG-GG, GL-GL and LG-LG) 
    where `zpairs_ab = zpairs_cd` and `ind_ab = ind_cd`.

    Args:
        cov_4D (np.ndarray): The 4D covariance matrix.
        nbl (int): The number of ell bins.
        zbins (int): The number of redshift bins.
        ind_ab (np.ndarray): The indices for the first pair of redshift bins.
        ind_cd (np.ndarray): The indices for the second pair of redshift bins.
        symmetrize_output_ab (bool): Whether to symmetrize the output cov block for the first pair of probes.
        symmetrize_output_cd (bool): Whether to symmetrize the output cov block for the second pair of probes.

    Returns:
        np.ndarray: The 6D covariance matrix.
    """

    assert ind_ab.shape[1] == ind_cd.shape[1], 'ind_ab and ind_cd must have the same number of columns'
    assert ind_ab.shape[1] == 2 or ind_ab.shape[1] == 4, 'ind_ab and ind_cd must have 2 or 4 columns'
    ncols = ind_ab.shape[1]

    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell2 in range(nbl):
        for ij in range(zpairs_ab):
            for kl in range(zpairs_cd):
                i, j, k, l = ind_ab[ij, ncols - 2], ind_ab[ij, ncols - 1], ind_cd[kl, ncols - 2], ind_cd[kl, ncols - 1]
                cov_6D[:, ell2, i, j, k, l] = cov_4D[:, ell2, ij, kl]

    # GL blocks are not symmetric
    # ! this part makes this function quite slow
    if symmetrize_output_ab:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])

    if symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


def cov_4D_to_6D_blocks_opt(cov_4D, nbl, zbins, ind_ab, ind_cd, symmetrize_output_ab, symmetrize_output_cd):
    assert ind_ab.shape[1] == ind_cd.shape[1], 'ind_ab and ind_cd must have the same number of columns'
    assert ind_ab.shape[1] in {2, 4}, 'ind_ab and ind_cd must have 2 or 4 columns'

    ncols = ind_ab.shape[1]
    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    ell2_indices, ij_indices, kl_indices = np.ogrid[:nbl, :zpairs_ab, :zpairs_cd]
    i_indices = ind_ab[ij_indices, ncols - 2]
    j_indices = ind_ab[ij_indices, ncols - 1]
    k_indices = ind_cd[kl_indices, ncols - 2]
    l_indices = ind_cd[kl_indices, ncols - 1]

    cov_6D[:, ell2_indices, i_indices, j_indices, k_indices,
           l_indices] = cov_4D[:, ell2_indices, ij_indices, kl_indices]

    if symmetrize_output_ab or symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                if symmetrize_output_ab:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])
                if symmetrize_output_cd:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


def return_combinations(A, B, C, D):
    print(f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}')


###########################
# @njit
def check_symmetric(array_2d, exact, rtol=1e-05):
    """
    :param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert type(exact) == bool, 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


def slice_cov_3x2pt_2D_ell_probe_zpair(cov_2D_ell_probe_zpair, nbl, zbins, probe):
    """ Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal matrix in ell, probe and zpair
    (unpacked in this order)"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cov_1D_ell_probe_zpair = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cov_1D_ell_probe_zpair[ell_bin] = cov_2D_ell_probe_zpair[start:stop, start:stop]

    cov_2D_ell_probe_zpair_sliced = scipy.linalg.block_diag(*cov_1D_ell_probe_zpair)

    return cov_2D_ell_probe_zpair_sliced


def slice_cl_3x2pt_1D_ell_probe_zpair(cl_3x2pt_1D_ell_probe_zpair, nbl, zbins, probe):
    """ Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal matrix in ell, probe and zpair
    (unpacked in this order)"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cl_1D_ell_probe_zpair_list = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cl_1D_ell_probe_zpair_list[ell_bin] = cl_3x2pt_1D_ell_probe_zpair[start:stop]

    cl_1D_ell_probe_zpair = np.array(list(itertools.chain(*cl_1D_ell_probe_zpair_list)))

    return cl_1D_ell_probe_zpair


# @njit
def cov_2D_to_4D(cov_2D, nbl, block_index, optimize=True, symmetrize=False):
    """ 
    Reshapes the covariance from 2D to 4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use zpair or ell as the outermost index (determined by the ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'C-style'] + ['zpair', 'F-style'], \
        'block_index must be "ell", "C-style" or "zpair", "F-style"'
    assert cov_2D.ndim == 2, 'the input covariance must be 2-dimensional'

    zpairs_AB = cov_2D.shape[0] // nbl
    zpairs_CD = cov_2D.shape[1] // nbl

    cov_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    if optimize:
        if block_index in ['ell', 'C-style']:
            cov_4D = cov_2D.reshape((nbl, zpairs_AB, nbl, zpairs_CD)).transpose((0, 2, 1, 3))
        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_4D = cov_2D.reshape((zpairs_AB, nbl, zpairs_CD, nbl)).transpose((1, 3, 0, 2))

    else:
        if block_index in ['ell', 'C-style']:
            for l1 in range(nbl):
                for l2 in range(nbl):
                    for ipair in range(zpairs_AB):
                        for jpair in range(zpairs_CD):
                            # block_index * block_size + running_index
                            cov_4D[l1, l2, ipair, jpair] = cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair]

        elif block_index in ['zpair', 'F-style']:
            for l1 in range(nbl):
                for l2 in range(nbl):
                    for ipair in range(zpairs_AB):
                        for jpair in range(zpairs_CD):
                            # block_index * block_size + running_index
                            cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]

    # mirror the upper triangle into the lower one
    if symmetrize:
        for l1 in range(nbl):
            for l2 in range(nbl):
                cov_4D[l1, l2, :, :] = symmetrize_2d_array(cov_4D[l1, l2, :, :])

    return cov_4D


# @njit
def cov_4D_to_2D(cov_4D, block_index, optimize=True):
    """ 
    Reshapes the covariance from 4D to 2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'C-style'] + ['zpair', 'F-style'], \
        'block_index must be "ell", "C-style" or "zpair", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = cov_4D.transpose(0, 2, 1, 3)

        elif block_index in ['zpair', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = cov_4D.transpose(2, 0, 3, 1)
        return cov_2D

    # I tested that the 2 methods give the same results. This code is kept to remember the
    # block_index * block_size + running_index unpacking
    if block_index in ['ell', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['zpair', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]

    return cov_2D


# @njit
def cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins, block_index='ell'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    ! Important note: block_index = 'ell' means that the overall ordering will be probe_ell_zpair. 
    ! Setting it to 'zpair' will give you the ordering probe_zpair_ell. 
    ! Bottom line: the probe is the outermost loop in any case.
    ! The ordering used by CLOE v2 is probe_ell_zpair, so block_index = 'ell' is the correct choice in this case.
    """

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], block_index, optimize=True)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], block_index, optimize=True)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], block_index, optimize=True)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], block_index, optimize=True)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], block_index, optimize=True)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], block_index, optimize=True)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], block_index, optimize=True)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], block_index, optimize=True)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], block_index, optimize=True)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


# @njit
def cov_2DCLOE_to_4D_3x2pt(cov_2D, nbl, zbins, block_index='ell'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # now I'm reshaping the full block diagonal matrix, not just the sub-blocks (cov_2D_to_4D works for both cases)
    lim_1 = zpairs_auto * nbl
    lim_2 = (zpairs_cross + zpairs_auto) * nbl
    lim_3 = zpairs_3x2pt * nbl

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_2D_to_4D(cov_2D[:lim_1, :lim_1], nbl, block_index)
    cov_LL_LG = cov_2D_to_4D(cov_2D[:lim_1, lim_1:lim_2], nbl, block_index)
    cov_LL_GG = cov_2D_to_4D(cov_2D[:lim_1, lim_2:lim_3], nbl, block_index)

    cov_LG_LL = cov_2D_to_4D(cov_2D[lim_1:lim_2, :lim_1], nbl, block_index)
    cov_LG_LG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_1:lim_2], nbl, block_index)
    cov_LG_GG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_2:lim_3], nbl, block_index)

    cov_GG_LL = cov_2D_to_4D(cov_2D[lim_2:lim_3, :lim_1], nbl, block_index)
    cov_GG_LG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_1:lim_2], nbl, block_index)
    cov_GG_GG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_2:lim_3], nbl, block_index)

    # here it is a little more difficult to visualize the stacking, but the probes are concatenated
    # along the 2 zpair_3x2pt-long axes
    cov_4D = np.zeros((nbl, nbl, zpairs_3x2pt, zpairs_3x2pt))

    zlim_1 = zpairs_auto
    zlim_2 = zpairs_cross + zpairs_auto
    zlim_3 = zpairs_3x2pt

    cov_4D[:, :, :zlim_1, :zlim_1] = cov_LL_LL
    cov_4D[:, :, :zlim_1, zlim_1:zlim_2] = cov_LL_LG
    cov_4D[:, :, :zlim_1, zlim_2:zlim_3] = cov_LL_GG

    cov_4D[:, :, zlim_1:zlim_2, :zlim_1] = cov_LG_LL
    cov_4D[:, :, zlim_1:zlim_2, zlim_1:zlim_2] = cov_LG_LG
    cov_4D[:, :, zlim_1:zlim_2, zlim_2:zlim_3] = cov_LG_GG

    cov_4D[:, :, zlim_2:zlim_3, :zlim_1] = cov_GG_LL
    cov_4D[:, :, zlim_2:zlim_3, zlim_1:zlim_2] = cov_GG_LG
    cov_4D[:, :, zlim_2:zlim_3, zlim_2:zlim_3] = cov_GG_GG

    return cov_4D


def cov_2d_dav_to_cloe(cov_2d_dav, nbl, zbins, block_index_in, block_index_out):
    """convert a 2D covariance matrix from the davide convention to the CLOE convention, that is, from the probe being
    unraveled in the first for loop to the probe being unraveled in the second for loop.
    example: from ell_probe_zpair (my convention) to probe_ell_zpair (CLOE).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first, or outermost, of the two)"""
    cov_4D = cov_2D_to_4D(cov_2d_dav, nbl, block_index=block_index_in, optimize=True)
    cov_2d_cloe = cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins=zbins, block_index=block_index_out)
    return cov_2d_cloe


def cov_2d_cloe_to_dav(cov_2d_cloe, nbl, zbins, block_index_in, block_index_out):
    """convert a 2D covariance matrix from the CLOE convention to the davide convention, that is, from the probe being
    unraveled in the second for loop to the probe being unraveled in the first for loop.
    example: from probe_ell_zpair (CLOE) to ell_probe_zpair (my convention).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first, or outermost, of the two)"""
    cov_4D = cov_2DCLOE_to_4D_3x2pt(cov_2d_cloe, nbl, zbins, block_index=block_index_in)
    cov_2d_dav = cov_4D_to_2D(cov_4D, block_index=block_index_out, optimize=True)
    return cov_2d_dav


def _cov2corr(covariance):
    """ Credit:
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def cov2corr(covariance):
    """Convert a covariance matrix to a correlation matrix."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)

    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = np.divide(covariance, outer_v)
        correlation[covariance == 0] = 0  # Ensure zero covariance entries are explicitly zero
        correlation[~np.isfinite(correlation)] = 0  # Set any NaN or inf values to 0

    # Ensure diagonal elements are exactly 1
    # np.fill_diagonal(correlation, 1)

    return correlation


def build_noise(zbins, n_probes, sigma_eps2, ng_shear, ng_clust):
    """Builds the noise power spectra.

    Parameters
    ----------
    zbins : int
        Number of redshift bins.
    n_probes : int 
        Number of probes.
    sigma_eps2 : float
        Square of the *total* ellipticity dispersion.
        sigma_eps2 = sigma_eps ** 2, with
        sigma_eps = sigma_eps_i * sqrt(2),
        sigma_eps_i being the ellipticity dispersion *per component*
    ng_shear : int, float or numpy.ndarray
        Galaxy density of sources, relevant for cosmic shear
        If a scalar, cumulative galaxy density number density, per arcmin^2. 
        This will assume equipopulated bins. 
        If an array, galaxy number density, per arcmin^2, per redshift bin. 
        Must have length zbins.
    ng_clust : int, float or numpy.ndarray
        Galaxy density of lenses, relevant for galaxy clustering
        If a scalar, cumulative galaxy density number density, per arcmin^2. 
        This will assume equipopulated bins. 
        If an array, galaxy number density, per arcmin^2, per redshift bin. 
        Must have length zbins.
    which_shape_noise : str
        Which shape noise to use. 
        'ISTF' for the "incorrect" shape noise (used in ISTF paper), for backwars-compatibility.
        'per_component' for the correct shape noise, taking into account EE-only noise.

    Returns
    -------
    noise_4d : ndarray, shape (n_probes, n_probes, zbins, zbins)
        Noise power spectra matrices

    Notes
    -----
    The noise N is defined as:
        N_LL = sigma_eps^2 / (2 * n_bar) 
        N_GG = 1 / n_bar
        N_GL = N_LG = 0

    """

    if type(ng_shear) == list:
        ng_shear = np.array(ng_shear)
    if type(ng_clust) == list:
        ng_clust = np.array(ng_clust)

    conversion_factor = (180 / np.pi * 60)**2  # deg^2 to arcmin^2

    assert isinstance(ng_shear, np.ndarray), 'ng_shear should an array'
    assert isinstance(ng_clust, np.ndarray), 'ng_clust should an array'
    assert np.all(ng_shear > 0), 'ng_shear should be positive'
    assert np.all(ng_clust > 0), 'ng_clust should be positive'

    # if ng is an array, n_bar == ng (this is a slight misnomer, since ng is the cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are quipopulated, the n_bar array should
    # have all entries almost identical.

    n_bar_shear = ng_shear * conversion_factor
    n_bar_clust = ng_clust * conversion_factor

    # create and fill N
    noise_4d = np.zeros((n_probes, n_probes, zbins, zbins))

    np.fill_diagonal(noise_4d[0, 0, :, :], sigma_eps2 / (2 * n_bar_shear))
    np.fill_diagonal(noise_4d[1, 1, :, :], 1 / n_bar_clust)
    noise_4d[0, 1, :, :] = 0
    noise_4d[1, 0, :, :] = 0

    return noise_4d
