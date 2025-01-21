from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from pathlib import Path
import pandas as pd
import getdist
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
from matplotlib import cm


from spaceborne import sb_lib as sl



# matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)

param_names_label = sl.mpl_other_dict['cosmo_labels_TeX']
ylabel_perc_diff_wrt_mean = sl.mpl_other_dict['ylabel_perc_diff_wrt_mean']
ylabel_sigma_relative_fid = sl.mpl_other_dict['ylabel_sigma_relative_fid']
# plt.rcParams['axes.axisbelow'] = True
# markersize = mpl_cfg.mpl_rcParams_dict['lines.markersize']


###############################################################################
################## CODE TO PLOT THE FISHER CONSTRAINTS ########################
###############################################################################


def plot_ell_cuts(ell_cuts_a, ell_cuts_b, ell_cuts_c, label_a, label_b, label_c, kmax_h_over_Mpc, zbins):
    # Get the global min and max values for the color scale
    vmin = min(ell_cuts_a.min(), ell_cuts_b.min(), ell_cuts_c.min())
    vmax = max(ell_cuts_a.max(), ell_cuts_b.max(), ell_cuts_c.min())

    # Create a gridspec layout
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.12])

    # Create axes based on the gridspec layout
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[3])

    ticks = np.arange(1, zbins + 1)
    # Set x and y ticks for both subplots
    for ax in [ax0, ax1, ax2]:
        ax.set_xticks(np.arange(zbins))
        ax.set_yticks(np.arange(zbins))
        ax.set_xticklabels(ticks, fontsize=15)
        ax.set_yticklabels(ticks, fontsize=15)
        ax.set_xlabel('$z_{\\rm bin}$', fontsize=15)
        ax.set_ylabel('$z_{\\rm bin}$', fontsize=15)

    # Display the matrices with the shared color scale
    cax0 = ax0.matshow(ell_cuts_a, vmin=vmin, vmax=vmax)
    cax1 = ax1.matshow(ell_cuts_b, vmin=vmin, vmax=vmax)
    cax2 = ax2.matshow(ell_cuts_c, vmin=vmin, vmax=vmax)

    # Add titles to the plots
    ax0.set_title(label_a, fontsize=18)
    ax1.set_title(label_b, fontsize=18)
    ax2.set_title(label_c, fontsize=18)
    fig.suptitle(f'kmax = {kmax_h_over_Mpc:.2f} h_over_mpc_tex', fontsize=18, y=0.85)

    # Add a shared colorbar on the right
    cbar = fig.colorbar(cax0, cax=cbar_ax)
    cbar.set_label('$\\ell^{\\rm max}_{ij}$', fontsize=15, loc='center', )
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


def bar_plot_old(uncert_gauss, uncert_SSC, difference):
    labels = ["$\\Omega_m$", "$\\Omega_b$", "$w_0$", "$w_a$", "$h$", "$n_s$", "$\\sigma_8$"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, uncert_gauss, width, color="mediumseagreen", label='Gauss only')
    ax.bar(x + width / 2, uncert_SSC, width, color="tomato", label='Gauss + SSC')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('relative uncertainties $\\sigma/ \\theta_{fid}$')
    ax.set_title(f'FM 1-$\\sigma$ parameter constraints, {probe} - lower is better')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("% uncertainty increase", color=color)  # we already handled the x-label with ax1
    ax2.plot(range(7), difference, "o--", color=color, markersize=markersize)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax.grid()
    plt.show()
    plt.savefig(fname=f'bar_plot_{probe}.png', dpi=300, figsize=[16, 9])


def bar_plot(data, title, label_list, divide_fom_by_10_plt, bar_width=0.18, nparams=7, param_names_label=None,
             second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=None,
             include_fom=False, figsize=None, grey_bars=False, alpha=1):
    """
    data: usually the percent uncertainties, but could also be the percent difference
    """

    no_cases = data.shape[0]
    no_params = data.shape[1]

    markers = ['^', '*', 'D', 'v', 'p', 'P', 'X', 'h', 'H', 'd', '8', '1', '2', '3', '4', 'x', '+']
    marker_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = markers[:no_cases]
    marker_colors = marker_colors[:no_cases]
    # zorders = np.arange(no_cases)  # this is because I want to revert this in the case of superimposed bars
    zorders = np.arange(1, no_cases + 1)  # this is because I want to revert this in the case of superimposed bars

    # colors = cm.Paired(np.linspace(0, 1, data.shape[1]))

    # Set position of bar on x-axis
    bar_centers = np.zeros(data.shape)

    if data.ndim == 1:  # just one vector
        data = data[None, :]
        bar_centers = np.arange(no_params)
        bar_centers = bar_centers[None, :]
    elif data.ndim != 1 and not superimpose_bars:
        for bar_idx in range(no_cases):
            if bar_idx == 0:
                bar_centers[bar_idx, :] = np.arange(no_params) - bar_width
            else:
                bar_centers[bar_idx, :] = [x + bar_idx * bar_width for x in bar_centers[0]]

    # in this case, I simply define the bar centers to be the same
    elif data.ndim != 1 and superimpose_bars:
        zorders = zorders[::-1]
        bar_centers = np.arange(no_params)
        bar_centers = bar_centers[None, :]
        bar_centers = np.repeat(bar_centers, no_cases, axis=0)

    if param_names_label is None:
        param_names_label = mpl_cfg.general_dict['cosmo_labels_TeX']
        fom_div_10_str = '/10' if divide_fom_by_10_plt else ''
        if include_fom:
            param_names_label = mpl_cfg.general_dict['cosmo_labels_TeX'] + [f'FoM{fom_div_10_str}']

    if ylabel is None:
        ylabel = ylabel_sigma_relative_fid

    if figsize is None:
        figsize = (12, 8)

    if grey_bars:
        bar_color = ['grey' for _ in range(no_cases)]
    else:
        bar_color = None

    if second_axis:

        # this check is quite obsolete...
        assert no_cases == 3, "data must have 3 rows to display the second axis"

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_axisbelow(True)

        for bar_idx in range(no_cases - no_second_axis_bars):
            ax.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey',
                   label=label_list[bar_idx])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel_sigma_relative_fid)
        ax.set_title(title)
        ax.set_xticks(range(nparams), param_names_label)

        # second axis
        ax2 = ax.twinx()
        # ax2.set_ylabel('(GS/GO - 1) $\\times$ 100', color='g')
        ax2.set_ylabel('% uncertainty increase')
        for bar_idx in range(1, no_second_axis_bars + 1):
            ax2.bar(bar_centers[-bar_idx, :], data[-bar_idx, :], width=bar_width, edgecolor='grey',
                    label=label_list[-bar_idx], color='g', alpha=alpha, zorder=zorders[bar_idx])
        ax2.tick_params(axis='y')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        return

    # elif not second_axis:
    plt.figure(figsize=figsize)
    plt.grid(zorder=0)
    plt.rcParams['axes.axisbelow'] = True

    for bar_idx in range(no_cases):
        label = label_list[bar_idx] if not superimpose_bars else None
        plt.bar(bar_centers[bar_idx, :], data[bar_idx, :], width=bar_width, edgecolor='grey', alpha=alpha,
                label=label, zorder=zorders[bar_idx], color=bar_color)
        if show_markers:
            plt.scatter(bar_centers[bar_idx, :], data[bar_idx, :], color=marker_colors[bar_idx],
                        marker=markers[bar_idx], label=label_list[bar_idx], zorder=zorders[bar_idx])

    plt.ylabel(ylabel)
    plt.xticks(range(nparams), param_names_label)
    plt.title(title)
    plt.legend()
    plt.show()


def triangle_plot_old(fm_backround, fm_foreground, fiducials, title, label_background, label_foreground,
                      param_names_labels, param_names_labels_toplot, param_names_labels_tex=None, rotate_param_labels=False):

    idxs_tokeep = [param_names_labels.index(param) for param in param_names_labels_toplot]

    # parameters' covariance matrix - first invert, then slice! Otherwise, you're fixing the nuisance parameters
    fm_inv_bg = np.linalg.inv(fm_backround)[np.ix_(idxs_tokeep, idxs_tokeep)]
    fm_inv_fg = np.linalg.inv(fm_foreground)[np.ix_(idxs_tokeep, idxs_tokeep)]

    fiducials = [fiducials[idx] for idx in idxs_tokeep]
    param_names_labels = [param_names_labels[idx] for idx in idxs_tokeep]

    if param_names_labels_tex is not None:
        warnings.warn('the user should make sure that the order of the param_names_labels_tex list is the same as \
                      the order of the param_names_labels:')
        print(param_names_labels_tex)
        print(param_names_labels)
        # remove all the "$" from param_names_labels_tex
        param_names_labels_tex = [param_name.replace('$', '') for param_name in param_names_labels_tex]

    bg_contours = GaussianND(mean=fiducials, cov=fm_inv_bg, names=param_names_labels, labels=param_names_labels_tex)
    fg_contours = GaussianND(mean=fiducials, cov=fm_inv_fg, names=param_names_labels, labels=param_names_labels_tex)

    g = plots.get_subplot_plotter(subplot_size=2.3)
    g.settings.subplot_size_ratio = 1
    g.settings.linewidth = 3
    g.settings.legend_fontsize = 20
    g.settings.linewidth_contour = 3
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.lab_fontsize = 25  # this is the x labels size
    g.settings.scaling = True  # prevent scaling down font sizes even though small subplots
    g.settings.tight_layout = True
    g.settings.axis_tick_x_rotation = 45
    g.settings.solid_colors = 'tab10'

    g.triangle_plot([bg_contours, fg_contours],
                    # names=param_names_labels,
                    filled=True, contour_lws=2, ls=['-', '-'],
                    legend_labels=[label_background, label_foreground], legend_loc='upper right',
                    contour_colors=['tab:blue', 'tab:orange'],
                    line_colors=['tab:blue', 'tab:orange'],
                    )

    if rotate_param_labels:
        # Rotate x and y parameter name labels.
        # * also useful if you want to simply align them, by setting rotation=0
        for ax in g.subplots[:, 0]:
            ax.yaxis.set_label_position("left")
            ax.set_ylabel(ax.get_ylabel(), rotation=45, labelpad=20, fontsize=30, ha='center')

        for ax in g.subplots[-1, :]:
            ax.set_xlabel(ax.get_xlabel(), rotation=45, labelpad=20, fontsize=30, ha='center', va='center')

    plt.suptitle(f'{title}', fontsize='x-large')
    plt.show()


def triangle_plot(fisher_matrices, fiducials, title, labels, param_names_labels, param_names_labels_toplot,
                  param_names_labels_tex=None, rotate_param_labels=False, contour_colors=None, line_colors=None):

    idxs_tokeep = [param_names_labels.index(param) for param in param_names_labels_toplot]

    # Invert and slice the Fisher matrices, ensuring to keep only the desired parameters
    inv_fisher_matrices = [np.linalg.inv(fm)[np.ix_(idxs_tokeep, idxs_tokeep)] for fm in fisher_matrices]

    fiducials = [fiducials[idx] for idx in idxs_tokeep]
    param_names_labels = [param_names_labels[idx] for idx in idxs_tokeep]

    if param_names_labels_tex is not None:
        warnings.warn('Ensure that the order of param_names_labels_tex matches param_names_labels.')
        param_names_labels_tex = [param_name.replace('$', '') for param_name in param_names_labels_tex]

    # Prepare GaussianND contours for each Fisher matrix
    contours = [GaussianND(mean=fiducials, cov=fm_inv, names=param_names_labels, labels=param_names_labels_tex)
                for fm_inv in inv_fisher_matrices]

    g = plots.get_subplot_plotter(subplot_size=2.3)
    g.settings.subplot_size_ratio = 1
    g.settings.linewidth = 3
    g.settings.legend_fontsize = 20
    g.settings.linewidth_contour = 3
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.lab_fontsize = 25  # this is the x labels size
    g.settings.scaling = True  # prevent scaling down font sizes even with small subplots
    g.settings.tight_layout = True
    g.settings.axis_tick_x_rotation = 45
    g.settings.solid_colors = 'tab10'

    # Set default colors if not provided
    if contour_colors is None:
        contour_colors = [f'tab:{color}' for color in ['blue', 'orange', 'green', 'red']]
    if line_colors is None:
        line_colors = contour_colors

    # Plot the triangle plot for all Fisher matrices
    g.triangle_plot(contours,
                    filled=True, contour_lws=2, ls=['-'] * len(fisher_matrices),
                    legend_labels=labels, legend_loc='upper right',
                    contour_colors=contour_colors[:len(fisher_matrices)],
                    line_colors=line_colors[:len(fisher_matrices)])

    if rotate_param_labels:
        # Rotate x and y parameter name labels
        for ax in g.subplots[:, 0]:
            ax.yaxis.set_label_position("left")
            ax.set_ylabel(ax.get_ylabel(), rotation=45, labelpad=20, fontsize=30, ha='center')

        for ax in g.subplots[-1, :]:
            ax.set_xlabel(ax.get_xlabel(), rotation=45, labelpad=20, fontsize=30, ha='center', va='center')

    plt.suptitle(f'{title}', fontsize='x-large')
    plt.show()


def contour_plot_chainconsumer(cov, trimmed_fid_dict):
    """
    example usage:
                # decide params to show in the triangle plot
                cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]
                shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                params_tot_list = cosmo_param_names + shear_bias_param_names

                trimmed_fid_dict = {param: fiducials_dict[param] for param in params_tot_list}

                # get the covariance matrix (careful on how you cut the FM!!)
                fm_idxs_tokeep = [list(fiducials_dict.keys()).index(param) for param in params_tot_list]
                cov = np.linalg.inv(fm)[fm_idxs_tokeep, :][:, fm_idxs_tokeep]

                plot_utils.contour_plot_chainconsumer(cov, trimmed_fid_dict)
    :param cov:
    :param trimmed_fid_dict:
    :return:
    """
    param_names = list(trimmed_fid_dict.keys())
    param_means = list(trimmed_fid_dict.values())

    c = ChainConsumer()
    c.add_covariance(param_means, cov, parameters=param_names, name="Cov")
    c.add_marker(param_means, parameters=param_names, name="fiducial", marker_style=".", marker_size=50, color="r")
    c.configure(usetex=False, serif=True)
    fig = c.plotter.plot()
    return fig


# parametri fiduciali
fid = np.array((0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1))
fidmn = np.array((0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1))

IST = {}
IST['WL_pes'] = np.array((0.044, 0.47, 0.16, 0.59, 0.21, 0.038, 0.019))
IST['3x2pt_pes'] = np.array((0.011, 0.054, 0.042, 0.17, 0.029, 0.010, 0.0048))
IST['GC_pes'] = np.array((0, 0, 0, 0, 0, 0, 0))

IST['WL_opt'] = np.array((0.034, 0.42, 0.14, 0.48, 0.20, 0.030, 0.013))
IST['3x2pt_opt'] = np.array((0.0059, 0.046, 0.027, 0.10, 0.020, 0.0039, 0.0022))
IST['GC_opt'] = np.array((0, 0, 0, 0, 0, 0, 0))


# xxx attento, non stai normalizzando wa (come posso farlo?)
# params = [Om, Ob, wz, wa, h, ns, s8, mn, ga, 7, 8, 9, 10]
# xxx occhio a ga e mn

def plot_FM_constr(FM, label, uncert_kind='relative'):
    rel_uncert = sl.uncertainties_FM(FM)[:7]
    plt.plot(range(7), rel_uncert * 100, "--", marker='o', label=label, markersize=markersize)


def plot_FM(general_config, covariance_config, plot_config, FM_dict):
    # plot settings:
    params = plot_config['params']
    plt.rcParams.update(params)
    markersize = plot_config['markersize']
    dpi = plot_config['dpi']
    pic_format = plot_config['pic_format']
    plot_sylvain = plot_config['plot_sylvain']
    plot_ISTF = plot_config['plot_ISTF']
    custom_label = plot_config['custom_label']

    # import settings:
    nbl = general_config['nbl']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    zbins = general_config['zbins']
    Cij_folder = general_config['cl_folder']
    which_forecast = general_config['which_forecast']

    ind_ordering = covariance_config['ind_ordering']

    case = plot_config['case']
    probe = plot_config['probe']
    GO_or_GS = plot_config['GO_or_GS']
    covmat_dav_flag = plot_config['covmat_dav_flag']
    which_plot = plot_config['which_plot']

    param_names_label = plot_config['param_names_label']

    ell_max_XC = ell_max_GC
    ########################### import ######################################

    # output_folder = sl.get_output_folder(ind_ordering, which_forecast)

    # OPTIONS
    # XXX
    # XXX
    # XXX
    """
    case = "opt"
    # case = "pes"
    
    probe = "3x2pt"
    probe = "GC"
    probe = "WL"
    
    GO_or_GS = "GS"
    # GO_or_GS = "GO"
    
    # covmat_dav_flag = "yes"                                                       # this is for GC FM computed by Sylvain using my covmat
    covmat_dav_flag = "no" 
    
    which_plot = "SSC_degradation_dav_vs_sylv"
    # which_plot = "SSC_degradation"
    which_plot = "dav_vs_sylv"
    which_plot = "bar_plot"
    # which_plot = "constraints_only"
    # which_plot = "none"
    
    """

    ############################### IMPORT FM DICTS ###############################

    # davide
    FM_dav_may = FM_dict

    # sylvain XXX renamed, careful!
    # XXX pay attention to the folder you're importing
    folder = project_path.parent / "common_data/sylvain/FM/common_ell_and_deltas/latest_downloads/renamed"
    FM_sylv_may = dict(sl.get_kv_pairs(folder, "txt"))

    # vincenzo may: to check whose GC is correct
    folder = project_path.parent / "common_data/vincenzo/14may/FishMat/EP10"
    FM_vinc_may = dict(sl.get_kv_pairs(folder, "dat"))

    ######################## COMPUTE UNCERTAINTIES ###############################
    probe_name = probe
    print(f'reminder: \nprobe = {probe} \nCij_folder = {Cij_folder}')

    if case == 'opt':
        if probe == "WL":
            ell_max = 5000
        else:
            ell_max = 3000
    elif case == 'pes':
        if probe == "WL":
            ell_max = 1500
        else:
            ell_max = 750

    # set ell_max name to "XC" in 3x2pt case (just naming)
    if probe != '3x2pt':
        probe_ell_max = probe
    if probe == '3x2pt':
        probe_ell_max = 'XC'

    # davide
    # XXX recheck
    # FM_dav_G   = FM_dav_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}"]
    # FM_dav_SSC = FM_dav_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}"]

    FM_dav_G = FM_dav_may[f"FM_{probe}_GO"]
    FM_dav_SSC = FM_dav_may[f"FM_{probe}_GS"]

    if probe == "GC" and covmat_dav_flag == "yes":  # sylvain's with my covmats
        FM_dav_G = FM_sylv_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide_DavCovmat"]
        FM_dav_SSC = FM_sylv_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide_DavCovmat"]

    # sylvain
    nbl = 30  # because in ISTF we used 30 ell bins
    FM_sylv_G = FM_sylv_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide"]
    FM_sylv_SSC = FM_sylv_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_ellDavide"]
    nbl = general_config['nbl']  # back to correct number of bins

    # uncertainties
    uncert_dav_G = sl.uncertainties_FM(FM_dav_G)[:7]
    uncert_dav_SSC = sl.uncertainties_FM(FM_dav_SSC)[:7]

    uncert_sylv_G = sl.uncertainties_FM(FM_sylv_G)[:7]
    uncert_sylv_SSC = sl.uncertainties_FM(FM_sylv_SSC)[:7]

    # if which_forecast != 'sylvain':
    # SEYFERT
    # if probe == '3x2pt':
    # FM_SEYF_G = FM_dav_may[f"FM_{probe}_G_lmax{probe_ell_max}{ell_max}_nbl{nbl}_SEYFERT"]
    # FM_SEYF_SSC = FM_dav_may[f"FM_{probe}_G+SSC_lmax{probe_ell_max}{ell_max}_nbl{nbl}_SEYFERT"]
    # uncertainties
    # uncert_SEYF_G   = sl.uncertainties_FM(FM_SEYF_G)[:7]
    # uncert_SEYF_SSC = sl.uncertainties_FM(FM_SEYF_SSC)[:7]
    # else:
    #     uncert_SEYF_G = np.zeros(7)
    #     uncert_SEYF_SSC = np.zeros(7)

    # from 2DCLOE:
    # FM_2DCLOE_G = np.genfromtxt(path / f'output/FM/{output_folder}/{Cij_folder}/FM_2DCLOE_G.txt')
    # FM_2DCLOE_GpSSC = np.genfromtxt(path / f'output/FM/{output_folder}/{Cij_folder}/FM_2DCLOE_G+SSC.txt')
    # # uncertainties
    # uncert_2DCLOE_G =  sl.uncertainties_FM(FM_2DCLOE_G)[:7]
    # uncert_2DCLOE_GpSSC =  sl.uncertainties_FM(FM_2DCLOE_GpSSC)[:7]

    ################################## COMPUTE AND PLOT ###########################

    if GO_or_GS == "GO":
        dav = uncert_dav_G
        sylv = uncert_sylv_G
        # if which_forecast != 'sylvain':
        # CLOE2D = uncert_2DCLOE_G
        # SEYF = uncert_SEYF_G

    elif GO_or_GS == "GS":
        dav = uncert_dav_SSC
        sylv = uncert_sylv_SSC
        # if which_forecast != 'sylvain':
        # CLOE2D = uncert_2DCLOE_GpSSC
        # SEYF = uncert_SEYF_SSC

    ##################################  PLOT ######################################

    if which_plot == "dav_vs_sylv":

        mean = (sylv + dav) / 2
        # mean = (SEYF + dav)/2 ; print('attention, using SEYF instead of sylvain')
        diff_dav = sl.percent_diff(dav, mean)
        diff_sylv = sl.percent_diff(sylv, mean)

        # plt.plot(range(7), diff_dav, "o--", label = f"davide {GO_or_GS}")
        # plt.plot(range(7), diff_sylv, "o--", label = f"sylvain {GO_or_GS}")
        # plt.plot(range(7), diff_SEYF, "o--", label = f"SEYF. {GO_or_GS}")

        plt.plot(range(7), diff_dav, "o", label=f"group A, {GO_or_GS}",
                 markersize=markersize)  # XXX delete, just different label
        plt.plot(range(7), diff_sylv, "o", label=f"group B, {GO_or_GS}",
                 markersize=markersize)  # XXX delete, just different label
        plt.fill_between(range(7), diff_dav, diff_sylv, color='gray', alpha=0.2)

        plt.ylabel("$ \\bar{\sigma}_\\alpha^i / \\bar{\sigma}^m_\\alpha -1$ [%]")
        # plt.title(f"FM forec., % diff w.r.t. mean, {probe}, {GO_or_GS}") # XXX ripristina
        plt.title(f"FM forec., % diff w.r.t. mean, {probe}, {case}.")  # XXX delete

    elif which_plot == "constraints_only":  # plot only constraints

        plt.plot(range(7), dav * 100, "o--", label=f"davide {custom_label} {GO_or_GS}")
        if plot_sylvain:
            plt.plot(range(7), sylv * 100, "o--", label=f"sylvain {custom_label} {GO_or_GS}")
        if plot_ISTF:
            plt.plot(range(7), IST[f'{probe}_{case}'] * 100, "o--", label=f"IST_{probe}_{case}")

        # if which_forecast != 'sylvain':
        # if probe == '3x2pt': plt.plot(range(7), SEYF, "o--", label = f"SEYF {GO_or_GS}")
        # if probe == '3x2pt': plt.plot(range(7), CLOE2D, "o--", label = f"uncert_2DCLOE {GO_or_GS}")

        plt.ylabel("$ \\sigma_\\alpha/ \\theta_{fid} [\\%]$")
        plt.title(f"FM forec., {probe}, {case}.")

    elif which_plot == "SSC_degradation_dav_vs_sylv":  # plot only constraints

        # noSSC vs SSC, dav vs sylv

        diff_dav = sl.percent_diff(uncert_dav_SSC, uncert_dav_G)
        diff_sylv = sl.percent_diff(uncert_sylv_SSC, uncert_sylv_G)

        # mean = (diff_dav + diff_sylv)/2

        # diff_dav  = sl.percent_diff(diff_dav, mean) # XXX careful
        # diff_sylv = sl.percent_diff(diff_sylv, mean)

        plt.plot(range(7), diff_dav, "o-", label=f"davide {probe}")
        plt.plot(range(7), diff_sylv, "o--", label=f"sylvain {probe}")

        plt.ylabel("$ \\sigma_{G+SSC}/ \\sigma_{G} -1$  [%]")
        plt.title(f"FM forec. SSC uncertainty increase, {probe}")

        ######################

    elif which_plot == "bar_plot":
        # davide
        diff = sl.percent_diff(uncert_dav_SSC, uncert_dav_G)
        bar_plot(uncert_dav_G, uncert_dav_SSC, diff)
        # sylvain, just to check - GC is different, remember! we agree on the
        # relative errors in the G and G + SSC cases, not on the % uncertainty increase!
        # diff = sl.percent_diff(uncert_sylv_SSC, uncert_sylv_G)
        # bar_plot(uncert_sylv_G, uncert_sylv_SSC, diff)

    elif which_plot == "SSC_degradation":
        # noSSC vs SSC
        diff_dav = sl.percent_diff(uncert_dav_SSC, uncert_dav_G)
        diff_sylv = sl.percent_diff(uncert_sylv_SSC, uncert_sylv_G)

        plt.plot(range(7), diff_dav, "o-", label=f"{probe}")
        # plt.plot(range(7), diff_sylv, "o--", label = f"sylvain {probe}")

        plt.ylabel("$ \\sigma_{G+SSC}/ \\sigma_{G} -1$  [%]")
        plt.title(f"FM forec. SSC uncertainty increase")

    elif which_plot == 'radar_plot':

        categories = param_names_label
        categories = [*categories, categories[0]]

        restaurant_1 = dav * 100
        restaurant_2 = sylv * 100
        restaurant_1 = [*restaurant_1, restaurant_1[0]]
        restaurant_2 = [*restaurant_2, restaurant_2[0]]

        fig = go.Figure(
            data=[
                go.Scatterpolar(r=restaurant_1, theta=categories, fill='toself',
                                name=f'davide {custom_label} {GO_or_GS}'),
                go.Scatterpolar(r=restaurant_2, theta=categories, fill='toself',
                                name=f'sylvain {custom_label} {GO_or_GS}'),
                # go.Scatterpolar(r=restaurant_3, theta=categories, fill='toself', name='Restaurant 3')
            ],
            layout=go.Layout(
                title=go.layout.Title(text='Restaurant comparison'),
                polar={'radialaxis': {'visible': True}},
                showlegend=True
            )
        )

        pyo.plot(fig)

    # cases
    # plot(uncert_mt, style = ".-")
    # plot(uncert_sylvain, style = "o-")
    # if SSC == "no" and probe != "GC":
    #     plot(IST)
    # plot(uncert_unif, style = "o--")
    # plot(uncert_GCph_3000, style = "o--")

    # FOM
    FoM = sl.compute_FoM(FM_dav_G)
    print(f'FoM davide G {probe}:\t{FoM:.2f}')
    FoM = sl.compute_FoM(FM_dav_SSC)
    print(f'FoM davide GpSSC {probe}:\t{FoM:.2f}')

    FoM = sl.compute_FoM(FM_sylv_G)
    print(f'FoM sylvain G {probe}:\t{FoM:.2f}')
    FoM = sl.compute_FoM(FM_sylv_SSC)
    print(f'FoM sylvain GpSSC {probe}:\t{FoM:.2f}')

    """
    values_array = np.zeros((3,3))
    
    probes = ['WL', 'GC', '3x2pt']
    
    
    # XXX BROKEN, DAVIDE GC HAS NOT THE CORRECT FM!!
    i = 0
    for (probe, ell_max) in zip(probes, (ell_max_WL, ell_max_GC, ell_max_XC)):
        
        print(probe, ell_max)
        fom = sl.compute_FoM(FM_dav_may[f"FM_{probe}_lmax{probe}{ell_max}_nbl{nbl}"])
        fom = sl.compute_FoM(FM_sylv_may["FM_WL_lmax5000_ellDavide_gauss"])
    
    
        print(f'fom G {probe}, nbl = {nbl}, l_max = {ell_max}, dav:', fom)
        fom_SSC = sl.compute_FoM(FM_dav_may[f"FM_{probe}_SSC_lmax{probe}{ell_max}_nbl{nbl}"])
        print(f'fom G+SSC {probe}, nbl = {nbl}, l_max = {ell_max}, dav:', fom_SSC)
        perc_decrease = (fom/fom_SSC-1)*100
        print(perc_decrease)
        values_array[i,:] = np.asarray((fom, fom_SSC, perc_decrease))
        i += 1
    
    a2l.to_ltx(values_array, frmt = '{:6.1f}', arraytype = 'array')
    """

    # for probe
    # a2l.to_ltx(A, frmt = '{:6.2f}', arraytype = 'array')

    if which_plot != 'bar_plot':
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.xticks(range(7), param_names_label)

        plt.show()
        # plt.yscale("log")

    # plt.savefig(path / f'output/plots/{which_plot}_{probe}_{GO_or_GS}.png', dpi = 500) #, figsize=[16,9])

    #################### ALTERNATIVE PLOT ##############

    ########## PRINT ################
    # print("sylvain WL: ", uncertainties(FM_WL_sylvain)[:7])
    # print("FM_XC_pess_master : ",  uncertainties(FM_XC_pess_master)[:7])
    # print("IST_XC_pess: ", IST_XC_pess)
    # print("perc_incr_sylvain: ", percent_diff(uncertainties(FM_WL_sylvain)[:7], SSC)[:7])

    # save on a file
    ###############################################################################
    """
    header = 'params: [$\Omega_m$, $\Omega_b$, $w_0$, $w_a$, $h$, $n_s$, $\sigma_8$]'
    uncert = {
        'params_order': '[$\Omega_m$, $\Omega_b$, $w_0$, $w_a$, $h$, $n_s$, $\sigma_8$]',
        'uncert_dav_G': list(uncert_dav_G),
        'uncert_dav_SSC': list(uncert_dav_SSC),
        'uncert_SEYF_G': list(uncert_SEYF_G),
        'uncert_SEYF_SSC': list(uncert_SEYF_SSC)
        }
    
    # import json
    # json.dump(
    #     uncert,
    #     open(f"{path}/output/uncertainties.json", "w"))
    
    import pickle
    a_file = open(f"{path}/output/uncertainties.pkl", "wb")
    pickle.dump(uncert, a_file)
    a_file.close()
    a_file = open(f"{path}/output/uncertainties.pkl", "rb")
    output = pickle.load(a_file)
    print(output)
    
    # TODO do it with pd when you have time
    # d = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pd.DataFrame(data=d)
    """
    ###############################################################################

    # new bit: vincenzo's 14 may data
    """
    FM_vinc = FM_vinc_may[f"fm{probe.lower()}o-GR-Flat-eNLA-NA-GO-Opt-EP10"]
    if probe == "GC":
        FM_vinc = np.delete(FM_vinc, (7,8,9), 0)
        FM_vinc = np.delete(FM_vinc, (7,8,9), 1)
    
    uncert_vinc_G_may = uncertainties(FM_vinc)[:7]
    
    # plt.plot(range(7), uncert_vinc_G_may, "o--", label = "uncert_vinc_G_may new")
    # plt.plot(range(7), uncert_dav_G_may, "o--", label = "uncert_dav_G_may new")
    # plt.plot(range(7), uncert_sylv_G_may, "o--", label = "uncert_sylv_G_may new")
    
    array1 = uncert_vinc_G_may
    array2 = uncert_sylv_G_may
    
    mean = (array1 + array2)/2
    diff_array1 = sl.percent_diff(array1, mean)
    diff_array2 = sl.percent_diff(array2, mean)
    plt.plot(range(7), diff_array1, "o--", label = sl.namestr(array1, globals()))
    plt.plot(range(7), diff_array2, "o--", label = sl.namestr(array2, globals()))
    
    plt.legend()
    
    sys.exit()
    """
