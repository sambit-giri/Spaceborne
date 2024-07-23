from functools import partial
import logging
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import delayed, Parallel
import multiprocessing as mp
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from scipy.special import spherical_jn
import pyccl as ccl
from tqdm import tqdm
# import PySSC

import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as csmlib


start_time = time.perf_counter()


# TODO play around with the k_grid_sigma2 and z_grid_sigma2
# TODO should we add more points at low z for sigma2? I get a strange behavior...
# TODO range(ell1_idx, nbl) to avoid computing symmetric ell elements (check first)


# * logbook
# - the vectorization is quite messy; the quad version accepts z_1 or z_2 as vector, but only
# setting sigma2_integrating_function=quad_vec
# - the simpson version is not vectorized in z1/z2, but it is much faster than the quad version (and much noisier!!)
# - find the optimal k_grid_sigma2 and just fix it


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# ! inputs needed by the code:
# * primary:
# - cosmo params
# - fsky
# - zbins
# - ell_min, ell_max, nbl (or ell_grid)

# * secondary:
# - growth_factor

# TODO finish implementing this function and test if if needed
# def sigma2_flatsky(z1, z2, k_perp_grid, k_par_grid, cosmo_ccl, Omega_S, theta_S):
#     """Compute the flatsky variance between two redshifts z1 and z2 for a cosmology given by cosmo_ccl."""

#     # Compute the comoving distance at the given redshifts
# from scipy.special import j1 as J1
#     a1 = 1 / (1 + z1)
#     a2 = 1 / (1 + z2)
#     r1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
#     r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

#     # Compute the growth factors at the given redshifts
#     growth_factor_z1 = ccl.growth_factor(cosmo_ccl, a1)
#     growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

#     # Compute the integrand over k_perp and k_par grids
#     def integrand(k_perp, k_par, r1, r2, theta_S):
#         k = np.sqrt(k_par**2 + k_perp**2)
#         bessel_term = J1(k_perp * theta_S * r1) * J1(k_perp * theta_S * r2) / (k_perp * theta_S * r1 * k_perp * theta_S * r2)
#         power_spectrum = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)
#         return k_perp * bessel_term * np.cos(k_par * (r1 - r2)) * power_spectrum

#     # Perform the double integral using Simpson's rule
#     integral_result_k_perp = np.array([
#         simps(integrand(k_perp, k_par_grid, r1, r2, theta_S), k_par_grid)
#         for k_perp in k_perp_grid
#     ])
#     integral_result = simps(integral_result_k_perp, k_perp_grid)

#     # Compute the final result
#     sigma2 = 1 / (2 * np.pi**2) * growth_factor_z1 * growth_factor_z2 * integral_result / Omega_S**2

#     return sigma2

def sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B, ell_mask=None, cl_mask=None):
    """ Computes the integral in k. The rest is in another function, to vectorize the call to the growth_factor.
    Note that the 1/Omega_S^2 factors are missing in this function!! This is consistent with the definition given in
    my and Fabien's paper."""

    # compute the comoving distance at the given redshifts
    a1 = csmlib.z_to_a(z1)
    a2 = csmlib.z_to_a(z2)

    # in Mpc
    r1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    # compute the growth factors at the given redshifts
    growth_factor_z1 = ccl.growth_factor(cosmo_ccl, a1)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

    # define the integrand as a function of k
    def integrand(k): return k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
        spherical_jn(0, k * r1) * spherical_jn(0, k * r2)

    integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2)

    # different integration methods; simps seems to be the best
    # if integrating_funct == 'simps':
    #     integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2)
    # elif integrating_funct == 'quad':
    #     integral_result = quad(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # elif integrating_funct == 'quad_vec':
    #     integral_result = quad_vec(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # else:
    #     raise ValueError('sigma2_integrating_function must be either "simps" or "quad" or "quad_vec"')

    if which_sigma2_B == 'full-curved-sky':
        result = 1 / (2 * np.pi ** 2) * growth_factor_z1 * growth_factor_z2 * integral_result
    elif which_sigma2_B == 'mask':
        fsky = np.sqrt(cl_mask[0] / (4 * np.pi))
        result = 1 / (4 * np.pi * fsky)**2 * np.sum((2 * ell_mask + 1) * cl_mask * 2 /
                                                    np.pi * growth_factor_z1 * growth_factor_z2 * integral_result)
    else:
        raise ValueError('which_sigma2_B must be either "full-curved-sky" or "mask"')

    return result


def sigma2_func_vectorized(z1_arr, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B, ell_mask=None, cl_mask=None):
    """
    Version of sigma2_func vectorized in z1.
    """
    
    a1_arr = csmlib.z_to_a(z1_arr)
    a2 = csmlib.z_to_a(z2)

    r1_arr = ccl.comoving_radial_distance(cosmo_ccl, a1_arr)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    growth_factor_z1_arr = ccl.growth_factor(cosmo_ccl, a1_arr)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

    # Define the integrand as a function of k
    def integrand(k):
        return k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
            spherical_jn(0, k * r1_arr[:, None]) * spherical_jn(0, k * r2)

    integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2, axis=1)

    if which_sigma2_B == 'full-curved-sky':
        result = 1 / (2 * np.pi ** 2) * growth_factor_z1_arr * growth_factor_z2 * integral_result
    elif which_sigma2_B == 'mask':
        fsky = np.sqrt(cl_mask[0] / (4 * np.pi))
        result = 1 / (4 * np.pi * fsky) ** 2 * \
            np.sum((2 * ell_mask + 1) * cl_mask * 2 / np.pi *
                   growth_factor_z1_arr[:, None] * growth_factor_z2 * integral_result, axis=1)
    else:
        raise ValueError('which_sigma2_B must be either "full-curved-sky" or "mask"')

    return result



def sigma2_func_homogenized(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B, ell_mask=None, cl_mask=None):
    """
    Homogenized version of sigma2_func that can handle both scalar and vector inputs for z1.
    """
    # Ensure z1 is an array for vectorized operations
    z1 = np.atleast_1d(z1)
    
    # Convert redshifts to scale factors
    a1 = csmlib.z_to_a(z1)
    a2 = csmlib.z_to_a(z2)
    
    # Compute comoving distances
    r1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)
    
    # Compute growth factors
    growth_factor_z1 = ccl.growth_factor(cosmo_ccl, a1)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)
    
    # Define the integrand as a function of k
    def integrand(k):
        return k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
               spherical_jn(0, k * r1[..., None]) * spherical_jn(0, k * r2)
    
    # Compute the integral
    integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2, axis=-1)
    
    # Compute the result based on which_sigma2_B
    if which_sigma2_B == 'full-curved-sky':
        result = 1 / (2 * np.pi ** 2) * growth_factor_z1 * growth_factor_z2 * integral_result
    elif which_sigma2_B == 'mask':
        fsky = np.sqrt(cl_mask[0] / (4 * np.pi))
        result = 1 / (4 * np.pi * fsky) ** 2 * \
                 np.sum((2 * ell_mask + 1) * cl_mask * 2 / np.pi *
                        growth_factor_z1[..., None] * growth_factor_z2 * integral_result, axis=-1)
    else:
        raise ValueError('which_sigma2_B must be either "full-curved-sky" or "mask"')
    
    # If z1 was originally a scalar, return a scalar result
    if np.isscalar(z1):
        return result[0]
    return result

def plot_sigma2(sigma2_arr, z_grid_sigma2):
    font_size = 28
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size

    plt.figure()
    pad = 0.4  # I don't want to plot sigma at the edges of the grid, it's too noisy
    for z_test in np.linspace(z_grid_sigma2.min() + pad, z_grid_sigma2.max() - pad, 4):
        z1_idx = np.argmin(np.abs(z_grid_sigma2 - z_test))
        z_1 = z_grid_sigma2[z1_idx]

        plt.plot(z_grid_sigma2, sigma2_arr[z1_idx, :], label=f'$z_1=%.2f$ ' % z_1)
        plt.axvline(z_1, color='k', ls='--', label='$z_1$')
    plt.xlabel('$z_2$')
    plt.ylabel('$\sigma^2(z_1, z_2)$')  # sigma2 is dimensionless!
    plt.legend()
    plt.show()

    font_size = 18
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size
    mm.matshow(sigma2_arr, log=True, abs_val=True, title='$\sigma^2(z_1, z_2)$')

    # z_steps_sigma2 = len(z_grid_sigma2)
    # plt.savefig(f'../output/plots/sigma2_spikes_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')
    # plt.savefig(f'../output/plots/sigma2_matshow_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')


def compute_sigma2(z_grid_sigma2, k_grid_sigma2, which_sigma2_B, cosmo_ccl, parallel=True, vectorize=False):
    print(f'computing sigma^2(z_1, z_2) for SSC...')

    if parallel:
        # ! parallelize with ray
        # start_time = time.perf_counter()
        # sigma2_func_remote = ray.remote(sigma2_func)
        # remote_calls = []
        # for z1 in tqdm(z_grid_sigma2):
        #     for z2 in z_grid_sigma2:
        #         remote_calls.append(sigma2_func_remote.remote(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B))
        # # Get the results from the remote function calls
        # sigma2_arr = ray.get(remote_calls)

        # ! with joblib (doesn't seem to work anymore, I still don't know why)
        # sigma2_arr = Parallel(n_jobs=-1, backend='loky')(delayed(sigma2_func)(
        # z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B) for z1 in tqdm(z_grid_sigma2) for z2 in z_grid_sigma2)

        # ! with pool.map
        # Share cosmo_ccl object using custom SharedCosmology
        # shm_cosmo_ccl = SharedCosmology([cosmo_ccl])

        # with mp.Pool() as pool:
        #     sigma2_arr = pool.starmap(
        #         partial(sigma2_func, k_grid_sigma2=k_grid_sigma2, cosmo_ccl=shm_cosmo_ccl[0], which_sigma2_B=which_sigma2_B),
        #         [(z1, z2) for z1 in z_grid_sigma2 for z2 in z_grid_sigma2]
        #     )

        # reshape result
        sigma2_arr = np.array(sigma2_arr).reshape((len(z_grid_sigma2), len(z_grid_sigma2)))
        print(f'sigma2 computed in: {(time.perf_counter() - start_time):.2f} s')

    # ! serial version
    else:
        sigma2_arr = np.zeros((len(z_grid_sigma2), len(z_grid_sigma2)))

        if not vectorize:
            for z1_idx, z1 in enumerate(tqdm(z_grid_sigma2)):
                for z2_idx, z2 in enumerate(z_grid_sigma2):
                    sigma2_arr[z1_idx, z2_idx] = sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B)
        elif vectorize:
            for z2_idx, z2 in enumerate(tqdm(z_grid_sigma2)):
                sigma2_arr[:, z2_idx] = sigma2_func_vectorized(
                    z_grid_sigma2, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_B, None, None)

    return sigma2_arr


# @ray.remote
def batch_sigma2_func(batch, k_grid_sigma2, cosmo_ccl, ):
    results = []
    for z1, z2 in batch:
        result = sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl, )
        results.append(result)
    return results


def compute_sigma2_batched(sigma2_cfg, cosmo_ccl, batch_size=10, parallel=True, parallel_code='joblib'):
    print(f'computing sigma^2(z_1, z_2) for SSC...')

    z_grid_sigma2 = np.linspace(sigma2_cfg['z_min_sigma2'], sigma2_cfg['z_max_sigma2'], sigma2_cfg['z_steps_sigma2'])
    k_grid_sigma2 = np.logspace(sigma2_cfg['log10_k_min_sigma2'], sigma2_cfg['log10_k_max_sigma2'],
                                sigma2_cfg['k_steps_sigma2'])

    if parallel:
        start_time = time.perf_counter()

        # Create a list of all pairs (z1, z2)
        all_pairs = [(z1, z2) for z1 in z_grid_sigma2 for z2 in z_grid_sigma2]

        # Split all_pairs into smaller chunks (batches)
        batches = [all_pairs[i:i + batch_size] for i in range(0, len(all_pairs), batch_size)]

        if parallel_code == 'ray':
            batch_sigma2_func_remote = ray.remote(batch_sigma2_func)
            remote_calls = [batch_sigma2_func_remote.remote(batch, k_grid_sigma2, cosmo_ccl) for batch in
                            tqdm(batches)]

            # Get the results from the remote function calls
            start_time_remote = time.perf_counter()
            results = ray.get(remote_calls)
            print(f'remote calls gathered in: {(time.perf_counter() - start_time_remote):.2f} s')

        elif parallel_code == 'joblib':
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(batch_sigma2_func)(batch, k_grid_sigma2, cosmo_ccl) for batch in tqdm(batches))
        else:
            raise ValueError('parallel_code must be either "ray" or "joblib"')

        sigma2_arr = np.concatenate(results)

        # Reshape result
        sigma2_arr = np.array(sigma2_arr).reshape((len(z_grid_sigma2), len(z_grid_sigma2)))
        print(f'sigma2 computed in: {(time.perf_counter() - start_time):.2f} s')

    else:
        # Serial version (unchanged)
        sigma2_arr = np.zeros((len(z_grid_sigma2), len(z_grid_sigma2)))
        for z1_idx, z1 in enumerate(tqdm(z_grid_sigma2)):
            for z2_idx, z2 in enumerate(z_grid_sigma2):
                sigma2_arr[z1_idx, z2_idx] = sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl)

    return sigma2_arr, z_grid_sigma2


def interpolate_sigma2_arr(sigma2_arr, z_grid_original, z_grid_new):
    """ Interpolate sigma2_arr from z_grid_original to z_grid_new. This is needed because the covmat is computed
    in a different z_grid than the one used to compute sigma2."""

    # TODO test this!

    sigma2_interp_func = RegularGridInterpolator((z_grid_original, z_grid_original), sigma2_arr, method='linear')

    z_grid_new_xx, z_grid_new_yy = np.meshgrid(z_grid_new, z_grid_new)
    sigma2_arr_interpolated = sigma2_interp_func((z_grid_new_xx, z_grid_new_yy)).T
    return sigma2_arr_interpolated


def sigma2_pyssc(z_arr, classy_cosmo_params):
    """ Compute sigma2 with PySSC. This is just for comparison, it is not used in the code."""
    if classy_cosmo_params is None:
        logging.info('Using default classy cosmo params from cosmo_lib')
        classy_cosmo_params = csmlib.cosmo_par_dict_classy
    if z_arr is None:
        # ! 1e-3 as zmin gives errors in classy, probably need to increse pk_max
        z_arr = np.linspace(1e-2, 3, 300)
    return PySSC.sigma2_fullsky(z_arr, cosmo_params=classy_cosmo_params, cosmo_Class=None)


def compare_sigma2_sb_vs_pyssc(z_arr_pyssc, sigma2_pyssc_arr, z_1_idx=100):
    path = f'{ROOT}/exact_SSC/output/integrand_arrays/sigma2'
    sigma2_sb = np.load(f'{path}/sigma2_zsteps3000_simps.npy')
    z_arr_sb = np.load(f'{path}/z_grid_sigma2_zsteps3000.npy')

    sigma2_sb_interp = RegularGridInterpolator((z_arr_sb, z_arr_sb), sigma2_sb, method='linear')
    z_arr_xx, z_arr_yy = np.meshgrid(z_arr_pyssc, z_arr_pyssc)
    sigma2_sb = sigma2_sb_interp((z_arr_xx, z_arr_yy)).T

    plt.figure()
    plt.plot(z_arr_pyssc, sigma2_pyssc_arr[z_1_idx, :], label='PySSC', marker='.')
    plt.plot(z_arr_pyssc, sigma2_sb[z_1_idx, :], label='Spaceborne', ls='--', marker='.')
    plt.axvline(z_arr_pyssc[z_1_idx], color='k', ls='--', label='$z_1$')
    plt.xlim(0, 2.5)
    plt.xlabel('$z_2$')
    plt.ylabel('$\sigma^2(z_1, z_2)$')
    plt.legend()

# TODO compute sigma_b with PyCCL for a rought comparison
# fsky = csmlib.deg2_to_fsky(cfg['sky_area_deg2'])
# sigma2_ccl = ccl.sigma2_B_disc(cosmo=cosmo_ccl, a=csmlib.z_to_a(z_grid_sigma2)[::-1], fsky=fsky, p_of_k_a=None)
# plt.plot(z_grid_sigma2, sigma2_ccl, label='PyCCL')
