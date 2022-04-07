# %%
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
from typing import Iterable, TypeVar, Callable, Any, Optional
import multiprocess as mp
from tqdm.auto import tqdm
import zstandard
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
import json
import pandas as pd

from common import *
T = TypeVar('T')

# %%
# config

# Point this to the output of preprocess.py
data_directory = pathlib.Path('./preprocessed_inputs')
plot_directory = pathlib.Path('./plot_output')

plot_directory.mkdir(parents=False, exist_ok=True)
# %% [markdown]
# # Function library
# ## Helper Fünctions

# %%
RERUN_ALL_CACHED = False  # you can toggle this at the beginning of a script to re-do everything
def cached(func: Callable[[],Any], filename: pathlib.Path, rerun: Optional[bool] = False):
    """save the result of running `func` to disk and reload file if run again"""
    global RERUN_ALL_CACHED
    ext = '.pkl.zstd'
    filename = filename.with_suffix(ext)
    if not filename.exists() or rerun or RERUN_ALL_CACHED:
        result = func()
        if result is None:
            raise ValueError('The function you passed did not return anything')
        with zstandard.open(filename, 'wb') as outfile:
            pickle.dump(result, outfile)
    else:
        with zstandard.open(filename, 'rb') as infile:
            result = pickle.load(infile)
    return result


def double_trapz(arr: np.array, dx=1) -> float:
    """2D numerical integral over array"""
    assert len(arr.shape) == 2
    return np.trapz(np.trapz(arr, dx=dx), dx=dx)


@np.vectorize
def center_of_index(length: int) -> float:
    """given an array with extent `length`, what index hits the center of the array?"""
    return (length-1)/2


def centered_grid(shape: tuple[int, int], filled=True) -> np.ndarray:
    ystop, xstop = center_of_index(shape)
    ystart, xstart = -ystop, -xstop

    if filled:
        return np.mgrid[ystart:ystop:1j*shape[0], xstart:xstop:1j*shape[1]]
    else:
        return np.ogrid[ystart:ystop:1j*shape[0], xstart:xstop:1j*shape[1]]


def read_pickle(pickle_path: pathlib.Path):
    with open(pickle_path, 'rb') as f:
        shift, psf = pickle.load(f)
    return shift, psf

# %% [markdown]
# ## Cramer-Rao bound calculation for PSFs

# %%
def psf_to_fisher(psf_img: np.ndarray, constant_noise_variance: float = 0, n_photons: int = 1, oversampling=1) -> np.ndarray:
    """
    calculate Fisher information matrix for a given psf grid. Multiply with N to get information for N counts
    """
    if not len(psf_img.shape) == 2:
        raise ValueError('only 2D arrays are supported')

    dx = 1 / oversampling
    normalized_psf = psf_img / double_trapz(psf_img, dx=dx)

    prefactor = n_photons / (normalized_psf + constant_noise_variance/n_photons)
    grad = np.gradient(normalized_psf, dx)

    res = np.empty((2, 2))
    res[0, 0] = double_trapz(prefactor * grad[0] ** 2, dx=dx)
    res[0, 1] = res[1, 0] = double_trapz(prefactor * grad[0] * grad[1], dx=dx)
    res[1, 1] = double_trapz(prefactor * grad[1] ** 2, dx=dx)
    return res

def psf_cramer_rao_bound(psf_img: np.ndarray, constant_noise_variance: float = 0, n_photons: int = 1, oversampling=1) -> float:
    """
    Calculate the expected centroid deviation when performing an astrometric fit with a PSF of the given shape
    in the presence of constant and possonian noise.
    """
    fisher_matrix = psf_to_fisher(psf_img, constant_noise_variance, n_photons, oversampling)

    # variance = 1/|I^-1|; just add contributions of x, y
    sigma = np.sqrt(np.sum(np.linalg.inv(fisher_matrix)))

    return sigma


# %% [markdown]
# ## Empirical bound calculation

# %%
def empirical_bound(psf: np.ndarray, flux=10_000_000, repetitions=1):
    psf = psf / double_trapz(psf)
    psf_extended = np.zeros(np.array(psf.shape)+16)
    psf_extended[8:-8, 8:-8] = psf
    y, x = centered_grid(psf_extended.shape, filled=False)

    interpolator = RectBivariateSpline(x, y, psf_extended, kx=5, ky=5)
    x, y = centered_grid(psf.shape, filled=False)

    fit_results = []
    for i in range(repetitions):
        rng = np.random.default_rng(seed=i)
        img = rng.poisson(flux * interpolator(x, y, grid=True))

        def objective(arg):
            xshift, yshift, flux = arg
            return (img - flux*interpolator(x-xshift, y-yshift, grid=True)).flatten()

        initial_parameters = [
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(flux-0.25*np.sqrt(flux), flux+0.25*np.sqrt(flux))
        ]
        lbounds = [-1,-1, 0]
        ubounds = [1, 1, 2*flux]

        fit_results.append(least_squares(objective, initial_parameters, bounds=(lbounds, ubounds)).x)

    fit_results = np.array(fit_results)
    positional_deviation = np.mean(np.sqrt(fit_results[:, 0]**2+fit_results[:, 1]**2))

    return positional_deviation


# %% [markdown]
# ## Plotting functions

# %%
def save_plot(outdir: pathlib.Path, name: str, dpi=300):
    plt.savefig(outdir/(name+'.pdf'), dpi=dpi)
    plt.savefig(outdir/(name+'.png'), dpi=dpi)
    with open(outdir/(name+'.mplf'), 'wb') as f:
        pickle.dump(plt.gcf(), f)


def do_line_plots(dataframe:pd.DataFrame):

    # deviation line plot
    plt.figure()
    for (mode, size), group in dataframe.groupby([MODE, SIZE]):
        group = group.sort_values(by='total_shift')
        plt.plot(group.total_shift, group.expected_deviation_cr, label=f'Cramer Rao {mode} {size}')
        #plt.plot(group.total_shift, group.expected_deviation_empirical, label=f'empirical {mode} {size}')
    plt.legend()
    plt.ylabel('expected positional deviation [sqrt(px)]')
    plt.xlabel('total off-axis shift [mm]')
    save_plot(plot_directory, 'deviation_line')

    # strehl line plot
    plt.figure()
    for (mode, size), group in dataframe.groupby([MODE, SIZE]):
        group = group.sort_values(by='total_shift')
        plt.plot(group.total_shift, group.strehl, label=f'Strehl {mode} {size}')
    plt.legend()
    #plt.title('')
    plt.ylabel('Strehl ratio')
    plt.xlabel('total off-axis shift [mm]')
    save_plot(plot_directory, 'strehl_line')

    plt.figure()
    for (mode, size), group in dataframe.groupby([MODE, SIZE]):
        group = group.sort_values(by='total_shift')
        plt.plot(group.total_shift, group.strehl*group.expected_deviation_cr, label=f'{mode} {size}')
    plt.legend()
    #plt.title('')
    plt.ylabel('Strehl * expected shift')
    plt.xlabel('total off-axis shift [mm]')
    save_plot(plot_directory, 'strehl_times_deviation')



def do_pcm_plots(dataframe: pd.DataFrame):
    for (mode, size), group in dataframe.groupby([MODE, SIZE]):


        group = group.sort_values(by=['xshift', 'yshift'])
        shape = [int(np.sqrt(len(group)))] * 2
        xshift, yshift, cr_bound, strehl = np.array((group.xshift, group.yshift, group.expected_deviation_cr, group.strehl))

        # so this complains because the positions don't really form a grid (only approximately).
        plt.figure()
        plt.pcolormesh(xshift.reshape(shape), yshift.reshape(shape), cr_bound.reshape(shape),
                       shading='nearest')
        plt.title(f'{mode}, {size}')
        plt.colorbar(label='expected deviation [~pixel]')
        save_plot(plot_directory, f'pcm_{mode}_{size}_cr')

        plt.figure()
        plt.pcolormesh(xshift.reshape(shape), yshift.reshape(shape), strehl.reshape(shape),
                       shading='nearest')
        plt.title(f'{mode}, {size}')
        plt.colorbar(label='strehl')
        save_plot(plot_directory, f'pcm_{mode}_{size}_strehl')

        plt.figure()
        plt.pcolormesh(xshift.reshape(shape), yshift.reshape(shape), (cr_bound*strehl).reshape(shape),
                       shading='nearest')
        plt.title(f'{mode}, {size}')
        plt.colorbar(label='strehl * expected deviation')
        save_plot(plot_directory, f'pcm_{mode}_{size}_strehl_times_deviation')

# %%
# Wrapper to run it all

def compute_cr_bounds(pickle_path: pathlib.Path) -> dict:
    shift, psf = read_pickle(pickle_path)
    expected_deviation = psf_cramer_rao_bound(psf)
    return {'shift': shift, 'expected_deviation_cr': expected_deviation}


def compute_empirical_bounds(pickle_path: pathlib.Path) -> dict:
    shift, psf = read_pickle(pickle_path)

    flux = 10_000_000
    repetitions = 1
    expected_deviation = empirical_bound(psf, flux=flux, repetitions=repetitions)
    return {'shift': shift,
            'expected_deviation_empirical': expected_deviation,
            'flux_empirical': flux,
            'repetitions_empirical': repetitions}


def compute_strehls(pickle_path: pathlib.Path) -> dict:
    shift, psf = read_pickle(pickle_path)
    return {'shift': shift, 'strehl': np.max(psf)/double_trapz(psf)}


def do_computations(data_dir: pathlib.Path = data_directory,
                    functions_to_apply: tuple[Callable[[T, T], T]] = (compute_cr_bounds, compute_empirical_bounds, compute_strehls)):
    records = []
    with mp.Pool(mp.cpu_count() - 1) as p:
        for dataset_dir in data_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            with open(dataset_dir/tags_filename) as f:
                tags = json.load(f)
            for function in functions_to_apply:
                args = tqdm(list(dataset_dir.glob('*.pkl')), desc=str(dataset_dir.name)+' '+function.__name__)
                computed = p.map(function, args, 1)
                # add tags to records
                records += [computed_rec | tags for computed_rec in computed]
    return records

# %% [markdown]
# # Main Course

# %%
if __name__ == '__main__':
    records = cached(lambda: do_computations(), pathlib.Path('./cached_records'), rerun=False)
    dataframe = pd.DataFrame.from_records(records)
    dataframe = dataframe.groupby([dataframe['shift'].astype(str), MODE, SIZE], as_index=False).first()
    dataframe['xshift'] = dataframe['shift'].apply(lambda x: x[0])
    dataframe['yshift'] = dataframe['shift'].apply(lambda x: x[1])
    dataframe['total_shift'] = np.sqrt(dataframe.xshift**2+dataframe.yshift**2)

    do_line_plots(dataframe)
    do_pcm_plots(dataframe)
    plt.show()

