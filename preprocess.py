"""Run this first on a machine with a lot of ram to break the big pickle into chunks"""
import pathlib
import pickle
import numpy as np
from common import *
from typing import Iterable, TypeVar
import json
from tqdm.auto import tqdm
T = TypeVar('T')
"""
This script takes large pickles containing
tuple[list[list[shifts]], list[list[psf_array]]] 
and splits them into individual files containing
tuple[shift, psf_array]
"""


def flatten(input_list: Iterable[list[T]]) -> list[T]:
    """helper to flatten a list of lists"""
    return [entry for sublist in input_list for entry in sublist]


def read_pickle(pkl_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    psfs = np.array(flatten(np.array(data[0])))
    shifts = np.array(flatten(data[1]))
    return shifts, psfs


# Define the location and metadata of input files
data_sets = [
    {NAME: 'plain_small', PATH: pathlib.Path('PSFs-Nas5.pkl'),
        TAGS: {SIZE: 'small', MODE: 'plain', PIXEL_SCALE: 1}},
    {NAME: 'ao_small', PATH: pathlib.Path('PSFs-Nas5-AO.pkl'),
        TAGS: {SIZE: 'small', MODE: 'ao', PIXEL_SCALE: 1}},
    {NAME: 'ao_big', PATH: pathlib.Path('PSFsMitAO.pkl'),
        TAGS: {SIZE: 'big', MODE: 'ao', PIXEL_SCALE: 1}},
    {NAME: 'plain_big', PATH: pathlib.Path('PSFsOhneAO.pkl'),
        TAGS: {SIZE: 'big', MODE: 'plain', PIXEL_SCALE: 1}},
    {NAME: 'bare_big', PATH: pathlib.Path('PSFsNurM4.pkl'),
        TAGS: {SIZE: 'big', MODE: 'bare', PIXEL_SCALE: 1}}
    ]

data_directory = pathlib.Path('preprocessed_inputs')

if __name__ == '__main__':
    for data_set in tqdm(data_sets):
        shifts, psfs = read_pickle(data_set[PATH])

        dataset_path = data_directory / data_set[NAME]
        dataset_path.mkdir(parents=True, exist_ok=True)

        with open(dataset_path/tags_filename, 'w') as f:
            json.dump(data_set[TAGS], f)

        for shift, psf in tqdm(zip(shifts, psfs)):
            file_name = f"{str(shift[0]).replace('.','_')},{str(shift[1]).replace('.','_')}"
            with open(dataset_path/f'{file_name}.pkl', 'wb') as f:
                pickle.dump((shift, psf), f)
