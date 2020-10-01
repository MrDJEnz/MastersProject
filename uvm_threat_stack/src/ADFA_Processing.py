#!/usr/bin/env python3
# -*- coding: utf-8 -*
# def get_adfa_sequence_group(group: str = 'train_att', seq_len: int = 8, skip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#     """Gets integer encoded and n-gramed ADFA-LD training, validation or attack sets
#
#     Args:
#         seq_len: Sequence length (n) to use for n-grams
#         skip: see generate_sequence_pairs()
#         group: ("attack" | "train" | "val")
#
#     Returns:
#         The following integer encoded and n-gramed NDArrays
#         [ x_train, y_train, x_val, y_val ]
#
#     """
#     root_path = Path("../data/ADFA-LD")
#     if group == 'train':
#         path = root_path / "Training_Data_Master/"
#     elif group == "val":
#         path = root_path / "Validation_Data_Master/"
#     elif group == "attack":
#         path = root_path / "Attack_Data_Master/"
#     elif group == "train_att":
#         print("loading")
#         path = "C:/Users/djenz/OneDrive/Desktop/TS/uvm_threat_stack/data/ADFA-LD/train_att/"
#     else:
#         raise ValueError('group must be one of "attack", "train", "val", "train_att"')
#
#     if not root_path.exists():
#         urllib.request.urlretrieve(
#             "https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-IDS-Datasets/ADFA-LD.zip", "data.zip")
#         with zipfile.ZipFile("data.zip", "r") as zip_ref:
#             zip_ref.extractall("../")
#         Path("data.zip").unlink()
#         shutil.rmtree('../__MACOSX')
#
#     Path("../data").mkdir(exist_ok=True)
#
#     encoder = Encoder("../data/encoder.npy")
#     vec_encode = np.vectorize(encoder.encode)
#     data = load_files(path)
#     data_x = []
#     data_y = []
#
#     for i, row in enumerate(data):
#         x, y = generate_sequence_pairs(vec_encode(row), seq_len, skip=skip)
#         data_x.append(x)
#         data_y.append(y)
#
#     data_x = np.concatenate(data_x)
#     data_y = np.concatenate(data_y)
#     return data_x, data_y
"""Provides pre-processing functionality for the ADFA-LD dataset, used by UVM.

Provided tools include:
    Fetching of dataset
    Integer encoding of dataset
    n-gramming of dataset into sequence pairs

"""

__author__ = "Duncan Enzmann modified from:John H. Ring IV"

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np  # type: ignore
from typing import List, Tuple


def load_files(file_group: str) -> List[List[int]]:
    """Fetches integer sequences form specified AFDA group

    Looks for ADFA-LD datset in ../data/ADFA-LD fetching it to that location if required.

    Args:
        file_group: ("attack" | "train" | "val")

    Returns:
        List of lists of integer sequences

    """
    Path("../data").mkdir(exist_ok=True)

    root_path = Path("../data/ADFA-LD")
    if file_group == 'train':
        path = root_path / "Training_Data_Master/"
    elif file_group == "val":
        path = root_path / "Validation_Data_Master/"
    elif file_group == "attack":
        path = root_path / "Attack_Data_Master/"
    elif file_group == "train_att":
        path = "C:/Users/djenz/OneDrive/Desktop/TS/uvm_threat_stack/data/ADFA-LD/train_att/"
    else:
        raise ValueError('group must be one of "attack", "train", "val", "train_att"')

    if not root_path.exists():
        urllib.request.urlretrieve(
            "https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-IDS-Datasets/ADFA-LD.zip", "data.zip")
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall("../data/")
        Path("data.zip").unlink()
        shutil.rmtree('../data/__MACOSX')

    out = []
    files = path.rglob('*.txt')
    for f in files:
        with open(str(f), 'r') as myFile:
            seq = [int(x) for x in myFile.read().strip().split(' ')]
            out.append(seq)
    return out



def generate_sequence_pairs(data: list, seq_len: int, skip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generates input and output sequences

    Args:
        data: List to generate sequences from
        seq_len: Length of input and output sequences
        skip: If True inputs start after the output of the previous gram, otherwise a sliding window of one is used

    Returns:
        Numpy arrays of the input and output sequences

    """
    if skip:
        stop = 2 * seq_len
        start = 0
        ngrams = []
        while stop <= len(data):
            ngrams.append(data[start:stop])
            if skip:
                start = stop
            else:
                start = start
            stop = start + 2 * seq_len
    else:
        ngrams = list(zip(*[data[i:] for i in range(2 * seq_len)]))

    x = np.zeros((len(ngrams), seq_len), dtype=np.int)
    y = np.zeros((len(ngrams), seq_len), dtype=np.int)
    for i, gram in enumerate(ngrams):
        x[i] = gram[:seq_len]
        y[i] = gram[seq_len:]
    return x, y


class Encoder(object):
    """Encodes items as integers

    Attributes:
        file_path: location to save/load syscall map
        syscall_map: mapping from item to encoded value

    """
    file_path = Path()
    syscall_map: dict = dict()

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        if self.file_path.exists():
            self.syscall_map = np.load(self.file_path, allow_pickle=True).item()

    def encode(self, syscall) -> int:
        """Encodes an individual item

        Unique items are sequentially encoded (ie first item -> 0 next unique item -> 1). The mapping dict is updated
        with new encodings as necessary and immediately written to disk.

        Args:
            syscall: item to encode, can be an arbitrary type

        Returns:
            integer encoding of syscall

        """
        if syscall in self.syscall_map:
            return self.syscall_map[syscall]
        syscall_enc = len(self.syscall_map)
        self.syscall_map[syscall] = syscall_enc
        np.save(self.file_path, self.syscall_map)

        return syscall_enc

def get_adfa_sequence_group(group: str = 'train', seq_len: int = 8, skip: bool = False, keep_nested: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Gets integer encoded and n-gramed ADFA-LD training, validation or attack sets

    Args:
        seq_len: Sequence length (n) to use for n-grams
        skip: see generate_sequence_pairs()
        group: ("attack" | "train" | "val")

    Returns:
        The following integer encoded and n-gramed NDArrays
        [ x_data, y_data ]
    """

    encoder = Encoder("../data/encoder.npy")
    vec_encode = np.vectorize(encoder.encode)
    data = load_files(group)
    data_x = []
    data_y = []

    for i, row in enumerate(data):
        x, y = generate_sequence_pairs(vec_encode(row), seq_len, skip=skip)
        data_x.append(x)
        data_y.append(y)

    if not keep_nested:
        data_x = np.concatenate(data_x)
        data_y = np.concatenate(data_y)
    return data_x, data_y

def get_adfa_offset_sequences(group: str = 'train', seq_len: int = 8, skip: bool = True, keep_nested: bool = False):
    encoder = Encoder("../data/encoder.npy")
    vec_encode = np.vectorize(encoder.encode)
    data = [vec_encode(x) for x in load_files(group)]

    data_x = []
    data_y = []
    for row in data:
        n_grams = generate_sequences(row, seq_len + 1, skip)
        data_x.append(np.array([gram[:-1] for gram in n_grams], dtype=int))
        data_y.append(np.array([gram[1:] for gram in n_grams], dtype=int))
    if not keep_nested:
        data_x = np.concatenate(data_x)
        data_y = np.concatenate(data_y)
    return data_x, data_y


def generate_sequences(data: list, seq_len: int, skip=True) -> np.ndarray:
    if skip:
        pad_len = seq_len - (len(data) % seq_len)
        if pad_len <= seq_len / 2:
            data = np.concatenate([data, np.zeros(pad_len, dtype=int)])
        stop = seq_len
        start = 0
        ngrams = []
        while stop <= len(data):
            ngrams.append(data[start:stop])
            start = stop
            stop = start + seq_len
    else:
        ngrams = list(zip(*[data[i:] for i in range(seq_len)]))
    return np.array(ngrams)


def get_adfa_raw_sequences(group: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    encoder = Encoder("../data/encoder.npy")
    vec_encode = np.vectorize(encoder.encode)
    data = [vec_encode(x) for x in load_files(group)]

    data_x = []
    data_y = []
    for row in data:
        data_x.append(row[:-1])
        data_y.append(row[1:])
    return data_x, data_y



if __name__ == '__main__':
    data_x, data_y = get_adfa_offset_sequences(seq_len=16, skip=False)
    print(data_x.shape)
    get_adfa_raw_sequences('val')
    get_adfa_raw_sequences('attack')

