import os, json
import struct
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import re
import profile
import time
import mmap
import contextlib
import numpy.distutils.system_info as sysinfo

#bin_filename = 'C:\\Users\\Geoffrey Barrett\\Desktop\\RAW Data\\20170823-3-CYLINDEROPENFIELD_Mult.bin'
# bin_filename = 'C:\\Users\\Geoffrey Barrett\\Desktop\\RAW Data\\20170823-2-CYLINDEROPENFIELD.bin'
#bin_filename = 'C:\\Users\\Taub Institute\\Desktop\\RAW Data\\20170823-3-CYLINDEROPENFIELD_Mult.bin'
#bin_filename = 'C:\\Users\\Taub Institute\\Desktop\\RAW Data\\20170823-2-CYLINDEROPENFIELD.bin'
bin_filename = "C:\\Users\\Taub Institute\\Desktop\\RAW Data\\ConvFromOtherDaq\\20170829-1-CYLINDEROPENFIELD.bin"
#bin_filename = "C:\\Users\\Geoffrey Barrett\\Desktop\\RAW Data\\ConvFromOtherDaq\\20170829-1-CYLINDEROPENFIELD.bin"

bin_directory = os.path.dirname(bin_filename)
session = os.path.basename(os.path.splitext(bin_filename)[0])
set_filename = os.path.join(bin_directory, '%s.set' % session)
pos_filename = os.path.join(bin_directory, '%s.pos' % session)


def get_lfp_bytes(iterations):
    """This function works, but the strategy I use didn't. I was going to slice the bytearray like you would
    a numpy array but that didn't work"""
    data_byte_len = 384
    indices = np.arange(data_byte_len)
    indices = np.tile(indices, (1, iterations))
    indices = indices.flatten()

    offset_indices = np.arange(iterations)
    offset_indices = offset_indices * 432 + 32
    offset_indices = np.tile(offset_indices, (384, 1))
    offset_indices = offset_indices.flatten(order='F')

    return indices + offset_indices


def get_lfp_indices(iterations):
    data_byte_len = 192
    indices = np.arange(data_byte_len)
    indices = np.tile(indices, (1, iterations))
    indices = indices.flatten()

    offset_indices = np.arange(iterations)
    offset_indices = offset_indices * 213 + 13
    offset_indices = np.tile(offset_indices, (192, 1))
    offset_indices = offset_indices.flatten(order='F')

    return indices + offset_indices


def find_n(iterations, optimal=1000):
    n = optimal

    while True:
        if iterations % n == 0:

            return n
        else:
            n -= 1
    return 'abort'


def get_remap_chan(chan_num):
    """There is re-mapping, thus to get the correct channel data, you need to incorporate re-mapping
    input will be a channel from 1 to 64, and will return the remapped channel"""

    remap_channels = np.array([32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
                               6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
                               12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
                               18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
                               24, 25, 26, 27, 28, 29, 30, 31])

    return remap_channels[chan_num - 1]


def get_channel_bytes(channel_number, samples):
    """This will get the indices of the data if it is just the lfp data (not the bytes header bytes or trailing bytes)"""
    remap_channel = get_remap_chan(channel_number)

    indices_scalar = np.multiply(np.arange(samples), 64)
    sample_indices = indices_scalar + np.multiply(np.ones(samples), remap_channel)

    # return np.array([remap_channel, 64 + remap_channel, 64*2 + remap_channel])
    return (indices_scalar + np.multiply(np.ones(samples), remap_channel)).astype(int)


def get_channel_from_tetrode(tetrode):
    """This function will take the tetrode number and return the Axona channel numbers
    i.e. Tetrode 1 = Ch1 -Ch4, Tetrode 2 = Ch5-Ch8, etc"""
    tetrode = int(tetrode)  # just in case the user gave a string as the tetrode

    return np.arange(1, 5) + 4 * (tetrode - 1)

bytes_per_iteration = 432


filename = 'C:\\Users\\Taub Institute\\Desktop\\RAW Data\\Raw30Min\\20170927-RAW-30MIN.bin'
# filename = 'C:\\Users\\Geoffrey Barrett\\Desktop\\RAW Data\\Raw30m\\20170927-RAW-30MIN.bin'
# filename = bin_filename

with open(filename, 'rb') as f:
    with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
        num_iterations = int(len(m) / 432)

        data = np.ndarray((1,), (np.uint16, (1, 192)), m, 32, strides=(432,))[0]

print(data)

x = 1