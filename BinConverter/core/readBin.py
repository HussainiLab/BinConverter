import os
import numpy as np
import contextlib
import mmap
import numpy.distutils.system_info as sysinfo


def get_active_tetrode(set_filename):
    """in the .set files it will say collectMask_X Y for each tetrode number to tell you if
    it is active or not. T1 = ch1-ch4, T2 = ch5-ch8, etc."""
    active_tetrode = []
    active_tetrode_str = 'collectMask_'

    with open(set_filename) as f:
        for line in f:

            # collectMask_X Y, where x is the tetrode number, and Y is eitehr on or off (1 or 0)
            if active_tetrode_str in line:
                tetrode_str, tetrode_status = line.split(' ')
                if int(tetrode_status) == 1:
                    # then the tetrode is saved
                    tetrode_str.find('_')
                    tet_number = int(tetrode_str[tetrode_str.find('_') + 1:])
                    active_tetrode.append(tet_number)

    return active_tetrode


def get_active_eeg(set_filename):
    """This will return a dictionary (cative_eeg_dict) where the keys
    will be eeg channels from 1->64 which will represent the eeg suffixes (2 = .eeg2, 3 = 2.eeg3, etc)
    and the key will be the channel that the EEG maps to (a channel from 0->63)"""
    active_eeg = []
    active_eeg_str = 'saveEEG_ch'

    eeg_map = []
    eeg_map_str = 'EEG_ch_'

    active_eeg_dict = {}

    with open(set_filename) as f:
        for line in f:

            if active_eeg_str in line:
                # saveEEG_ch_X Y, where x is the eeg number, and Y is eitehr on or off (1 or 0)
                _, status = line.split(' ')
                active_eeg.append(int(status))
            elif eeg_map_str in line:
                # EEG_ch_X Y
                _, chan = line.split(' ')
                eeg_map.append(int(chan))

                # active_eeg = np.asarray(active_eeg)
                # eeg_map = np.asarray(eeg_map)

    for i, status in enumerate(active_eeg):
        if status == 1:
            active_eeg_dict[i + 1] = eeg_map[i] - 1

    return active_eeg_dict


def get_bin_data(bin_filename, channels=None, tetrode=None):
    """This function will be used to acquire the actual lfp data given the .bin filename,
    and the tetrode or channels (from 1-64) that you want to get"""

    if tetrode is not None:
        channels = get_channel_from_tetrode(tetrode)
    else:
        channels = np.array(channels)  # just in case it isn't an np.array

    bytes_per_iteration = 432

    with open(bin_filename, 'rb') as f:
        # pass
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            num_iterations = int(len(m)/bytes_per_iteration)

            data = np.ndarray((num_iterations,), (np.int16, (1,192)), m, 32, (bytes_per_iteration,)).reshape((-1, 1)).flatten()
            data = samples_to_array(data, channels=channels.tolist())

    return data


def samples_to_array(A, channels=[]):
    """This will take data matrix A, and convert it into a numpy array, there are three samples of
    64 channels in this matrix, however their channels do need to be re-mapped"""

    if channels == []:
        channels = np.arange(64) + 1
    else:
        channels = np.asarray(channels)

    A = np.asarray(A)

    sample_num = int(len(A) / 64)  # get the sample numbers

    sample_array = np.zeros((len(channels), sample_num))  # creating a 64x3 array of zeros (64 channels, 3 samples)

    for i, channel in enumerate(channels):
        sample_array[i, :] = A[get_sample_indices(channel, sample_num)]

    return sample_array


def get_sample_indices(channel_number, samples):
    remap_channel = get_remap_chan(channel_number)

    indices_scalar = np.multiply(np.arange(samples), 64)
    sample_indices = indices_scalar + np.multiply(np.ones(samples), remap_channel)

    # return np.array([remap_channel, 64 + remap_channel, 64*2 + remap_channel])
    return (indices_scalar + np.multiply(np.ones(samples), remap_channel)).astype(int)


def get_remap_chan(chan_num):
    """There is re-mapping, thus to get the correct channel data, you need to incorporate re-mapping
    input will be a channel from 1 to 64, and will return the remapped channel"""

    remap_channels = np.array([32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
                               6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
                               12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
                               18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
                               24, 25, 26, 27, 28, 29, 30, 31])

    return remap_channels[chan_num - 1]


def get_channel_from_tetrode(tetrode):
    """This function will take the tetrode number and return the Axona channel numbers
    i.e. Tetrode 1 = Ch1 -Ch4, Tetrode 2 = Ch5-Ch8, etc"""
    tetrode = int(tetrode)  # just in case the user gave a string as the tetrode

    return np.arange(1, 5) + 4 * (tetrode - 1)


def get_raw_pos(bin_filename, mode='mmap'):
    """This will get the raw position data from the .bin file in the following format:
    video timestamp, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused value

    The video timestamp is the time since the camera has been turned on, thus this value is
    irrelevent

    The .bin file is written of packets of 432 bytes containing position data, three samples of
    electrophys data (per channel), and etc.

    A packet only has a valid position if it has an "ADU2" tag in the header. You'll notice that
    the positions are sampled at twice the normal rate to avoid aliasing.

    There are two modes, one will use memory mapping, and the other will not: 'mmap' or 'non'
    """
    # pos_sample_num = 0

    byte_count = os.path.getsize(bin_filename)
    bytes_per_iteration = 432
    iteration_count = int(byte_count / bytes_per_iteration)
    # sample_count = iteration_count * 192  # each iteration has 192 samples (64*3)

    optimal_iteration = 1000000
    if optimal_iteration >= iteration_count:
        simul_iterations = iteration_count
    else:
        simul_iterations = find_n(iteration_count, optimal=optimal_iteration)

    n = int(iteration_count / simul_iterations)  # finds how many loops to do
    byte_chunksize = int(simul_iterations * bytes_per_iteration)

    DaqFs = 48000
    duration = iteration_count * 3 / DaqFs
    duration = np.ceil(duration)

    pos_Fs = 50
    n_samples = int(duration * pos_Fs)

    # Reading the Data

    # header_byte_len = 32
    # data_byte_len = 384
    # trailer_byte_len = 16

    raw_pos = np.array([]).astype(float)

    if sysinfo.platform_bits != 64:
        mode = 'non'  # do not mmap with a 32 bit python

    with open(bin_filename, 'rb') as f:

        if mode == 'mmap':
            # we will use memory mapped objects

            with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                num_iterations = int(len(m) / bytes_per_iteration)

                byte_ids = np.ndarray((num_iterations,), 'S4', m, 0, bytes_per_iteration)

                pos_bool = np.where(byte_ids == b'ADU2')  # valid position packets have an ADU2 header b'ADU2' in bytes

                valid_iterations = np.arange(len(byte_ids))[pos_bool]  # getting the packet number

                # timestamp starts at the 12th bit from the start
                time_stamp = np.ndarray((num_iterations,), np.int32, m, 12, bytes_per_iteration)[pos_bool].reshape(
                    (-1, 1))  # gettinge the time stamp data

                # position values start from the 16th bit
                raw_pos = np.ndarray((num_iterations,), (np.int16, (1, 8)), m, 16,
                                     (bytes_per_iteration,)).reshape((-1, 8))[pos_bool][:]

                raw_pos = np.hstack((valid_iterations.reshape((-1, 1)), time_stamp, raw_pos)).astype(
                    float)  # stacking all these values to create one matrix

        else:

            # in some cases using memory mapping will be slow, such as using 32 bit python

            iteration_start = 0
            # we will iterate
            for i in range(n):
                data = f.read(byte_chunksize)
                num_iterations = int(len(data) / bytes_per_iteration)

                byte_ids = np.ndarray((num_iterations,), 'S4', data, 0, 432)

                pos_bool = np.where(byte_ids == b'ADU2')  # valid position packets have an ADU2 header b'ADU2' in bytes

                valid_iterations = np.arange(len(byte_ids))[pos_bool]  # getting the packet number

                time_stamp = np.ndarray((num_iterations,), np.uint32, data, 12, 432)[pos_bool].reshape(
                    (-1, 1))  # gettinge the time stamp data

                positions = np.ndarray((num_iterations,), (np.uint16, (1, 8)), data, 16, (432,)).reshape((-1, 8))[
                                pos_bool][:]

                i = np.add(valid_iterations,
                           iteration_start)  # offsetting the samples depending on the chunk number (n)

                positions = np.hstack((i.reshape((-1, 1)), time_stamp, positions)).astype(
                    float)  # stacking all these values to create one matrix

                iteration_start += simul_iterations

                if len(raw_pos) != 0:

                    raw_pos = np.vstack((raw_pos, positions))

                else:

                    raw_pos = positions

        # raw_pos[:, 1] = np.divide(raw_pos[:, 1], 50)  # converting from a frame count to a time value in seconds

        # raw_pos is structured as: packet #, video timestamp, y1, x1, y2, x2, numpix1, numpix2, total_pix, unused value

        # the X and Y values are reverse piece-wise so lets switch the format from
        # packet #, video timestamp, y1, x1, y2, x2, numpix1, numpix2, total_pix, unused value
        # to
        # packet #, video timestamp, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused value

        raw_pos[:, 2:6] = raw_pos[:, [3, 2, 5, 4]]

        # find the first valid sample, since the data is sampled at 48kHz, and there are 3 samples per packet, the packet
        # rate is 16kHz. The positions are sampled at 50 Hz thus there is a valid position ever 320 packets. The valid position
        # will essentially take the last ADU2 headered packet values

        first_sample_index = len(
            np.where(raw_pos[:, 0] <= 320 - 1)[0]) - 1  # subtract one since indices in python start at 0

        raw_pos = raw_pos[first_sample_index:, :]

        # there should be twice the number of samples since they double sampled to stop aliasing
        # if there is not 2 * n_samples, append the last recorded sample to the end (we will assume the animal remained there
        # for the rest of the time)

        if raw_pos.shape[0] < 2 * n_samples:
            missing_n = int(2 * n_samples - raw_pos.shape[0])
            last_location = raw_pos[-1, :]
            missing_samples = np.tile(last_location, (missing_n, 1))

            raw_pos = np.vstack((raw_pos, missing_samples))

        # now we will set the oversampled data to 1023 (NaN) as that is how the converter treats the double sampled data

        # indices = np.arange(1,raw_pos.shape[0], 2)
        indices = np.arange(0, raw_pos.shape[0], 2)
        if indices[-1] >= raw_pos.shape[0]:
            indices = indices[:-1]

        # raw_pos[indices, 2:4] = 1023
        raw_pos = raw_pos[indices, :]

        raw_pos = raw_pos[:, 1:]  # don't need the packet index anymore

    return raw_pos


def find_n(iterations, optimal=1000):
    n = optimal

    while True:
        if iterations % n == 0:

            return n
        else:
            n -= 1
    return 'abort'
