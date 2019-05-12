import os
import numpy as np
import struct
from .Tint_Matlab import int16toint8, get_setfile_parameter
import scipy
import scipy.signal
from .conversion_utils import get_set_header


def fir_hann(data, Fs, cutoff, n_taps=101, showresponse=0):
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2

    b = scipy.signal.firwin(n_taps, cutoff / nyq_rate, window='hann')

    a = 1.0
    # Use lfilter to filter x with the FIR filter.
    data = scipy.signal.lfilter(b, a, data)

    '''
    if showresponse == 1:
        w, h = scipy.signal.freqz(b, a, worN=8000)  # returns the requency response h, and the angular frequencies
        # w in radians/sec
        # w (radians/sec) * (1 cycle/2pi*radians) = Hz
        # f = w / (2 * np.pi)  # Hz

        plt.figure(figsize=(20, 15))
        plt.subplot(211)
        plt.semilogx((w / np.pi) * nyq_rate, np.abs(h), 'b')
        plt.xscale('log')
        plt.title('%s Filter Frequency Response')
        plt.xlabel('Frequency(Hz)')
        plt.ylabel('Gain [V/V]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='green')
    '''

    return data, n_taps


def create_eeg(filename, data, Fs, DC_Blocker=True):
    # data is given in int16

    if os.path.exists(filename):
        return

    Fs_EGF = int(4.8e3)  # sampling rate of .EGF files
    Fs_EEG = int(250)

    data = data[:, 0::int(Fs / Fs_EGF)]

    # append zeros to make the duration a round number
    duration_round = np.ceil(data.shape[1] / Fs_EGF)  # the duration should be rounded up to the nearest integer
    missing_samples = int(duration_round * Fs_EGF - data.shape[1])
    if missing_samples != 0:
        missing_samples = np.tile(np.array([0]), (1, missing_samples))
        data = np.hstack((data, missing_samples))

    # ensure the full last second of data is equal to zero
    data[0, -int(Fs_EGF):] = 0

    data = data[0, :-Fs_EGF]

    # data = np.rint(data)  # convert the data to integers
    data = data.astype(np.int32)  # convert the data to integers

    # ensuring the appropriate range of the values
    data[np.where(data > 32767)] = 32767
    data[np.where(data < -32768)] = -32768

    data, N = fir_hann(data, Fs_EGF, 125, n_taps=101, showresponse=0)  # FIR filter to remove anti aliasing

    data = int16toint8(data)

    data = EEG_downsample(data)

    # append 1 second of 0's like tint does
    data = np.hstack((data.flatten(), np.zeros((250, 1)).flatten()))
    ##################################################################################################
    # ---------------------------Writing the EEG Data-------------------------------------------
    ##################################################################################################

    write_eeg(filename, data, Fs_EEG)


def EEG_downsample(EEG):
    """The EEG data is created from the EGF files which involves a 4.8k to 250 Hz conversion"""
    EEG = EEG.flatten()

    i = -1
    # i = 0

    # indices = [i]
    indices = []
    while i < len(EEG) - 1:
        indices.extend([(i + 19), (i + 19 * 2), (i + 19 * 3), (i + 19 * 4), (i + 19 * 4 + 20)])
        # indices.extend([(i+20), (i+20+19), (i+20+19*2), (i+20+19*3), (i+20+19*4)])
        i += (19 * 4 + 20)

    indices = np.asarray(indices)

    indices = indices[np.where(indices <= len(EEG) - 1)]

    return EEG[indices]


def create_egf(filename, data, Fs, DC_Blocker=True):
    if os.path.exists(filename):
        return

    Fs_EGF = int(4.8e3)  # sampling rate of .EGF files

    data = data[:, 0::int(Fs / Fs_EGF)]

    # notch filter the data
    # data = sp.Filtering().notch_filt(data, Fs_EGF, freq=60, band=10, order=3)

    # append zeros to make the duration a round number
    duration_round = np.ceil(data.shape[1] / Fs_EGF)  # the duration should be rounded up to the nearest integer
    missing_samples = int(duration_round * Fs_EGF - data.shape[1])
    if missing_samples != 0:
        missing_samples = np.tile(np.array([0]), (1, missing_samples))
        data = np.hstack((data, missing_samples))

    # ensure the full last second of data is equal to zero
    data[0, -int(Fs_EGF):] = 0

    # data = np.rint(data)  # convert the data to integers
    data = data.astype(np.int32)  # convert the data to integers

    # ensuring the appropriate range of the values
    data[np.where(data > 32767)] = 32767
    data[np.where(data < -32768)] = -32768

    # data is already in int16 which is what the final unit should be in

    ##################################################################################################
    # ---------------------------Writing the EGF Data-------------------------------------------
    ##################################################################################################

    write_eeg(filename, data, Fs_EGF)


def write_eeg(filepath, data, Fs):

    data = data.flatten()

    session_path, session_filename = os.path.split(filepath)

    tint_basename = os.path.splitext(session_filename)[0]

    set_filename = os.path.join(session_path, '%s.set' % tint_basename)
    header = get_set_header(set_filename)

    num_samples = int(len(data))

    if '.egf' in session_filename:

        # EEG_Fs = 4800
        egf = True

    else:

        # EEG_Fs = 250
        egf = False

    # if the duration before the set file was overwritten wasn't a round number, it rounded up and thus we need
    # to append values to the EEG (we will add 0's to the end)
    duration = int(get_setfile_parameter('duration', set_filename))  # get the duration from the set file

    EEG_expected_num = int(Fs * duration)

    if num_samples < EEG_expected_num:
        missing_samples = EEG_expected_num - num_samples
        data = np.hstack((data, np.zeros((1, missing_samples)).flatten()))
        num_samples = EEG_expected_num

    with open(filepath, 'w') as f:

        num_chans = 'num_chans 1'

        if egf:
            sample_rate = '\nsample_rate %d Hz' % (int(Fs))
            data = struct.pack('<%dh' % (num_samples), *[np.int(data_value) for data_value in data.tolist()])
            b_p_sample = '\nbytes_per_sample 2'
            num_EEG_samples = '\nnum_EGF_samples %d' % (num_samples)

        else:
            sample_rate = '\nsample_rate %d.0 hz' % (int(Fs))
            data = struct.pack('>%db' % (num_samples), *[np.int(data_value) for data_value in data.tolist()])
            b_p_sample = '\nbytes_per_sample 1'
            num_EEG_samples = '\nnum_EEG_samples %d' % (num_samples)

        eeg_p_position = '\nEEG_samples_per_position %d' % (5)

        start = '\ndata_start'

        if egf:
            write_order = [header, num_chans, sample_rate,
                           b_p_sample, num_EEG_samples, start]
        else:
            write_order = [header, num_chans, sample_rate, eeg_p_position,
                           b_p_sample, num_EEG_samples, start]

        f.writelines(write_order)

    with open(filepath, 'rb+') as f:
        f.seek(0, 2)
        f.writelines([data, bytes('\r\ndata_end\r\n', 'utf-8')])

