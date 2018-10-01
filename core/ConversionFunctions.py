import struct, scipy, datetime, os, time
from PyQt4 import QtGui
import numpy as np
from scipy import signal
# from scipy.interpolate import InterpolatedUnivariateSpline
# from scipy import interpolate
# import core.filtering as filt
from BatchTint.KlustaFunctions import klusta
from core.Tint_Matlab import int16toint8
import matplotlib.pyplot as plt
from core.readBin import get_bin_data, get_raw_pos, get_channel_from_tetrode, get_active_tetrode, get_active_eeg


def find_sub(string, sub):
    '''finds all instances of a substring within a string and outputs a list of indices'''
    result = []
    k = 0
    while k < len(string):
        k = string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1  # change to k += len(sub) to not search overlapping results
    return result


def getScalar(channels, set_filename, mode='8bit'):
    """input channels are from the range of 1->64, but the set file records gains in the channels
    numbered from 0->63

    This method which will return a scalar which when multiplied by the data will produce the data
    in uV and not bits."""
    adc_fullscale = float(get_setfile_parameter('ADC_fullscale_mv', set_filename))

    channel_gains = np.zeros(len(channels))

    for i, channel in enumerate(channels):
        channel_gains[i] = float(get_setfile_parameter('gain_ch_%d' % int(channel-1), set_filename))

    if mode == '8bit':
        scalars = np.divide(adc_fullscale * 1000, channel_gains * 128)
    elif mode == '16bit':
        scalars = np.divide(adc_fullscale * 1000, channel_gains * 32768)

    return scalars


def uV2bits(data, gain, ADC_Fullscale=1500, mode='8bit'):
    """converts the data to bits:
    mode='default', for EEG and tetrode data
    mode='egf', for EGF data
    """

    if mode == '8bit':
        scalar = (ADC_Fullscale * 1000) / (gain * 128)
    else:
        scalar = (ADC_Fullscale * 1000) / (gain * 32768)

    data = np.divide(data, scalar)

    if mode == '8bit':
        data[np.where(data > 127)] = 127
        data[np.where(data < -128)] = -128
    else:
        data[np.where(data > 32767)] = 32767
        data[np.where(data < -32768)] = -32768

    data = np.rint(data)

    return data


def find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values


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


def get_setfile_parameter(parameter, set_filename):
    if not os.path.exists(set_filename):
        return

    with open(set_filename, 'r+') as f:
        for line in f:
            if parameter in line:
                if line.split(' ')[0] == parameter:
                    # prevents part of the parameter being in another parameter name
                    new_line = line.strip().split(' ')
                    if len(new_line) == 2:
                        return new_line[-1]
                    else:
                        return ' '.join(new_line[1:])


def create_pos(pos_filename, pos_data):
    n = int(pos_data.shape[0])

    session_path, session_filename = os.path.split(pos_filename)
    tint_basename = os.path.splitext(session_filename)[0]
    set_filename = os.path.join(session_path, '%s.set' % tint_basename)

    if os.path.exists(pos_filename):
        return

    header = get_set_header(set_filename)

    with open(pos_filename, 'wb+') as f:  # opening the .pos file
        write_list = []

        [write_list.append(bytes('%s\r\n' % value, 'utf-8')) for value in header.split('\n') if value != '']

        header_vals = ['num_colours %d' % 4]
        if 'abid' in header.lower():
            header_vals.extend(
                ['\r\nmin_x %d' % 0,
                 '\r\nmax_x %d' % 768,
                 '\r\nmin_y %d' % 0,
                 '\r\nmax_y %d' % 574])
        elif 'gus' in header.lower():
            header_vals.extend(
                ['\r\nmin_x %d' % 0,
                 '\r\nmax_x %d' % 640,
                 '\r\nmin_y %d' % 0,
                 '\r\nmax_y %d' % 480]

            )
        header_vals.extend(
            ['\r\nwindow_min_x %d' % int(get_setfile_parameter('xmin', set_filename)),
            '\r\nwindow_max_x %d' % int(get_setfile_parameter('xmax', set_filename)),
            '\r\nwindow_min_y %d' % int(get_setfile_parameter('ymin', set_filename)),
            '\r\nwindow_max_y %d' % int(get_setfile_parameter('ymax', set_filename)),
            '\r\ntimebase %d hz' % 50,
            '\r\nbytes_per_timestamp %d' % 4,
            '\r\nsample_rate %.1f hz' % 50.0,
            '\r\nEEG_samples_per_position %d' % 5,
            '\r\nbearing_colour_1 %d' % 0,
            '\r\nbearing_colour_2 %d' % 0,
            '\r\nbearing_colour_3 %d' % 0,
            '\r\nbearing_colour_4 %d' % 0,
            '\r\npos_format t,x1,y1,x2,y2,numpix1,numpix2',
            '\r\nbytes_per_coord %d' % 2,
            '\r\npixels_per_metre %f' % float(
               get_setfile_parameter('tracker_pixels_per_metre', set_filename)),
            '\r\nnum_pos_samples %d' % n,
            '\r\ndata_start'])

        for value in header_vals:
            write_list.append(bytes(value, 'utf-8'))

        onespot = 1  # this is just in case we decide to add other modes.

        # write_list = [bytes(headers, 'utf-8')]

        # write_list.append()

        if onespot:
            position_format_string = 'i8h'
            position_format_string = '>%s' % (n * position_format_string)
            write_list.append(struct.pack(position_format_string, *pos_data.astype(int).flatten()))

        write_list.append(bytes('\r\ndata_end\r\n', 'utf-8'))
        f.writelines(write_list)


def write_tetrode(filepath, data, Fs):

    session_path, session_filename = os.path.split(filepath)
    tint_basename = os.path.splitext(session_filename)[0]
    set_filename = os.path.join(session_path, '%s.set' % tint_basename)

    n = len(data)

    write_order = []

    header = get_set_header(set_filename)

    [write_order.append(bytes('%s\n' % value, 'utf-8')) for value in header.split('\n') if value != '']

    with open(filepath, 'w') as f:
        num_chans = 'num_chans 4'
        timebase_head = '\ntimebase %d hz' % (96000)
        bp_timestamp = '\nbytes_per_timestamp %d' % (4)
        # samps_per_spike = '\nsamples_per_spike %d' % (int(Fs*1e-3))
        samps_per_spike = '\nsamples_per_spike %d' % (50)
        sample_rate = '\nsample_rate %d hz' % (Fs)
        b_p_sample = '\nbytes_per_sample %d' % (1)
        # b_p_sample = '\nbytes_per_sample %d' % (4)
        spike_form = '\nspike_format t,ch1,t,ch2,t,ch3,t,ch4'
        num_spikes = '\nnum_spikes %d' % (n)
        start = '\ndata_start'

        write_order = [header, num_chans, timebase_head,
                       bp_timestamp,
                       samps_per_spike, sample_rate, b_p_sample, spike_form, num_spikes, start]

        f.writelines(write_order)

    # rearranging the data to have a flat array of t1, waveform1, t2, waveform2, t3, waveform3, etc....
    spike_times = np.asarray(sorted(data.keys()))
    #print(spike_times.shape)
    #print(spike_times[:50])
    # the spike times are repeated for each channel so lets tile this
    spike_times = np.tile(spike_times, (4,1))
    spike_times = spike_times.flatten(order='F')

    #print(spike_times[:50])

    spike_values = np.asarray([value for (key, value) in sorted(data.items())])

    # this will create a (n_samples, n_channels, n_samples_per_spike) => (n, 4, 50) sized matrix, we will create a
    # matrix of all the samples and channels going from ch1 -> ch4 for each spike time
    # time1 ch1_data
    # time1 ch2_data
    # time1 ch3_data
    # time1 ch4_data
    # time2 ch1_data
    # time2 ch2_data
    # .
    # .
    # .

    spike_values = spike_values.reshape((n * 4, 50))  # create the 4nx50 channel data matrix

    # make the first column the time values

    spike_array = np.hstack((spike_times.reshape(len(spike_times), 1), spike_values))

    data = None
    spike_times = None
    spike_values = None

    spike_n = spike_array.shape[0]

    '''
    with open(filepath, 'rb+') as f:
        for spike_t, spike_data in sorted(data.items()):
            write_list = []
            for i in range(spike_data.shape[0]):
                write_list.append(struct.pack('>i', int(spike_t)))
                write_list.append(struct.pack('<%db' % (50),
                                              *[int(sample) for sample in spike_data[i, :].tolist()]))

            f.seek(0, 2)
            f.writelines(write_list)
    '''
    t_packed = struct.pack('>%di' % spike_n, *spike_array[:, 0].astype(int))
    spike_array = spike_array[:, 1:]  # removing time data from this matrix to save memory

    spike_data_pack = struct.pack('<%db' % (spike_n*50), *spike_array.astype(int).flatten())

    spike_array = None

    # now we need to combine the lists by alternating

    comb_list = [None] * (2*spike_n)
    comb_list[::2] = [t_packed[i:i + 4] for i in range(0, len(t_packed), 4)]  # breaks up t_packed into a list,
    # each timestamp is one 4 byte integer
    comb_list[1::2] = [spike_data_pack[i:i + 50] for i in range(0, len(spike_data_pack), 50)]  # breaks up spike_data_
    # pack and puts it into a list, each spike is 50 one byte integers

    t_packed = None
    spike_data_pack = None

    write_order = []
    with open(filepath, 'rb+') as f:

        write_order.extend(comb_list)
        write_order.append(bytes('\r\ndata_end\r\n', 'utf-8'))

        f.seek(0, 2)
        f.writelines(write_order)

    '''
    with open(filepath, 'rb+') as f:
        f.seek(0, 2)
        f.write(bytes('\r\ndata_end\r\n', 'utf-8'))
    '''


def convert_basename(self, set_filename):
    """This function will convert the .bin file to the TINT format"""

    directory = os.path.dirname(set_filename)

    tint_basename = os.path.basename(os.path.splitext(set_filename)[0])

    bin_filename = os.path.join(directory, '%s.bin' % tint_basename)

    # the first created file of the session will be the basename for tint

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Converting the the following session: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], set_filename))

    # -----------------------------------Overwrite the Set File ---------------------------------------

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Overwriting the following set file\'s duration: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], set_filename))

    overwrite_setfile(set_filename)

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Finished converting the following basename: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], tint_basename))

    # --------------------------------------Read Position Data----------------------------------------

    position_filename = os.path.join(directory, tint_basename + '.pos')

    if not os.path.exists(position_filename):
        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Analyzing the following position file: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], position_filename))

        # Fs_pos = 50  # Hz, position sampling frequency

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Reading in the position data!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))

        raw_position = get_raw_pos(bin_filename)  # vid time, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Creating the .pos file!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))

        create_pos(position_filename, raw_position)

    else:
        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: The following position file already exists: %s!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], position_filename))

    # --------------------------------------Read Tetrode Data----------------------------------------

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Converting the session one tetrode at a time!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8]))

    Fs = get_Fs(set_filename)  # read the sampling frequency from the .set file, most like 48k

    active_tetrodes = get_active_tetrode(set_filename)

    # converts the data one tetrode at a time so we can eliminate memory errors

    pre_spike_samples = int(get_setfile_parameter('pretrigSamps', set_filename))
    post_spike_samples = int(get_setfile_parameter('spikeLockout', set_filename))
    rejstart = int(get_setfile_parameter('rejstart', set_filename))
    rejthreshtail = int(get_setfile_parameter('rejthreshtail', set_filename))
    rejthreshupper = int(get_setfile_parameter('rejthreshupper', set_filename))
    rejthreshlower = int(get_setfile_parameter('rejthreshlower', set_filename))

    for tetrode in active_tetrodes:

        tetrode = int(tetrode)
        # check if this tetrode exists already

        tetrode_filename = os.path.join(directory, '%s.%d' % (tint_basename, tetrode))
        if os.path.exists(tetrode_filename):
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: The following tetrode has already been converted, skipping: %d!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], tetrode))
            continue

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Currently Converting Data Related to Tetrode: %d!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], tetrode))

        tetrode_channels = get_channel_from_tetrode(tetrode)  # get the channels (from range of 1->64)

        data = get_bin_data(bin_filename, tetrode=tetrode)  # 16bit, get data associated with the tetrode

        # converting data to uV

        n_samples = data.shape[1]
        # create a time array that represents the 48kHz sampled data times
        t = np.arange(0, n_samples) / Fs  # creates a time array of the signal starting from 0 (in seconds)

        if not os.path.exists(tetrode_filename):
            '''
            # --------------------------------Notch Filter if necessary --------------------------------------

            # Applying Notch Filter

            # ---------------------------Band Pass Filter the Unit Data --------------------------------------

            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: High Pass Filtering the Spike Data for T%d!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], tetrode))

            #data = filt.iirfilt(bandtype='band', data=data, Fs=Fs, Wp=300, Ws=7000, order=3, automatic=0,
            #                              Rp=0.1, As=60, filttype='butter', showresponse=0)

            # data = filt.custom_cheby1(data, Fs, 3, 0.1, 300, Ws=7e3, filtresponse='bandpass')
            '''

            # ---------------------------Find the spikes in the unit data --------------------------------------

            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Finding the spikes for the tetrode %d!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], tetrode))

            tetrode_spikes = {}  # creates an empty dictionary to hold the spike times
            # for each tetrode, find the spikes

            k = 0

            # data = int16toint8(data)  # converting the data into int8

            tetrode_thresholds = []
            for channel_index, channel in enumerate(tetrode_channels):
                k += 1
                self.LogAppend.myGUI_signal.emit(
                    '[%s %s]: Finding the spikes within T%dCh%d!' %
                    (str(datetime.datetime.now().date()),
                     str(datetime.datetime.now().time())[:8], tetrode, k))

                '''Auto thresholding technique incorporated by:
                Quian Quiroga in 2014 - Unsupervised Spike Detection and Sorting with Wavelets and
                Superparamagnetic Clustering

                Thr = 4*sigma, sigma = median(abs(x)/0.6745)
                '''
                standard_deviations = float(self.threshold.text())

                sigma_n = np.median(np.divide(np.abs(data[channel_index, :]), 0.6745))
                # threshold = sigma_n / channel_max
                # threshold = standard_deviations * sigma_n
                tetrode_thresholds.append(standard_deviations * sigma_n)

            # threshold = int(17152)
            threshold = int(16640)
            tetrode_thresholds = [threshold, threshold, threshold, threshold]

            valid_spikes = get_spikes(data, tetrode_thresholds)

            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Number of spikes found: %d!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], len(valid_spikes)))

            # threshold is done in 16 bit values, but the rejection is done in 8bit, so we convert here
            # data = int16toint8(data)  # converting the data into int8

            data = int16toint8(data)  # converting the data into int8

            tetrode_spikes = validate_spikes(self, tetrode, valid_spikes, data, t, pre_spike_samples,
                                             post_spike_samples, rejstart, rejthreshtail, rejthreshupper,
                                             rejthreshlower)

            # write the tetrode data to create the .N file
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Writing the following tetrode file: %s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], tetrode_filename))

            write_tetrode(tetrode_filename, tetrode_spikes, Fs)
        else:
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: The following tetrode already exists, skipping: %s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], tetrode_filename))

        data = None
        tetrode_spikes = None
        valid_spikes = None

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Finished Converting the Following Tetrode: %d!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], tetrode))

    ##################################################################################################
    # ----------------------------Load LFP Data then Create EEG/EGF and then Filter --------------------
    ##################################################################################################

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Converting the EEG/EGF Files One at a Time!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8]))

    active_eeg_channels = get_active_eeg(set_filename)
    active_eeg_channel_numbers = np.asarray(
        list(active_eeg_channels.values())) + 1  # makes the eeg channels from 1-> 64

    # check if any of the tetrode data is an EEG channel

    for channel in active_eeg_channel_numbers:

        for eeg_number, eeg_chan_value in sorted(active_eeg_channels.items()):
            if eeg_chan_value == channel - 1:
                break

        if eeg_number == 1:
            eeg_str = ''
        else:
            eeg_str = str(eeg_number)

        if eeg_number == 1:
            eeg_filename = os.path.join(directory, tint_basename + '.eeg')
            egf_filename = os.path.join(directory, tint_basename + '.egf')
        else:
            eeg_filename = os.path.join(directory, tint_basename + '.eeg%d' % (eeg_number))
            egf_filename = os.path.join(directory, tint_basename + '.egf%d' % (eeg_number))

        if os.path.exists(eeg_filename):

            EEG = np.array([])
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: The following EEG file has already been created, skipping: %s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], eeg_filename))
        else:
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: Creating the following EEG file: .eeg%s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], eeg_str))

            # load data
            EEG = get_bin_data(bin_filename, channels=[channel])

            '''
            # append zeros to make the duration a round number
            duration = np.ceil(EEG.shape[1] / Fs)  # the duration should be rounded up to the nearest integer
            missing_samples = int(duration * Fs - EEG.shape[1])
            if missing_samples != 0:
                missing_samples = np.tile(np.array([0]), (1, missing_samples))
                EEG = np.hstack((EEG, missing_samples))
            '''
            create_eeg(eeg_filename, EEG, Fs, DC_Blocker=self.dc_blocker.isChecked())
            # EEG = None

        if os.path.exists(egf_filename):
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: The following EGF file has already been created, skipping: %s!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], eeg_filename))
        else:

            if len(EEG) != 0:
                # then the EEG has already been loaded in
                create_egf(egf_filename, EEG, Fs, DC_Blocker=self.dc_blocker.isChecked())
                EEG = None
            else:
                self.LogAppend.myGUI_signal.emit(
                    '[%s %s]: Creating the following EGF file: .egf%s!' %
                    (str(datetime.datetime.now().date()),
                     str(datetime.datetime.now().time())[:8], eeg_str))

                # then the EEG hasn't been read in (EEG was already created), read the data
                EGF = get_bin_data(bin_filename, channels=[channel])

                '''
                # append zeros to make the duration a round number
                duration = np.ceil(EGF.shape[1] / Fs)
                missing_samples = int(duration * Fs - EGF.shape[1])
                if missing_samples != 0:
                    missing_samples = np.tile(np.array([0]), (1, missing_samples))
                    EGF = np.hstack((EGF, missing_samples))
                '''

                create_egf(egf_filename, EGF, Fs, DC_Blocker=self.dc_blocker.isChecked())
                EGF = None

            # session_parameters['SamplesPerSpike'] = 50
            # session_parameters['timebase'] = 96000
        EEG = None

    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: Finished converting the following: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], tint_basename))


def matching_ind(haystack, needle):
    idx = np.searchsorted(haystack, needle)
    mask = idx < haystack.size
    mask[mask] = haystack[idx[mask]] == needle[mask]
    idx = idx[mask]
    return idx


def batchtint(main_window, directory):

    check_chosen_settings_directory(main_window)

    if main_window.choice == QtGui.QMessageBox.Abort:
        main_window.LogAppend.myGUI_signal.emit(
            '[%s %s]: Batch-Tint of the following directory has been aborted: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))
        main_window.StopConversion()
        return

    check_valid_settings_directory(main_window)

    if main_window.choice == QtGui.QMessageBox.Abort:
        main_window.LogAppend.myGUI_signal.emit(
            '[%s %s]: Batch-Tint of the following directory has been aborted: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))
        main_window.StopConversion()
        return

    main_window.settings_fname = main_window.parameters['tintsettings']
    #klusta_ready = check_klusta_ready(main_window, directory, main_window.parameters['tintsettings'])
    klusta_ready = True

    if klusta_ready:
        main_window.LogAppend.myGUI_signal.emit(
            '[%s %s]: Analyzing the following directory: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))
        # ------------- find all subdirectories within directory -------------------------------------

        sub_directories = [d for d in os.listdir(directory) if
                           os.path.isdir(os.path.join(directory, d)) and d not in ['Processed', 'Converted']]

        if not sub_directories:

            # message that shows how many files were found
            main_window.LogAppend.myGUI_signal.emit(
                '[%s %s]: There are no files to analyze in this directory!' % (str(datetime.datetime.now().date()),
                                                                               str(datetime.datetime.now().time())[
                                                                               :8]))
            return

        else:
            # message that shows how many files were found
            main_window.LogAppend.myGUI_signal.emit(
                '[%s %s]: Found %d files in the directory!' % (str(datetime.datetime.now().date()),
                                                               str(datetime.datetime.now().time())[
                                                               :8], len(sub_directories)))

        # ----------- cycle through each file and find the tetrode files ------------------------------------------

        for sub_directory in sub_directories:  # finding all the folders within the directory
            try:
                dir_new = os.path.join(directory, sub_directory)  # sets a new filepath for the directory
                f_list = os.listdir(dir_new)  # finds the files within that directory
                set_file = [file for file in f_list if '.set' in file]  # finds the set file

                if not set_file:  # if there is no set file it will return as an empty list
                    # message saying no .set file
                    main_window.LogAppend.myGUI_signal.emit(
                        '[%s %s]: The following folder contains no \'.set\' file: %s' % (
                            str(datetime.datetime.now().date()),
                            str(datetime.datetime.now().time())[
                            :8], str(sub_directory)))
                    continue
                # runs the function that will perform the klusta'ing
                klusta(main_window, sub_directory, directory)

            except NotADirectoryError:
                # if the file is not a directory it prints this message
                main_window.LogAppend.myGUI_signal.emit(
                    '[%s %s]: %s is not a directory!' % (
                        str(datetime.datetime.now().date()),
                        str(datetime.datetime.now().time())[
                        :8], str(sub_directory)))
                continue


def check_chosen_settings_directory(main_window):
    # checks if the settings are appropriate to run analysis
    if 'Choose the Batch-Tint' in main_window.parameters['tintsettings']:
        main_window.choice == ''
        main_window.directory_chosen = False
        main_window.LogError.myGUI_signal.emit('NoTintSettings')

        while main_window.choice == '':
            time.sleep(0.5)

        if main_window.choice == QtGui.QMessageBox.Ok:

            while not main_window.directory_chosen:
                time.sleep(0.5)

            # main_window.new_file()
            # main_window.parameters['tintsettings'] = main_window.batchtintsettings_edit.text()
            main_window.parameters['tintsettings'] = os.path.join(
                main_window.batchtintsettings_edit.text(), 'settings.json')


def check_valid_settings_directory(main_window):

    while True:
        if 'Choose the Batch-Tint' in main_window.parameters['tintsettings']:
            check_chosen_settings_directory()

            if main_window.choice == QtGui.QMessageBox.Abort:
                return

            if 'Choose the Batch-Tint' in main_window.parameters['tintsettings']:
                main_window.choice == ''
                main_window.directory_chosen = False
                main_window.LogError.myGUI_signal.emit('DefaultTintSettings')

                while main_window.choice == '':
                    time.sleep(0.5)

                if main_window.choice == QtGui.QMessageBox.Yes:
                    main_window.SETTINGS_DIR = main_window.BATCH_TINT_DIR
                    main_window.parameters['tintsettings'] = os.path.join(main_window.SETTINGS_DIR, 'settings.json')
                else:
                    return
        else:
            if not os.path.exists(os.path.join(main_window.parameters['tintsettings'])):
                main_window.choice == ''
                main_window.directory_chosen = False
                main_window.LogError.myGUI_signal.emit('InvalidTintSettings')

                while main_window.choice == '':
                    time.sleep(0.5)

                if main_window.choice == QtGui.QMessageBox.Yes:

                    while not main_window.directory_chosen:
                        time.sleep(0.5)

                    main_window.parameters['tintsettings'] = os.path.join(
                        main_window.batchtintsettings_edit.text(), 'settings.json')

                elif main_window.choice == QtGui.QMessageBox.Default:
                    main_window.SETTINGS_DIR = main_window.BATCH_TINT_DIR
                    main_window.parameters['tintsettings'] = os.path.join(main_window.SETTINGS_DIR, 'settings.json')
                else:
                    return
            else:
                return


def is_converted(set_filename):
    """This method will check if the file has been converted already"""

    # find active tetrodes

    active_tetrodes = get_active_tetrode(set_filename)

    # check if each of these active .N files have been created yet

    set_basename = os.path.basename(os.path.splitext(set_filename)[0])

    set_directory = os.path.dirname(set_filename)

    all_tetrodes_written = True
    all_eeg_written = True
    all_egf_written = True
    pos_written = True

    if not os.path.exists(os.path.join(set_directory, '%s.pos' % (set_basename))):
        pos_written = False

    for tetrode in active_tetrodes:

        if not os.path.exists(os.path.join(set_directory, '%s.%d' % (set_basename, int(tetrode)))):
            all_tetrodes_written = False
            break

    active_eeg = get_active_eeg(set_filename)

    for eeg in active_eeg.keys():
        if eeg == 1:
            if not os.path.exists(os.path.join(set_directory, '%s.eeg' % (set_basename))):
                all_eeg_written = False
                break

        elif not os.path.exists(os.path.join(set_directory, '%s.eeg%d' % (set_basename, int(eeg)))):
            all_eeg_written = False
            break

    if is_egf_active(set_filename):
        for egf in active_eeg.keys():
            if egf == 1:
                if not os.path.exists(os.path.join(set_directory, '%s.egf' % (set_basename))):
                    all_egf_written = False
                    break

            elif not os.path.exists(os.path.join(set_directory, '%s.egf%d' % (set_basename, int(egf)))):
                all_egf_written = False
                break

    return all_tetrodes_written and all_eeg_written and all_egf_written and pos_written


def is_egf_active(set_filename):
    active_egf_str = 'saveEGF'

    with open(set_filename) as f:
        for line in f:

            if active_egf_str in line:
                _, egf_status = line.split(' ')

                if int(egf_status) == 1:
                    return True

        return False


def fir_hann(data, Fs, cutoff, n_taps=101, showresponse=0):
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2

    b = scipy.signal.firwin(n_taps, cutoff / nyq_rate, window='hann')

    a = 1.0
    # Use lfilter to filter x with the FIR filter.
    data = scipy.signal.lfilter(b, a, data)
    # data = scipy.signal.filtfilt(b, a, data)

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

    return data, n_taps


def has_files(set_filename):
    """This method will check if all the necessary files exist"""

    # the only thing it needs is a .bin file

    tint_basename = os.path.basename(os.path.splitext(set_filename)[0])

    directory = os.path.dirname(set_filename)

    if os.path.exists(os.path.join(directory, '%s.bin' % tint_basename)):
        return True

    return False


'''
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


def get_channel_bytes(channel_number, samples):
    """This will get the indices of the data if it is just the lfp data (not the bytes header bytes or trailing bytes)"""
    remap_channel = get_remap_chan(channel_number)

    indices_scalar = np.multiply(np.arange(samples), 64)
    sample_indices = indices_scalar + np.multiply(np.ones(samples), remap_channel)

    # return np.array([remap_channel, 64 + remap_channel, 64*2 + remap_channel])
    return (indices_scalar + np.multiply(np.ones(samples), remap_channel)).astype(int)
'''


def get_Fs(set_filename):
    fs_str = 'rawRate'

    with open(set_filename) as f:
        for line in f:

            if fs_str in line:
                _, Fs = line.split(' ')

                try:
                    return int(Fs)
                except:
                    return float(Fs)
    return


def get_set_header(set_filename):
    with open(set_filename, 'r+') as f:
        header = ''
        for line in f:
            header += line
            if 'sw_version' in line:
                break
    return header


def find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values


def remove_appended_zeros(data):
    """This method will remove the zeros that were added to the end of the end of the data"""

    zero_ind = find_consec(np.where((data.flatten() == 0) | (data.flatten() == 0.0))[0])[-1]

    if len(data.flatten()) - 1 not in zero_ind:
        return []

    return zero_ind


def get_spikes(data, threshold):
    all_spikes = np.array([])

    for i, channel_data in enumerate(data):
        spike_indices = np.where(channel_data >= threshold[i])[0]

        if len(spike_indices) == 0:
            continue

        spike_indices = find_consec(spike_indices)

        spike_indices = np.asarray([value[0] for value in spike_indices])

        if len(all_spikes) == 0:
            # this is the first iteration of the tetrode, no need to sort
            unadded_spikes = spike_indices
        else:
            idx = matching_ind(all_spikes, spike_indices)
            if len(idx) == 0:
                unadded_spikes = spike_indices
            else:
                unadded_spikes = np.setdiff1d(spike_indices, all_spikes[idx])

        if len(all_spikes) != 0:
            all_spikes = np.sort(np.concatenate((all_spikes, unadded_spikes)))
            unadded_spikes = None
        else:
            all_spikes = np.array(unadded_spikes)

    return all_spikes


def validate_spikes(self, tetrode, spikes, data, t, pre_spike_samples=10, post_spike_samples=40, rejstart=30,
                    rejthreshtail=43, rejthreshupper=100, rejthreshlower=-100):
    latest_spike = None

    spike_count = 0
    percentage_values = [int(value) for value in np.rint(np.linspace(0, len(spikes), num=21)).tolist()]

    n_max = data.shape[1]

    tetrode_spikes = {}

    for spike in sorted(spikes):
        # iterate through each spike and validate to ensure no spikes occur at the same time or within the
        # refractory period

        spike_count += 1

        if spike_count in percentage_values:
            self.LogAppend.myGUI_signal.emit(
                '[%s %s]: %d%% completed adding spikes from T%d!' %
                (str(datetime.datetime.now().date()),
                 str(datetime.datetime.now().time())[:8], int(np.rint(100 * spike_count / len(spikes))),
                 tetrode))

        if spike - pre_spike_samples + 1 < 0:
            continue

        elif spike + post_spike_samples >= n_max:
            continue

        if latest_spike is not None:
            if spike != latest_spike:
                if spike in spike_refractory:
                    # ensures no overlapping spikes
                    continue
        else:
            pass

        latest_spike = spike
        spike_refractory = list(np.arange(spike + 1, spike + post_spike_samples + 1))

        # spike_time = t[int(spike)]
        spike_time = t[int(spike)]

        # waveform_indices = np.where((t>=spike_time-250/1e6) & (t<=spike_time+850/1e6))[0]  # too slow
        waveform_indices = np.arange(spike - pre_spike_samples + 1, spike + post_spike_samples + 1).astype(int)

        # spike_t = t[waveform_indices] - spike_time  # making the times from -200 us to 800 us

        # spike_waveform = np.zeros((len(tetrode_channels), 50))

        spike_waveform = data[:, waveform_indices]

        spike_time = spike_time * 96000  # multiply it by the timebase to get the frame count

        spike_waveform = np.rint(spike_waveform)

        # artifact rejection

        if sum(spike_waveform[:, rejstart:].flatten() > rejthreshtail) > 0:
            # this is 33% above baseline (0)
            continue

        # check if the first sample is well above or well below baseline
        elif sum(spike_waveform[:, 0].flatten() > rejthreshupper) > 0:
            # the first sample is >100
            continue

        elif sum(spike_waveform[:, 0].flatten() < rejthreshlower) > 0:
            # or < -100
            continue

        tetrode_spikes[spike_time] = spike_waveform

        # latest_spike = spike
        # spike_refractory = list(np.arange(spike + 1, spike + post_spike_samples + 1))

    return tetrode_spikes


def create_eeg(filename, data, Fs, DC_Blocker=True):
    # data is given in int16

    if os.path.exists(filename):
        return

    Fs_EGF = int(4.8e3)  # sampling rate of .EGF files
    Fs_EEG = int(250)
    """
    if DC_Blocker:
        data = sp.Filtering().dcblock(data, 0.1, Fs)

    # LP at 500
    data = sp.Filtering().iirfilt(bandtype='low', data=data, Fs=Fs, Wp=500, order=6,
                                  automatic=0, Rp=0.1, filttype='cheby1', showresponse=0)

    # notch filter the data
    # data = sp.Filtering().notch_filt(data, Fs, freq=60, band=10, order=3)
    data = filt.notch_filt(data, Fs, freq=60, band=10, order=2)

    # downsample to 4.8khz signal for EGF signal (EEG is derived from EGF data)
    """

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

    """
    if DC_Blocker:
        data = sp.Filtering().dcblock(data, 0.1, Fs)

    # LP at 500
    data = sp.Filtering().iirfilt(bandtype='low', data=data, Fs=Fs, Wp=500, order=6,
                                  automatic=0, Rp=0.1, filttype='cheby1', showresponse=0)

    # notch filter the data
    # data = sp.Filtering().notch_filt(data, Fs, freq=60, band=10, order=3)
    data = filt.notch_filt(data, Fs, freq=60, band=10, order=2)

    # downsample to 4.8khz signal for EGF signal (EEG is derived from EGF data)
    """

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


def overwrite_setfile(set_filename):
    with open(set_filename, 'r+') as f:

        header = ''
        footer = ''

        header_values = True

        for line in f:

            if header_values:
                if 'duration' in line:
                    line = line.strip()
                    duration = int(np.ceil(float(line.split(' ')[-1])))
                    header += 'duration %d\n' % duration
                    continue
                header += line

            else:
                footer += line

    with open(set_filename, 'w') as f:

        f.writelines([header, footer])

