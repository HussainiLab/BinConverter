import datetime, os, time
from PyQt5 import QtWidgets
import numpy as np
from BatchTINTV3.core.klusta_functions import klusta
from .Tint_Matlab import int16toint8, get_setfile_parameter
from .readBin import get_bin_data, get_raw_pos, get_channel_from_tetrode, get_active_tetrode, get_active_eeg
from .CreatePos import create_pos
from .ConvertTetrode import write_tetrode
from .CreateEEG import create_eeg, create_egf


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
            # threshold = int(16640)
            # tetrode_thresholds = [threshold, threshold, threshold, threshold]

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

    if main_window.choice == QtWidgets.QMessageBox.Abort:
        main_window.LogAppend.myGUI_signal.emit(
            '[%s %s]: Batch-Tint of the following directory has been aborted: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))
        main_window.StopConversion()
        return

    check_valid_settings_directory(main_window)

    if main_window.choice == QtWidgets.QMessageBox.Abort:
        main_window.LogAppend.myGUI_signal.emit(
            '[%s %s]: Batch-Tint of the following directory has been aborted: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))
        main_window.StopConversion()
        return

    main_window.settings_fname = main_window.parameters['tintsettings']

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
        main_window.choice = None
        main_window.directory_chosen = False
        main_window.LogError.myGUI_signal.emit('NoTintSettings')

        while main_window.choice is None:
            time.sleep(0.5)

        if main_window.choice == QtWidgets.QMessageBox.Ok:

            while not main_window.directory_chosen:
                time.sleep(0.1)

            main_window.parameters['tintsettings'] = os.path.join(
                main_window.batchtintsettings_edit.text(), 'settings.json')


def check_valid_settings_directory(main_window):

    while True:
        if 'Choose the Batch-Tint' in main_window.parameters['tintsettings']:
            check_chosen_settings_directory()

            if main_window.choice == QtWidgets.QMessageBox.Abort:
                return

            if 'Choose the Batch-Tint' in main_window.parameters['tintsettings']:
                main_window.choice = None
                main_window.directory_chosen = False
                main_window.LogError.myGUI_signal.emit('DefaultTintSettings')

                while main_window.choice is None:
                    time.sleep(0.5)

                if main_window.choice == QtWidgets.QMessageBox.Yes:
                    main_window.SETTINGS_DIR = main_window.BATCH_TINT_DIR
                    main_window.parameters['tintsettings'] = os.path.join(main_window.SETTINGS_DIR, 'settings.json')
                else:
                    return
        else:
            if not os.path.exists(os.path.join(main_window.parameters['tintsettings'])):
                main_window.choice = None
                main_window.directory_chosen = False
                main_window.LogError.myGUI_signal.emit('InvalidTintSettings')

                while main_window.choice is None:
                    time.sleep(0.5)

                if main_window.choice == QtWidgets.QMessageBox.Yes:

                    while not main_window.directory_chosen:
                        time.sleep(0.5)

                    main_window.parameters['tintsettings'] = os.path.join(
                        main_window.batchtintsettings_edit.text(), 'settings.json')

                elif main_window.choice == QtWidgets.QMessageBox.Default:
                    main_window.SETTINGS_DIR = main_window.BATCH_TINT_DIR
                    main_window.parameters['tintsettings'] = os.path.join(main_window.SETTINGS_DIR, 'settings.json')
                else:
                    return
            else:
                return


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

