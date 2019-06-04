from __future__ import division, print_function
import numpy as np
import struct, os
import numpy.matlib
from scipy.io import savemat
import mmap
import contextlib


def int16toint8(value):
    """Converts int16 data to int8"""
    value = np.divide(value, 256).astype(int)
    value[np.where(value > 127)] = 127
    value[np.where(value < -128)] = -128

    return value


def remEEGShift(t, EEG):
    """This function removes the shift that the EEG undergoes"""
    EEG = EEG.flatten()
    Fs_EGF = int(4.8e3)
    # normal delay is 0.5(n_taps-1) / Fs_EGF, n_taps is 101 for the FIR filter
    # there is already a shift of 18 as the EEG starts off using the 18th EGF data
    delay = (0.5 * (101 - 1) - 18) / Fs_EGF

    t = t - delay

    t_bool = np.where(t >= 0)[0]
    t = t[t_bool]
    EEG = EEG[t_bool]

    return t, EEG


def MatlabNumSeq(start, stop, step, exclude=True):
    """In Matlab you can type:

    start:step:stop and easily create a numerical sequence

    if exclude is true it will exclude any values greater than the stop value
    """

    '''np.arange(start, stop, step) works good most of the time

    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if exclude:
        if seq[-1] > stop:
            seq = seq[:-1]

    return seq


class TintException(Exception):
    def __init___(self, message):
        Exception.__init__(self, "%s" % message)
        self.message = message


def get_good_cells(units):
    """
    In Tint we have a method of determining the 'good cells'. A good tetrode must begin
    with cell #1, and the set of consecutive cell numberes are all good. Essentially we leave
    a blank cell separating the good cells from the bad. This function will return the good cells.
    """
    units = np.unique(units)

    if 1 not in units:
        return np.array([])

    good_cells = [1]

    found_separation = False
    for index, unit in enumerate(units):
        if unit > 1:
            if unit == units[index - 1] + 1 and not found_separation:
                good_cells.append(unit)
            else:  #
                # no longer consecutive units, so set value to false to ensure no more appending
                found_separation = True

    return np.asarray(good_cells)


def get_setfile_parameter(parameter, set_filename):
    """
    This function will return the parameter value of a given parameter name for a given set filename.

    Example:
        set_fullpath = 'C:\\example\\tetrode_1.1'
        parameter_name = 'duration
        duration = get_setfile_parameter(parameter_name, set_fullpath)

    Args:
        parameter (str): the name of the set file parameter that you want to obtain.
        set_filename (str): the full path of the .set file that you want to obtain the parameter value from.


    Returns:
        parameter_value (str): the value for the given parameter

    """

    if not os.path.exists(set_filename):
        return

    # adding the encoding because tint data is created via windows and if you want to run this in linux, you need
    # to explicitly say this
    with open(set_filename, 'r+', encoding='cp1252') as f:
        for line in f:
            if parameter in line:
                if line.split(' ')[0] == parameter:
                    # prevents part of the parameter being in another parameter name
                    new_line = line.strip().split(' ')
                    if len(new_line) == 2:
                        return new_line[-1]
                    else:
                        return ' '.join(new_line[1:])


def getpos(pos_fpath, arena, method='', flip_y=True):
    '''getpos function:
    ---------------------------------------------
    variables:
    -pos_fpath: the full path (C:\example\session.pos)

    output:
    t: column numpy array of the time stamps
    x: a column array of the x-values (in pixels)
    y: a column array of the y-values (in pixels)
    '''

    with open(pos_fpath, 'rb+') as f:  # opening the .pos file
        headers = ''  # initializing the header string
        for line in f:  # reads line by line to read the header of the file
            # print(line)
            if 'data_start' in str(line):  # if it reads data_start that means the header has ended
                headers += 'data_start'
                break  # break out of for loop once header has finished
            elif 'duration' in str(line):
                headers += line.decode(encoding='UTF-8')
            elif 'num_pos_samples' in str(line):
                num_pos_samples = int(line.decode(encoding='UTF-8')[len('num_pos_samples '):])
                headers += line.decode(encoding='UTF-8')
            elif 'bytes_per_timestamp' in str(line):
                bytes_per_timestamp = int(line.decode(encoding='UTF-8')[len('bytes_per_timestamp '):])
                headers += line.decode(encoding='UTF-8')
            elif 'bytes_per_coord' in str(line):
                bytes_per_coord = int(line.decode(encoding='UTF-8')[len('bytes_per_coord '):])
                headers += line.decode(encoding='UTF-8')
            elif 'timebase' in str(line):
                timebase = (line.decode(encoding='UTF-8')[len('timebase '):]).split(' ')[0]
                headers += line.decode(encoding='UTF-8')
            elif 'pixels_per_metre' in str(line):
                ppm = float(line.decode(encoding='UTF-8')[len('pixels_per_metre '):])
                headers += line.decode(encoding='UTF-8')
            elif 'min_x' in str(line) and 'window' not in str(line):
                min_x = int(line.decode(encoding='UTF-8')[len('min_x '):])
                headers += line.decode(encoding='UTF-8')
            elif 'max_x' in str(line) and 'window' not in str(line):
                max_x = int(line.decode(encoding='UTF-8')[len('max_x '):])
                headers += line.decode(encoding='UTF-8')
            elif 'min_y' in str(line) and 'window' not in str(line):
                min_y = int(line.decode(encoding='UTF-8')[len('min_y '):])
                headers += line.decode(encoding='UTF-8')
            elif 'max_y' in str(line) and 'window' not in str(line):
                max_y = int(line.decode(encoding='UTF-8')[len('max_y '):])
                headers += line.decode(encoding='UTF-8')
            elif 'pos_format' in str(line):
                headers += line.decode(encoding='UTF-8')
                if 't,x1,y1,x2,y2,numpix1,numpix2' in str(line):
                    two_spot = True
                else:
                    two_spot = False
                    print('The position format is unrecognized!')

            elif 'sample_rate' in str(line):
                sample_rate = float(line.decode(encoding='UTF-8').split(' ')[1])
                headers += line.decode(encoding='UTF-8')

            else:
                headers += line.decode(encoding='UTF-8')

    if two_spot:

        '''Run when two spot mode is on, (one_spot has the same format so it will also run here)'''
        with open(pos_fpath, 'rb+') as f:
            '''get_pos for one_spot'''
            pos_data = f.read()  # all the position data values (including header)
            pos_data = pos_data[len(headers):-12]  # removes the header values

            byte_string = 'i8h'

            pos_data = np.asarray(struct.unpack('>%s' % (num_pos_samples * byte_string), pos_data))
            pos_data = pos_data.astype(float).reshape((num_pos_samples, 9))  # there are 8 words and 1 time sample

        x = pos_data[:, 1]
        y = pos_data[:, 2]
        t = pos_data[:, 0]

        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))
        t = t.reshape((len(t), 1))

        if method == 'raw':
            return x, y, t, sample_rate

        t = np.divide(t, np.float(timebase))  # converting the frame number from Axona to the time value

        # values that are NaN are set to 1023 in Axona's system, replace these values by NaN's

        x[np.where(x == 1023)] = np.nan
        y[np.where(y == 1023)] = np.nan

        didFix, fixedPost = fixTimestamps(t)

        if didFix:
            t = fixedPost

        t = t - t[0]

        x, y = arena_config(x, y, arena, conversion=ppm, center=np.asarray([np.mean([np.amin(x), np.amax(x)]),
                                                                            np.mean([np.amin(y), np.amax(y)])]),
                            flip_y=flip_y)

        # remove any NaNs at the end of the file
        x, y, t = removeNan(x, y, t)

    else:
        print("Haven't made any code for this part yet.")

    return x.reshape((len(x), 1)), y.reshape((len(y), 1)), t.reshape((len(t), 1)), sample_rate


def is_tetrode(file, session):
    """"
    Determines if the file is a tetrode, essentially will look at the extension and see if it ends in an integer
    or not. Also checks that the tetrode belongs to a given session.
    """
    if os.path.splitext(file)[0] == session:
        try:
            tetrode_number = int(os.path.splitext(file)[1][1:])
            return True
        except ValueError:
            return False
    else:
        return False


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


def is_egf_active(set_filename):
    active_egf_str = 'saveEGF'

    with open(set_filename) as f:
        for line in f:

            if active_egf_str in line:
                _, egf_status = line.split(' ')

                if int(egf_status) == 1:
                    return True

        return False


def find_tetrodes(set_fullpath):
    """finds the tetrode files available for a given .set file if there is a  .cut file existing"""

    tetrode_path, session = os.path.split(set_fullpath)
    session, _ = os.path.splitext(session)

    # getting all the files in that directory
    file_list = os.listdir(tetrode_path)

    # acquiring only a list of tetrodes that belong to that set file
    tetrode_list = [os.path.join(tetrode_path, file) for file in file_list
                    if is_tetrode(file, session)]

    # if the .cut file doesn't exist remove list

    tetrode_list = [file for file in tetrode_list if os.path.exists(
        os.path.join(tetrode_path, '%s_%s.cut' % (os.path.splitext(file)[0], os.path.splitext(file)[1][1:])))]

    return tetrode_list


def find_unit(tetrode_list):
    """Inputs:
    tetrode_list: list of tetrodes to find the units that are in the tetrode_path
    example [r'C:Location\of\File\filename.1', r'C:Location\of\File\filename.2' ],
    -------------------------------------------------------------
    Outputs:
    cut_list: an nx1 list for n-tetrodes in the tetrode_list containing a list of unit numbers that each spike belongs to
    """

    input_list = True
    if type(tetrode_list) != list:
        input_list = False
        tetrode_list = [tetrode_list]

    cut_list = []
    unique_cell_list = []
    for tetrode_file in tetrode_list:
        directory = os.path.dirname(tetrode_file)

        try:
            tetrode = int(os.path.splitext(tetrode_file)[1][1:])
        except ValueError:
            raise ValueError("The following file is invalid: %s" % tetrode_file)

        tetrode_base = os.path.splitext(os.path.basename(tetrode_file))[0]

        cut_filename = os.path.join(directory, '%s_%d.cut' % (tetrode_base, tetrode))

        cut_values = read_cut(cut_filename)
        cut_list.append(cut_values)
        unique_cell_list.append(np.unique(cut_values))

    return cut_list


def read_cut(cut_filename):
    """This function will read the given cut file, and output the """
    cut_values = None
    if os.path.exists(cut_filename):
        extract_cut = False
        with open(cut_filename, 'r') as f:
            for line in f:
                if 'Exact_cut' in line:  # finding the beginning of the cut values
                    extract_cut = True
                if extract_cut:  # read all the cut values
                    cut_values = str(f.readlines())
                    for string_val in ['\\n', ',', "'", '[', ']']:  # removing non base10 integer values
                        cut_values = cut_values.replace(string_val, '')
                    cut_values = [int(val) for val in cut_values.split()]
        cut_values = np.asarray(cut_values)
    return cut_values


def arena_config(posx, posy, arena, conversion='', center='', flip_y=True):
    """
    :param posx:
    :param posy:
    :param arena:
    :param conversion:
    :param center:
    :param flip_y: bool value that will determine if you want to flip y or not. When recording on Intan we inverted the
    positions due to the camera position. However in the virtualmaze you might not want to flip y values.
    :return:
    """
    if 'BehaviorRoom' in arena:
        center = np.array([314.75, 390.5])
        conversion = 495.5234
    elif 'BehaviorRoom2' in arena:
        # added december 12th, 2018
        center = np.array([314.75, 390.5])
        conversion = 485.1185
    elif 'DarkRoom' in arena:
        center = np.array([346.5, 273.5])
        conversion = 711.3701
    elif 'room4' in arena:
        center = np.array([418, 186])
        conversion = 313
    elif arena in ['Linear Track', 'Circular Track', 'Four Leaf Clover Track', 'Simple Circular Track',
                   'Parallel Linear Global Track', 'Parallel Linear Rate Track']:
        center = center
        conversion = conversion
    else:
        print("Room: " + arena + ", is an unknown room!")

    posx = 100 * (posx - center[0]) / conversion

    if flip_y:
        posy = 100 * (-posy + center[1]) / conversion

    return posx, posy


def ReadEEG(eeg_fname):
    """input:
    eeg_filename: the fullpath to the eeg file that is desired to be read.
    Example: C:\Location\of\eegfile.eegX

    Output:
    The EEG waveform, and the sampling frequency"""

    with open(eeg_fname, 'rb') as f:

        is_eeg = False
        if 'eeg' in eeg_fname:
            is_eeg = True
            # Fs = 250
        # else:
        #    Fs = 4.8e3

        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            # find the data_start
            start_index = int(m.find(b'data_start') + len('data_start'))  # start of the data
            stop_index = int(m.find(b'\r\ndata_end'))  # end of the data

            sample_rate_start = m.find(b'sample_rate')
            sample_rate_end = m[sample_rate_start:].find(b'\r\n')
            Fs = float(m[sample_rate_start:sample_rate_start + sample_rate_end].decode('utf-8').split(' ')[1])

            m = m[start_index:stop_index]

            if is_eeg:
                EEG = np.fromstring(m, dtype='>b')
            else:
                EEG = np.fromstring(m, dtype='<h')

            return EEG, int(Fs)


def EEG_to_Mat(input_filename, output_filename):
    EEG, Fs = ReadEEG(input_filename)

    if Fs > 250:
        save_dictionary = {'EGF': EEG, 'Fs': Fs}
    else:
        save_dictionary = {'EEG': EEG, 'Fs': Fs}

    savemat(output_filename, save_dictionary)


def removeNan(posx, posy, post):
    """Remove any NaNs from the end of the array"""
    removeNan = True
    while removeNan:
        if np.isnan(posx[-1]):
            posx = posx[:-1]
            posy = posy[:-1]
            post = post[:-1]
        else:
            removeNan = False
    return posx, posy, post


def centerBox(posx, posy):
    # must remove Nans first because the np.amin will return nan if there is a nan
    posx = posx[~np.isnan(posx)]  # removes NaNs
    posy = posy[~np.isnan(posy)]  # remove Nans

    NE = np.array([np.amax(posx), np.amax(posy)])
    NW = np.array([np.amin(posx), np.amax(posy)])
    SW = np.array([np.amin(posx), np.amin(posy)])
    SE = np.array([np.amax(posx), np.amin(posy)])

    return findCenter(NE, NW, SW, SE)


def findCenter(NE, NW, SW, SE):
    """Finds the center point (x,y) of the position boundaries"""

    x = np.mean([np.amax([NE[0], SE[0]]), np.amin([NW[0], SW[0]])])
    y = np.mean([np.amax([NW[1], NE[1]]), np.amin([SW[1], SE[1]])])
    return np.array([x, y])


def bits2uV(data, data_fpath, set_fpath=''):
    '''

    :param data:
    :param data_fpath: example: 'C:\example\filepath.whatever'
    :param set_fpath:
    :return:
    '''
    path = os.path.split(data_fpath)[0]

    if set_fpath == '':
        set_fpath = os.path.join(path, ''.join([os.path.splitext(os.path.basename(data_fpath))[0],'.set']))

    ext = os.path.splitext(data_fpath)[1]

    if not os.path.exists(set_fpath):
        error_message = 'The following setpath does not exist, cannot convert to uV: %s' % (set_fpath)
        raise TintException(error_message)
        #return error_message, 0

    # create a tetrode map that has rows of channels that correspond to the same tetrode
    tet_map = np.asarray([np.arange(start,start+4) for start in np.arange(0, 32)*4])

    chan_gains = np.array([])
    saved_eeg = np.array([])
    eeg_chan_map = np.array([])

    with open(set_fpath, 'r') as f:
        for line in f:

            if 'ADC_fullscale_mv' in line:
                ADC_fullscale_mv = int(line.split(" ")[1])
            elif 'gain_ch_' in line:
                # create an array of channel gains [channel_number, channels_gain]
                if len(chan_gains) == 0:
                    chan_gains = np.array([int(line[len('gain_ch_'):line.find(" ")]), int(line.split(" ")[1])], ndmin=2)
                else:
                    chan_gains = np.append(chan_gains, np.array([int(line[len('gain_ch_'):line.find(" ")]), int(line.split(" ")[1])], ndmin=2), axis=0)
            elif 'saveEEG_ch_' in line:
                # create an array of EEG channels that are saved
                if int(line.split(" ")[1]) == 1:
                    if len(chan_gains) == 0:
                        saved_eeg = np.array([int(line[len('saveEEG_ch_'):line.find(" ")])])
                    else:
                        saved_eeg = np.append(saved_eeg, np.array([int(line[len('saveEEG_ch_'):line.find(" ")])]))
            elif 'EEG_ch_' in line and 'BPF' not in line:
                if len(eeg_chan_map) == 0:
                    eeg_chan_map = np.array([int(line[len('EEG_ch_'):line.find(" ")]), int(line.split(" ")[1])], ndmin=2)
                else:
                    eeg_chan_map = np.append(eeg_chan_map, np.array([int(line[len('EEG_ch_'):line.find(" ")]), int(line.split(" ")[1])], ndmin=2), axis=0)

    if '.eeg' in ext:
        if len(ext) == len('.eeg'):
            chan_num = 1
        else:
            chan_num = int(ext[len('.eeg'):])

        for index, value in enumerate(eeg_chan_map[:]):
            if value[0] == chan_num:
                eeg_chan = value[1] - 1
                break

        for index, value in enumerate(chan_gains):
            if value[0] == eeg_chan:
                gain = value[1]

        scalar = ADC_fullscale_mv*1000/(gain*128)
        if len(data) == 0:
            data_uV = []
        else:
            data_uV = np.multiply(data, scalar)
            #print(data_uV)

    elif '.egf' in ext:
        if len(ext) == len('.egf'):
            chan_num = 1
        else:
            chan_num = int(ext[len('.egf'):])

        for index, value in enumerate(eeg_chan_map[:]):
            if value[0] == chan_num:
                eeg_chan = value[1] - 1
                break

        for index, value in enumerate(chan_gains):
            if value[0] == eeg_chan:
                gain = value[1]
                break

        scalar = ADC_fullscale_mv*1000/(gain*32768)

        if len(data) == 0:
            data_uV = []
        else:
            data_uV = np.multiply(data, scalar)

    else:
        tetrode_num = int(ext[1:])

        tet_chans = tet_map[tetrode_num-1]

        gain = np.asarray([[gains[1] for gains in chan_gains if gains[0] == chan] for chan in tet_chans])

        scalar = (ADC_fullscale_mv*1000/(gain*128).reshape((1, len(gain))))[0]

        if len(data) == 0:
            data_uV = []
        else:
            data_uV = np.multiply(data, scalar)

    return data_uV, scalar


def getspikes(fullpath):
    """
    This function will return the spike data, spike times, and spike parameters from Tint tetrode data.

    Example:
        tetrode_fullpath = 'C:\\example\\tetrode_1.1'
        ts, ch1, ch2, ch3, ch4, spikeparam = getspikes(tetrode_fullpath)

    Args:
        fullpath (str): the fullpath to the Tint tetrode file you want to acquire the spike data from.

    Returns:
        ts (ndarray): an Nx1 array for the spike times, where N is the number of spikes.
        ch1 (ndarray) an NxM matrix containing the spike data for channel 1, N is the number of spikes,
            and M is the chunk length.
        ch2 (ndarray) an NxM matrix containing the spike data for channel 2, N is the number of spikes,
            and M is the chunk length.
        ch3 (ndarray) an NxM matrix containing the spike data for channel 3, N is the number of spikes,
            and M is the chunk length.
        ch4 (ndarray) an NxM matrix containing the spike data for channel 4, N is the number of spikes,
            and M is the chunk length.
        spikeparam (dict): a dictionary containing the header values from the tetrode file.
    """
    spikes, spikeparam = importspikes(fullpath)
    ts = spikes['t']
    nspk = spikeparam['num_spikes']
    spikelen = spikeparam['samples_per_spike']

    ch1 = spikes['ch1']
    ch2 = spikes['ch2']
    ch3 = spikes['ch3']
    ch4 = spikes['ch4']

    return ts, ch1, ch2, ch3, ch4, spikeparam


def importspikes(filename):
    """Reads through the tetrode file as an input and returns two things, a dictionary containing the following:
    timestamps, ch1-ch4 waveforms, and it also returns a dictionary containing the spike parameters"""

    with open(filename, 'rb') as f:
        for line in f:
            if 'data_start' in str(line):
                spike_data = np.fromstring((line + f.read())[len('data_start'):-len('\r\ndata_end\r\n')], dtype='uint8')
                break
            elif 'num_spikes' in str(line):
                num_spikes = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'bytes_per_timestamp' in str(line):
                bytes_per_timestamp = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'samples_per_spike' in str(line):
                samples_per_spike = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'bytes_per_sample' in str(line):
                bytes_per_sample = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'timebase' in str(line):
                timebase = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'duration' in str(line):
                duration = int(line.decode(encoding='UTF-8').split(" ")[1])
            elif 'sample_rate' in str(line):
                samp_rate = int(line.decode(encoding='UTF-8').split(" ")[1])

                # calculating the big-endian and little endian matrices so we can convert from bytes -> decimal
    big_endian_vector = 256 ** np.arange(bytes_per_timestamp - 1, -1, -1)
    little_endian_matrix = np.arange(0, bytes_per_sample).reshape(bytes_per_sample, 1)
    little_endian_matrix = 256 ** numpy.matlib.repmat(little_endian_matrix, 1, samples_per_spike)

    number_channels = 4

    # calculating the timestamps
    t_start_indices = np.linspace(0, num_spikes * (bytes_per_sample * samples_per_spike * 4 +
                                                   bytes_per_timestamp * 4), num=num_spikes, endpoint=False).astype(
        int).reshape(num_spikes, 1)
    t_indices = t_start_indices

    for chan in np.arange(1, number_channels):
        t_indices = np.hstack((t_indices, t_start_indices + chan))

    t = spike_data[t_indices].reshape(num_spikes, bytes_per_timestamp)  # acquiring the time bytes
    t = np.sum(np.multiply(t, big_endian_vector), axis=1) / timebase  # converting from bytes to float values
    t_indices = None

    waveform_data = np.zeros((number_channels, num_spikes, samples_per_spike))  # (dimensions, rows, columns)

    bytes_offset = 0
    # read the t,ch1,t,ch2,t,ch3,t,ch4

    for chan in range(number_channels):  # only really care about the first time that gets written
        chan_start_indices = t_start_indices + chan * samples_per_spike + bytes_per_timestamp + bytes_per_timestamp * chan
        for spike_sample in np.arange(1, samples_per_spike):
            chan_start_indices = np.hstack((chan_start_indices, t_start_indices +
                                            chan * samples_per_spike + bytes_per_timestamp +
                                            bytes_per_timestamp * chan + spike_sample))
        waveform_data[chan][:][:] = spike_data[chan_start_indices].reshape(num_spikes, samples_per_spike).astype(
            'int8')  # acquiring the channel bytes
        waveform_data[chan][:][:][np.where(waveform_data[chan][:][:] > 127)] -= 256
        waveform_data[chan][:][:] = np.multiply(waveform_data[chan][:][:], little_endian_matrix)

    spikeparam = {'timebase': timebase, 'bytes_per_sample': bytes_per_sample, 'samples_per_spike': samples_per_spike,
                  'bytes_per_timestamp': bytes_per_timestamp, 'duration': duration, 'num_spikes': num_spikes,
                  'sample_rate': samp_rate}

    return {'t': t.reshape(num_spikes, 1), 'ch1': np.asarray(waveform_data[0][:][:]),
            'ch2': np.asarray(waveform_data[1][:][:]),
            'ch3': np.asarray(waveform_data[2][:][:]), 'ch4': np.asarray(waveform_data[3][:][:])}, spikeparam


def speed2D(x, y, t):
    '''calculates an averaged/smoothed speed'''

    N = len(x)
    v = np.zeros((N, 1))

    for index in range(1, N-1):
        v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
        t[index + 1] - t[index - 1])

    v[0] = v[1]
    v[-1] = v[-2]

    return v


def fixTimestamps(post):
    first = post[0]
    N = len(post)
    uniquePost = np.unique(post)

    if len(uniquePost) != N:
        didFix = True
        numZeros = 0
        # find the number of zeros at the end of the file

        while True:
            if post[-1 - numZeros] == 0:
                numZeros += 1
            else:
                break
        last = first + (N-1-numZeros)*0.02
        fixedPost = np.arange(first, last+0.02, 0.02)
        fixedPost = fixedPost.reshape((len(fixedPost), 1))

    else:
        didFix = False
        fixedPost = []

    return didFix, fixedPost


def remBadTrack(x, y, t, threshold):
    """function [x,y,t] = remBadTrack(x,y,t,treshold)

    % Indexes to position samples that are to be removed
   """

    remInd = []
    diffx = np.diff(x, axis=0)
    diffy = np.diff(y, axis=0)
    diffR = np.sqrt(diffx ** 2 + diffy ** 2)

    # the MATLAB works fine without NaNs, if there are Nan's just set them to threshold they will be removed later
    diffR[np.isnan(diffR)] = threshold # setting the nan values to threshold
    ind = np.where((diffR > threshold))[0]

    if len(ind) == 0:  # no bad samples to remove
        return x, y, t

    if ind[-1] == len(x):
        offset = 2
    else:
        offset = 1

    for index in range(len(ind) - offset):
        if ind[index + 1] == ind[index] + 1:
            # A single sample position jump, tracker jumps out one sample and
            # then jumps back to path on the next sample. Remove bad sample.
            remInd.append(ind[index] + 1)
        else:
            ''' Not a single jump. 2 possibilities:
             1. Tracker jumps out, and stay out at the same place for several
             samples and then jumps back.
             2. Tracker just has a small jump before path continues as normal,
             unknown reason for this. In latter case the samples are left
             untouched'''
            idx = np.where(x[ind[index] + 1:ind[index + 1] + 1 + 1] == x[ind[index] + 1])[0]
            if len(idx) == len(x[ind[index] + 1:ind[index + 1] + 1 + 1]):
                remInd.extend(
                    list(range(ind[index] + 1, ind[index + 1] + 1 + 1)))  # have that extra since range goes to end-1

    # keep_ind = [val for val in range(len(x)) if val not in remInd]
    keep_ind = np.setdiff1d(np.arange(len(x)), remInd)

    x = x[keep_ind]
    y = y[keep_ind]
    t = t[keep_ind]

    return x.reshape((len(x), 1)), y.reshape((len(y), 1)), t.reshape((len(t), 1))


def visitedBins(x, y, mapAxis):

    binWidth = mapAxis[1]-mapAxis[0]

    N = len(mapAxis)
    visited = np.zeros((N, N))

    for col in range(N):
        for row in range(N):
            px = mapAxis[col]
            py = mapAxis[row]
            distance = np.sqrt((px-x)**2 + (py-y)**2)

            if np.amin(distance) <= binWidth:
                visited[row, col] = 1

    return visited


def spikePos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):

    randtime = 0

    if shuffleSpks:

        # create a random sample to shuffle from -20 to 20 (not including 0)
        randsamples = np.asarray([sample_num for sample_num in range(-20, 21) if sample_num != 0])

        if shuffleCounter:
            randtime = 0
        else:
            randtime = np.random.choice(randsamples, replace=False)

            maxts = max(ts)
            ts += randtime

            if np.sign(randtime) < 0:
                for index in range(len(ts)):
                    if ts[index] < 0:
                        ts[index] = maxts + np.absolute(randtime) + ts[index]

            elif np.sign(randtime) > 0:
                for index in range(len(ts)):
                    if ts[index] > maxts:
                        ts[index] = ts[index] - maxts

            ts = np.sort(ts)
    else:
        ts = np.roll(ts, randtime)

    N = len(ts)
    spkx = np.zeros((N, 1))
    spky = np.zeros_like(spkx)
    newTs = np.zeros_like(spkx)
    count = -1 # need to subtract 1 because the python indices start at 0 and MATLABs at 1

    for index in range(N):
        tdiff = (t - ts[index])**2
        tdiff2 = (cPost-ts[index])**2
        m = np.amin(tdiff)
        ind = np.where(tdiff == m)[0]

        m2 = np.amin(tdiff2)
        #ind2 = np.where(tdiff2 == m2)[0]

        if m == m2:
            count += 1
            spkx[count] = x[ind[0]]
            spky[count] = y[ind[0]]
            newTs[count] = ts[index]

    spkx = spkx[:count + 1]
    spky = spky[:count + 1]
    newTs = newTs[:count + 1]

    return spkx, spky, newTs, randtime


def ratemap(spike_x, spike_y, posx, posy, post, h, yAxis, xAxis):
    invh = 1/h
    map = np.zeros((len(xAxis), len(yAxis)))
    pospdf = np.zeros_like(map)

    current_Y = -1
    for Y in yAxis:
        current_Y +=1
        current_X = -1
        for X in xAxis:
            current_X += 1
            map[current_Y, current_X], pospdf[current_Y, current_X] = rate_estimator(spike_x, spike_y, X, Y, invh, posx, posy, post)
    pospdf = pospdf / np.sum(np.sum(pospdf))
    return map, pospdf


def rate_estimator(spike_x, spike_y, x, y, invh, posx, posy, post):
    '''Calculate the rate for one position value.
    edge-corrected kernel density estimator'''
    conv_sum = np.sum(gaussian_kernel((spike_x-x)*invh, (spike_y-y)*invh))
    edge_corrector = np.trapz(gaussian_kernel(((posx-x)*invh),((posy-y)*invh)), post, axis=0)
    r = (conv_sum / (edge_corrector + 0.0001)) + 0.0001 # regularised firing rate for "wellbehavedness" i.e. no division by zero or log of zero
    return r, edge_corrector


def gaussian_kernel(x, y):
    '''Gaussian kernel for the rate calculation:
    % k(u) = ((2*pi)^(-length(u)/2)) * exp(u'*u)'''
    r = 0.15915494309190 * np.exp(-0.5 * (np.multiply(x,x) + np.multiply(y,y)));
    return r


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def get_spike_color(cell_number):

    """This method will match the cell number with the color it should be RGB in Tint.

    These cells are numbered from 1-30 (there is technically a zeroth cell, but that isn't plotted"""
    spike_colors = [(1, 8, 184), (93, 249, 75), (234, 8, 9),
                         (229, 22, 239), (80, 205, 243), (27, 164, 0),
                         (251, 188, 56), (27, 143, 167), (127, 41, 116),
                         (191, 148, 23), (185, 9, 17), (231, 223, 67),
                         (144, 132, 145), (34, 236, 228), (217, 20, 145),
                         (172, 64, 80), (176, 106, 138), (199, 194, 167),
                         (216, 204, 105), (160, 204, 61), (187, 81, 88),
                         (45, 216, 122), (242, 136, 25), (50, 164, 161),
                         (249, 67, 16), (252, 232, 147), (114, 156, 238),
                         (241, 212, 179), (129, 62, 162), (235, 133, 126)]

    return spike_colors[int(cell_number)-1]