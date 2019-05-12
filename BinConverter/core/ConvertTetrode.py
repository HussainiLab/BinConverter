import os
from .conversion_utils import get_set_header
import numpy as np
import struct


def write_tetrode(filepath, data, Fs):

    session_path, session_filename = os.path.split(filepath)
    tint_basename = os.path.splitext(session_filename)[0]
    set_filename = os.path.join(session_path, '%s.set' % tint_basename)

    n = len(data)

    header = get_set_header(set_filename)

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

    # the spike times are repeated for each channel so lets tile this
    spike_times = np.tile(spike_times, (4, 1))
    spike_times = spike_times.flatten(order='F')

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
