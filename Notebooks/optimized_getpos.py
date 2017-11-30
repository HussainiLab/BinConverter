import os
import numpy as np

bin_filename = "C:\\Users\\Geoffrey Barrett\\Desktop\\RAW Data\\ConvFromOtherDaq\\20170829-1-CYLINDEROPENFIELD.bin"


def get_valid_pos_indices(valid_iterations):
    """This method will produce the indices of the unpacked byte data relating to the valid positions,
    as well as the the iteration (packet) number that will be later used to calculate the starting position"""

    num_pos_values = 9
    position_offset = 4  # number of values before the position values start

    indices = np.arange(num_pos_values)  # there are 8 position words and 1 frame count for the time
    indices = np.tile(indices, (1, len(valid_iterations)))
    indices = indices.flatten()

    offset_indices = valid_iterations * 213 + position_offset
    offset_indices = np.tile(offset_indices, (num_pos_values, 1))
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


@profile
def get_raw_pos2(bin_filename):
    """This will get the raw position data from the .bin file in the following format:
    video timestamp, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused value

    The video timestamp is the time since the camera has been turned on, thus this value is
    irrelevent

    The .bin file is written of packets of 432 bytes containing position data, three samples of
    electrophys data (per channel), and etc.

    A packet only has a valid position if it has an "ADU2" tag in the header. You'll notice that
    the positions are sampled at twice the normal rate to avoid aliasing.
    """
    pos_sample_num = 0

    byte_count = os.path.getsize(bin_filename)
    bytes_per_iteration = 432
    iteration_count = int(byte_count / bytes_per_iteration)
    sample_count = iteration_count * 192  # each iteration has 192 samples (64*3)

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

    header_byte_len = 32
    data_byte_len = 384
    trailer_byte_len = 16

    raw_pos = np.array([])

    # iteration_string = '%dh' % (8)
    iteration_string = '%di%dhi%dh' % (2, 2, 8 + 192 + 1 + 1 + 5 + 1)

    pos_data = b''

    positions = np.array([])

    with open(bin_filename, 'rb') as f:

        iteration_start = 0

        for i in range(n):
            data = f.read(byte_chunksize)

            # find the indices of ADU2, ADU2 is where the position is valid
            valid_iterations = (np.asarray([instance.start() for instance in re.finditer(b'ADU2', data)]) / 432).astype(
                int)

            # return data
            data = np.asarray(struct.unpack('<%s' % (simul_iterations * iteration_string), data))

            # get the indices that the positions are in
            pos_indices = get_valid_pos_indices(valid_iterations)

            i = valid_iterations + iteration_start

            data = data[pos_indices].reshape(
                (-1, 9))  # There are 8 values for positions and 1 for the frame count (timestamp)

            data = np.hstack((i.reshape((-1, 1)), data))

            iteration_start += simul_iterations

            if len(positions) != 0:

                positions = np.vstack((positions, data))

            else:

                positions = data

        positions[:, 1] = positions[:, 1] / 50  # converting from a frame count to a time value in seconds

        # positions is structured as: packet #, video timestamp, y1, x1, y2, x2, numpix1, numpix2, total_pix, unused value

        # the X and Y values are reverse piece-wise so lets switch the format from
        # packet #, video timestamp, y1, x1, y2, x2, numpix1, numpix2, total_pix, unused value
        # to
        # packet #, video timestamp, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused value

        positions[:, 2:6] = positions[:, [3, 2, 5, 4]]

        # find the first valid sample, since the data is sampled at 48kHz, and there are 3 samples per packet, the packet
        # rate is 16kHz. The positions are sampled at 50 Hz thus there is a valid position ever 320 packets. The valid position
        # will essentially take the last ADU2 headered packet values

        first_sample_index = len(
            np.where([positions[:, 0] <= 320 - 1])[0]) - 1  # subtract one since indices in python start at 0

        indices = np.arange(first_sample_index, len(positions) + 1, 2)  # step of 2 because it needs to be downsampled
        positions = positions[indices, 1:]  # don't need the packet index anymore

        # if there are less than the correct number of positions, just say the rest of the time was spent at the last recorded location

        if positions.shape[0] < n_samples:
            missing_n = n_samples - positions.shape[0]
            last_location = positions[-1, :]
            missing_samples = np.tile(last_location, (missing_n, 1))

            positions = np.vstack((positions, missing_samples))

    return positions
