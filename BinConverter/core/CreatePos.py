import os
from .conversion_utils import get_set_header
from .Tint_Matlab import get_setfile_parameter
import struct


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
