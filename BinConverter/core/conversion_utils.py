import os
from .readBin import get_active_tetrode, get_active_eeg


def get_set_header(set_filename):
    with open(set_filename, 'r+') as f:
        header = ''
        for line in f:
            header += line
            if 'sw_version' in line:
                break
    return header


def is_egf_active(set_filename):
    active_egf_str = 'saveEGF'

    with open(set_filename) as f:
        for line in f:

            if active_egf_str in line:
                _, egf_status = line.split(' ')

                if int(egf_status) == 1:
                    return True

        return False


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


def has_files(set_filename):
    """This method will check if all the necessary files exist"""

    # the only thing it needs is a .bin file

    tint_basename = os.path.basename(os.path.splitext(set_filename)[0])

    directory = os.path.dirname(set_filename)

    if os.path.exists(os.path.join(directory, '%s.bin' % tint_basename)):
        return True

    return False
