import os
from PyQt5 import QtWidgets
import time
import threading
from .conversion_utils import has_files, is_converted


threadLock = threading.Lock()


def addSessions(self):
    """Adds any sessions that are not already on the list"""

    if self is not None:
        # doesn't add sessions when re-ordering
        while self.modifying_list:
            # pauses add Sessions when the individual is reordering
            time.sleep(0.1)

    directory_added = False

    directory = self.directory_edit.text()
    # finds the sub directories within the chosen directory
    try:
        sub_directories = [d for d in os.listdir(directory)
                           if os.path.isdir(os.path.join(directory, d)) and
                           len([file for file in os.listdir(os.path.join(directory, d))
                                if '.set' in file]) != 0 and
                               d not in ['Processed', 'Converted']]
    except OSError:
        return

    iterator = QtWidgets.QTreeWidgetItemIterator(self.recording_queue)
    # loops through all the already added sessions
    added_directories = []
    while iterator.value():
        item = iterator.value()
        if not os.path.exists(os.path.join(directory, item.data(0, 0))):
            # then remove from the list since it doesn't exist anymore

            while threadLock:
                root = self.recording_queue.invisibleRootItem()
                for child_index in range(root.childCount()):
                    if root.child(child_index) == item:
                        self.RemoveChildItem.myGUI_signal_QTreeWidgetItem.emit(item)
        else:
            added_directories.append(item.data(0, 0))

        iterator += 1

    for sub_dir in sub_directories:

        if sub_dir in added_directories:
            # the directory has already been added, skip
            continue

        directory_item = QtWidgets.QTreeWidgetItem()
        directory_item.setText(0, sub_dir)

        try:
            self.sessions = FindSessions(os.path.join(directory, sub_dir))
        except FileNotFoundError:
            return

        # add the sessions to the TreeWidget
        for session in self.sessions:
            if isinstance(session, str):
                session = [session]  # needs to be a list for the sorted()[] line

            tint_basename = os.path.basename(os.path.splitext(sorted(session, reverse=False)[0])[0])

            # only adds the sessions that haven't been added already

            session_item = QtWidgets.QTreeWidgetItem()
            session_item.setText(0, tint_basename)

            for file in session:
                session_file_item = QtWidgets.QTreeWidgetItem()
                session_file_item.setText(0, file)
                session_item.addChild(session_file_item)

            directory_item.addChild(session_item)

        if directory_item.childCount() != 0:
            # makes sure that it only adds sessions that have sessions to convert
            self.recording_queue.addTopLevelItem(directory_item)

            directory_added = True

    if directory_added:
        pass


def FindSessions(directory):
    """This function will find the sessions"""

    directory_file_list = os.listdir(
        directory)  # making a list of all files within the specified directory

    set_filenames = []

    [set_filenames.append(file) for file in directory_file_list if
     '.set' in file and has_files(os.path.join(directory, file)) and not
     is_converted(os.path.join(directory, file))]

    return set_filenames


def RepeatAddSessions(main_window):
    """
    This will repeat adding the sessions for BatchTINTV3 so that it is continuously looking for files to analyze within
    a chosen directory.

    :param main_window: this is again the self of the main window for batchTINTV3
    :return:
    """

    # the thread is active, set to true

    main_window.repeat_thread_active = True

    while True:

        with threadLock:
            if main_window.reset_add_thread:
                main_window.repeat_thread_active = False
                main_window.reset_add_thread = False
                return

        if main_window.directory_changed:
            # then we have changed the append cut value
            main_window.recording_queue.clear()
            while (time.time() - main_window.change_directory_time) < 0.5:
                time.sleep(0.25)
                main_window.directory_changed = False

        try:
            with threadLock:
                main_window.adding_session = True
            addSessions(main_window)

            with threadLock:
                main_window.adding_session = False
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass

