import os
from PyQt5 import QtWidgets
import time
import threading
from .conversion_utils import has_files, is_converted


threadLock = threading.Lock()


def get_added(queue):
    added = []

    item_count = queue.topLevelItemCount()
    for item_index in range(item_count):
        parent_item = queue.topLevelItem(item_index)
        parent_directory = parent_item.data(0, 0)
        for child_i in range(parent_item.childCount()):
            added.append(os.path.join(parent_directory, parent_item.child(child_i).data(0, 0)))

    return added


def addSessions(self):
    """Adds any sessions that are not already on the list"""

    # TODO: Add the removal of a session if you re-check that the file is not longer valid

    if self is not None:
        # doesn't add sessions when re-ordering
        while self.modifying_list:
            # pauses add Sessions when the individual is reordering
            time.sleep(0.1)

    directory_added = False

    current_directory = self.directory_edit.text()
    # finds the sub directories within the chosen directory
    try:
        sub_directories = [d for d in os.listdir(current_directory)
                           if os.path.isdir(os.path.join(current_directory, d)) and
                           len([file for file in os.listdir(os.path.join(current_directory, d))
                                if '.set' in file]) != 0 and
                               d not in ['Processed', 'Converted']]
    except OSError:
        return

    iterator = QtWidgets.QTreeWidgetItemIterator(self.recording_queue)
    # loops through all the already added sessions
    added_directories = []
    while iterator.value():
        item = iterator.value()

        check_directory = os.path.join(current_directory, item.data(0, 0))

        if not os.path.exists(check_directory) and '.set' not in item.data(0, 0):
            # then remove from the list since it doesn't exist anymore
            while threadLock:
                if os.path.basename(check_directory) != self.current_subdirectory:
                    root = self.recording_queue.invisibleRootItem()
                    for child_index in range(root.childCount()):
                        if root.child(child_index) == item:
                            self.RemoveChildItem.myGUI_signal_QTreeWidgetItem.emit(item)
        else:
            added_directories.append(item.data(0, 0))

        iterator += 1

    for directory in sub_directories:

        if directory in added_directories:
            # the directory has already been added, determine if we need to add more
            # find the treewidget item
            iterator = QtWidgets.QTreeWidgetItemIterator(self.recording_queue)
            while iterator.value():
                directory_item = iterator.value()
                if directory_item.data(0, 0) == directory:
                    break
                iterator += 1

            # find added sessions
            added_sessions = []
            try:
                iterator = QtWidgets.QTreeWidgetItemIterator(directory_item)
            except UnboundLocalError:
                return
            except RuntimeError:
                return

            # compile list of added sessions

            added = get_added(self.recording_queue)
            added_sessions += added

            # find sessions to add
            try:
                sessions = FindSessions(os.path.join(current_directory, directory))
            except FileNotFoundError:
                return
            except PermissionError:
                return

            for session in sessions:
                if os.path.join(directory, session) not in added_sessions:
                    tint_basename = os.path.basename(session)

                    # only adds the sessions that haven't been added already

                    session_item = QtWidgets.QTreeWidgetItem()
                    session_item.setText(0, tint_basename)

                    directory_item.addChild(session_item)
                else:
                    pass

        else:
            directory_item = QtWidgets.QTreeWidgetItem()
            directory_item.setText(0, directory)

            # find sessions to add
            try:
                sessions = FindSessions(os.path.join(current_directory, directory))
            except FileNotFoundError:
                return
            except PermissionError:
                return

            # add the sessions to the TreeWidget
            for session in sessions:
                tint_basename = os.path.basename(session)

                # only adds the sessions that haven't been added already

                session_item = QtWidgets.QTreeWidgetItem()
                session_item.setText(0, tint_basename)

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
            while (time.time() - main_window.change_directory_time) < 0.5:
                time.sleep(0.1)
            main_window.recording_queue.clear()
            main_window.directory_changed = False

        try:
            with threadLock:
                main_window.adding_session = True
            addSessions(main_window)

            with threadLock:
                main_window.adding_session = False
                time.sleep(0.1)
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass

