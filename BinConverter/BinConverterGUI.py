import sys, shutil, os, time, datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from core.ConversionFunctions import convert_basename
from distutils.dir_util import copy_tree
from BatchTINTV3.core.settings import Settings_Window
from BatchTINTV3.core.klusta_functions import klusta
from core.defaultParameters import default_batchtint, default_filename, default_move_converted, default_threshold
from core.utils import background, Worker, center, Communicate, project_name, raise_w
from core.AddSessions import RepeatAddSessions
import json


class Window(QtWidgets.QWidget):  # defines the window class (main window)

    def __init__(self):  # initializes the main window
        super(Window, self).__init__()
        background(self)  # acquires some features from the background function we defined earlier
        # sets the title of the window

        self.setWindowTitle("%s - Main Window" % project_name)

        # self.current_session = None
        self.current_subdirectory = None
        self.directory_changed = False
        self.modifying_list = False
        self.reset_add_thread = False
        self.repeat_thread_active = True
        self.conversion = False
        self.choice = None
        self.file_chosen = False

        # this signal will append errors/messages to the Log object
        self.LogAppend = Communicate()
        self.LogAppend.myGUI_signal.connect(self.AppendLog)

        # this signal will raise any errors
        self.LogError = Communicate()
        self.LogError.myGUI_signal.connect(self.raiseError)

        # this will remove the top level item from the queue
        self.RemoveQueueItem = Communicate()
        self.RemoveQueueItem.myGUI_signal.connect(self.takeTopLevel)

        self.RemoveSessionItem = Communicate()
        self.RemoveSessionItem.myGUI_signal.connect(self.takeChild)

        self.RemoveSessionData = Communicate()
        self.RemoveSessionData.myGUI_signal.connect(self.takeChildData)

        self.SetSessionItem = Communicate()
        self.SetSessionItem.myGUI_signal.connect(self.setChild)

        self.RemoveChildItem = Communicate()
        self.RemoveChildItem.myGUI_signal_QTreeWidgetItem.connect(self.removeChild)

        self.RepeatAddSessionsThread = QtCore.QThread()
        self.convert_thread = QtCore.QThread()

        self.home()  # runs the home function

    def home(self):  # defines the home function (the main window)

        # read any previous directories chosen so we don't have to
        try:  # attempts to open previous directory catches error if file not found
            # No saved directory's need to create file
            with open(self.directory_settings, 'r+') as filename:  # opens the defined file
                directory_data = json.load(filename)  # loads the directory data from file
                if os.path.exists(directory_data['directory']):
                    current_directory_name = directory_data['directory']  # defines the data
                else:
                    current_directory_name = default_filename  # states that no directory was chosen

        except FileNotFoundError:  # runs if file not found
            with open(self.directory_settings, 'w') as filename:  # opens a file
                current_directory_name = default_filename  # states that no directory was chosen
                directory_data = {'directory': current_directory_name}  # creates a dictionary
                json.dump(directory_data, filename)  # writes the dictionary to the file

        # ------ buttons + widgets -----------------------------

        quit_btn = QtWidgets.QPushButton("Quit", self)
        quit_btn.clicked.connect(self.close_app)
        quit_btn.setShortcut("Ctrl+Q")
        quit_btn.setToolTip('Click to quit (or press Ctrl+Q)')

        self.convert_button = QtWidgets.QPushButton('Convert', self)
        self.convert_button.clicked.connect(self.Convert)
        self.convert_button.setToolTip('Click to start the conversion.')

        self.batch_tint_settings_window = None
        self.batch_tint_settings_button = QtWidgets.QPushButton("Batch Tint Settings")
        self.batch_tint_settings_button.clicked.connect(self.open_batch_tint_settings)

        btn_layout = QtWidgets.QHBoxLayout()

        button_order = [self.convert_button, self.batch_tint_settings_button, quit_btn]
        for button in button_order:
            btn_layout.addWidget(button)

        # Version information -------------------------------------------
        vers_label = QtWidgets.QLabel("%s V1.0.3" % (project_name, ))

        # ------------------ widget layouts ----------------
        self.choose_directory_btn = QtWidgets.QPushButton('Choose Directory', self)
        self.choose_directory_btn.clicked.connect(self.new_directory)

        # the label that states that the line-edit corresponds to the current directory
        directory_label = QtWidgets.QLabel('Current Directory')
        directory_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # the line-edit that displays the current directory
        self.directory_edit = QtWidgets.QLineEdit()
        self.directory_edit.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # aligning the text
        self.directory_edit.setText(current_directory_name)  # default text
        # updates the directory every time the text changes
        self.directory_edit.textChanged.connect(self.changed_directory)

        self.batch_tint_checkbox = QtWidgets.QCheckBox("Batch Tint")
        self.batch_tint_checkbox.toggle()

        self.move_converted_checkbox = QtWidgets.QCheckBox("Move Converted")
        self.move_converted_checkbox.toggle()

        # creating the layout for the text + line-edit so that they are aligned appropriately
        current_directory_layout = QtWidgets.QHBoxLayout()
        current_directory_layout.addWidget(directory_label)
        current_directory_layout.addWidget(self.directory_edit)
        current_directory_layout.addWidget(self.batch_tint_checkbox)
        current_directory_layout.addWidget(self.move_converted_checkbox)

        # creating a layout with the line-edit/text + the button so that they are all together
        directory_layout = QtWidgets.QHBoxLayout()
        directory_layout.addWidget(self.choose_directory_btn)
        directory_layout.addLayout(current_directory_layout)

        # creates the queue of recording sessions to convert
        self.recording_queue = QtWidgets.QTreeWidget()
        self.recording_queue.headerItem().setText(0, "Recording Session:")
        recording_queue_label = QtWidgets.QLabel("Conversion Queue:")
        recording_queue_label.setFont(QtGui.QFont("Arial", 10, weight=QtGui.QFont.Bold))

        recording_queue_layout = QtWidgets.QVBoxLayout()
        recording_queue_layout.addWidget(recording_queue_label)
        recording_queue_layout.addWidget(self.recording_queue)

        # adding the layout for the log
        self.log = QtWidgets.QTextEdit()
        log_label = QtWidgets.QLabel('Log:')
        log_label.setFont(QtGui.QFont("Arial", 10, weight=QtGui.QFont.Bold))
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log)

        # adding the thresholding portion fo the layout

        threshold_label = QtWidgets.QLabel('Threshold(SD\'s)')
        self.threshold = QtWidgets.QLineEdit()
        self.threshold.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.threshold.setText(str(default_threshold))
        self.threshold.setToolTip("This will determine the standard deviations away from the baseline value " +
                                  "that you want to use for the thresholding")

        # check if you want to perform a DC block on the EEG and EGF data
        self.dc_blocker = QtWidgets.QCheckBox("DC Blocking Filter")
        self.dc_blocker.toggle()  # set the default to on

        threshold_layout = QtWidgets.QHBoxLayout()
        for widget in [threshold_label, self.threshold]:
            threshold_layout.addWidget(widget)

        # extra parameters layout

        parameters_layout = QtWidgets.QHBoxLayout()
        for parameter in [threshold_layout, self.dc_blocker]:
            if 'Layout' in parameter.__str__():
                parameters_layout.addLayout(parameter)
            else:
                parameters_layout.addWidget(parameter)

        # ------------- layout ------------------------------

        layout = QtWidgets.QVBoxLayout()

        layout_order = [directory_layout, recording_queue_layout, log_layout,
                        parameters_layout, btn_layout]

        for order in layout_order:
            if 'Layout' in order.__str__():
                layout.addLayout(order)
                layout.addStretch(1)
            else:
                layout.addWidget(order, 0, QtCore.Qt.AlignCenter)
                layout.addStretch(1)

        layout.addStretch(1)  # adds stretch to put the version info at the button
        layout.addWidget(vers_label)  # adds the date modification/version number

        self.setSettings()

        self.setLayout(layout)

        center(self)

        self.set_batch_tint_settings()

        self.show()

        # start thread that will search for new files to convert
        self.RepeatAddSessionsThread.start()
        self.RepeatAddSessionsWorker = Worker(RepeatAddSessions, self)
        self.RepeatAddSessionsWorker.moveToThread(self.RepeatAddSessionsThread)
        self.RepeatAddSessionsWorker.start.emit("start")

    def setCheckbox(self, widget, value):

        if widget.isChecked() and not value:
            # if the checkbox is checked and it isn't supposed to be, untoggle
            widget.toggle()
        elif not widget.isChecked and value:
            # if checkbox isn't checked, and it's supposed to be, toggle
            widget.toggle()

    def setValues(self, settings):
        self.setCheckbox(self.batch_tint_checkbox, settings['batchtint'])
        self.setCheckbox(self.move_converted_checkbox, settings['move_converted'])
        self.threshold.setText(str(settings['threshold']))

    def setSettings(self):
        try:
            with open(self.settings_fname, 'r+') as filename:
                settings = json.load(filename)
        except FileNotFoundError:
            self.setDefaultSettings()
            with open(self.settings_fname, 'r+') as filename:
                settings = json.load(filename)

        try:
            self.setValues(settings)
        except KeyError:
            # re-create settings file, likely you have an outdated version
            self.setDefaultSettings()
            self.setValues(settings)

    def setDefaultSettings(self):
        settings = {}

        settings['batchtint'] = default_batchtint
        settings['move_converted'] = default_move_converted
        settings['threshold'] = default_threshold

        with open(self.settings_fname, 'w') as filename:
            json.dump(settings, filename)

    def getSettings(self):
        settings = {}

        settings['batchtint'] = self.batch_tint_checkbox.isChecked()
        settings['move_converted'] = self.move_converted_checkbox.isChecked()
        settings['threshold'] = self.threshold.text()

        return settings

    def restart_add_sessions_thread(self):

        self.reset_add_thread = True
        if not hasattr(self, 'repeat_thread_active'):
            return

        while self.repeat_thread_active:
            time.sleep(0.1)

        self.RepeatAddSessionsThread.setTerminationEnabled(True)
        self.RepeatAddSessionsThread.start()

        self.RepeatAddSessionsWorker = Worker(RepeatAddSessions, self)
        self.RepeatAddSessionsWorker.moveToThread(self.RepeatAddSessionsThread)
        self.RepeatAddSessionsWorker.start.emit("start")

    def set_batch_tint_settings(self):
        """This method will be used for the Batch Tint Settings window.
        It will define the window, as well as raise the window if it is already defined"""

        if self.batch_tint_settings_window is None:
            batchtint_filename = os.path.join(self.SETTINGS_DIR, 'batchtint_settings_filename.json')
            self.batch_tint_settings_window = Settings_Window(settings_fname=batchtint_filename)
            self.batch_tint_settings_window.backbtn.clicked.connect(lambda: raise_w(self,
                                                                                    self.batch_tint_settings_window))
            self.batch_tint_settings_window.backbtn2.clicked.connect(lambda: raise_w(self,
                                                                                     self.batch_tint_settings_window))

    def open_batch_tint_settings(self):

        if self.batch_tint_settings_window is None:
            self.set_batch_tint_settings()

        self.batch_tint_settings_window.raise_window()

    def changed_directory(self):

        self.directory_changed = True
        self.change_directory_time = time.time()
        # Find the sessions, and populate the conversion queue

    def AppendLog(self, message):
        """
        A function that will append the Log field of the main window (mainly
        used as a slot for a custom pyqt signal)
        """
        if '#' in message:
            message = message.split('#')
            color = message[-1].lower()
            message = message[0]
            message = '<span style="color:%s">%s</span>' % (color, message)

        self.log.append(message)

    def raiseError(self, error_val):
        '''raises an error window given certain errors from an emitted signal'''

        if 'NoDir' in error_val:
            self.choice = QtWidgets.QMessageBox.question(self, "No Chosen Directory",
                                                     "You have not chosen a directory,\n"
                                                     "please choose one to continue!",
                                                     QtWidgets.QMessageBox.Ok)

        elif 'NoPos' in error_val:
            session_pos_filename = error_val[error_val.find('!')+1:]
            self.choice = QtWidgets.QMessageBox.question(self, "No '.pos' file!",
                                                     "There '.pos' file for this '.rhd' session:\n" +
                                                     '%s\n' %session_pos_filename +
                                                     "was not found. would you like a dummy '.pos' file \n"
                                                     "to be created for you?\n",
                                                     QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        elif 'NoTintSettings' in error_val:
            self.choice = QtWidgets.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You have not chosen the settings directory for Batch-Tint,\n"
                                                     "in the main directory that holds all the Batch-Tint files\n"
                                                     "there will be a directory entitled 'settings' that holds\n"
                                                     "all the '.json' files, please choose this folder to continue!",
                                                     QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Abort)
            while self.choice is None:
                time.sleep(0.5)

            if self.choice == QtWidgets.QMessageBox.Ok:
                self.new_settings_directory()
            else:
                return

        elif 'StillNoTintSettings' in error_val:
            self.choice = QtWidgets.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You still have not chosen the settings directory for Batch-Tint,\n"
                                                     "please choose it now.\n",
                                                     QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Abort)
            while self.choice is None:
                time.sleep(0.5)

            if self.choice == QtWidgets.QMessageBox.Ok:
                self.new_settings_directory()
            else:
                return

        elif 'DefaultTintSettings' in error_val:
            self.choice = QtWidgets.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You still have not chosen the settings directory for Batch-Tint,\n"
                                                     "Do you want to try with the default batch-tint settings?\n",
                                                     QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Abort)

        elif 'InvalidTintSettings' in error_val:
            self.choice = QtWidgets.QMessageBox.question(self, "Invalid Batch-Tint Settings file!",
                                                     "You chose an invalid Batch-Tint settings directory,\n"
                                                     "Do you want to choose another directory?\n"
                                                     "Note: Press Default to use the default Batch-Tint Settings!",
                                                     QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Default | QtWidgets.QMessageBox.Abort)

            if self.choice == QtWidgets.QMessageBox.Yes:
                self.new_settings_directory()
            else:
                return

    def close_app(self):

        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtWidgets.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def Convert(self):
        self.choice = None
        # self.current_session = None
        self.current_subdirectory = None

        self.convert_button.setText('Stop Conversion')
        self.convert_button.setToolTip('Click to stop the conversion.')  # defining the tool tip for the start button
        self.convert_button.clicked.disconnect()
        self.convert_button.clicked.connect(self.StopConversion)

        self.conversion = True
        self.position_overwritten = False
        # start conversion threads

        self.convert_thread.start()

        self.convert_thread_worker = Worker(self.convert_queue)
        self.convert_thread_worker.moveToThread(self.convert_thread)
        self.convert_thread_worker.start.emit("start")

    def convert_queue(self):

        if default_filename in self.directory_edit.text():
            self.LogError.myGUI_signal.emit('NoDir')
            self.StopConversion()
            return

        if self.recording_queue.topLevelItemCount() == 0:
            pass

        while self.conversion:

            self.session_item = self.recording_queue.topLevelItem(0)

            if not self.session_item:
                continue
            else:
                # check if the path exists
                sessionpath = os.path.join(self.directory_edit.text(), self.session_item.data(0, 0))
                if not os.path.exists(sessionpath):
                    self.top_level_taken = False
                    self.RemoveQueueItem.myGUI_signal.emit(str(0))
                    while not self.top_level_taken:
                        time.sleep(0.1)
                    continue

            self.current_subdirectory = sessionpath

            # overwrite the settings file so it saves these settings for next time
            self.settings = self.getSettings()
            with open(self.settings_fname, 'w') as f:
                json.dump(self.settings, f)

            # overwrite the directory name
            with open(self.directory_settings, 'w') as f:
                json.dump({'directory': self.directory_edit.text()}, f)

            ConvertSession(self.session_item.data(0, 0), self.settings, self)

    def StopConversion(self):
        self.convert_button.setText('Convert')
        self.convert_button.setToolTip('Click to start the conversion.')  # defining the tool tip for the start button
        self.convert_button.clicked.disconnect()
        self.convert_button.clicked.connect(self.Convert)

        # self.convert_thread.quit()
        self.convert_thread.terminate()
        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Conversion terminated!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))

        self.conversion = False

    def new_directory(self):
        """
        A function that will be used from the Choose Set popup window that will
        produce a popup so the user can pick a filename for the .set file
        """
        # prompt user to pick a .set file

        current_directory_name = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Directory!")

        # if no file chosen, skip
        if current_directory_name == '':
            return

        # change the line-edit that contains the directory information
        self.directory_edit.setText(current_directory_name)

    def new_file(self):
        """
        this method is no longer necessary, decided to have
        the user choose the settings directory instead of the file
        """

        cur_file_name, file_ext = QtWidgets.QFileDialog.getOpenFileName(self, "Select your Batch-Tint Settings File!", '',
                                              'Settings Files (*settings.json)')

        # if no file chosen, skip
        if cur_file_name == '':
            return

        # replace the current .set field in the choose .set window with chosen filename
        self.batchtintsettings_edit.setText(cur_file_name)
        self.file_chosen = True

    def new_settings_directory(self):
        current_directory_name = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Directory!")

        if current_directory_name != '':
            # replace the current .set field in the choose .set window with chosen filename
            self.batchtintsettings_edit.setText(current_directory_name)
            self.SETTINGS_DIR = current_directory_name

        self.directory_chosen = True

    def takeTopLevel(self, item_count):
        item_count = int(item_count)
        self.recording_queue.takeTopLevelItem(item_count)
        self.top_level_taken = True

    def setChild(self, child_count):
        """
        this method will set the self.child_session value to a provided child number of the
        self.session_item item.

        :param child_count:
        :return:
        """
        self.child_session = self.session_item.child(int(child_count)).clone()
        self.child_set = True

    def takeChild(self, child_count):
        self.child_session = self.session_item.takeChild(int(child_count)).clone()
        self.child_taken = True
        # return child_session

    def takeChildData(self, child_count):
        self.child_session = self.session_item.takeChild(int(child_count)).data(0, 0)
        self.child_data_taken = True

    def removeChild(self, QTreeWidgetItem):
        root = self.recording_queue.invisibleRootItem()
        (QTreeWidgetItem.parent() or root).removeChild(QTreeWidgetItem)
        self.child_removed = True


def ConvertSession(directory, bin_settings, self=None):
    """
    This function will be run on all the QTreeWidgetItems which contain the directory and session names for files that
    we want converted.

    :param directory: string, containing the name of the directory that is currently being analyzed
    :param bin_settings: contains settings, such as if the user wants to batchtint their data, or not.
    :param self: object, reference to the class's self.
    :return:
    """
    if default_filename in directory:
        if self is None:
            self.LogError.myGUI_signal.emit('NoDir')

    """This function will take in a session files and then convert the files associated with this session"""

    ##################################################################################################
    # ------------------Concatenates all the LFP/time data for each session---------------------------
    ##################################################################################################

    # main_window.current_session = directory

    session_aborted = False

    # remove the appropriate session from the TreeWidget
    iterator = QtWidgets.QTreeWidgetItemIterator(self.recording_queue)
    item_found = False
    # loops through the tree to see if the session is already there

    while iterator.value() and not item_found:

        # find the next file in the list via the iterator, in this case this will represent the directory
        self.item = iterator.value()

        # find all sessions within the directory that we will need to analyze

        if self.item.data(0, 0) == directory:
            # iterate through each item in the queue to find the appropriate directory to analyze. We will also find
            # the item_count which will help us later remove this directory once we have finalized the conversion.
            for item_count in range(self.recording_queue.topLevelItemCount()):
                if self.item == self.recording_queue.topLevelItem(item_count):
                    # these items match, therefore
                    item_found = True  # this will insure that we break the parented while loop early.

                    # initialize the tint_basenames list which will contain session basenames to analyze
                    tint_basenames = []
                    # adding the .set files to a list of session_files
                    for child_count in range(self.item.childCount()):
                        tint_basenames.append(self.item.child(child_count).data(0, 0))
                    break
        else:
            iterator += 1

    # iterate through each of the basenames within the directory
    for child_index, basename in enumerate(tint_basenames):

        if self.item.child(0).data(0, 0) == basename:
            self.child_set = False  # setting the child_set value to false

            # set the self.child_session value
            self.SetSessionItem.myGUI_signal.emit(str(0))

            # the self.child_set value will be set to True once the child has been set
            while not self.child_set:
                time.sleep(0.1)

        else:
            # considering the order through which these tint_basename values are set
            # we should not have to worry about the above if statement being false.
            # theoretically we should be performing a for loop of all the children
            # within this item and finding the item that matches.
            pass

        # obtaining the corresponding .set filename
        set_filename = os.path.join(self.directory_edit.text(), directory, basename)

        # convert the corresponding .bin file of the obtained .set file, if any errors have occurred the returned
        # converted value should be an 'Aborted' string value.
        converted = convert_basename(self, set_filename, bin_settings['threshold'])

        # remove the session's QTreeWidgetItem since we have analyzed it already
        self.child_taken = False  # initialize the child taken value
        self.RemoveSessionItem.myGUI_signal.emit(str(0))  # remove the child

        # wait for child to be removed
        while not self.child_taken:
            time.sleep(0.1)

        # set the child_session to None since we are done converting the previous session
        self.child_session = None

        # check if there are any child items within the directory, if not then we can remove the directory from
        # the queue
        if self.item.childCount() == 0:
            self.top_level_taken = False
            self.RemoveQueueItem.myGUI_signal.emit(str(item_count))
            # wait for directory item to be removed
            while not self.top_level_taken:
                time.sleep(0.1)

        # check if there were any errors in the conversion
        if isinstance(converted, str):
            # ensures that the aborted session does not go into batchTint or get moved
            if 'Aborted' in converted:
                session_aborted = True
                continue
                # return

    if session_aborted:
        # in the case that a session was aborted we simply finish the conversion process. Maybe in the future we
        # should just continue on our way and just remove this session from the queue. It is possible that this session
        # is corrupted.
        return

    # determine if the user wants to sort the newly converted data
    if bin_settings['batchtint']:
        # if batch-tint is checked, don't move to converted, just convert run BatchTINTV3 here

        # import the settings values
        with open(self.batch_tint_settings_window.settings_fname, 'r') as f:
            settings = json.load(f)

        settings['move_processed'] = 0  # we will override the move_processed functionality here.

        smtp_settings = {}
        smtp_settings['Notification'] = 0  # 1 for send e-mails, 0 for don't send

        '''
        # if you have the notifications set to 0 you don't have to worry about this.
        # we will need an e-mail to send these experimenter's e-mails from
        smtp_settings['Username'] = 'example@gmail.com'
        smtp_settings['Password'] = 'password'  # associated password
        smtp_settings['ServerName'] = 'smtp.gmail.com'  # the smtp server name, 'smtp.gmail.com' for gmail
        smtp_settings['Port'] = 587  # 587 default for gmail
        '''

        experimenter_settings = {
            # 'example': 'example@gmail.com'  # can do [email1@.., email2@..] if you want it sent to more than 1
        }

        analyzed_set_files = klusta([set_filename], settings,
                                    smtp_settings=smtp_settings,
                                    experimenter_settings=experimenter_settings,
                                    append=None, self=self)

    # move the files to a Converted folder if the user wants to
    if bin_settings['move_converted']:
        # if the user wants to move the converted files.

        # move to the converted file
        convert_fpath = os.path.join(self.directory_edit.text(), 'Converted')
        if not os.path.exists(convert_fpath):
            os.mkdir(convert_fpath)

        directory_source = os.path.join(self.directory_edit.text(), directory)
        directory_destination = os.path.join(convert_fpath, directory)

        if os.path.exists(directory_destination):
            try:
                copy_tree(directory_source, directory_destination)
            except FileNotFoundError:
                return
            try:
                shutil.rmtree(directory_source)
            except PermissionError:
                self.LogAppend.myGUI_signal.emit(
                    '[%s %s]: The current directory could not be deleted, close files and then delete the directory!#red' %
                    (str(datetime.datetime.now().date()),
                     str(datetime.datetime.now().time())[:8]))
        else:
            shutil.move(directory_source, convert_fpath)

    # finished converting
    self.LogAppend.myGUI_signal.emit(
        '[%s %s]: The following directory has finished converting: %s!' %
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8], self.directory_edit.text()))


def run():
    app = QtWidgets.QApplication(sys.argv)

    main_window = Window()  # calling the main window
    main_window.raise_()  # making the main window on top

    sys.exit(app.exec_())  # prevents the window from immediately exiting out


if __name__ == '__main__':
    run()

