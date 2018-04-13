import sys, shutil
from PyQt4 import QtCore, QtGui
from core.ConversionFunctions import *
from distutils.dir_util import copy_tree

_author_ = "Geoffrey Barrett"  # defines myself as the author

Large_Font = ("Arial", 11)  # defines two fonts for different purposes (might not be used
Small_Font = ("Arial", 8)


class Worker(QtCore.QObject):
    '''This worker object will act to ensure that the QThreads are not within the Main Thread'''
    # def __init__(self, main_window, thread):
    def __init__(self, function, *args, **kwargs):
        '''takes in a function, and the arguments and keyword arguments for that function'''
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start.connect(self.run)

    start = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        self.function(*self.args, **self.kwargs)


def background(self):  # defines the background for each window
    """providing the background info for each window"""
    # Acquiring information about geometry
    if getattr(sys, 'frozen', False):
        # frozen
        self.PROJECT_DIR = os.path.dirname(os.path.abspath(sys.executable))  # project directory
    else:
        # unfrozen
        self.PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # project directory

    self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')  # image directory
    self.CORE_DIR = os.path.join(self.PROJECT_DIR, 'core')  # core directory
    # self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')  # settings directory
    self.BATCH_TINT_DIR = os.path.join(self.PROJECT_DIR, 'BatchTint')
    self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'cumc-crown.png')))  # declaring the icon image
    self.deskW, self.deskH = QtGui.QDesktopWidget().availableGeometry().getRect()[2:]  # gets the window resolution
    # self.setWindowState(QtCore.Qt.WindowMaximized) # will maximize the GUI
    self.setGeometry(0, 0, self.deskW/2, self.deskH/1.75)

    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))


class Window(QtGui.QWidget):  # defines the window class (main window)

    def __init__(self):  # initializes the main window
        super(Window, self).__init__()
        # self.setGeometry(50, 50, 500, 300)
        background(self)  # acquires some features from the background function we defined earlier
        # sets the title of the window
        #self.child_session = None
        if getattr(sys, 'frozen', False):
            # frozen
            self.setWindowTitle(os.path.splitext(os.path.basename(sys.executable))[0] + " - Main Window")
        else:
            self.setWindowTitle(os.path.splitext(os.path.basename(__file__))[0] + " - Main Window")

        self.current_session = ''
        self.conversion = False
        self.choice = ''
        self.file_chosen = False
        self.LogAppend = Communicate()
        self.LogAppend.myGUI_signal.connect(self.AppendLog)

        self.LogError = Communicate()
        self.LogError.myGUI_signal.connect(self.raiseError)

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

        #self.q = queue.Queue()

        self.home()  # runs the home function

    def home(self):  # defines the home function (the main window)

        # ------ buttons + widgets -----------------------------

        quit_btn = QtGui.QPushButton("Quit", self)
        quit_btn.clicked.connect(self.close_app)
        quit_btn.setShortcut("Ctrl+Q")
        quit_btn.setToolTip('Click to quit (or press Ctrl+Q)')

        self.convert_button = QtGui.QPushButton('Convert', self)
        self.convert_button.clicked.connect(self.Convert)
        self.convert_button.setToolTip('Click to start the conversion.')

        btn_layout = QtGui.QHBoxLayout()

        button_order = [self.convert_button, quit_btn]
        for button in button_order:
            btn_layout.addWidget(button)

        # Version information -------------------------------------------
        if getattr(sys, 'frozen', False):
            # frozen
            dir_ = os.path.dirname(sys.executable)
            mod_date = time.ctime(os.path.getmtime(dir_))  # finds the modification date of the program
            vers_label = QtGui.QLabel(
                os.path.splitext(os.path.basename(sys.executable))[0] + " V1.0 - Last Updated: " + mod_date)
        else:
            # unfrozen
            dir_ = os.path.dirname(os.path.realpath(__file__))
            mod_date = time.ctime(os.path.getmtime(dir_))  # finds the modification date of the program
            vers_label = QtGui.QLabel(
                os.path.splitext(os.path.basename(__file__))[0] + " V1.0 - Last Updated: " + mod_date)


        # ------------------ widget layouts ----------------
        '''
        self.conversion_widgets = ['Choose Directory', '', 'Current Directory:', '',
                                'Recording Sessions:', '', '', '',
                                'Log:', '', '', '']
        '''

        self.choose_directory_btn = QtGui.QPushButton('Choose Directory', self)
        self.choose_directory_btn.clicked.connect(self.new_directory)

        # the label that states that the line-edit corresponds to the current directory
        directory_label = QtGui.QLabel('Current Directory')
        directory_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # the line-edit that displays the current directory
        self.directory_edit = QtGui.QLineEdit()
        self.directory_edit.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # aligning the text
        self.directory_edit.setText("Choose a Directory!")  # default text
        # updates the directory every time the text changes
        self.directory_edit.textChanged.connect(self.changed_directory)
        self.directory = self.directory_edit.text()

        # creating the layout for the text + line-edit so that they are aligned appropriately
        current_directory_layout = QtGui.QHBoxLayout()
        current_directory_layout.addWidget(directory_label)
        current_directory_layout.addWidget(self.directory_edit)
        #current_directory_layout.addWidget(self.batch_tint_checkbox)

        # batch tint settings filename widgets

        self.choose_batchtint_btn = QtGui.QPushButton('Choose Batch-Tint Settings', self)
        self.choose_batchtint_btn.clicked.connect(self.new_settings_directory)

        # the label that states that the line-edit corresponds to the current directory
        batch_tint_label = QtGui.QLabel('Current Settings File:')
        batch_tint_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # the line-edit that displays the current directory
        self.batchtintsettings_edit = QtGui.QLineEdit()
        self.batchtintsettings_edit.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # aligning the text
        self.batchtintsettings_edit.setText("Choose the Batch-Tint Settings directory!")  # default text
        # updates the directory every time the text changes
        self.batchtintsettings_edit.textChanged.connect(self.changed_directory)
        self.batchtintsettings = self.batchtintsettings_edit.text()

        self.batch_tint_checkbox = QtGui.QCheckBox('Batch-Tint?')
        self.batch_tint_checkbox.setToolTip('Leave Checked to Convert Run Batch-Tint after Conversion.')

        # creating the layout for the text + line-edit so that they are aligned appropriately
        current_settings_layout = QtGui.QHBoxLayout()
        current_settings_layout.addWidget(batch_tint_label)
        current_settings_layout.addWidget(self.batchtintsettings_edit)
        current_settings_layout.addWidget(self.batch_tint_checkbox)

        batch_settings_layout = QtGui.QHBoxLayout()
        batch_settings_layout.addWidget(self.choose_batchtint_btn)
        batch_settings_layout.addLayout(current_settings_layout)

        # creating a layout with the line-edit/text + the button so that they are all together
        directory_layout = QtGui.QHBoxLayout()
        directory_layout.addWidget(self.choose_directory_btn)
        directory_layout.addLayout(current_directory_layout)

        # creates the queue of recording sessions to convert
        self.recording_queue = QtGui.QTreeWidget()
        self.recording_queue.headerItem().setText(0, "Recording Session:")
        recording_queue_label = QtGui.QLabel("Conversion Queue:")
        recording_queue_label.setFont(QtGui.QFont("Arial", 10, weight=QtGui.QFont.Bold))

        recording_queue_layout = QtGui.QVBoxLayout()
        recording_queue_layout.addWidget(recording_queue_label)
        recording_queue_layout.addWidget(self.recording_queue)

        # adding the layout for the log
        self.log = QtGui.QTextEdit()
        log_label = QtGui.QLabel('Log:')
        log_label.setFont(QtGui.QFont("Arial", 10, weight=QtGui.QFont.Bold))
        log_layout = QtGui.QVBoxLayout()
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log)


        # adding the thresholding portion fo the layout

        threshold_label = QtGui.QLabel('Threshold(SD\'s)')
        self.threshold = QtGui.QLineEdit()
        self.threshold.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.threshold.setText('3')
        self.threshold.setToolTip("This will determine the standard deviations away from the baseline value " +
                                  "that you want to use for the thresholding")

        # check if you want to perform a DC block on the EEG and EGF data
        self.dc_blocker = QtGui.QCheckBox("DC Blocking Filter")
        self.dc_blocker.toggle() # set the default to on

        threshold_layout = QtGui.QHBoxLayout()
        for widget in [threshold_label, self.threshold]:
            threshold_layout.addWidget(widget)

        # extra parameters layout

        parameters_layout = QtGui.QHBoxLayout()
        for parameter in [threshold_layout, self.dc_blocker]:
            if 'Layout' in parameter.__str__():
                parameters_layout.addLayout(parameter)
            else:
                parameters_layout.addWidget(parameter)

        # ------------- layout ------------------------------

        layout = QtGui.QVBoxLayout()

        layout_order = [directory_layout, batch_settings_layout, recording_queue_layout, log_layout,
                        parameters_layout, btn_layout]

        for order in layout_order:
            if 'Layout' in order.__str__():
                layout.addLayout(order)
                layout.addStretch(1)
            else:
                layout.addWidget(order, 0, QtCore.Qt.AlignCenter)
                layout.addStretch(1)

        layout.addStretch(1)  # adds stretch to put the version info at the buttom
        layout.addWidget(vers_label)  # adds the date modification/version number

        self.set_parameters('Default')

        self.setLayout(layout)

        center(self)

        self.show()

        # start thread that will search for new files to convert

        self.RepeatAddSessionsThread = QtCore.QThread()
        self.RepeatAddSessionsThread.start()

        self.RepeatAddSessionsWorker = Worker(self.FindSessionsRepeat)
        self.RepeatAddSessionsWorker.moveToThread(self.RepeatAddSessionsThread)
        self.RepeatAddSessionsWorker.start.emit("start")

    def changed_directory(self):
        self.directory = self.directory_edit.text()

        # Find the sessions, and populate the conversion queue

        self.recording_queue.clear()

    def AppendLog(self, message):
        self.log.append(message)

    def raiseError(self, error_val):
        '''raises an error window given certain errors from an emitted signal'''

        if 'NoDir' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "No Chosen Directory",
                                                     "You have not chosen a directory,\n"
                                                     "please choose one to continue!",
                                                     QtGui.QMessageBox.Ok)

        elif 'NoPos' in error_val:
            session_pos_filename = error_val[error_val.find('!')+1:]
            self.choice = QtGui.QMessageBox.question(self, "No '.pos' file!",
                                                     "There '.pos' file for this '.rhd' session:\n" +
                                                     '%s\n' %session_pos_filename +
                                                     "was not found. would you like a dummy '.pos' file \n"
                                                     "to be created for you?\n",
                                                     QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        elif 'NoTintSettings' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You have not chosen the settings directory for Batch-Tint,\n"
                                                     "in the main directory that holds all the Batch-Tint files\n"
                                                     "there will be a directory entitled 'settings' that holds\n"
                                                     "all the '.json' files, please choose this folder to continue!",
                                                     QtGui.QMessageBox.Ok | QtGui.QMessageBox.Abort)
            while self.choice == '':
                time.sleep(0.5)

            if self.choice == QtGui.QMessageBox.Ok:
                self.new_settings_directory()
            else:
                return

        elif 'StillNoTintSettings' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You still have not chosen the settings directory for Batch-Tint,\n"
                                                     "please choose it now.\n",
                                                     QtGui.QMessageBox.Ok | QtGui.QMessageBox.Abort)
            while self.choice == '':
                time.sleep(0.5)

            if self.choice == QtGui.QMessageBox.Ok:
                self.new_settings_directory()
            else:
                return

        elif 'DefaultTintSettings' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "No Batch-Tint Settings Directory!",
                                                     "You still have not chosen the settings directory for Batch-Tint,\n"
                                                     "Do you want to try with the default batch-tint settings?\n",
                                                     QtGui.QMessageBox.Yes, QtGui.QMessageBox.Abort)

        elif 'InvalidTintSettings' in error_val:
            self.choice = QtGui.QMessageBox.question(self, "Invalid Batch-Tint Settings file!",
                                                     "You chose an invalid Batch-Tint settings directory,\n"
                                                     "Do you want to choose another directory?\n"
                                                     "Note: Press Default to use the default Batch-Tint Settings!",
                                                     QtGui.QMessageBox.Yes, QtGui.QMessageBox.Default | QtGui.QMessageBox.Abort)

            if self.choice == QtGui.QMessageBox.Yes:
                self.new_settings_directory()
            else:
                return

    def close_app(self):

        # pop up window that asks if you really want to exit the app ------------------------------------------------

        choice = QtGui.QMessageBox.question(self, "Quitting ",
                                            "Do you really want to exit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            sys.exit()  # tells the app to quit
        else:
            pass

    def addSessions(self):
        """Adds any sessions that are not already on the list"""

        directory_added = False
        # finds the sub directories within the chosen directory
        sub_directories = [d for d in os.listdir(self.directory)
                           if os.path.isdir(os.path.join(self.directory, d)) and
                           len([file for file in os.listdir(os.path.join(self.directory, d))
                                if '.set' in file]) != 0]

        iterator = QtGui.QTreeWidgetItemIterator(self.recording_queue)
        # loops through all the already added sessions
        added_directories = []
        while iterator.value():
            item = iterator.value()
            if not os.path.exists(os.path.join(self.directory, item.data(0, 0))):
                # then remove from the list since it doesn't exist anymore

                root = self.recording_queue.invisibleRootItem()
                for child_index in range(root.childCount()):
                    if root.child(child_index) == item:
                        self.RemoveChildItem.myGUI_signal_QTreeWidgetItem.emit(item)
                        # root.removeChild(directory_item)
            else:
                added_directories.append(item.data(0, 0))

            iterator += 1

        for directory in sub_directories:

            if 'Converted' in directory or 'Processed' in directory:
                continue

            if directory in added_directories:
                # the directory has already been added, skip
                continue

            directory_item = QtGui.QTreeWidgetItem()
            directory_item.setText(0, directory)

            self.sessions = self.FindSessions(os.path.join(self.directory, directory))

            # add the sessions to the TreeWidget
            for session in self.sessions:
                if isinstance(session, str):
                    session = [session]  # needs to be a list for the sorted()[] line

                tint_basename = os.path.basename(os.path.splitext(sorted(session, reverse=False)[0])[0])

                #if session == self.current_session:
                #    already_added = True

                # only adds the sessions that haven't been added already

                session_item = QtGui.QTreeWidgetItem()
                session_item.setText(0, tint_basename)

                # directory_item.addChild(session_item)

                for file in session:
                    session_file_item = QtGui.QTreeWidgetItem()
                    session_file_item.setText(0, file)
                    session_item.addChild(session_file_item)

                directory_item.addChild(session_item)

            if directory_item.childCount() != 0:
                # makes sure that it only adds sessions that have sessions to convert
                self.recording_queue.addTopLevelItem(directory_item)

                directory_added = True
                # self.q.put(tint_basename)
                # self.q.put(directory)

        if directory_added:
            pass

    def Convert(self):
        self.choice == ''
        self.current_session = ''
        self.parameters = self.get_paramters()

        self.convert_button.setText('Stop Conversion')
        self.convert_button.setToolTip('Click to stop the conversion.')  # defining the tool tip for the start button
        self.convert_button.clicked.disconnect()
        self.convert_button.clicked.connect(self.StopConversion)

        # self.addSessions()

        self.conversion = True
        self.position_overwritten = False
        # start conversion threads

        self.convert_thread = QtCore.QThread()
        self.convert_thread.start()

        self.convert_thread_worker = Worker(self.convert_queue)
        self.convert_thread_worker.moveToThread(self.convert_thread)
        self.convert_thread_worker.start.emit("start")

    def convert_queue(self):

        if 'Choose a Directory' in self.directory:
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
                sessionpath = os.path.join(self.directory, self.session_item.data(0, 0))
                if not os.path.exists(sessionpath):
                    self.top_level_taken = False
                    self.RemoveQueueItem.myGUI_signal.emit(str(0))
                    while not self.top_level_taken:
                        time.sleep(0.1)
                    continue

            ConvertSession(self, self.session_item.data(0, 0), self.parameters)

    def StopConversion(self):
        self.convert_button.setText('Convert')
        self.convert_button.setToolTip('Click to start the conversion.')  # defining the tool tip for the start button
        self.convert_button.clicked.disconnect()
        self.convert_button.clicked.connect(self.Convert)

        self.convert_thread.quit()
        self.conversion = False

    def FindSessionsRepeat(self):
        """This will continuously look for files to add to the Queue"""

        while True:
            time.sleep(0.1)  # wait X seconds before adding sessions to create less stress on the machine

            if os.path.exists(self.directory):
                self.addSessions()

    def FindSessions(self, directory):
        """This function will find the sessions"""

        directory_file_list = os.listdir(
            directory)  # making a list of all files within the specified directory


        set_filenames = []

        [set_filenames.append(file) for file in directory_file_list if
         '.set' in file and has_files(os.path.join(directory, file)) and not
         is_converted(os.path.join(directory, file))]

        return set_filenames

    def new_directory(self):
        '''A function that will be used from the Choose Set popup window that will
        produce a popup so the user can pick a filename for the .set file'''
        # prompt user to pick a .set file

        current_directory_name = str(QtGui.QFileDialog.getExistingDirectory(self, "Select a Directory!"))

        # if no file chosen, skip
        if current_directory_name == '':
            return

        self.recording_queue.clear()

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: New Directory Chosen: %s' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], current_directory_name))

        # change the line-edit that contains the directory information
        self.directory_edit.setText(current_directory_name)

        self.LogAppend.myGUI_signal.emit(
            '[%s %s]: Finding Intan Recording Sessions Within Chosen Directory!' %
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8]))

        #self.sessions = self.FindSessions(directory)

        # self.addSessions()

        #self.AppendLog('Found %d sessions within this directory!' % len(self.sessions))

        #add the sessions to the TreeWidget
        '''
        self.recording_queue.clear()
        for session in self.sessions:
            tint_basename = os.path.basename(os.path.splitext(sorted(session, reverse=False)[0])[0])

            session_item = QtGui.QTreeWidgetItem()
            session_item.setText(0, tint_basename)

            for file in session:
                session_file_item = QtGui.QTreeWidgetItem()
                session_file_item.setText(0, file)
                session_item.addChild(session_file_item)

            self.recording_queue.addTopLevelItem(session_item)
            self.q.put(tint_basename)
        '''

    def new_file(self):
        '''this method is no longer necessary, decided to have
        the user choose the settings directory instead of the file'''
        cur_file_name = str(
            QtGui.QFileDialog.getOpenFileName(self, "Select your Batch-Tint Settings File!", '',
                                              'Settings Files (*settings.json)'))

        # if no file chosen, skip
        if cur_file_name == '':
            return

        # replace the current .set field in the choose .set window with chosen filename
        self.batchtintsettings_edit.setText(cur_file_name)
        self.file_chosen = True

    def new_settings_directory(self):
        current_directory_name = str(QtGui.QFileDialog.getExistingDirectory(self, "Select a Directory!"))

        if current_directory_name != '':
            # replace the current .set field in the choose .set window with chosen filename
            self.batchtintsettings_edit.setText(current_directory_name)
            # self.settings_fname = os.path.join(current_directory_name, 'settings.json')
            self.SETTINGS_DIR = current_directory_name

        self.directory_chosen = True

    def set_parameters(self, mode):

        if 'default' in mode.lower():

            default_settings = {'Batch-Tint': 0,
                                }

            for key, value in default_settings.items():

                if 'Batch-Tint' in key:
                    if value == 1:
                        # automatically check
                        if not self.batch_tint_checkbox.isChecked():
                            self.batch_tint_checkbox.toggle()
                    else:
                        # automatically uncheck
                        if self.batch_tint_checkbox.isChecked():
                            self.batch_tint_checkbox.toggle()

        elif 'previous' in mode.lower():
            pass

    def get_paramters(self):

        parameters = {}

        if self.batch_tint_checkbox.isChecked():
            parameters['batchtint'] = True
        else:
            parameters['batchtint'] = False

        parameters['tintsettings'] = self.batchtintsettings_edit.text()

        return parameters

    def takeTopLevel(self, item_count):
        item_count = int(item_count)
        self.recording_queue.takeTopLevelItem(item_count)
        self.top_level_taken = True

    def setChild(self, child_count):
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

@QtCore.pyqtSlot()
def raise_w(new_window, old_window):
    """ raise the current window"""
    new_window.raise_()
    new_window.show()
    time.sleep(0.1)
    old_window.hide()


class Communicate(QtCore.QObject):
    '''A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window'''
    myGUI_signal = QtCore.pyqtSignal(str)
    myGUI_signal_QTreeWidgetItem = QtCore.pyqtSignal(QtGui.QTreeWidgetItem)


def center(self):
    """centers the window on the screen"""
    frameGm = self.frameGeometry()
    screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
    centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


def ConvertSession(main_window, directory, parameters):

    if 'Choose a Directory!' in directory:
        main_window.LogError.myGUI_signal.emit('NoDir')

    """This function will take in a session files and then convert the files associated with this session"""

    ##################################################################################################
    # ------------------Concatenates all the LFP/time data for each session---------------------------
    ##################################################################################################

    # tint_basename = os.path.basename(os.path.splitext(sorted(session_files, reverse=False)[0])[0])
    # current_session = directory

    main_window.current_session = directory

    # remove the appropriate session from the TreeWidget
    iterator = QtGui.QTreeWidgetItemIterator(main_window.recording_queue)
    item_found = False
    # loops through the tree to see if the session is already there

    while iterator.value() and not item_found:
        main_window.item = iterator.value()

        if main_window.item.data(0, 0) == directory:
            for item_count in range(main_window.recording_queue.topLevelItemCount()):
                if main_window.item == main_window.recording_queue.topLevelItem(item_count):
                    # main_window.item = main_window.recording_queue.takeTopLevelItem(item_count)
                    item_found = True

                    # adding the .set files to a list of session_files
                    tint_basenames = []
                    for child_count in range(main_window.item.childCount()):
                        tint_basenames.append(main_window.item.child(child_count).data(0, 0))
                    break
        else:
            iterator += 1

    for child_index, basename in enumerate(tint_basenames):

        # if not item.child(child_index):
        #    return

        if main_window.item.child(0).data(0, 0) == basename:
            main_window.child_set = False
            main_window.SetSessionItem.myGUI_signal.emit(str(0))
            while not main_window.child_set:
                time.sleep(0.1)

            # new_item = main_window.item.takeChild(0)

            for child_count in range(main_window.child_session.childCount()):
                set_fname = main_window.child_session.child(child_count).data(0, 0)

        set_filename = os.path.join(main_window.directory, directory, set_fname)
        converted = convert_basename(main_window, set_filename)

        main_window.child_taken = False
        main_window.RemoveSessionItem.myGUI_signal.emit(str(0))
        while not main_window.child_taken:
            time.sleep(0.1)
        main_window.child_session = None

        if main_window.item.childCount() == 0:
            # main_window.recording_queue.takeTopLevelItem(item_count)
            main_window.top_level_taken = False
            main_window.RemoveQueueItem.myGUI_signal.emit(str(item_count))
            while not main_window.top_level_taken:
                time.sleep(0.1)

        if isinstance(converted, str):
            # ensures that the aborted session does not go into batchTint or get moved
            if 'Aborted' in converted:
                continue
                # return

    if parameters['batchtint']:
        # if batch-tint is checked, don't move to converted, just convert run batchtint here
        # main_window.settings_fname = parameters['tintsettings']
        batchtint(main_window, main_window.directory)

    else:
        # move to the converted file
        convert_fpath = os.path.join(main_window.directory, 'Converted')
        if not os.path.exists(convert_fpath):
            os.mkdir(convert_fpath)

        directory_source = os.path.join(main_window.directory, directory)
        directory_destination = os.path.join(convert_fpath, directory)

        if os.path.exists(directory_destination):
            try:
                copy_tree(directory_source, directory_destination)
            except FileNotFoundError:
                return
            try:
                shutil.rmtree(directory_source)
            except PermissionError:
                main_window.LogAppend.myGUI_signal.emit(
                    '[%s %s]: The following directory could not be deleted, close files and then delete the directory.' %
                    (str(datetime.datetime.now().date()),
                     str(datetime.datetime.now().time())[:8]))
        else:
            shutil.move(directory_source, convert_fpath)


def run():
    app = QtGui.QApplication(sys.argv)

    main_window = Window()  # calling the main window
    main_window.raise_()  # making the main window on top

    sys.exit(app.exec_())  # prevents the window from immediately exiting out

if __name__ == '__main__':
    run()
