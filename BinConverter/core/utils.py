import sys, os
from PyQt5 import QtWidgets, QtGui, QtCore
import time


_author_ = "Geoffrey Barrett"  # defines myself as the author

Large_Font = ("Arial", 11)  # defines two fonts for different purposes (might not be used
Small_Font = ("Arial", 8)

project_name = 'BinConverter'


@QtCore.pyqtSlot()
def raise_w(new_window, old_window):
    """ raise the current window"""

    if "Settings_Window" in str(new_window):
        new_window.raise_window()
        old_window.hide()

    elif "Settings_Window" in str(old_window):
        new_window.raise_()
        new_window.show()

        old_window.backbtn_function()
        old_window.hide()
    else:
        new_window.raise_()
        new_window.show()
        time.sleep(0.1)
        old_window.hide()


class Communicate(QtCore.QObject):
    '''A custom pyqtsignal so that errors and popups can be called from the threads
    to the main window'''
    myGUI_signal = QtCore.pyqtSignal(str)
    myGUI_signal_str = myGUI_signal
    myGUI_signal_QTreeWidgetItem = QtCore.pyqtSignal(QtWidgets.QTreeWidgetItem)


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
    project_dir = os.path.dirname(os.path.abspath("__file__"))

    if os.path.basename(project_dir) != project_name:
        project_dir = os.path.dirname(sys.argv[0])

    # defining the directory filepaths
    self.PROJECT_DIR = project_dir  # project directory

    self.IMG_DIR = os.path.join(self.PROJECT_DIR, 'img')  # image directory
    self.CORE_DIR = os.path.join(self.PROJECT_DIR, 'core')  # core directory
    self.SETTINGS_DIR = os.path.join(self.PROJECT_DIR, 'settings')  # settings directory
    self.BATCH_TINT_DIR = os.path.join(self.PROJECT_DIR, 'BatchTint')
    self.setWindowIcon(QtGui.QIcon(os.path.join(self.IMG_DIR, 'GEBA_Logo.png')))  # declaring the icon image
    self.deskW, self.deskH = QtWidgets.QDesktopWidget().availableGeometry().getRect()[2:]  # gets the window resolution
    # self.setWindowState(QtCore.Qt.WindowMaximized) # will maximize the GUI
    self.setGeometry(0, 0, self.deskW/2, self.deskH/1.75)

    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))


def center(self):
    """centers the window on the screen"""
    frameGm = self.frameGeometry()
    screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
    centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())


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

