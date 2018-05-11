import sys, os
from PyQt4 import QtCore, QtGui


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


def center(self):
    """centers the window on the screen"""
    frameGm = self.frameGeometry()
    screen = QtGui.QApplication.desktop().screenNumber(QtGui.QApplication.desktop().cursor().pos())
    centerPoint = QtGui.QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())