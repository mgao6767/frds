"""MainWindow class"""

from importlib.resources import open_text
from PyQt5.QtCore import QUrl
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDesktopServices, QStandardItemModel
from PyQt5.QtWidgets import QMessageBox, QFileSystemModel
from frds.settings import FRDS_HOME_PAGE
import frds.ui.designs
from frds.ui.components import Preferences, TreeViewMeasures
from frds.utils.settings import get_root_dir
from frds.multiprocessing.threads import ThreadsManager, ThreadWorker

ui = open_text(frds.ui.designs, "MainWindow.ui")


class MainWindow(*uic.loadUiType(ui)):
    def __init__(self):
        super().__init__()
        super().setupUi(self)
        self.threadpool = ThreadsManager(self)

        # Preference settings
        self.pref_window = Preferences(self)

        # File explorer
        self.filesystermModel = QFileSystemModel()
        self.filesystermModel.setRootPath(get_root_dir())
        self.treeViewFilesystem.setModel(self.filesystermModel)
        self.treeViewFilesystem.setRootIndex(
            self.filesystermModel.index(get_root_dir())
        )
        # Setup treeView of measures
        self.tabCorpFinc.layout().addWidget(TreeViewMeasures(self))
        # Tabify dock widgets
        self.tabifyDockWidget(self.dockWidgetFilesystem,
                              self.dockWidgetHistory)
        self.dockWidgetFilesystem.raise_()
        # Connect signals
        self.actionAbout_Qt.triggered.connect(
            lambda: QMessageBox.aboutQt(self))
        self.actionRestoreViews.triggered.connect(self.restoreAllViews)
        self.actionFile_Explorer.triggered.connect(self.toggleFileExplorer)
        self.actionPreferences.triggered.connect(self.pref_window.show)
        self.actionDocumentation.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(FRDS_HOME_PAGE))
        )
        self.threadpool.status.connect(self.statusbar.showMessage)

    def restoreAllViews(self):
        self.dockWidgetFilesystem.show()
        self.dockWidgetFilesystem.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidgetFilesystem)

    def toggleFileExplorer(self):
        if self.actionFile_Explorer.isChecked():
            self.dockWidgetFilesystem.show()
        else:
            self.dockWidgetFilesystem.hide()
