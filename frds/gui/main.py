from PyQt5 import QtWidgets, QtCore, QtGui
from frds.gui.multiprocessing.processes import ProcessManager
from frds.gui.ui_components import (
    MainWindow,
    ProgressWindow,
    DialogSettings,
    DialogAbout,
)
from frds.settings import FRDS_HOME_PAGE


class FRDSApplication:
    """The FRDS application class that holds all UI components together"""

    def __init__(self, *args, **kwargs):
        self.app = QtWidgets.QApplication(*args, **kwargs)
        self.manager = ProcessManager()
        # Init UI components
        self.main_window = MainWindow()
        self.progress_monitor = ProgressWindow(
            manager=self.manager, parent=self.main_window
        )
        self.settings_dialog = DialogSettings()
        self.about_dialog = DialogAbout()
        # Connect signals to slots
        self._connect_signals_slots()

        self.test_progress_monitor()  # test only

    def _connect_signals_slots(self):
        self.main_window.actionExit.triggered.connect(self.close)
        self.main_window.actionGeneral.triggered.connect(self.settings_dialog.show)
        self.main_window.actionAbout.triggered.connect(self.about_dialog.show)
        self.main_window.actionProgressMonitor.triggered.connect(
            self.progress_monitor.show
        )
        self.main_window.actionDocumentation.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(FRDS_HOME_PAGE))
        )

    def add_worker_job(self, job):
        self.manager.add_estimation_job(job)

    def run(self):
        self.main_window.show()
        self.app.exec()
        # Code below executed when the main window is closed
        self.manager.close()

    def close(self):
        if self.manager.running_jobs > 0:
            btn = QtWidgets.QMessageBox.question(
                self.main_window,
                "Warning",
                "There are running processes. Do you want to wait for them to finish?",
                defaultButton=QtWidgets.QMessageBox.Yes,
            )
            if btn == QtWidgets.QMessageBox.Yes:
                pass
            elif btn == QtWidgets.QMessageBox.No:
                self.main_window.close()
        else:
            self.main_window.close()

    def test_progress_monitor(self):
        from frds.gui.multiprocessing import Worker
        from frds.measures_func import roa

        for i in range(50):
            job = Worker(job_id=f"id_{i+1}", fn=roa.estimation, n=100,)
            self.add_worker_job(job)
