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
        self.main_window.actionGeneral.triggered.connect(
            self.settings_dialog.show
        )
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

    def test_progress_monitor(self):
        from frds.gui.multiprocessing import Worker

        for i in range(50):
            job = Worker(job_id=f"id{i}", fn=job_test, n=100,)
            self.add_worker_job(job)


def job_test(job_id, queue, n):
    print(f"{job_id=}, {n=}")
    import time
    import random

    for i in range(n):
        queue.put((job_id, int((i + 1) / n * 100)))
        time.sleep(0.03)

    if random.random() > 0.8:
        raise ValueError

    print(f"{job_id=}, finished")
    return 1
