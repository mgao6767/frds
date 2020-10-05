from PyQt5 import QtWidgets, QtCore, QtGui
from frds.gui.multiprocessing.processes import ProcessManager
from frds.gui.multiprocessing.threads import ThreadsManager
from frds.gui.ui_components import (
    MainWindow,
    ProgressWindow,
    DialogSettings,
    DialogAbout,
    DialogMeasuresSelection,
    DataDownloadWindow,
    DialogTRTHDataLoading,
)
from frds.gui.ui_components.generated_py_files import resources_rc
from frds.settings import FRDS_HOME_PAGE
from frds.measures import MeasureCategory
from .works import make_worker_list_wrds_libraries


class FRDSApplication:
    """The FRDS application class that holds all UI components together"""

    def __init__(self, *args, **kwargs):
        self.app = QtWidgets.QApplication(*args, **kwargs)
        self.app.setWindowIcon(QtGui.QIcon(":/images/frds_icon.png"))
        # Process-based manager for CPU-bound tasks
        self.process_manager = ProcessManager()
        # Thread-based manager for I/O-bound tasks
        self.thread_manager = ThreadsManager()
        # Init UI components
        self.main_window = MainWindow()
        self.progress_monitor = ProgressWindow(
            manager=self.process_manager, parent=self.main_window
        )
        self.settings_dialog = DialogSettings()
        self.about_dialog = DialogAbout()
        self.wrds_data_download_window = DataDownloadWindow(
            parent=self.main_window, thread_manager=self.thread_manager
        )
        self.trth_data_download_window = DialogTRTHDataLoading(
            parent=self.main_window, thread_manager=self.thread_manager
        )
        self.measures_selection_dialog_corp_finc = DialogMeasuresSelection(
            parent=self.main_window,
            measures_category=MeasureCategory.CORPORATE_FINANCE,
            progress_monitor=self.progress_monitor,
        )
        self.measures_selection_dialog_mkt_structure = DialogMeasuresSelection(
            parent=self.main_window,
            measures_category=MeasureCategory.MARKET_MICROSTRUCTURE,
            progress_monitor=self.progress_monitor,
        )
        self.measures_selection_dialog_banking = DialogMeasuresSelection(
            parent=self.main_window,
            measures_category=MeasureCategory.BANKING,
            progress_monitor=self.progress_monitor,
        )
        # Connect signals to slots
        self._connect_signals_slots()

        # Start background tasks
        self._start_background_workers()
        # self.test_progress_monitor()  # test only

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
        self.main_window.actionCorporate_finance_measures.triggered.connect(
            self.measures_selection_dialog_corp_finc.show
        )
        self.main_window.actionBanking_measures.triggered.connect(
            self.measures_selection_dialog_banking.show
        )
        self.main_window.actionMarket_micro_structure_measures.triggered.connect(
            self.measures_selection_dialog_mkt_structure.show
        )
        self.main_window.actionWRDSData.triggered.connect(
            self.initWRDSDataDownloadWindow
        )
        self.main_window.actionTRTHData.triggered.connect(
            self.initTRTHDataDownloadWindow
        )

    def _start_background_workers(self):
        worker = make_worker_list_wrds_libraries()
        worker.signals.result.connect(
            # discard the first parameter job_id from the worker.signals
            lambda job_id, result: self.wrds_data_download_window.display_libraries(
                result
            ),
        )
        self.thread_manager.enqueue(worker)

    def initWRDSDataDownloadWindow(self):
        self.wrds_data_download_window.show()

    def initTRTHDataDownloadWindow(self):
        self.trth_data_download_window.show()

    def add_worker_job(self, job):
        self.process_manager.add_estimation_job(job)

    def run(self):
        self.main_window.show()
        self.app.exec()
        # Code below executed when the main window is closed

    def close(self):
        # FIXME: quit when there're still running processes
        if self.process_manager.running_jobs > 0:
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
        # self.process_manager.close()

    def test_progress_monitor(self):
        from frds.gui.multiprocessing import Worker
        from frds.measures_func import roa

        for i in range(10):
            job = Worker(job_id=f"id_{i+1}", fn=roa.estimation, n=100,)
            self.add_worker_job(job)
