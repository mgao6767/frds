"""Qt-based concurrent executors"""
import abc
from typing import Callable
from PyQt5 import QtCore

STATUS_WAITING = "waiting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"
STATUS_COMPLETE = "complete"
STATUS_COLORS = {
    STATUS_RUNNING: "#33a02c",
    STATUS_ERROR: "#e31a1c",
    STATUS_COMPLETE: "#b2df8a",
}
DEFAULT_STATE = {"progress": 0, "status": STATUS_WAITING}


class WorkerSignals(QtCore.QObject):
    """
    Supported signals are:
    
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc())
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    """

    error = QtCore.pyqtSignal(str, str)
    result = QtCore.pyqtSignal(str, object)  # We can send anything back.
    finished = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    status = QtCore.pyqtSignal(str, str)


class Worker:
    def __init__(self, job_id: str, fn: Callable, *args, **kwargs):
        self.job_id = job_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class ThreadWorker(QtCore.QRunnable):
    def __init__(self, job_id: str, fn: Callable, *args, **kwargs):
        super().__init__()
        self.job_id = job_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.signals.status.emit(self.job_id, STATUS_WAITING)

    @QtCore.pyqtSlot()
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.
        """

        self.signals.status.emit(self.job_id, STATUS_RUNNING)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(self.job_id, str(e))
            self.signals.status.emit(self.job_id, STATUS_ERROR)
        else:
            self.signals.result.emit(self.job_id, result)
            self.signals.status.emit(self.job_id, STATUS_COMPLETE)
        self.signals.finished.emit(self.job_id)
