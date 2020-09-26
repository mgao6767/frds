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
