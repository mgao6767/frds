"""Qt-based concurrent executors"""
from PyQt5 import QtCore


class Status:
    Waiting = "waiting"
    Running = "running"
    Error = "error"
    Complete = "complete"
    Colors = {
        Running: "#33a02c",
        Error: "#e31a1c",
        Complete: "#b2df8a",
    }
    Default = {"progress": 0, "status": Waiting}


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
    status
        `Status`
    log
        `str` of log info
    """

    error = QtCore.pyqtSignal(str, str)
    result = QtCore.pyqtSignal(str, object)  # We can send anything back.
    finished = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    status = QtCore.pyqtSignal(str, str)
    log = QtCore.pyqtSignal(str, str)

