from typing import Callable
from PyQt5 import QtCore
from frds.settings import MAX_WORKERS, PROGRESS_UPDATE_INTERVAL_SECONDS
from frds.multiprocessing import WorkerSignals, Status


class ThreadWorker(QtCore.QRunnable):
    def __init__(
        self,
        job_id: str,
        fn: Callable,
        *args,
        enable_progress_callback=False,
        **kwargs
    ):
        super().__init__()
        self.job_id = job_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.signals.status.emit(self.job_id, Status.Waiting)
        if enable_progress_callback:
            self.kwargs["progress_callback"] = self.signals.log

    @QtCore.pyqtSlot()
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.
        """
        self.signals.status.emit(self.job_id, Status.Running)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(self.job_id, str(e))
            self.signals.status.emit(self.job_id, Status.Error)
        else:
            self.signals.result.emit(self.job_id, result)
            self.signals.status.emit(self.job_id, Status.Complete)
        self.signals.finished.emit(self.job_id)


class ThreadsManager(QtCore.QAbstractListModel):
    """
    Manager to handle our worker queues and state.
    Also functions as a Qt data model for a view
    displaying progress for each worker.

    """

    _workers = {}
    _state = {}

    status = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a threadpool for our workers.
        self.threadpool = QtCore.QThreadPool(*args, **kwargs)
        self.threadpool.setMaxThreadCount(MAX_WORKERS)
        self.max_threads = self.threadpool.maxThreadCount()
        print("Multithreading with maximum %d threads" % self.max_threads)

        self.status_timer = QtCore.QTimer()
        self.status_timer.setInterval(PROGRESS_UPDATE_INTERVAL_SECONDS)
        self.status_timer.timeout.connect(self.notify_status)
        self.status_timer.start()

    def notify_status(self):
        n_workers = len(self._workers)
        running = min(n_workers, self.max_threads)
        waiting = max(0, n_workers - self.max_threads)
        self.status.emit(
            "{} running, {} waiting, {} threads".format(
                running, waiting, self.max_threads
            )
        )

    def enqueue(self, worker):
        """
        Enqueue a worker to run (at some point) by passing it to the QThreadPool.
        """
        worker.signals.error.connect(self.receive_error)
        worker.signals.status.connect(self.receive_status)
        worker.signals.progress.connect(self.receive_progress)
        worker.signals.finished.connect(self.done)

        self.threadpool.start(worker)
        self._workers[worker.job_id] = worker

        # Set default status to waiting, 0 progress.
        self._state[worker.job_id] = Status.Default.copy()

        self.layoutChanged.emit()

    def receive_status(self, job_id, status):
        self._state[job_id]["status"] = status
        self.layoutChanged.emit()

    def receive_progress(self, job_id, progress):
        self._state[job_id]["progress"] = progress
        self.layoutChanged.emit()

    def receive_error(self, job_id, message):
        print(job_id, message)

    def done(self, job_id):
        """
        Task/worker complete. Remove it from the active workers
        dictionary. We leave it in worker_state, as this is used to
        to display past/complete workers too.
        """
        del self._workers[job_id]
        self.layoutChanged.emit()

    def cleanup(self):
        """
        Remove any complete/failed workers from worker_state.
        """
        for job_id, state in list(self._state.items()):
            if state["status"] in (Status.Complete, Status.Error):
                del self._state[job_id]
        self.layoutChanged.emit()

    # Model interface
    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            # See below for the data structure.
            job_ids = list(self._state.keys())
            job_id = job_ids[index.row()]
            return job_id, self._state[job_id]

    def rowCount(self, index):
        return len(self._state)
