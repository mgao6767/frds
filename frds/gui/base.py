import inspect
from PyQt5.QtCore import (
    Qt,
    QRunnable,
    QObject,
    pyqtSignal,
    pyqtSlot,
    QSize,
    QUrl,
)
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QGridLayout,
    QCheckBox,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
import frds.measures
import frds.run
from frds.gui import MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT


class Worker(QRunnable):
    """Worker thread for running background tasks."""

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exception:
            print(exception)
            self.signals.error.emit(str(exception))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `Exception`
    result
        `object` data returned from processing, anything
    """

    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(str)


class TabBase(QWidget):
    def __init__(self, parent, category):
        super().__init__(parent)
        self.category = category
        self.app = parent.app
        self.status_bar = parent.status_bar
        self._measures = {
            name: {"measure": measure, "url": QUrl(measure.description()),}
            for name, measure in self.collect_measures()
        }

        self.descripion_browser = QWebEngineView()
        useragent = self.descripion_browser.page().profile().httpUserAgent()
        self.descripion_browser.page().profile().setHttpUserAgent(
            f"{useragent} FRDS"
        )
        self.descripion_browser.setZoomFactor(0.75)

        layout = QVBoxLayout()

        # Control button
        self.all_measures_btn = QCheckBox("All Measures")
        self.all_measures_btn.setCheckState(Qt.Checked)
        self.start_btn = QPushButton("Start")
        (ctrl_layout := QHBoxLayout()).addWidget(self.start_btn)
        # Measure selection
        self.list_of_measures = QListWidget()
        self.measure_selection = self.create_measure_selection_layout()
        # Description and measure params
        self.measure_params = self.create_measure_params_widget()
        measure_selection_and_params = QGridLayout()
        measure_selection_and_params.addWidget(
            self.measure_selection, 1, 1, 1, 1
        )
        measure_selection_and_params.addWidget(self.measure_params, 1, 2, 1, 2)

        layout.addLayout(measure_selection_and_params)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

        # Connect
        self.start_btn.clicked.connect(self.on_start_btn_clicked)
        self.all_measures_btn.clicked.connect(self.on_all_measures_btn_clicked)

    def collect_measures(self):
        for name, measure in inspect.getmembers(frds.measures, inspect.isclass):
            if (
                not inspect.isabstract(measure)
                and hasattr(measure, "category")
                and measure.category() is self.category
            ):
                yield name, measure

    def on_all_measures_btn_clicked(self) -> None:
        """Select and deselect all measures"""
        checked = self.all_measures_btn.isChecked()
        for i in range(self.list_of_measures.count()):
            item = self.list_of_measures.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def create_measure_params_widget(self) -> QGroupBox:
        layout = QVBoxLayout()
        layout.addWidget(self.descripion_browser)
        measures_params = QGroupBox("Description")
        measures_params.setLayout(layout)
        return measures_params

    def create_measure_selection_layout(self) -> QGroupBox:
        layout = QVBoxLayout()
        layout.addWidget(self.all_measures_btn)
        for name, _ in self.collect_measures():
            self.list_of_measures.addItem(name)
        h = self.list_of_measures.height()
        for i in range(self.list_of_measures.count()):
            item = self.list_of_measures.item(i)
            item.setSizeHint(QSize(0, int(h / 20)))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
        self.list_of_measures.itemPressed.connect(self.on_measure_selected)
        layout.addWidget(self.list_of_measures)
        measure_selection = QGroupBox("Measures")
        measure_selection.setLayout(layout)
        measure_selection.setMaximumWidth(MAIN_WINDOW_WIDTH // 3 - 20)
        measure_selection.setMinimumHeight(int(MAIN_WINDOW_HEIGHT * 0.7))
        return measure_selection

    def on_measure_selected(self, item: QListWidgetItem) -> None:
        measure_name = item.text()
        url = self._measures.get(measure_name).get("url")
        self.descripion_browser.setUrl(url)

    def on_start_btn_clicked(self) -> None:
        """Start running estimation"""
        self.app.stopped = False
        self.measure_selection.setDisabled(True)
        self.start_btn.setDisabled(True)
        self.start_btn.setText("Running")
        measures_to_estimate = []
        for i in range(self.list_of_measures.count()):
            item = self.list_of_measures.item(i)
            if item.checkState() == Qt.Checked:
                measures_to_estimate.append(item.text())
        worker = Worker(
            frds.run.main, measures_to_estimate=measures_to_estimate, gui=True,
        )
        worker.signals.finished.connect(self.on_completed)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.error.connect(self.update_progress)
        self.app.threadpool.start(worker)

    def on_completed(self) -> None:
        """Callback when estimation is completed"""
        self.measure_selection.setDisabled(False)
        self.start_btn.setDisabled(False)
        self.start_btn.setText("Start")

    def update_progress(self, msg: str) -> None:
        """Update progress"""
        self.status_bar.showMessage(msg)

