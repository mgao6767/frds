import sys
import os
import inspect
from typing import List
from PyQt5.QtCore import (
    Qt,
    QRunnable,
    QObject,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QApplication,
    QScrollArea,
    QWidget,
    QDialog,
    QStatusBar,
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QGridLayout,
    QCheckBox,
    QLineEdit,
)
from PyQt5.QtGui import QIcon
from frds import wrds_username, wrds_password, data_dir, result_dir
import frds.measures
import frds.run


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


class GUI(QDialog):
    """Main GUI"""

    def __init__(self):
        super(GUI, self).__init__(parent=None)

        author = "Mingze Gao"
        my_email = "mingze.gao@sydney.edu.au"
        my_site_url = "https://mingze-gao.com/"
        github_url = "https://github.com/mgao6767/frds/"
        frds_title = "FRDS - Financial Research Data Services"
        style_name = "Fusion"
        intro_html = f"""
        <p>Estimate a collection of corporate finance metrics on one click!</p>
        <ol>
            <li>Select measures to estimate.</li>
            <li>Enter your WRDS username and password, click "Start".</li>
            <li>Output datasets will be saved in the result directory.</li>
        </ol>
        <p>Author: <a href="{my_site_url}">{author}</a> |
        Email: <a href="mailto:{my_email}">{my_email}</a> |
        Source code: <a href="{github_url}">{github_url}</a></p><hr>"""

        self.measures: List[(str, QCheckBox)] = []
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Status: Ready.")
        self.threadpool = QThreadPool()
        self.stopped = True

        # Set window title
        self.setWindowTitle(frds_title)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(script_dir, "favicon.ico")
        self.setWindowIcon(QIcon(icon_path))
        # Set style
        QApplication.setStyle(style_name)

        # Intro layout
        (intro_label := QLabel(intro_html)).setOpenExternalLinks(True)
        (intro_layout := QHBoxLayout()).addWidget(intro_label)

        # Control button
        self.all_measures_btn = QCheckBox("All Measures")
        self.all_measures_btn.setCheckState(Qt.Checked)
        self.start_btn = QPushButton("Start")
        (ctrl_layout := QHBoxLayout()).addWidget(self.start_btn)

        # Measure selection
        self.measure_selection = self.create_measure_selection_layout()
        # Configuration
        self.configuration = self.create_configuration_layout()

        # Main layout
        main_layout = QGridLayout()
        # - Top intro section
        main_layout.addLayout(intro_layout, 0, 0, 1, 4)
        # - Left panel for measure selection
        main_layout.addWidget(self.measure_selection, 1, 0, 2, 1)
        # - Right panel top: configuration
        main_layout.addWidget(self.configuration, 1, 1, 1, 3)
        # - Right panel bottom: controls
        main_layout.addLayout(ctrl_layout, 2, 1, 1, 3)

        (layout := QVBoxLayout()).addLayout(main_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

        # Connect
        self.start_btn.clicked.connect(self.on_start_btn_clicked)
        self.all_measures_btn.clicked.connect(self.on_all_measures_btn_clicked)

    def on_all_measures_btn_clicked(self) -> None:
        """Select and deselect all measures"""
        checked = self.all_measures_btn.isChecked()
        for _, check_box in self.measures:
            check_box.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def create_measure_selection_layout(self) -> QGroupBox:
        """Create layout for measure selection

        It also populates `self.measures`, a list of (measure_name, QCheckBox).

        Returns
        -------
        QGroupBox
            Layout for measure selection
        """
        layout = QVBoxLayout()
        layout.addWidget(self.all_measures_btn)
        widget = QWidget()
        _layout = QVBoxLayout()
        for name, measure in inspect.getmembers(frds.measures, inspect.isclass):
            if not inspect.isabstract(measure):
                (check_box := QCheckBox(name)).setCheckState(Qt.Checked)
                _layout.addWidget(check_box)
                self.measures.append((name, check_box))
        widget.setLayout(_layout)
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        layout.addWidget(scroll)
        (measure_selection := QGroupBox("Measures")).setLayout(layout)
        measure_selection.setMaximumWidth(300)
        return measure_selection

    def create_configuration_layout(self) -> QGroupBox:
        """Create layout for configuration

        It uses the default configuration from `config.ini`.

        Returns
        -------
        QGroupBox
            Layout for configuration
        """
        self.wrds_username_qline = QLineEdit(wrds_username)
        self.wrds_password_qline = QLineEdit(wrds_password)
        self.wrds_password_qline.setEchoMode(QLineEdit.Password)
        self.frds_data_dir = QLineEdit(str(data_dir))
        self.frds_result_dir = QLineEdit(str(result_dir))

        layout = QGridLayout()
        layout.addWidget(QLabel("WRDS Username"), 0, 0, 1, 1)
        layout.addWidget(self.wrds_username_qline, 0, 1, 1, 2)
        layout.addWidget(QLabel("WRDS Password"), 1, 0, 1, 1)
        layout.addWidget(self.wrds_password_qline, 1, 1, 1, 2)
        layout.addWidget(QLabel("Data directory"), 2, 0, 1, 1)
        layout.addWidget(self.frds_data_dir, 2, 1, 1, 2)
        layout.addWidget(QLabel("Result directory"), 3, 0, 1, 1)
        layout.addWidget(self.frds_result_dir, 3, 1, 1, 2)

        (configuration_layout := QGroupBox("Configuration")).setLayout(layout)
        return configuration_layout

    def on_start_btn_clicked(self) -> None:
        """Start running estimation"""
        self.stopped = False
        self.measure_selection.setDisabled(True)
        self.configuration.setDisabled(True)
        self.start_btn.setDisabled(True)
        self.start_btn.setText("Running")
        # TODO: modify config.ini and read config each time in frds.run.main?
        config = dict(
            wrds_username=self.wrds_username_qline.text(),
            wrds_password=self.wrds_password_qline.text(),
            data_dir=self.frds_data_dir.text(),
            result_dir=self.frds_result_dir.text(),
        )
        measures_to_estimate = [
            m for m, check_box in self.measures if check_box.isChecked()
        ]
        worker = Worker(
            frds.run.main,
            measures_to_estimate=measures_to_estimate,
            gui=True,
            config=config,
        )
        worker.signals.finished.connect(self.on_completed)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.error.connect(self.update_progress)
        self.threadpool.start(worker)

    def on_completed(self) -> None:
        """Callback when estimation is completed"""
        self.measure_selection.setDisabled(False)
        self.configuration.setDisabled(False)
        self.start_btn.setDisabled(False)
        self.start_btn.setText("Start")

    def update_progress(self, msg: str) -> None:
        """Update progress"""
        self.status_bar.showMessage(msg)


if __name__ == "__main__":
    app = QApplication([])
    (gui := GUI()).show()
    sys.exit(app.exec_())
