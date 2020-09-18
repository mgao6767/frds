import sys
import os
import inspect
import webbrowser
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
    QMainWindow,
    QTabWidget,
    QDialogButtonBox,
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
from frds import credentials
from frds import data_dir, result_dir
import frds.measures
import frds.run


MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT = 800, 600
author = "Mingze Gao"
my_email = "mingze.gao@sydney.edu.au"
my_site_url = "https://mingze-gao.com/"
github_url = "https://github.com/mgao6767/frds/"
homepage = "https://frds.io"
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
Source code: <a href="{github_url}">{github_url}</a></p>"""
intro_home = f'<em>Made for better and easier finance research, \
    by <a href="{my_site_url}">{author}</a>.</em>'


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


class App(QMainWindow):
    """Main entrance of the application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(frds_title)
        self.setFixedSize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
        self.threadpool = QThreadPool()
        self.stopped = True
        QApplication.setStyle(style_name)
        self.setCentralWidget(GUI(self))
        self.show()


class GUI(QWidget):
    """Main GUI"""

    def __init__(self, parent):
        super(GUI, self).__init__(parent)
        self.app = parent

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Status: Ready.")

        # Intro layout
        intro_layout = QVBoxLayout()
        (lbl := QLabel(intro_home)).setOpenExternalLinks(True)
        intro_layout.addWidget(lbl)
        buttons = QHBoxLayout()
        intro_layout.addLayout(buttons)
        homepage_button = QPushButton("Homepage")
        abt_button = QPushButton("About")
        config_button = QPushButton("Settings")
        buttons.addWidget(homepage_button)
        buttons.addWidget(abt_button)
        buttons.addWidget(config_button)
        homepage_button.clicked.connect(lambda: webbrowser.open(homepage))
        abt_button.clicked.connect(self.on_about_btn_clicked)
        config_button.clicked.connect(self.on_config_btn_clicked)

        self.tabs = QTabWidget()
        self.tab_corporate_finance = TabCorporateFinance(self)
        # self.tab_2 = TabCorporateFinance(self)
        # self.tab_3 = TabCorporateFinance(self)
        self.tabs.addTab(self.tab_corporate_finance, "Corporate Finance")
        # self.tabs.addTab(self.tab_2, "Market Microstructure")
        # self.tabs.addTab(self.tab_3, "Banking")

        # Main layout
        main_layout = QGridLayout()
        main_layout.addLayout(intro_layout, 0, 0, 1, 4)
        main_layout.addWidget(self.tabs, 1, 0, 2, 4)

        (layout := QVBoxLayout()).addLayout(main_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

    def on_about_btn_clicked(self) -> None:
        about = DialogAbout(self)
        about.exec_()

    def on_config_btn_clicked(self) -> None:
        config = DialogConfig(self)
        config.exec_()


class TabCorporateFinance(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.app = parent.app
        self.status_bar = parent.status_bar
        self.measures: List[(str, QCheckBox)] = []
        layout = QVBoxLayout()

        # Control button
        self.all_measures_btn = QCheckBox("All Measures")
        self.all_measures_btn.setCheckState(Qt.Checked)
        self.start_btn = QPushButton("Start")
        (ctrl_layout := QHBoxLayout()).addWidget(self.start_btn)
        # Measure selection
        self.measure_selection = self.create_measure_selection_layout()

        layout.addWidget(self.create_measure_selection_layout())
        layout.addLayout(ctrl_layout)
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

    def on_start_btn_clicked(self) -> None:
        """Start running estimation"""
        self.app.stopped = False
        self.measure_selection.setDisabled(True)
        self.start_btn.setDisabled(True)
        self.start_btn.setText("Running")
        measures_to_estimate = [
            m for m, check_box in self.measures if check_box.isChecked()
        ]
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


class DialogAbout(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(int(MAIN_WINDOW_WIDTH * 0.8))
        # Intro layout
        (intro_label := QLabel(intro_html)).setOpenExternalLinks(True)
        (intro_layout := QVBoxLayout()).addWidget(intro_label)
        btn = QDialogButtonBox.Ok
        btn_box = QDialogButtonBox(btn)
        intro_layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        self.setLayout(intro_layout)
        self.setWindowTitle("About FRDS")


class DialogConfig(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(int(MAIN_WINDOW_WIDTH * 0.8))
        layout = QVBoxLayout()
        layout.addLayout(self.create_configuration())
        self.setLayout(layout)
        btn = QDialogButtonBox.Save | QDialogButtonBox.Cancel
        btn_box = QDialogButtonBox(btn)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.on_save_btn_clicked)
        btn_box.rejected.connect(self.reject)

    def on_save_btn_clicked(self) -> None:
        # TODO: save settings to `config.ini`
        self.accept()

    def create_configuration(self):
        self.frds_data_dir = QLineEdit(str(data_dir))
        self.frds_result_dir = QLineEdit(str(result_dir))
        self.wrds_username_qline = QLineEdit(credentials.get("wrds_username"))
        self.wrds_password_qline = QLineEdit(credentials.get("wrds_password"))
        self.wrds_password_qline.setEchoMode(QLineEdit.Password)
        self.dss_username_qline = QLineEdit()
        self.dss_password_qline = QLineEdit()
        self.dss_password_qline.setEchoMode(QLineEdit.Password)

        layout = QGridLayout()
        layout.addWidget(QLabel("Data directory"), 0, 0, 1, 1)
        layout.addWidget(QLabel("Result directory"), 1, 0, 1, 1)
        layout.addWidget(self.frds_data_dir, 0, 1, 1, 2)
        layout.addWidget(self.frds_result_dir, 1, 1, 1, 2)
        general = QGroupBox("General Settings")
        general.setLayout(layout)

        login_layout = QGridLayout()
        login_layout.addWidget(QLabel("WRDS Username"), 0, 0, 1, 1)
        login_layout.addWidget(QLabel("WRDS Password"), 1, 0, 1, 1)
        login_layout.addWidget(QLabel("DSS Username"), 2, 0, 1, 1)
        login_layout.addWidget(QLabel("DSS Username"), 3, 0, 1, 1)
        login_layout.addWidget(self.wrds_username_qline, 0, 1, 1, 2)
        login_layout.addWidget(self.wrds_password_qline, 1, 1, 1, 2)
        login_layout.addWidget(self.dss_username_qline, 2, 1, 1, 2)
        login_layout.addWidget(self.dss_password_qline, 3, 1, 1, 2)
        login = QGroupBox("Database Credentials")
        login.setLayout(login_layout)

        configuration_layout = QHBoxLayout()
        configuration_layout.addWidget(general)
        configuration_layout.addWidget(login)
        return configuration_layout


if __name__ == "__main__":
    app = QApplication([])
    script_dir = os.path.dirname(os.path.realpath(__file__))
    icon_path = os.path.join(script_dir, "favicon.ico")
    app.setWindowIcon(QIcon(icon_path))
    ex = App()
    sys.exit(app.exec_())
