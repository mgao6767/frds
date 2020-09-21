import os
from frds import data_dir
from frds.data.datascope.trth_parser import parse_to_data_dir

from datetime import datetime
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QCheckBox,
    QDateEdit,
    QSizePolicy,
    QTextBrowser,
)
from .base import TabBase, Worker, WorkerSignals
import frds.measures
from frds.gui import MAIN_WINDOW_WIDTH
from frds.data import TRTHConnection
from frds.data.datascope.utils import (
    extract_index_components_ric,
    NASDAQ_RIC,
    SP500_RIC,
    NYSE_RIC,
)
import frds


class TabMarketMicrostructure(TabBase):
    def __init__(self, parent):
        super().__init__(parent, frds.measures.Category.MARKET_MICROSTRUCTURE)

    def on_start_btn_clicked(self) -> None:
        """Start data loading dialog"""
        data_loading_dialog = DialogDataLoading(self)
        data_loading_dialog.exec_()


class DialogDataLoading(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.app = parent.app
        self.setFixedWidth(int(MAIN_WINDOW_WIDTH * 0.9))
        self.setWindowTitle("Loading Data")

        main_layout = QGridLayout()
        self.data_params = self.create_data_params_layout()
        self.log_area = self.create_log_area()
        main_layout.addWidget(self.data_params, 1, 1, 1, 1)
        main_layout.addWidget(self.log_area, 1, 2, 1, 2)

        self.btn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.btn_box = QDialogButtonBox(self.btn)
        self.btn_box.accepted.connect(self.on_accepted)
        self.btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addWidget(self.btn_box)

        self.setLayout(layout)

    def create_data_params_layout(self) -> QGroupBox:
        layout = QVBoxLayout()
        # Securities
        layout.addWidget(QLabel("Securities"))
        self._sp500_components_checkbox = QCheckBox("S&&P500 Index componenets")
        self._nasdaq_components_checkbox = QCheckBox(
            "NASDAQ Composite components"
        )
        self._sp500_components_checkbox.setCheckState(Qt.Checked)
        layout.addWidget(self._sp500_components_checkbox)
        layout.addWidget(self._nasdaq_components_checkbox)
        # Data period
        layout.addWidget(QLabel("Data Period"))
        self.date_start_picker = QDateEdit(calendarPopup=True)
        self.date_start_picker.setDateTime(
            QDateTime.currentDateTime().addMonths(-1)
        )
        self.date_end_picker = QDateEdit(calendarPopup=True)
        self.date_end_picker.setDateTime(QDateTime.currentDateTime())
        hbox = QHBoxLayout()
        hbox.addWidget(self.date_start_picker)
        _to = QLabel("to")
        (_hpolicy := QSizePolicy()).setHorizontalStretch(1)
        _to.setSizePolicy(_hpolicy)
        hbox.addWidget(_to)
        hbox.addWidget(self.date_end_picker)
        layout.addLayout(hbox)
        # Order type
        layout.addWidget(QLabel("Data Type"))
        self._type_quote_checkbox = QCheckBox("Quote")
        self._type_trans_checkbox = QCheckBox("Transaction")
        self._type_quote_checkbox.setCheckState(Qt.Checked)
        self._type_trans_checkbox.setCheckState(Qt.Checked)
        self._type_quote_checkbox.setEnabled(False)
        self._type_trans_checkbox.setEnabled(False)
        layout.addWidget(self._type_quote_checkbox)
        layout.addWidget(self._type_trans_checkbox)
        # Frequency
        layout.addWidget(QLabel("Data Frequency"))
        self._freq_intraday_checkbox = QCheckBox("Intraday - All")
        self._freq_daily_checkbox = QCheckBox("Daily")
        self._freq_intraday_checkbox.setCheckState(Qt.Checked)
        self._freq_intraday_checkbox.setEnabled(False)
        self._freq_daily_checkbox.setEnabled(False)
        layout.addWidget(self._freq_intraday_checkbox)
        layout.addWidget(self._freq_daily_checkbox)
        # Putting things together
        data_params = QGroupBox("Data Parameters")
        data_params.setLayout(layout)
        return data_params

    def create_log_area(self) -> QGroupBox:
        layout = QVBoxLayout()
        self.logs = QTextBrowser()
        self.logs.append("Status: Ready.")
        layout.addWidget(self.logs)
        log_area = QGroupBox("Progress")
        log_area.setLayout(layout)
        return log_area

    def on_abort(self, button):
        self.data_params.setEnabled(True)

    def on_accepted(self):
        self.selected_index = []
        if self._nasdaq_components_checkbox.checkState():
            self.selected_index.append(NASDAQ_RIC)
        if self._sp500_components_checkbox.checkState():
            self.selected_index.append(SP500_RIC)
        if not self.selected_index:
            return
        self.data_params.setEnabled(False)
        self.btn_box.clear()
        self.btn_box.addButton(QDialogButtonBox.Abort)
        self.btn_box.clicked.connect(self.on_abort)
        worker = Worker(self.get_index_components)
        worker.signals.finished.connect(self.on_completed)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.error.connect(self.update_progress)
        self.app.threadpool.start(worker)

    def get_index_components(self, progress_callback=None):
        # progress_func = progress_callback.emit
        progress_func = lambda msg: progress_callback.emit(
            f"{datetime.now().strftime('%H:%M:%S')}: {msg}"
        )
        progress_func("Connecting to TRTH...")
        trth = TRTHConnection(
            frds.credentials["dss_username"],
            frds.credentials["dss_password"],
            token=None,
            progress_callback=progress_func,
        )

        date_start = self.date_start_picker.date().toString(Qt.ISODate)
        date_end = self.date_end_picker.date().toString(Qt.ISODate)
        start_date = f"{date_start}T00:00:00.000Z"
        end_date = f"{date_end}T00:00:00.000Z"

        progress_func("Loading index components...")
        self._index_components = extract_index_components_ric(
            trth.get_index_components(
                self.selected_index, start_date, end_date
            ),
            mkt_index=self.selected_index,
        )
        progress_func(f"Loaded {len(self._index_components)} securities.")

        progress_func("Downloading data...")
        data = trth.get_table(self._index_components, start_date, end_date)
        # data = trth.get_table(["A.N"], start_date, end_date)
        temp_dir = os.path.join(data_dir, "TRTH", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(
            temp_dir,
            f"{'_'.join(self.selected_index)}.{date_start}.{date_end}.csv.gz",
        )
        progress_func(f"Saving data to {path}...")
        trth.save_results(data, path)
        progress_func("Downloading finished.")

        progress_func("Start parsing the downloaded data...")
        parse_to_data_dir(path, os.path.join(data_dir, "TRTH"), "1")
        progress_func("Parsing completed.")

    def update_progress(self, msg: str):
        self.logs.append(msg)

    def on_completed(self):
        self.data_params.setEnabled(True)
        self.btn_box.clear()
        self.btn_box.addButton(QDialogButtonBox.Ok)
        self.btn_box.addButton(QDialogButtonBox.Cancel)
