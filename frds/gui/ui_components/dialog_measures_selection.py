import typing
import datetime
from PyQt5 import QtWidgets, QtCore

from .generated_py_files.Ui_MeasureSelectionWidget import Ui_Measures
from frds.measures import MeasureCategory
from frds.utils.measures import (
    get_all_measures,
    get_estimation_function_of_measure,
    get_name_of_measure,
    get_doc_url_of_measure,
)
from frds.gui.multiprocessing import Worker


class DialogMeasuresSelection(QtWidgets.QDialog, Ui_Measures):
    def __init__(
        self,
        measures_category=MeasureCategory.CORPORATE_FINANCE,
        progress_monitor=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        super().setupUi(self)
        self.measures_category = measures_category
        self.measures_doc_urls = {}
        self.measures_module = {}
        self.progress_monitor = progress_monitor
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setText("Start")
        self.initOtherComponents()
        self.connect_signals()

    def initOtherComponents(self):
        self.setWindowTitle(self.measures_category.value)
        for _, module in get_all_measures(self.measures_category):
            measure_name = get_name_of_measure(module)
            self.listWidgetMeasures.addItem(measure_name)
            self.measures_doc_urls.update(
                {measure_name: QtCore.QUrl(get_doc_url_of_measure(module))}
            )
            self.measures_module.update({measure_name: module})
        for i in range(self.listWidgetMeasures.count()):
            item = self.listWidgetMeasures.item(i)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
        self.checkBoxSelectAll.setCheckState(QtCore.Qt.Checked)

        useragent = self.webEngineView.page().profile().httpUserAgent()
        self.webEngineView.page().profile().setHttpUserAgent(f"{useragent} FRDS")

    def connect_signals(self):
        self.checkBoxSelectAll.clicked.connect(self.on_all_measures_btn_clicked)
        self.listWidgetMeasures.itemPressed.connect(self.on_measure_selected)
        self.buttonBox.accepted.connect(self.start_estimation)
        self.buttonBox.rejected.connect(self.close)
        self.progress_monitor.workers.status.connect(self.toggle_start_button)

    def on_all_measures_btn_clicked(self) -> None:
        """Select and deselect all measures"""
        checked = self.checkBoxSelectAll.isChecked()
        for i in range(self.listWidgetMeasures.count()):
            item = self.listWidgetMeasures.item(i)
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def on_measure_selected(self, item: QtWidgets.QListWidgetItem) -> None:
        measure_name = item.text()
        url = self.measures_doc_urls.get(measure_name)
        self.webEngineView.setUrl(url)

    def start_estimation(self):
        self.progress_monitor.show()
        self.progress_monitor.raise_()  # For Mac, bring monitor to front
        self.progress_monitor.activateWindow()  # For Windows, bring monitor to front
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        for i in range(self.listWidgetMeasures.count()):
            item = self.listWidgetMeasures.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                job_id = item.text()
                fn = self.measures_module.get(job_id).estimation
                # TODO: Custom parameters to pass to the estiamtion function
                self.update_estimation_func_params(fn)
                worker = Worker(job_id, fn)
                self.progress_monitor.workers.add_estimation_job(worker)

    def toggle_start_button(self, message):
        btn = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        btn.setEnabled("0 running" in message)

    def update_estimation_func_params(self, fn):
        # TODO: make GUI for params selection based on its function parameters
        for param_name, param_type in typing.get_type_hints(fn).items():
            if param_name == "return":  # don't care about the return type
                continue
            print(param_name, param_type)
            if param_type is datetime.datetime:
                print("is datetime")
