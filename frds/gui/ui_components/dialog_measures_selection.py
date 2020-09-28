from PyQt5 import QtWidgets, QtCore

from .generated_py_files.Ui_MeasureSelectionWidget import Ui_Measures
from frds.measures_func import MeasureCategory
from frds.utils.measures import (
    get_all_measures,
    get_estimation_function_of_measure,
    get_name_of_measure,
    get_doc_url_of_measure,
)


class DialogMeasuresSelection(QtWidgets.QDialog, Ui_Measures):
    def __init__(
        self, measures_category=MeasureCategory.CORPORATE_FINANCE, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        super().setupUi(self)
        self.measures_category = measures_category
        self.measures_doc_urls = {}
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
