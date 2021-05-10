"""Estimation class"""

from importlib.resources import open_text
from PyQt5 import uic
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.Qt import Qt
import frds.ui.designs
from .treewidget_measures import TreeViewMeasures
from frds.multiprocessing.threads import ThreadWorker

ui = open_text(frds.ui.designs, "Estimation.ui")


class Estimation(*uic.loadUiType(ui)):
    def __init__(self, parent, threadpool):
        super().__init__(parent)
        super().setupUi(self)
        self.threadpool = threadpool
        self.treeWidget = TreeViewMeasures(self)
        self.groupBoxMeasures.layout().addWidget(self.treeWidget)
        self.buttonBox.removeButton(self.buttonBox.button(QDialogButtonBox.Ok))
        self.start_btn = self.buttonBox.addButton("Start", QDialogButtonBox.ActionRole)
        self.start_btn.clicked.connect(self.start)

    def start(self):
        # self.start_btn.setEnabled(False)
        print("Start estimation")

        for i in range(self.treeWidget.topLevelItemCount()):
            m = self.treeWidget.topLevelItem(i)
            for idx in range(m.childCount()):
                measure = m.child(idx)
                name = measure.data(TreeViewMeasures.Name, Qt.DisplayRole)
                w = ThreadWorker(name, print, name)
                w.signals.finished.connect(
                    lambda name=name: self.textEditLog.insertPlainText(f"{name}\n")
                )
                self.threadpool.enqueue(w)
        # print("Finish estimation")
        # self.start_btn.setEnabled(True)
