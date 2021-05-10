"""Estimation class"""
from datetime import datetime
from importlib.resources import open_text
from PyQt5 import uic
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.Qt import Qt
import frds.ui.designs
from .treewidget_measures import TreeViewMeasures
from frds.multiprocessing import Status
from frds.multiprocessing.threads import ThreadWorker

ui = open_text(frds.ui.designs, "Estimation.ui")


class Estimation(*uic.loadUiType(ui)):
    def __init__(self, parent, threadpool):
        super().__init__(parent)
        super().setupUi(self)
        self.running = False
        self.__full_log = []
        self.threadpool = threadpool
        self.treeWidget = TreeViewMeasures(self)
        self.groupBoxMeasures.layout().addWidget(self.treeWidget)
        self.buttonBox.removeButton(self.buttonBox.button(QDialogButtonBox.Ok))
        self.start_btn = self.buttonBox.addButton("Start", QDialogButtonBox.ActionRole)
        self.start_btn.clicked.connect(self.start)
        self.checkBoxWarningsOnly.stateChanged.connect(self.__log_warnings_only)

    def start(self):
        # self.start_btn.setEnabled(False)
        print("Start estimation")
        self.running = True
        for i in range(self.treeWidget.topLevelItemCount()):
            m = self.treeWidget.topLevelItem(i)
            for idx in range(m.childCount()):
                measure = m.child(idx)
                name = measure.data(TreeViewMeasures.Name, Qt.DisplayRole)
                w = ThreadWorker(name, self.test_estimation_func, name)
                w.signals.log.connect(self.__log)
                w.signals.error.connect(
                    lambda job_id, msg: self.__log(job_id, msg, Status.Error)
                )
                self.threadpool.enqueue(w)
        # print("Finish estimation")
        # self.start_btn.setEnabled(True)

    def __log(self, job_id, msg, status=Status.Running):
        time = datetime.now().strftime("%H:%M:%S")
        if status == Status.Running:
            color = "blue"
            level = "info"
        elif status == Status.Error:
            color = "red"
            level = "warning"
        log = f"{time} <span style='color:{color};'>{level}</span>: <strong>{job_id}</strong> {msg}"
        self.__full_log.append(log)
        if self.checkBoxWarningsOnly.checkState() == Qt.Checked:
            if level == "warning":
                self.textEditLog.append(log)
        else:
            self.textEditLog.append(log)

    def __log_warnings_only(self, state):
        self.textEditLog.clear()
        if state == Qt.Checked:
            for log in self.__full_log:
                if "warning" in log:
                    self.textEditLog.append(log)
        elif state == Qt.Unchecked:
            for log in self.__full_log:
                self.textEditLog.append(log)

    @staticmethod
    def test_estimation_func(data):
        import time
        import random

        time.sleep(random.random() * 2)

        # Fake error
        if random.random() < 0.1:
            raise ValueError("Failed due to reason XYZ!")
