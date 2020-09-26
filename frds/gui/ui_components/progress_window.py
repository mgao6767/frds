from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QStyledItemDelegate
from PyQt5.QtGui import QBrush, QPen, QColor

from frds.gui.multiprocessing import STATUS_COLORS

from .generated_py_files.Ui_ProgressWindow import Ui_ProgressWindow


class ProgressWindow(QtWidgets.QMainWindow, Ui_ProgressWindow):
    def __init__(self, manager: QtCore.QAbstractListModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setupUi(self)

        self.workers = manager
        self.listViewProgresses.setModel(self.workers)
        self.listViewProgresses.setItemDelegate(ProgressBarDelegate())

        self._init_other_components()
        self._connect_actions()
        self._connect_signals()

    def _init_other_components(self):
        pass

    def _connect_actions(self):
        pass

    def _connect_signals(self):
        self.workers.status.connect(self.statusBar().showMessage)

    def display_result(self, job_id, data):
        self.textBrowserLogs.append("Worker %s: %s" % (job_id, data))


class ProgressBarDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # data is our status dict, containing progress, id, status
        job_id, data = index.model().data(index, Qt.DisplayRole)
        if data["progress"] > 0:
            color = QColor(STATUS_COLORS[data["status"]])
            brush = QBrush()
            brush.setColor(color)
            brush.setStyle(Qt.SolidPattern)
            width = option.rect.width() * data["progress"] // 100
            rect = QRect(option.rect)  # Copy of the rect, so we can modify.
            rect.setWidth(width)
            painter.fillRect(rect, brush)
            pen = QPen()
            pen.setColor(Qt.black)
            painter.drawText(option.rect, Qt.AlignLeft, job_id)
