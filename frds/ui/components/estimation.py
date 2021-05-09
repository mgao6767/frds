"""Estimation class"""

from importlib.resources import open_text
from PyQt5 import uic
import frds.ui.designs
from .treewidget_measures import TreeViewMeasures

ui = open_text(frds.ui.designs, "Estimation.ui")


class Estimation(*uic.loadUiType(ui)):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setupUi(self)
        self.treeWidget = TreeViewMeasures(self)
        self.groupBoxMeasures.layout().addWidget(self.treeWidget)
