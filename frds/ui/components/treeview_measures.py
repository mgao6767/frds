from os import name
import os.path
import pkgutil
import importlib
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QHeaderView, QTreeView

N_COLUMNS = 3
Name, Description, Reference = range(N_COLUMNS)


class TreeViewMeasures(QTreeView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = QStandardItemModel(0, N_COLUMNS, *args, **kwargs)
        self.model.setHorizontalHeaderLabels(
            ['Name', 'Description', 'Reference'])
        self.setModel(self.model)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(Description, QHeaderView.Stretch)

    def addMeasures(self, module, parent):
        pkgpath = os.path.dirname(module.__file__)
        for _, name, _ in pkgutil.walk_packages([pkgpath]):
            mod = importlib.import_module(f'.{name}', module.__name__)
            modpath = os.path.dirname(mod.__file__)
            name = mod.name if hasattr(mod, 'name') else name.upper()
            it = QStandardItem(name)
            parent.appendRow(it)
            for _, version, _ in pkgutil.walk_packages([modpath]):
                ver = importlib.import_module(f'.{version}', mod.__name__)
                vername = ver.name if hasattr(ver, 'name') else name
                verdesc = ver.description if hasattr(
                    ver, 'description') else ''
                verref = ver.reference if hasattr(ver, 'reference') else ''
                it.appendRow([QStandardItem(vername), QStandardItem(
                    verdesc), QStandardItem(verref)])
        self.resizeColumnToContents(Name)
