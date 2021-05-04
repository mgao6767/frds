import os.path
import pkgutil
import importlib
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QFont
from PyQt5.QtWidgets import (
    QHeaderView,
    QTreeView,
    QAbstractItemView,
    QStyledItemDelegate,
)

N_COLUMNS = 5
Name, Frequency, Description, Source, Reference = range(N_COLUMNS)


class BoldDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # decide here if item should be bold and set font weight to bold if needed
        it = index.model().itemFromIndex(index)
        if not it.parent():
            option.font.setWeight(QFont.Bold)
        QStyledItemDelegate.paint(self, painter, option, index)


class TreeViewMeasures(QTreeView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = QStandardItemModel(0, N_COLUMNS, *args, **kwargs)
        self.model.setHorizontalHeaderLabels(
            ["Name", "Frequency", "Description", "Source", "Reference"]
        )
        self.setModel(self.model)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(Description, QHeaderView.Stretch)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setItemDelegate(BoldDelegate(self))

    def addMeasures(self, module, parent):
        pkgpath = os.path.dirname(module.__file__)
        for _, name, _ in pkgutil.walk_packages([pkgpath]):
            mod = importlib.import_module(f".{name}", module.__name__)
            modpath = os.path.dirname(mod.__file__)
            name = mod.name if hasattr(mod, "name") else name.upper()
            it = QStandardItem(name)
            parent.appendRow(it)
            for _, version, _ in pkgutil.walk_packages([modpath]):
                ver = importlib.import_module(f".{version}", mod.__name__)
                vername = ver.name if hasattr(ver, "name") else name
                verfreq = ver.frequency if hasattr(ver, "frequency") else ""
                verdesc = ver.description if hasattr(ver, "description") else ""
                verref = ver.reference if hasattr(ver, "reference") else ""
                if hasattr(ver, "source"):
                    if isinstance(ver.source, str):
                        versrc = ver.source
                    elif isinstance(ver.source, list):
                        versrc = "/".join(sorted(ver.source))
                else:
                    versrc = ""
                it.appendRow(
                    [
                        QStandardItem(vername),
                        QStandardItem(verfreq),
                        QStandardItem(verdesc),
                        QStandardItem(versrc),
                        QStandardItem(verref),
                    ]
                )
        self.resizeColumnToContents(Name)
        self.resizeColumnToContents(Frequency)
