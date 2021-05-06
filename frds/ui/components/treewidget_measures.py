import os.path
import pkgutil
import importlib
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QStyledItemDelegate,
)
from PyQt5.Qt import Qt

N_COLUMNS = 5
Name, Frequency, Description, Source, Reference = range(N_COLUMNS)


class TreeViewMeasures(QTreeWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setColumnCount(N_COLUMNS)
        self.setHeaderLabels(
            ["Name", "Frequency", "Description", "Source", "Reference"]
        )
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(Description, QHeaderView.Stretch)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.setItemDelegate(BoldDelegate(self))

    def addMeasures(self, module):
        pkgpath = os.path.dirname(module.__file__)
        for _, name, _ in pkgutil.walk_packages([pkgpath]):
            mod = importlib.import_module(f".{name}", module.__name__)
            modpath = os.path.dirname(mod.__file__)
            name = mod.name if hasattr(mod, "name") else name.upper()
            it = QTreeWidgetItem(self)
            it.setText(Name, name)
            it.setFlags(it.flags() | Qt.ItemIsTristate |
                        Qt.ItemIsUserCheckable)
            it.setCheckState(Name, Qt.Unchecked)
            f = QFont()
            f.setWeight(QFont.DemiBold)  # less than Bold
            it.setFont(Name, f)
            self.addTopLevelItem(it)
            for _, version, _ in pkgutil.walk_packages([modpath]):
                ver = importlib.import_module(f".{version}", mod.__name__)
                vername = ver.name if hasattr(ver, "name") else name
                verfreq = ver.frequency if hasattr(ver, "frequency") else ""
                verdesc = ver.description if hasattr(
                    ver, "description") else ""
                if hasattr(ver, "source"):
                    if isinstance(ver.source, str):
                        versrc = ver.source
                    elif isinstance(ver.source, list):
                        versrc = "/".join(sorted(ver.source))
                else:
                    versrc = ""
                if hasattr(ver, "reference"):
                    if isinstance(ver.reference, str):
                        verref = ver.reference
                    elif isinstance(ver.reference, list):
                        verref = "; ".join(sorted(ver.reference))
                else:
                    verref = ""
                child = QTreeWidgetItem(it)
                child.setText(Name, vername)
                child.setText(Frequency, verfreq)
                child.setText(Description, verdesc)
                child.setText(Source, versrc)
                child.setText(Reference, verref)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(Name, Qt.Unchecked)
                it.addChild(child)
        self.resizeColumnToContents(Name)
        self.resizeColumnToContents(Frequency)
