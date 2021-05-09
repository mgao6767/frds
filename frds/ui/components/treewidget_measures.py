import os.path
import pkgutil
import importlib
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
)
from PyQt5.Qt import Qt
from frds.settings import FRDS_MEASURES_PAGE

N_COLUMNS = 6

class TreeViewMeasures(QTreeWidget):

    Name, Frequency, Description, Source, Reference, DocUrl = range(N_COLUMNS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setColumnCount(N_COLUMNS)
        self.setHeaderLabels(
            ["Name", "Frequency", "Description", "Source", "Reference", "DocUrl"]
        )
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(
            TreeViewMeasures.Description, QHeaderView.Stretch)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setColumnHidden(TreeViewMeasures.DocUrl, True)
        self.setExpandsOnDoubleClick(False)

    def addMeasures(self, module):
        pkgpath = os.path.dirname(module.__file__)
        for _, name, _ in pkgutil.walk_packages([pkgpath]):
            mod = importlib.import_module(f".{name}", module.__name__)
            modpath = os.path.dirname(mod.__file__)
            name = mod.name if hasattr(mod, "name") else name.upper()
            doc_url = mod.doc_url if hasattr(mod, "doc_url") else FRDS_MEASURES_PAGE
            it = QTreeWidgetItem(self)
            it.setText(TreeViewMeasures.Name, name)
            it.setFlags(it.flags() | Qt.ItemIsTristate |
                        Qt.ItemIsUserCheckable)
            it.setCheckState(TreeViewMeasures.Name, Qt.Unchecked)
            f = QFont()
            f.setWeight(QFont.DemiBold)  # less than Bold
            it.setFont(TreeViewMeasures.Name, f)
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
                child.setText(TreeViewMeasures.Name, vername)
                child.setText(TreeViewMeasures.Frequency, verfreq)
                child.setText(TreeViewMeasures.Description, verdesc)
                child.setText(TreeViewMeasures.Source, versrc)
                child.setText(TreeViewMeasures.Reference, verref)
                child.setText(TreeViewMeasures.DocUrl, doc_url)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(TreeViewMeasures.Name, Qt.Unchecked)
                it.addChild(child)
        self.resizeColumnToContents(TreeViewMeasures.Name)
        self.resizeColumnToContents(TreeViewMeasures.Frequency)
        self.sortByColumn(TreeViewMeasures.Name, Qt.AscendingOrder)
