from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QHeaderView, QTreeView

N_COLUMNS = 4
Name, Description, Reference, Contributor = range(N_COLUMNS)


class TreeViewMeasures(QTreeView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = QStandardItemModel(0, N_COLUMNS, *args, **kwargs)
        self.model.setHeaderData(Name, Qt.Horizontal, "Name")
        self.model.setHeaderData(
            Description, Qt.Horizontal, "Short Description")
        self.model.setHeaderData(Reference, Qt.Horizontal, "Reference")
        self.model.setHeaderData(Contributor, Qt.Horizontal, "Contributor")
        self.setModel(self.model)
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.header().setStretchLastSection(False)
        # self.header().setSectionResizeMode(Name, QHeaderView.ResizeToContents)
        self.header().setSectionResizeMode(Description, QHeaderView.Stretch)
        # self.header().setSectionResizeMode(Reference, QHeaderView.ResizeToContents)

        self.addMeasure("ROA", "Return on Assets", "")
        self.addMeasure("ROE", "Return on Equity", "")
        self.addMeasure("Firm Size", "Logarithm of total assets", "")
        self.addMeasure("Z-score", "Z-score", "")
        self.addMeasure("Asset Tangibility", "Logarithm of PPENT", "")

    def addMeasure(self, name, description, reference, contributor="Mingze Gao"):
        self.model.insertRow(0)
        self.model.setData(self.model.index(0, Name), name)
        self.model.setData(self.model.index(0, Description), description)
        self.model.setData(self.model.index(0, Reference), reference)
        self.model.setData(self.model.index(0, Contributor), contributor)
        self.sortByColumn(0, Qt.AscendingOrder)
