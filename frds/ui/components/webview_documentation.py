from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from frds.settings import FRDS_MEASURES_PAGE

class Documentation(QWebEngineView):


    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.home = QWebEnginePage()
        self.home.setUrl(QUrl(FRDS_MEASURES_PAGE))

    def displayDoc(self):
        self.setPage(self.home)