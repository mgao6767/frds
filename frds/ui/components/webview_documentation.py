from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from frds.settings import FRDS_HOME_PAGE, FRDS_MEASURES_PAGE

class WebEnginePage(QWebEnginePage):

    def acceptNavigationRequest(self, url,  _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            if FRDS_HOME_PAGE not in url.url():
                QDesktopServices.openUrl(url)
                return False
        return True

class Documentation(QWebEngineView):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.home = WebEnginePage()
        self.home.setUrl(QUrl(FRDS_MEASURES_PAGE))

    def displayDoc(self):
        self.setPage(self.home)