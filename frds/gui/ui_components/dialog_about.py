from PyQt5 import QtWidgets
from .generated_py_files.Ui_DialogAbout import Ui_DialogAbout

from frds.settings import ABOUT_FRDS


class DialogAbout(QtWidgets.QDialog, Ui_DialogAbout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setupUi(self)

        self.textBrowser.setMarkdown(ABOUT_FRDS)
