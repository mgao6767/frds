"""Preferences class"""

from importlib.resources import open_text
from PyQt5 import uic
import frds.ui.designs

ui = open_text(frds.ui.designs, "Preferences.ui")


class Preferences(*uic.loadUiType(ui)):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setupUi(self)

