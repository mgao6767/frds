import sys
from importlib.resources import read_binary
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QPixmap
from frds.ui.components.main_window import MainWindow
import frds.ui.designs


class FRDSApplication:
    """The FRDS application class that holds all UI components together"""

    def __init__(self, *args, **kwargs):
        self.app = QApplication(*args, **kwargs)
        self.main_window = MainWindow()

        # Set window's icon
        logo = QPixmap()
        logo.loadFromData(read_binary(frds.ui.designs, "frds_icon.png"))
        self.app.setWindowIcon(QIcon(logo))

    def run(self):
        """Display main window and start running
        """
        self.main_window.show()
        self.connect_signals()
        self.app.exec()
        # Code below executed when the main window is closed

    def connect_signals(self):
        """Connect all signals
        """


def run():
    """Entry point for starting the application
    """
    FRDSApplication(sys.argv).run()


if __name__ == "__main__":
    run()
