import sys
import os
import webbrowser
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QDialogButtonBox,
    QWidget,
    QDialog,
    QStatusBar,
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
)
from PyQt5.QtGui import QIcon
from frds import credentials, data_dir, result_dir
from frds.gui import (
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_WIDTH,
    frds_title,
    style_name,
    intro_html,
    intro_home,
    homepage,
)
from .tab_banking import TabBanking
from .tab_corporate_finance import TabCorporateFinance
from .tab_market_microstructure import TabMarketMicrostructure


class App(QMainWindow):
    """Main entrance of the application"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(frds_title)
        self.setFixedSize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
        self.threadpool = QThreadPool()
        self.stopped = True
        QApplication.setStyle(style_name)
        self.setCentralWidget(GUI(self))
        self.show()


class GUI(QWidget):
    """Main GUI"""

    def __init__(self, parent):
        super(GUI, self).__init__(parent)
        self.app = parent

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Status: Ready.")

        # Intro layout
        intro_layout = QVBoxLayout()
        (lbl := QLabel(intro_home)).setOpenExternalLinks(True)
        intro_layout.addWidget(lbl)
        buttons = QHBoxLayout()
        intro_layout.addLayout(buttons)
        homepage_button = QPushButton("Homepage")
        abt_button = QPushButton("About")
        config_button = QPushButton("Settings")
        buttons.addWidget(homepage_button)
        buttons.addWidget(abt_button)
        buttons.addWidget(config_button)
        homepage_button.clicked.connect(lambda: webbrowser.open(homepage))
        abt_button.clicked.connect(self.on_about_btn_clicked)
        config_button.clicked.connect(self.on_config_btn_clicked)

        self.tabs = QTabWidget()
        self.tab_corporate_finance = TabCorporateFinance(self)
        self.tab_banking = TabBanking(self)
        self.tab_market_microstructure = TabMarketMicrostructure(self)
        self.tabs.addTab(self.tab_corporate_finance, "Corporate Finance")
        self.tabs.addTab(self.tab_banking, "Banking")
        self.tabs.addTab(
            self.tab_market_microstructure, "Market Microstructure"
        )

        # Main layout
        main_layout = QGridLayout()
        main_layout.addLayout(intro_layout, 0, 0, 1, 4)
        main_layout.addWidget(self.tabs, 1, 0, 2, 4)

        (layout := QVBoxLayout()).addLayout(main_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

    def on_about_btn_clicked(self) -> None:
        about = DialogAbout(self)
        about.exec_()

    def on_config_btn_clicked(self) -> None:
        config = DialogConfig(self)
        config.exec_()


class DialogAbout(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(int(MAIN_WINDOW_WIDTH * 0.8))
        # Intro layout
        (intro_label := QLabel(intro_html)).setOpenExternalLinks(True)
        (intro_layout := QVBoxLayout()).addWidget(intro_label)
        btn = QDialogButtonBox.Ok
        btn_box = QDialogButtonBox(btn)
        intro_layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        self.setLayout(intro_layout)
        self.setWindowTitle("About FRDS")


class DialogConfig(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(int(MAIN_WINDOW_WIDTH * 0.8))
        layout = QVBoxLayout()
        layout.addLayout(self.create_configuration())
        self.setLayout(layout)
        btn = QDialogButtonBox.Save | QDialogButtonBox.Cancel
        btn_box = QDialogButtonBox(btn)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.on_save_btn_clicked)
        btn_box.rejected.connect(self.reject)

    def on_save_btn_clicked(self) -> None:
        # TODO: save settings to `config.ini`
        self.accept()

    def create_configuration(self):
        self.frds_data_dir = QLineEdit(str(data_dir))
        self.frds_result_dir = QLineEdit(str(result_dir))
        self.wrds_username_qline = QLineEdit(credentials.get("wrds_username"))
        self.wrds_password_qline = QLineEdit(credentials.get("wrds_password"))
        self.wrds_password_qline.setEchoMode(QLineEdit.Password)
        self.dss_username_qline = QLineEdit(credentials.get("dss_username"))
        self.dss_password_qline = QLineEdit(credentials.get("dss_password"))
        self.dss_password_qline.setEchoMode(QLineEdit.Password)

        layout = QGridLayout()
        layout.addWidget(QLabel("Data directory"), 0, 0, 1, 1)
        layout.addWidget(QLabel("Result directory"), 1, 0, 1, 1)
        layout.addWidget(self.frds_data_dir, 0, 1, 1, 2)
        layout.addWidget(self.frds_result_dir, 1, 1, 1, 2)
        general = QGroupBox("General Settings")
        general.setLayout(layout)

        login_layout = QGridLayout()
        login_layout.addWidget(QLabel("WRDS Username"), 0, 0, 1, 1)
        login_layout.addWidget(QLabel("WRDS Password"), 1, 0, 1, 1)
        login_layout.addWidget(QLabel("DSS Username"), 2, 0, 1, 1)
        login_layout.addWidget(QLabel("DSS Username"), 3, 0, 1, 1)
        login_layout.addWidget(self.wrds_username_qline, 0, 1, 1, 2)
        login_layout.addWidget(self.wrds_password_qline, 1, 1, 1, 2)
        login_layout.addWidget(self.dss_username_qline, 2, 1, 1, 2)
        login_layout.addWidget(self.dss_password_qline, 3, 1, 1, 2)
        login = QGroupBox("Database Credentials")
        login.setLayout(login_layout)

        configuration_layout = QHBoxLayout()
        configuration_layout.addWidget(general)
        configuration_layout.addWidget(login)
        return configuration_layout


if __name__ == "__main__":
    app = QApplication([])
    script_dir = os.path.dirname(os.path.realpath(__file__))
    icon_path = os.path.join(script_dir, "frds_icon.png")
    app.setWindowIcon(QIcon(icon_path))
    ex = App()
    sys.exit(app.exec_())
