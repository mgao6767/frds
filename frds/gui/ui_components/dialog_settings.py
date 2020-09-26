from PyQt5 import QtWidgets
from .generated_py_files.Ui_DialogSettings import Ui_DialogSettings
from frds.utils.settings import (
    save_data_source_credentials,
    read_data_source_credentials,
    save_general_settings,
    read_general_settings,
)


class DialogSettings(QtWidgets.QDialog, Ui_DialogSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setupUi(self)

        self.initOtherComponents()
        self.connect_signals()

    def initOtherComponents(self):
        generalSettings = read_general_settings()
        credentialSettings = read_data_source_credentials()
        self.lineEditDataDirectory.setText(generalSettings.get("data_dir"))
        self.lineEditResultDirectory.setText(generalSettings.get("result_dir"))
        self.lineEditUsernameWRDS.setText(credentialSettings.get("wrds_username"))
        self.lineEditPsswordWRDS.setText(credentialSettings.get("wrds_password"))
        self.lineEditUsernameDSS.setText(credentialSettings.get("dss_username"))
        self.lineEditPasswordDSS.setText(credentialSettings.get("dss_password"))

    def connect_signals(self):
        btn_apply = self.buttonBox.button(QtWidgets.QDialogButtonBox.Apply)
        btn_apply.clicked.connect(self.save_settings)
        self.buttonBox.accepted.connect(self.save_settings)

    def save_settings(self):
        config_general = dict(
            data_dir=self.lineEditDataDirectory.text(),
            result_dir=self.lineEditResultDirectory.text(),
        )
        config_credentials = dict(
            wrds_username=self.lineEditUsernameWRDS.text(),
            wrds_password=self.lineEditPsswordWRDS.text(),
            dss_username=self.lineEditUsernameDSS.text(),
            dss_password=self.lineEditPasswordDSS.text(),
        )
        save_data_source_credentials(config_credentials)
        save_general_settings(config_general)

