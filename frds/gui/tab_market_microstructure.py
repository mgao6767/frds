from .base import TabBase
import frds.measures


class TabMarketMicrostructure(TabBase):
    def __init__(self, parent):
        super().__init__(parent, frds.measures.Category.MARKET_MICROSTRUCTURE)

    def on_start_btn_clicked(self) -> None:
        """Start data loading dialog"""
        # self.app.stopped = False
        # self.measure_selection.setDisabled(True)
        # self.start_btn.setDisabled(True)
        # self.start_btn.setText("Running")
        # measures_to_estimate = []
        # for i in range(self.list_of_measures.count()):
        #     item = self.list_of_measures.item(i)
        #     if item.checkState() == Qt.Checked:
        #         measures_to_estimate.append(item.text())
        # worker = Worker(
        #     frds.run.main, measures_to_estimate=measures_to_estimate, gui=True,
        # )
        # worker.signals.finished.connect(self.on_completed)
        # worker.signals.progress.connect(self.update_progress)
        # worker.signals.error.connect(self.update_progress)
        # self.app.threadpool.start(worker)
        pass
        # data_loading_dialog = DialogAbout(self)
        # data_loading_dialog.exec_()
