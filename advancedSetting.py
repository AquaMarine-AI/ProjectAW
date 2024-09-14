import ast, os
from PySide6 import QtWidgets
from PySide6.QtCore import QSettings
from advancedSetting_ui import Ui_AdvancedSetting

class AdvancedSettingWindow(QtWidgets.QDialog):
    aiSettings = ("imageDirectory", "input", "output", "sequenceLength", "units", "epochs", "batchSize", "outlierCheck", "outlierColumn", "outlierWeight", "verbose")

    def __init__(self):
        super().__init__()

        """Fixed works of Runtime"""
        self.ui = Ui_AdvancedSetting()
        self.ui.setupUi(self)       # self = parent of Ui_AdvancedSetting

        """Dynamic works of Runtime"""
        #load Setting Data and Write Data to Widgets (TextEdit ... etc)
        self.aiSettings = QSettings("JS", "WaterAnalysis")
        self.ui.imageDirEdit.setText(self.aiSettings.value("imageDirectory", "./resultImage"))
        self.ui.inputEdit.setText(self.aiSettings.value("input", "[\"feed pressure\"]"))
        self.ui.outputEdit.setText(self.aiSettings.value("output", "[\"feed pressure\"]"))
        self.ui.sequenceLengthEdit.setText(self.aiSettings.value("sequenceLength", "50"))
        self.ui.unitsEdit.setText(self.aiSettings.value("units", "20"))
        self.ui.epochsEdit.setText(self.aiSettings.value("epochs", "20"))
        self.ui.batchSizeEdit.setText(self.aiSettings.value("batchSize", "16"))
        self.ui.outlierCheckBox.setChecked(self.aiSettings.value("outlierCheck", "True") == "True")
        self.ui.outlierColumnEdit.setText(self.aiSettings.value("outlierColumn", "feed pressure"))
        self.ui.outlierWeightEdit.setText(self.aiSettings.value("outlierWeight", "1.5"))
        self.ui.verboseEdit.setText(self.aiSettings.value("verbose", "1"))


    #predefined function in Ui_AdvancedSetting
    def accept(self) -> None:
        
        #Check Setting Data Validation
        try:
            index = 0
            if not os.path.exists(self.ui.imageDirEdit.text()):
                raise Exception("imageDirectory is not Exists")
            index = 1
            ast.literal_eval(self.ui.inputEdit.text())
            index = 2
            ast.literal_eval(self.ui.outputEdit.text())
            index = 3
            int(self.ui.sequenceLengthEdit.text())
            index = 4
            int(self.ui.unitsEdit.text())
            index = 5
            int(self.ui.epochsEdit.text())
            index = 6
            int(self.ui.batchSizeEdit.text())
            index = 9
            float(self.ui.outlierWeightEdit.text())
            index = 10
            int(self.ui.verboseEdit.text())
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please Check " + AdvancedSettingWindow.aiSettings[index] + " Value")
            print(e)
            return
        
        #save Setting Data from Widgets (TextEdit ... etc)
        self.aiSettings.setValue("imageDirectory", self.ui.imageDirEdit.text())
        self.aiSettings.setValue("input", self.ui.inputEdit.text())
        self.aiSettings.setValue("output", self.ui.outputEdit.text())
        self.aiSettings.setValue("sequenceLength", self.ui.sequenceLengthEdit.text())
        self.aiSettings.setValue("units", self.ui.unitsEdit.text())
        self.aiSettings.setValue("epochs", self.ui.epochsEdit.text())
        self.aiSettings.setValue("batchSize", self.ui.batchSizeEdit.text())
        self.aiSettings.setValue("outlierCheck", "True" if self.ui.outlierCheckBox.isChecked() else "False")
        self.aiSettings.setValue("outlierColumn", self.ui.outlierColumnEdit.text())
        self.aiSettings.setValue("outlierWeight", self.ui.outlierWeightEdit.text())
        self.aiSettings.setValue("verbose", self.ui.verboseEdit.text())

        print("Advanced Setting Accepted")
        return super().accept()
    
    #predefined function in Ui_AdvancedSetting
    def reject(self) -> None:
        print("Advanced Setting Rejected")
        return super().reject()

    


