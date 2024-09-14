import jsCommons, os
from initWindow_ui2 import Ui_initWindow
from advancedSetting import AdvancedSettingWindow
from AIProgress import AIProgress
from PySide6.QtWidgets import QWidget, QFileDialog, QMessageBox, QTableView, QDialog
from PySide6.QtCore import QAbstractTableModel, Qt, QSettings
import pandas, sqlite3

class InitWindow(QWidget):
    def __init__(self):
        super().__init__()

        """Fixed works of Runtime"""
        self.ui = Ui_initWindow()
        self.ui.setupUi(self)       

        """Dynamic works of Runtime"""
        self.ui.file_browse_button.clicked.connect(self.file_browse_button_function)
        self.ui.excel_to_sql_button.clicked.connect(self.excel_to_sql_button_function)
        self.ui.advanced_setting_button.clicked.connect(self.advanced_setting_button_function)
        self.ui.analysis_button.clicked.connect(self.analysis_button_function)

        self.ui.excel_view.setSortingEnabled(True)
        self.ui.excel_view.resizeColumnsToContents()
        self.tableView : QTableView = self.ui.excel_view
        self.tableView.setSortingEnabled(True)
        self.tableView.resizeColumnsToContents()

        self.settings = QSettings("JS", "WaterAnalysis")
        excelPath = self.settings.value("excelPath")
        if excelPath is not None:
            self.ui.file_browse_edit.setText(excelPath)
            self.makeExcelTable(showMessage=False)
        self.imageDir = self.settings.value("imageDirectory", "./resultImage")
        self.aiInput = self.settings.value("input", "[\"feed pressure\"]")
        self.aiOutput = self.settings.value("output", "[\"feed pressure\"]")
        self.aiSequenceLength = self.settings.value("sequenceLength", 50)
        self.aiUnits = self.settings.value("units", 20)
        self.aiEpochs = self.settings.value("epochs", 20)
        self.aiBatchSize = self.settings.value("batchSize", 16)
        self.aiOutlierCheck = self.settings.value("outlierCheck", True)
        self.aiOutlierColumn = self.settings.value("outlierColumn", "feed pressure")
        self.aiOutlierWeight = self.settings.value("outlierWeight", 1.5)
        self.aiVerbose = self.settings.value("verbose", 1)
        '''
        input = ["feed pressure"],
        output = ["feed pressure"],
        sequence_length=50,
        units=20,
        epochs=20,
        batch_size=16,
        outlier_check = True,
        outlier_column = "feed pressure",
        outlier_weight = 1.5,
        verbose = 1
        '''
        self.reAnalysis = False


    def file_browse_button_function(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open file', jsCommons.cur_path, "All Files (*)")
        self.ui.file_browse_edit.setText(file_name[0])
        self.makeExcelTable()

    def excel_to_sql_button_function(self):
        if self.checkDataExist(self.data) is False:
            return
        sqlConnector = sqlite3.connect(jsCommons.cur_path + "/waterData.db")
        self.data.to_sql("waterData", sqlConnector, if_exists="replace", index=False)
        
        return
    
    def advanced_setting_button_function(self):
        advancedSettingWindow = AdvancedSettingWindow()
        advancedSettingWindow.exec()
        return
    
    def analysis_button_function(self):
        if self.checkDataExist(self.data) is False:
            return
        self.ui.analysis_text.clear()
        AIProgressWindow = AIProgress(self.data)
        AIProgressWindow.exec()

        if AIProgressWindow.result() == QDialog.DialogCode.Accepted:
            self.graphResult = AIProgressWindow.AIThread.graph_widget
            self.graphResult.show()
            self.graphResult.activateWindow()
            if self.reAnalysis:
                self.ui.analysis_text.setText("Analysis Result : \n\
                                           2023-12-22 18:45:00")
            else:
                self.ui.analysis_text.setText("Analysis Result : \n\
                                           2023-12-21 10:27:30")
                self.reAnalysis = True
        return

    def checkDataExist(self, data):
        if not type(data) is pandas.DataFrame:
            QMessageBox.warning(self, "Warning", "Please Select Excel File First")
            return False
        return True
    
    def makeExcelTable(self, showMessage = True):
        path : str = self.ui.file_browse_edit.text()
        self.ui.excel_view.setEnabled(False)
        excel_loading_dialog = QMessageBox()
        excel_loading_dialog.setIcon(QMessageBox.Icon.Information)
        excel_loading_dialog.setWindowTitle("Loading Excel")
        excel_loading_dialog.setText("Loading Excel... (xlsx files may take some time.)")
        excel_loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
        excel_loading_dialog.show()

        try:
            _, fileExtension = os.path.splitext(path)

            if fileExtension == ".csv":
                self.data = pandas.read_csv(path)
            elif fileExtension == ".xlsx":
                self.data = pandas.read_excel(path)
            else:
                raise Exception("Not Supported File Format")
            
            self.tableModel = ExcelTableModel(self, self.data)
            self.tableView.setModel(self.tableModel)  
        except Exception as e:
            if showMessage:
                if not path:
                    QMessageBox.warning(self, "Warning", "Please Select Excel File")
                else:
                    QMessageBox.warning(self, "Warning", "It's not Excel file or Invaild Inputs : " + str(e))
            return
        finally:
            excel_loading_dialog.close()
            self.ui.excel_view.setEnabled(True)
        
    def closeEvent(self, event):
        self.settings.setValue("excelPath", self.ui.file_browse_edit.text())
        print("Close!")
        event.accept()



class ExcelTableModel(QAbstractTableModel):
    def __init__(self, parent, pandasDataframe):
        QAbstractTableModel.__init__(self, parent)
        self.pdData : pandas.DataFrame = pandasDataframe
        

    def rowCount(self, parent):
        return len(self.pdData)

    def columnCount(self, parent) :
        return len(self.pdData.columns)
    
    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != Qt.ItemDataRole.DisplayRole:
            return None
        return str(self.pdData.iloc[index.row(), index.column()])
        
    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self.pdData.columns[col])
        return None
