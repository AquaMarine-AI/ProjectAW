# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'initWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTableView, QVBoxLayout,
    QWidget)

class Ui_initWindow(object):
    def setupUi(self, initWindow):
        if not initWindow.objectName():
            initWindow.setObjectName(u"initWindow")
        initWindow.resize(800, 600)
        self.horizontalLayout_2 = QHBoxLayout(initWindow)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox = QGroupBox(initWindow)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.path_label = QLabel(self.groupBox)
        self.path_label.setObjectName(u"path_label")

        self.horizontalLayout.addWidget(self.path_label)

        self.file_browse_edit = QLineEdit(self.groupBox)
        self.file_browse_edit.setObjectName(u"file_browse_edit")

        self.horizontalLayout.addWidget(self.file_browse_edit)

        self.file_browse_button = QPushButton(self.groupBox)
        self.file_browse_button.setObjectName(u"file_browse_button")

        self.horizontalLayout.addWidget(self.file_browse_button)

        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.excel_view = QTableView(self.groupBox)
        self.excel_view.setObjectName(u"excel_view")

        self.verticalLayout_3.addWidget(self.excel_view)

        self.excel_to_sql_button = QPushButton(self.groupBox)
        self.excel_to_sql_button.setObjectName(u"excel_to_sql_button")

        self.verticalLayout_3.addWidget(self.excel_to_sql_button)


        self.verticalLayout_2.addLayout(self.verticalLayout_3)

        self.verticalLayout_2.setStretch(1, 1)

        self.horizontalLayout_2.addWidget(self.groupBox)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.SettingGroup = QGroupBox(initWindow)
        self.SettingGroup.setObjectName(u"SettingGroup")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SettingGroup.sizePolicy().hasHeightForWidth())
        self.SettingGroup.setSizePolicy(sizePolicy)
        self.horizontalLayout_3 = QHBoxLayout(self.SettingGroup)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.update_setting_button = QPushButton(self.SettingGroup)
        self.update_setting_button.setObjectName(u"update_setting_button")

        self.horizontalLayout_3.addWidget(self.update_setting_button)

        self.advanced_setting_button = QPushButton(self.SettingGroup)
        self.advanced_setting_button.setObjectName(u"advanced_setting_button")

        self.horizontalLayout_3.addWidget(self.advanced_setting_button)


        self.verticalLayout.addWidget(self.SettingGroup)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.analysis_button = QPushButton(initWindow)
        self.analysis_button.setObjectName(u"analysis_button")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.analysis_button.sizePolicy().hasHeightForWidth())
        self.analysis_button.setSizePolicy(sizePolicy1)
        self.analysis_button.setMinimumSize(QSize(14, 28))
        self.analysis_button.setStyleSheet(u"background-color: rgb(85, 170, 255);")

        self.verticalLayout.addWidget(self.analysis_button)


        self.horizontalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(initWindow)

        QMetaObject.connectSlotsByName(initWindow)
    # setupUi

    def retranslateUi(self, initWindow):
        initWindow.setWindowTitle(QCoreApplication.translate("initWindow", u"Water Analysis", None))
        self.groupBox.setTitle(QCoreApplication.translate("initWindow", u"\uc5d1\uc140 \ub370\uc774\ud130", None))
        self.path_label.setText(QCoreApplication.translate("initWindow", u"Path:", None))
        self.file_browse_button.setText(QCoreApplication.translate("initWindow", u"Browse..", None))
        self.excel_to_sql_button.setText(QCoreApplication.translate("initWindow", u"Extract as SQL", None))
        self.SettingGroup.setTitle(QCoreApplication.translate("initWindow", u"Settings", None))
        self.update_setting_button.setText(QCoreApplication.translate("initWindow", u"\uc2e4\uc2dc\uac04 \uac31\uc2e0 \uc124\uc815", None))
        self.advanced_setting_button.setText(QCoreApplication.translate("initWindow", u"Advanced...", None))
        self.analysis_button.setText(QCoreApplication.translate("initWindow", u"Analysis", None))
    # retranslateUi

