# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'advancedSetting.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
    QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_AdvancedSetting(object):
    def setupUi(self, AdvancedSetting):
        if not AdvancedSetting.objectName():
            AdvancedSetting.setObjectName(u"AdvancedSetting")
        AdvancedSetting.resize(600, 600)
        AdvancedSetting.setBaseSize(QSize(600, 600))
        self.verticalLayout = QVBoxLayout(AdvancedSetting)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.aiSettingGroupBox = QGroupBox(AdvancedSetting)
        self.aiSettingGroupBox.setObjectName(u"aiSettingGroupBox")
        self.verticalLayout_2 = QVBoxLayout(self.aiSettingGroupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.as0_info = QLabel(self.aiSettingGroupBox)
        self.as0_info.setObjectName(u"as0_info")

        self.verticalLayout_2.addWidget(self.as0_info)

        self.as1_imageDir = QHBoxLayout()
        self.as1_imageDir.setObjectName(u"as1_imageDir")
        self.label_10 = QLabel(self.aiSettingGroupBox)
        self.label_10.setObjectName(u"label_10")

        self.as1_imageDir.addWidget(self.label_10)

        self.imageDirEdit = QLineEdit(self.aiSettingGroupBox)
        self.imageDirEdit.setObjectName(u"imageDirEdit")

        self.as1_imageDir.addWidget(self.imageDirEdit)

        self.as1_imageDir.setStretch(0, 1)
        self.as1_imageDir.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as1_imageDir)

        self.as2_input = QHBoxLayout()
        self.as2_input.setObjectName(u"as2_input")
        self.label_9 = QLabel(self.aiSettingGroupBox)
        self.label_9.setObjectName(u"label_9")

        self.as2_input.addWidget(self.label_9)

        self.inputEdit = QLineEdit(self.aiSettingGroupBox)
        self.inputEdit.setObjectName(u"inputEdit")

        self.as2_input.addWidget(self.inputEdit)

        self.as2_input.setStretch(0, 1)
        self.as2_input.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as2_input)

        self.as3_output = QHBoxLayout()
        self.as3_output.setObjectName(u"as3_output")
        self.label_8 = QLabel(self.aiSettingGroupBox)
        self.label_8.setObjectName(u"label_8")

        self.as3_output.addWidget(self.label_8)

        self.outputEdit = QLineEdit(self.aiSettingGroupBox)
        self.outputEdit.setObjectName(u"outputEdit")

        self.as3_output.addWidget(self.outputEdit)

        self.as3_output.setStretch(0, 1)
        self.as3_output.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as3_output)

        self.as4_sequenceLength = QHBoxLayout()
        self.as4_sequenceLength.setObjectName(u"as4_sequenceLength")
        self.label_7 = QLabel(self.aiSettingGroupBox)
        self.label_7.setObjectName(u"label_7")

        self.as4_sequenceLength.addWidget(self.label_7)

        self.sequenceLengthEdit = QLineEdit(self.aiSettingGroupBox)
        self.sequenceLengthEdit.setObjectName(u"sequenceLengthEdit")

        self.as4_sequenceLength.addWidget(self.sequenceLengthEdit)

        self.as4_sequenceLength.setStretch(0, 1)
        self.as4_sequenceLength.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as4_sequenceLength)

        self.as5_units = QHBoxLayout()
        self.as5_units.setObjectName(u"as5_units")
        self.label_6 = QLabel(self.aiSettingGroupBox)
        self.label_6.setObjectName(u"label_6")

        self.as5_units.addWidget(self.label_6)

        self.unitsEdit = QLineEdit(self.aiSettingGroupBox)
        self.unitsEdit.setObjectName(u"unitsEdit")

        self.as5_units.addWidget(self.unitsEdit)

        self.as5_units.setStretch(0, 1)
        self.as5_units.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as5_units)

        self.as6_epochs = QHBoxLayout()
        self.as6_epochs.setObjectName(u"as6_epochs")
        self.label = QLabel(self.aiSettingGroupBox)
        self.label.setObjectName(u"label")

        self.as6_epochs.addWidget(self.label)

        self.epochsEdit = QLineEdit(self.aiSettingGroupBox)
        self.epochsEdit.setObjectName(u"epochsEdit")

        self.as6_epochs.addWidget(self.epochsEdit)

        self.as6_epochs.setStretch(0, 1)
        self.as6_epochs.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as6_epochs)

        self.as7_batchSize = QHBoxLayout()
        self.as7_batchSize.setObjectName(u"as7_batchSize")
        self.label_2 = QLabel(self.aiSettingGroupBox)
        self.label_2.setObjectName(u"label_2")

        self.as7_batchSize.addWidget(self.label_2)

        self.batchSizeEdit = QLineEdit(self.aiSettingGroupBox)
        self.batchSizeEdit.setObjectName(u"batchSizeEdit")

        self.as7_batchSize.addWidget(self.batchSizeEdit)

        self.as7_batchSize.setStretch(0, 1)
        self.as7_batchSize.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as7_batchSize)

        self.as8_outlierCheck = QHBoxLayout()
        self.as8_outlierCheck.setObjectName(u"as8_outlierCheck")
        self.outlierCheckBox = QCheckBox(self.aiSettingGroupBox)
        self.outlierCheckBox.setObjectName(u"outlierCheckBox")
        self.outlierCheckBox.setLayoutDirection(Qt.LeftToRight)

        self.as8_outlierCheck.addWidget(self.outlierCheckBox)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.as8_outlierCheck.addItem(self.horizontalSpacer)


        self.verticalLayout_2.addLayout(self.as8_outlierCheck)

        self.as9_outlierColumn = QHBoxLayout()
        self.as9_outlierColumn.setObjectName(u"as9_outlierColumn")
        self.label_3 = QLabel(self.aiSettingGroupBox)
        self.label_3.setObjectName(u"label_3")

        self.as9_outlierColumn.addWidget(self.label_3)

        self.outlierColumnEdit = QLineEdit(self.aiSettingGroupBox)
        self.outlierColumnEdit.setObjectName(u"outlierColumnEdit")

        self.as9_outlierColumn.addWidget(self.outlierColumnEdit)

        self.as9_outlierColumn.setStretch(0, 1)
        self.as9_outlierColumn.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as9_outlierColumn)

        self.as10_outlierWeight = QHBoxLayout()
        self.as10_outlierWeight.setObjectName(u"as10_outlierWeight")
        self.label_4 = QLabel(self.aiSettingGroupBox)
        self.label_4.setObjectName(u"label_4")

        self.as10_outlierWeight.addWidget(self.label_4)

        self.outlierWeightEdit = QLineEdit(self.aiSettingGroupBox)
        self.outlierWeightEdit.setObjectName(u"outlierWeightEdit")

        self.as10_outlierWeight.addWidget(self.outlierWeightEdit)

        self.as10_outlierWeight.setStretch(0, 1)
        self.as10_outlierWeight.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as10_outlierWeight)

        self.as11_verbose = QHBoxLayout()
        self.as11_verbose.setObjectName(u"as11_verbose")
        self.label_5 = QLabel(self.aiSettingGroupBox)
        self.label_5.setObjectName(u"label_5")

        self.as11_verbose.addWidget(self.label_5)

        self.verboseEdit = QLineEdit(self.aiSettingGroupBox)
        self.verboseEdit.setObjectName(u"verboseEdit")

        self.as11_verbose.addWidget(self.verboseEdit)

        self.as11_verbose.setStretch(0, 1)
        self.as11_verbose.setStretch(1, 4)

        self.verticalLayout_2.addLayout(self.as11_verbose)


        self.verticalLayout.addWidget(self.aiSettingGroupBox)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.buttonBox = QDialogButtonBox(AdvancedSetting)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(AdvancedSetting)
        self.buttonBox.accepted.connect(AdvancedSetting.accept)
        self.buttonBox.rejected.connect(AdvancedSetting.reject)

        QMetaObject.connectSlotsByName(AdvancedSetting)
    # setupUi

    def retranslateUi(self, AdvancedSetting):
        AdvancedSetting.setWindowTitle(QCoreApplication.translate("AdvancedSetting", u"Advanced Setting", None))
        self.aiSettingGroupBox.setTitle(QCoreApplication.translate("AdvancedSetting", u"AI HyperParameter", None))
        self.as0_info.setText(QCoreApplication.translate("AdvancedSetting", u"\uac00\ub2a5\ud558\uba74 \uac74\ub4e4\uc9c0 \uc54a\ub294 \uac83\uc774 \uc88b\uc2b5\ub2c8\ub2e4.", None))
        self.label_10.setText(QCoreApplication.translate("AdvancedSetting", u"Image Directory", None))
        self.label_9.setText(QCoreApplication.translate("AdvancedSetting", u"Input (List)", None))
        self.label_8.setText(QCoreApplication.translate("AdvancedSetting", u"Output (List)", None))
        self.label_7.setText(QCoreApplication.translate("AdvancedSetting", u"Sequence Length", None))
        self.label_6.setText(QCoreApplication.translate("AdvancedSetting", u"Units", None))
        self.label.setText(QCoreApplication.translate("AdvancedSetting", u"Epochs", None))
        self.label_2.setText(QCoreApplication.translate("AdvancedSetting", u"Batch Size", None))
        self.outlierCheckBox.setText(QCoreApplication.translate("AdvancedSetting", u"Outlier Check?", None))
        self.label_3.setText(QCoreApplication.translate("AdvancedSetting", u"Outlier Column", None))
        self.label_4.setText(QCoreApplication.translate("AdvancedSetting", u"Outlier Weight (IQR)", None))
        self.label_5.setText(QCoreApplication.translate("AdvancedSetting", u"Verbose", None))
    # retranslateUi

