#!/usr/bin/env python
# coding=utf-8

"""
@Time: 11/2/2023 11:22 AM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: myDimenDialog.py
@Software: PyCharm
"""
import sys
from PyQt5.QtCore import pyqtSlot
from QDataLabel_select2 import Ui_DataLabel_select2
from PyQt5.QtWidgets import QApplication,QDialog

class QmyLabelDialog(QDialog):
    def __init__(self, parent=None, column=None):
        super().__init__(parent)  # Call the parent class constructor to create a form
        self.ui = Ui_DataLabel_select2()  # Creating UI Objects
        self.ui.setupUi(self)  # Constructing the UI interface

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(column)
        self.index = -1


    def __del__(self):  # Destructor
        print("QmyDialogHeaders: The object was deleted")

    @pyqtSlot(bool)
    def on_Yes_clicked(self):
        self.index = self.ui.comboBox.currentIndex()

    @pyqtSlot(bool)
    def on_Cancel_clicked(self):
        self.index = -1


##  ====================Form Tester========================
if  __name__ == "__main__":         # For testing the current form
   app = QApplication(sys.argv)     # Creating GUI Applications
   form=QmyLabelDialog(column=['1','2','3','4','5'])          # Create a form
   form.show()
   sys.exit(app.exec_())

