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
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_DataLabel_select2()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(column)
        self.index = -1


    def __del__(self):  # 析构函数
        print("QmyDialogHeaders 对象被删除了")

    @pyqtSlot(bool)
    def on_Yes_clicked(self):
        self.index = self.ui.comboBox.currentIndex()

    @pyqtSlot(bool)
    def on_Cancel_clicked(self):
        self.index = -1


##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyLabelDialog(column=['1','2','3','4','5'])          #创建窗体
   form.show()
   sys.exit(app.exec_())

