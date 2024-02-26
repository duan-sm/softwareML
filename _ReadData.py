#!/usr/bin/env python
# coding=utf-8

"""
@Time: 10/31/2023 11:38 AM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Outlier.py
@Software: PyCharm
"""


import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt
import math
from FunctionDataPreprocessing import draw_cross_curves
def DataInputBtn(self):
    print('DataInputBtn|'*10)
    self.ui.NondimenBtn.setEnabled(True)
    self.ui.DimenComBtn.setEnabled(True)
    self.ui.NondimenBtn.setEnabled(True)
    self.ui.FeatureFig.figure.clear()  # 清除图表
    self.ui.FeatureFig.figure.clf()
    # self.ui.FeatureFig.figure.cla()
    self.ui.OtherFig.figure.clear()  # 清除图表
    self.ui.OtherFig.figure.clf()
    # self.ui.OtherFig.figure.cla()
    self.ui.FeatureFig2.figure.clear()  # 清除图表
    self.ui.FeatureFig2.figure.clf()
    # self.ui.FeatureFig2.figure.cla()
    self.ui.Fig3.figure.clear()  # 清除图表
    self.ui.Fig3.figure.clf()
    # self.ui.Fig3.figure.cla()
    self.ui.LossFig.figure.clear()  # 清除图表
    self.ui.LossFig.figure.clf()
    # self.ui.LossFig.figure.cla()
    self.ui.ResultsText1.clear()
    self.ui.StartWD.setText('0')
    self.ui.EndWD.setText('0')
    self.ui.StartWD.setEnabled(True)
    self.ui.EndWD.setEnabled(True)
    self.ui.DataOutlierDraw1.setCheckable(True)
    self.ui.DataOutlierDraw2.setCheckable(False)
    self.ui.ResultsText2.setText('It shows here'
                                 ':  Variance results/correlation results/chi-square test results/mutual information results'
                                 '  (Chi-square/HE tests require target features to perform, and other processes do not'
                                 ' require target features.)  (Here all processing results are saved by default, and'
                                 ' the save button represents the initial use of the data of the current step for subsequent'
                                 ' data)  ')

    # self.ui.DataReplaceBtn.setEnabled(True)
    # self.ui.DataReplaceBtn2.setEnabled(True)
    self.ui.tabWidget.setTabEnabled(3, True)
    self.Wellname = self.ui.WellName.text()
    self.size_ = 0
    self._DimDialog = None
    self.all_y = None
    self.nondim = 'No'
    self.all_y_index = None
    self.state = -1
    print('0|' * 10)
    if self.Wellname == '例：呼111' or self.Wellname == '请输入井号' or self.Wellname == '' or len(self.Wellname)<1:
        # self.ui.WellName.setPlaceholderText('请输入井号')
        self.ui.WellName.setPlaceholderText('Please input FileName')
        # self.ui.Results.setText('请输入井号，确定后续文件保存路径，否则无法继续')
        self.ui.Results.setText('Enter the folder name to determine the path for saving subsequent files.')
        return 0
    # if not os.path.isdir(self.Wellname):
    #     os.mkdir(self.Wellname)
    if not os.path.exists(r'./%s'%self.Wellname):
        os.makedirs(r'./%s'%self.Wellname)
    # if not os.path.isdir(self.Wellname):
    #     os.mkdir(self.Wellname)
    # fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.'
    #                                        , '数据文件(*.csv *.xlsx *.xls)'
    #                                        )
    print('1|' * 10)
    fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.'
                                           , 'Data file(*.csv *.xlsx *.xls)'
                                           )
    self.ui.InputDataTableView.clear()
    print('2|' * 10)
    index_col = 0
    try:
        if fname:
            self.ui.InputDataTableView.clearContents()
            print('fname', fname)
            # 打开文件
            if fname[-3:]=='csv':
                self.filetype = '.csv'
                self.workbook = pd.read_csv(fname
                                       ,encoding='gb18030'
                                       )
                # print(workbook.iloc[:,0].values)
                workbook_first_col1 = self.workbook.iloc[:, 0].values == np.arange(len(self.workbook))
                workbook_first_col2 = self.workbook.iloc[:,0].values == np.arange(1,len(self.workbook)+1)
                if sum(workbook_first_col1)==len(workbook_first_col1):
                    self.workbook = self.workbook.iloc[:,1:]
                elif sum(workbook_first_col2)==len(workbook_first_col2):
                    self.workbook = self.workbook.iloc[:, 1:]
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='lsx':
                print('3|' * 10)
                self.filetype = '.xlsx'
                self.workbook = pd.read_excel(fname,index_col=index_col)
                print('4|' * 10)
                workbook_first_col1 = self.workbook.iloc[:, 0].values == np.arange(len(self.workbook))
                workbook_first_col2 = self.workbook.iloc[:, 0].values == np.arange(1, len(self.workbook) + 1)
                print('5|' * 10)
                if sum(workbook_first_col1) == len(workbook_first_col1):
                    workbook = self.workbook.iloc[:, 1:]
                elif sum(workbook_first_col2) == len(workbook_first_col2):
                    workbook = self.workbook.iloc[:, 1:]
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='xls':
                print('3|' * 10)
                self.filetype = '.xlsx'
                self.workbook = pd.read_excel(fname,skiprows=1,sheet_name=0)
                print('4|' * 10)
                self.workbook = self.workbook.dropna(axis=0, how='all')
                col_o = self.workbook.columns
                col_c = []
                for i in range(len(col_o)):
                    str_ = ''
                    new_c = col_o[i].split('\n')
                    if len(new_c) > 1:
                        new_1 = str_.join(new_c[:-1])
                        new_2 = new_c[-1]
                        new_c = new_1 + '\n' + new_2
                        col_c.append(new_c)
                    else:
                        col_c.append(new_c[0])
                print('5|' * 10)
                self.workbook.columns = col_c
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            else:
                # self.ui.Results.setText('打开文件格式为%s，不满足要求'%(fname.split('.')[1]))
                self.ui.Results.setText('The file format should be: xls, csv, xlsx' % (fname.split('/')[-1]))
            # 获取整行和整列的值（数组）
            columns = [i.split('\n')[0] for i in self.workbook.columns]
            print(columns)
            rows = len(self.workbook)
            workbook_np = self.workbook.values
            # print(sheet1.nrows)
            self.ui.InputDataTableView.setRowCount(rows)
            self.ui.InputDataTableView.setColumnCount(len(columns))
            for i in range(len(columns)):
                # print(i)
                headerItem = QTableWidgetItem(columns[i])
                font = headerItem.font()
                ##  font.setBold(True)
                font.setPointSize(18)
                headerItem.setFont(font)
                headerItem.setForeground(QBrush(Qt.red))  # 前景色，即文字颜色
                self.ui.InputDataTableView.setHorizontalHeaderItem(i, headerItem)

            self.ui.InputDataTableView.resizeRowsToContents()
            self.ui.InputDataTableView.resizeColumnsToContents()
            # 初始化可以选择绘制的特征
            for num in range(2,5):
                eval('self.ui.SelectFeature%d.clear()' % num)  # 清除图表
                eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # 清除图表

            # 在table中显示数据
            for i in range(rows):
                rowslist = workbook_np[i,:]  # 获取excel每行内容
                # print(rowslist)
                for j in range(len(rowslist)):
                    # 在tablewidget中添加行
                    row = self.ui.InputDataTableView.rowCount()
                    self.ui.InputDataTableView.insertRow(row)
                    # 把数据写入tablewidget中
                    newItem = QTableWidgetItem(str(rowslist[j]))
                    self.ui.InputDataTableView.setItem(i, j, newItem)
            self.ui.InputDataTableView.setAlternatingRowColors(True)
            print(self.size_)
            print(self.workbook.columns)
    except:
        self.ui.Results.setText('Something is wrong')



def merge(self):
    self.Wellname = self.ui.WellName.text()
    if self.Wellname == '例：呼111' or self.Wellname == '请输入井号' or self.Wellname == '' or len(
            self.Wellname) < 1:
        # self.ui.WellName.setPlaceholderText('请输入井号')
        # self.ui.Results.setText('请输入井号，确定后续文件保存路径，否则无法继续')
        self.ui.WellName.setPlaceholderText('Please input FileName')
        self.ui.Results.setText('Enter the folder name to determine the path for saving subsequent files.')
        return 0
    if not os.path.isdir(self.Wellname):
        os.mkdir(self.Wellname)

    try:
        self.size_ = 1
        # fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.'
        #                                        , '数据文件(*.csv *.xlsx *.xls)'
        #                                        )
        fname = QFileDialog.getExistingDirectory(self, "Select Folder", "/"
                                               )
        allFileName = os.listdir(fname)
        if len(allFileName) == 0:
            self.ui.Results.setText('Error: This is an empty folder')
            return 0
        else:
            if allFileName[0][-3:] == 'csv':
                self.filetype = '.csv'
                data = pd.read_csv(fname+'\\'+allFileName[0]
                                   , encoding='gb18030'
                                   )
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            elif allFileName[0][-3:] == 'lsx':
                self.filetype = '.xlsx'
                data = pd.read_excel(fname+'\\'+allFileName[0], sheet_name=0)
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            elif allFileName[0][-3:] == 'xls':
                self.filetype = '.xlsx'
                data = pd.read_excel(fname+'\\'+allFileName[0], sheet_name=0)
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            else:
                # self.ui.Results.setText('打开文件格式为%s，不满足要求' % (fname.split('.')[1]))
                self.ui.Results.setText('The file format should be: xls, csv, xlsx' % allFileName[0])
            print('first is OK')
            for fileIndex in range(1,len(allFileName)):
                print('fileIndex: %d'%fileIndex)
                # 打开文件
                if allFileName[fileIndex][-3:] == 'csv':
                    self.filetype = '.csv'
                    data1 = pd.read_csv(fname+'\\'+allFileName[fileIndex]
                                       , encoding='gb18030'
                                       )
                    # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                elif allFileName[fileIndex][-3:] == 'lsx':
                    self.filetype = '.xlsx'
                    data1 = pd.read_excel(fname+'\\'+allFileName[fileIndex], sheet_name=0)
                    # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                elif allFileName[fileIndex][-3:] == 'xls':
                    self.filetype = '.xlsx'
                    data1 = pd.read_excel(fname+'\\'+allFileName[fileIndex], sheet_name=0)
                    # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                else:
                    # self.ui.Results.setText('打开文件格式为%s，不满足要求' % (fname.split('.')[1]))
                    self.ui.Results.setText('The file format should be: xls, csv, xlsx' % allFileName[fileIndex])
                print('*'*10)
                print(data, data1)
                data = pd.concat([data, data1])
            print('he bing wan cheng')
            if self.filetype == '.csv':
                data.to_csv(fname+'\\'+'allData.csv', encoding='gb18030')
                self.ui.Results.setText('Merge file as: allData.csv')
            elif self.filetype == '.xlsx':
                data.to_excel(fname+'\\'+'allData.xlsx')
                self.ui.Results.setText('Merge file as: allData.xlsx')
            print('Success')

    except:
        print('self.size_ = 0')
        # self.ui.Results.setText('文件错误')
        self.ui.Results.setText('Error: file')
        self.size_ = 0

def btnDelCols(self):
    try:
        row0, col0 = self.workbook.shape
        self.workbook = self.workbook.dropna(axis=1, how='all')
        row1, col1 = self.workbook.shape
        cha = col0 - col1
        self.ui.delColsNum.setText(str(cha))
        # self.ui.Results.setText('完成删除空列，数量为:%d' % cha)
        self.ui.Results.setText('Complete deletion of empty columns, the number is:%d' % cha)
    except:
        # self.ui.Results.setText('请输入数据后再点击删除')
        self.ui.Results.setText('Please input the data and click Delete')

def btnDelRows(self):
    try:
        row0, col0 = self.workbook.shape
        self.workbook = self.workbook.dropna(axis=0, how='any')
        row1, col1 = self.workbook.shape
        cha = row0 - row1
        self.ui.delRowsNum.setText(str(cha))
        # self.ui.Results.setText('完成删除空行，数量为:%d' % cha)
        self.ui.Results.setText('Complete deletion of empty lines, the number is:%d' % cha)
    except:
        # self.ui.Results.setText('请输入数据后再点击删除')
        self.ui.Results.setText('Please input the data and click Delete')

def btnSaveData(self):
    try:
        if self.filetype == '.csv':
            print('file style is .csv')
            self.workbook.to_csv(os.getcwd() + '\\' + self.Wellname + '\\' + '%s-0NoneNull.csv' % self.Wellname
                                 , encoding='gb18030')
            # self.ui.Results.setText('保存文件：0处理空值%s.csv' % self.Wellname)
            self.ui.Results.setText('Save file: %s-0NoneNull' % self.Wellname)
        else:
            print('file style is .xlsx')
            self.workbook.to_excel(os.getcwd() + '\\' + self.Wellname + '\\' + '%s-0NoneNull.xlsx' % self.Wellname
                                   )
            # self.ui.Results.setText('保存文件：0处理空值%s.xlsx' % self.Wellname)
            self.ui.Results.setText('Save file: %s-0NoneNull' % self.Wellname)


        self.ui.InputDataTableView.clearContents()
        columns = [i.split('\n')[0] for i in self.workbook.columns]
        workbook_np = self.workbook.values
        rows, cols = workbook_np.shape
        print('行数，列数')
        print(rows, cols)
        self.ui.InputDataTableView.setRowCount(rows)
        self.ui.InputDataTableView.setColumnCount(len(columns))
        for i in range(len(columns)):
            print('columns=', i, columns[i])
            headerItem = QTableWidgetItem(columns[i])
            font = headerItem.font()
            ##  font.setBold(True)
            font.setPointSize(18)
            headerItem.setFont(font)
            headerItem.setForeground(QBrush(Qt.red))  # 前景色，即文字颜色
            self.ui.InputDataTableView.setHorizontalHeaderItem(i, headerItem)

        self.ui.InputDataTableView.resizeRowsToContents()
        self.ui.InputDataTableView.resizeColumnsToContents()
        # 初始化可以选择绘制的特征
        for num in range(1, 5):
            eval('self.ui.SelectFeature%d.clear()' % num)  # 清除图表
            eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # 清除图表

        for i in range(rows):
            rowslist = workbook_np[i, :]  # 获取excel每行内容
            # print(rowslist)
            for j in range(len(rowslist)):
                # 在tablewidget中添加行
                row = self.ui.InputDataTableView.rowCount()
                self.ui.InputDataTableView.insertRow(row)
                # 把数据写入tablewidget中
                newItem = QTableWidgetItem(str(rowslist[j]))
                self.ui.InputDataTableView.setItem(i, j, newItem)
        self.ui.InputDataTableView.setAlternatingRowColors(True)
        self.ui.Results.setText('Delete empty values of the table has been saved')
    except:
        # self.ui.Results.setText('请输入数据处理后再点击保存')
        self.ui.Results.setText('Please input the data and then click Save')


def DrawFig(self):
    bwith = 1
    fontsize=13
    self.ui.FeatureFig.figure.clear()  # 清除图表

    # index = self.ui.SelectFeature1.currentIndex()
    # print('绘图 begin')
    length = float(self.ui.length.text())
    Angle = float(self.ui.Angle.text())
    StartBent = float(self.ui.StartBent.text())
    # print('length,Angle,StartBent',length,Angle,StartBent)
    RadiusBent = float(self.ui.RadiusBent.text())
    # print('RadiusBent', RadiusBent)
    DCircle = float(self.ui.DCircle.text())
    # print('DCircle', DCircle)
    number = float(self.ui.number.text())
    print('read data is finished')
    AllLength = float(self.ui.AllLength.text())
    # StartBent = 240in/6.096m
    # Angle=90
    # DCircle = 30in/0.762m
    # RadiusBent = 60in/1.524m
    Region = self.ui.Region.currentIndex()
    Style = self.ui.Style0.currentIndex()
    # baozheng canshu shuru fanwei zhengque
    if Region == 0:
        if number <= StartBent and number>=0:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 0 is wrong. Please input the number again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region == 1:
        if number <= 90 and number>=0:
            number = StartBent + np.pi * RadiusBent * number / 180
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 1 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region == 2:
        if number <= AllLength and number >= StartBent + np.pi * RadiusBent * Angle / 180:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 2 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0

    # dui meiyige qiepain de dian jisuan
    if number<=StartBent:
        print('number<=StartBent')
        o1 = np.where(self.workbook.values[:, 0] < number + length)[0]
        o2 = np.where(self.workbook.values[:, 0] >= number)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        data_ori = self.workbook.iloc[original_position_index, :]
        x_y_z = data_ori.iloc[:, [0, 1, 2]].values
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 30
        azim = 0
        # Style = pressure / stress
        if Style == 0:
            p = data_ori.loc[:, ['Points_0','Points_1','Points_2','p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < 0:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0','Points_1','Points_2','wallShearStress_0'
                                         ,'wallShearStress_1','wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < 0:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]
        z_special = 0
    elif number<=(StartBent+np.pi*RadiusBent*Angle/180):
        print('number<=(StartBent+np.pi*RadiusBent*Angle/180)')
        remain_len = number - StartBent
        sita = remain_len/(np.pi*RadiusBent) #
        centre_x = StartBent+RadiusBent*np.sin(sita)
        centre_y = 0
        centre_z = RadiusBent * (1 - np.cos(sita))
        x_min = centre_x - DCircle * np.sin(sita) - length / 2
        x_max = centre_x + DCircle * np.sin(sita) + length / 2
        z_min = centre_z - DCircle * np.cos(sita) - length / 2
        z_max = centre_z + DCircle * np.cos(sita) + length / 2

        o1 = np.where(self.workbook.values[:, 0] <= x_max)[0]
        o2 = np.where(self.workbook.values[:, 0] >= x_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index1 = []
        for point in o1:
            if point in o2:
                original_position_index1.append(point)
        o1 = np.where(self.workbook.values[:, 2] <= z_max)[0]
        o2 = np.where(self.workbook.values[:, 2] >= z_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index2 = []
        for point in o1:
            if point in o2:
                original_position_index2.append(point)
        original_position_index = []
        for point in original_position_index1:
            if point in original_position_index2:
                original_position_index.append(point)
        data_ori = self.workbook.iloc[original_position_index, :]
        data = data_ori.iloc[:, [0,1,2]].values
        # position_index = []
        # for aim_point in range(len(data_ori)):
        #     DCircle_2 = ((data[aim_point, 0] - centre_x) ** 2 + (data[aim_point, 1] - centre_y) ** 2
        #      + (data[aim_point, 2] - centre_z) ** 2)
        #     if (DCircle_2-DCircle**2)<0.02**2:
        #         position_index.append(aim_point)
        # data_ori = data_ori.iloc[position_index, :]
        x_y_z = data
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        # y_ = DCircle * np.sin(omega)
        # x_ = DCircle * np.cos(omega) * np.sin(sita) #x +-
        # z_ = DCircle * np.cos(omega) * np.cos(sita)  # z +-
        elev = 45
        azim = 0
        # Style = pressure / stress
        if Style == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < centre_z:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'wallShearStress_0'
                                         , 'wallShearStress_1', 'wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < centre_z:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]
        z_special = centre_z
    # region 2
    else:
        print('else')
        print('number, (StartBent + np.pi * RadiusBent * Angle / 180)')
        print(number, StartBent, np.pi, RadiusBent, Angle , 180)
        remain_len = number - (StartBent + np.pi * RadiusBent * Angle / 180)
        print('remain_len',remain_len)
        # z_real = RadiusBent + DCircle / 2
        z_centre = RadiusBent + remain_len
        print('z',z_centre)
        o1 = np.where(self.workbook.values[:, 2] < z_centre+length)[0]
        o2 = np.where(self.workbook.values[:, 2] >= z_centre)[0]
        print('shape',self.workbook.values.shape)
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        print(original_position_index)
        data_ori = self.workbook.iloc[original_position_index, :]
        x_y_z = data_ori.iloc[:, [0, 1, 2]].values
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 90
        azim = 0
        # Style = pressure / stress
        if Style == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 0] < StartBent + RadiusBent:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            print('Style == 1')
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'wallShearStress_0'
                                         , 'wallShearStress_1', 'wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            print('1'*10)
            print(stress_y.shape)
            for i in range(len(stress_y)):
                if stress_y[i, 0] < StartBent + RadiusBent:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            print('2' * 10)
            stress = stress_y[new_i, :]
            print('3' * 10)
        z_special = z_centre
    # columns = [i.split('\n')[0] for i in self.workbook.columns]
    print('length', length)
    print('number', number)
    # print('self.workbook', self.workbook)
    # plt.figure(figsize=(2.5, 2.5), dpi=300)
    # plt.scatter(x, y, s=2)

    # print('11112')
    # fea = self.workbook.colum.ns[index].split('\n')[0]

    # print('fea=',fea)
    # print(self.workbook.iloc[:,index].values, self.workbook.iloc[:,xaxis].values)
    # self.ui.FeatureFig.figure.dpi = 200
    print('test')
    ax1 = self.ui.FeatureFig.figure.add_subplot(1, 1, 1, label='plot3D',projection='3d')
    print('000000')

    ax1.scatter3D(x, y, z, c='r', s=1)
    # print(x, y, z)
    print('111111')
    print('x',x)
    print('y', y)
    print('z', z)
    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    print(np.min(z), np.max(z))
    try:
        print(math.floor((np.min(x) * 1000)) / 1000, math.ceil((np.max(x) * 1000)) / 1000)
        print(math.floor((np.min(y) * 1000)) / 1000, math.ceil((np.max(y) * 1000)) / 1000)
        print(math.floor((np.min(z) * 1000)) / 1000, math.ceil((np.max(z) * 1000)) / 1000)
    except:
        pass

    ax1.set_xlim((math.floor((np.min(x) * 10)) / 10, math.ceil((np.max(x) * 10)) / 10))
    ax1.set_ylim((math.floor(np.min(y) * 10) / 10, math.ceil(np.max(y) * 10) / 10))
    ax1.set_zlim((math.floor(np.min(z) * 10) / 10, math.ceil(np.max(z) * 10) / 10))
    print('222222')
    ax1.set_xlabel('x axis', fontsize=fontsize)
    ax1.set_ylabel('y axis', fontsize=fontsize)
    ax1.set_zlabel('z axis', fontsize=fontsize)
    ax1.set_title('Slice', fontsize=fontsize)
    print('3333333')
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    print('4444444')
    # ax1.invert_yaxis()
    # print('11113')
    # ax1.set_xticks(fontproperties='Times New Roman')
    # ax1.set_yticks(fontproperties='Times New Roman')

    ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    ax1.view_init(elev=elev, azim=azim)
    self.ui.FeatureFig.figure.tight_layout()
    self.ui.Results.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    # self.ui.Results.setText('%s绘图完成')#%self.workbook.columns[index].split('\n')[0]
    # ax1.show()
    self.ui.FeatureFig.figure.canvas.draw()

    print('****'*3)
    self.ui.CurveFig.figure.clear()  # 清除图表
    print('test')
    ax1 = self.ui.CurveFig.figure.add_subplot(1, 1, 1, label='plot3D')

    # shujudian  paixu
    if Style==0:
        if Region == 0 or Region == 1:
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 2] < z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 2] < z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 2] > z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 2] > z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))

            print('000000')
            ax1.scatter(x_axis, p[:,3], c='r', s=1, label='Pressure')
        else:
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 0] < StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 0] < StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 0] > StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 0] > StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))

            print('000000')
            ax1.scatter(x_axis, p[:, 3], c='r', s=1, label='Pressure')
    elif Style==1:
        if Region == 0 or Region == 1:
            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 2] < z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 2] < z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 2] > z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 2] > z_special)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))
            stress_magnitude = np.sqrt(stress[:, 3]**2+stress[:, 4]**2+stress[:, 5]**2)
            ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='wallShearStress_magnitude')
        else:

            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 0] < StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            print('stress[index, 1]')
            print(stress[index, 1])
            print('np.max(stress[:, 1])')
            print(np.max(stress[:, 1]))
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 0] < StartBent + RadiusBent)[0]
            print('*'*100)
            print(stress[:, 1])
            print('*' * 100)
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            print('stress[index, 1]')
            print(stress[index, 1])
            print('np.max(stress[:, 1])')
            print(np.max(stress[:, 1]))
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 0] > StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            print('stress[index, 1]')
            print(stress[index, 1])
            print('np.max(stress[:, 1])')
            print(np.max(stress[:, 1]))
            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 0] > StartBent + RadiusBent)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))
            stress_magnitude = np.sqrt(stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)
            ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='wallShearStress_magnitude')

    print('111111')
    print('x', x_axis)
    ax1.set_xlabel('x axis (Angle)', fontsize=fontsize)

    if Style == 0:
        ax1.set_title('Pressure', fontsize=fontsize)
        ax1.set_ylabel('y axis (Pressure)', fontsize=fontsize)
    else:
        ax1.set_title('Stress', fontsize=fontsize)
        ax1.set_ylabel('y axis (Stress)', fontsize=fontsize)
    print('3333333')
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    print('4444444')
    # ax1.invert_yaxis()
    # print('11113')
    # ax1.set_xticks(fontproperties='Times New Roman')
    # ax1.set_yticks(fontproperties='Times New Roman')

    ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    ax1.legend(loc="best", edgecolor='white', facecolor=None, framealpha =0)
    self.ui.CurveFig.figure.tight_layout()
    self.ui.Results.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    # self.ui.Results.setText('%s绘图完成')#%self.workbook.columns[index].split('\n')[0]
    # ax1.show()
    self.ui.CurveFig.figure.canvas.draw()

    print('****' * 3)
    # print(os.getcwd() + '\\' + self.Wellname + self.workbook.columns[index].split('\n')[0])
    # draw_cross_curves(self.workbook
    #                   , x_=xaxis
    #                   , factor=[self.workbook.columns[index]]
    #                   , fontsize=10
    #                   , xfigsize=(4, 3)
    #                   , xdpi=300
    #                   , bwith=1.5
    #                   , point_num=10
    #                   , mode='auto'
    #                   , x_label=r'井深$\mathrm{(m)}$'
    #                   , y_label=r'%s$' % self.workbook.columns[index].split('\n')[0]
    #                   , truncate=False
    #                   , label=[self.workbook.columns[index].split('\n')[0]]
    #                   , c_=['b', 'r', 'k', 'g']
    #                   , lins=['-', '-']
    #                   # , yinv=True
    #                   , picture_name=self.workbook.columns[index].split('\n')[0]
    #                   , path=os.getcwd()+'\\'+self.Wellname
    #                   , trend=False
    #                   # , s_m='M'
    #                   )

def DrawWholePipe(self):

    print('DrawWholePipe')
    print('1'*20)
    bwith = 1
    fontsize=13
    self.ui.OtherFig.figure.clear()  # 清除图表
    print('2' * 20)
    # index = self.ui.SelectFeature1.currentIndex()
    # print('绘图 begin')
    length = float(self.ui.length.text())
    Angle = float(self.ui.Angle.text())
    StartBent = float(self.ui.StartBent.text())
    # print('length,Angle,StartBent',length,Angle,StartBent)
    RadiusBent = float(self.ui.RadiusBent.text())
    # print('RadiusBent', RadiusBent)
    print('3' * 20)
    DCircle = float(self.ui.DCircle.text())
    print('3.1' * 20)
    # print('DCircle', DCircle)
    number = float(self.ui.number.text())
    print('read data is finished')
    AllLength = float(self.ui.AllLength.text())
    # StartBent = 240in/6.096m
    # Angle=90
    # DCircle = 30in/0.762m
    # RadiusBent = 60in/1.524m
    Region = self.ui.Region.currentIndex()
    Style = self.ui.Style0.currentIndex()
    if Region == 0:
        if number <= StartBent and number >= 0:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 0 is wrong. Please input the number again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region == 1:
        if number <= 90 and number >= 0:
            number = StartBent + np.pi * RadiusBent * number / 180
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 1 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region == 2:
        if number <= AllLength and number >= StartBent + np.pi * RadiusBent * Angle / 180:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 2 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0

    # StartBent = 240in/6.096m
    # Angle=90
    # DCircle = 30in/0.762m
    # RadiusBent = 60in/1.524m
    EndBent_length = StartBent + np.pi * RadiusBent * Angle / 180
    StartBent_point = int(StartBent//length)
    EndBent_points = int(EndBent_length//length)
    all_points = int(AllLength//length)
    self.ui.label_37.setText('Range Region 0: 0.00-%.2f' % StartBent)
    self.ui.label_36.setText('Range Region 1: %.2f-%.2f' % (StartBent, EndBent_length))
    self.ui.label_38.setText('Range Region 2: %.2f-%.2f' % (EndBent_length, AllLength))

    print(StartBent_point, EndBent_points, all_points)
    print('StartBent_point, EndBent_points, all_points')
    x = np.zeros(all_points)
    y = np.zeros(all_points)
    z = np.zeros(all_points)
    for i in range(1, all_points):
        #
        if i <= StartBent_point:
            print(i)
            x[i] = x[i - 1] + length
        elif i <= EndBent_points:
            print(i)
            now_length = (i - StartBent_point) * length
            radian = now_length/RadiusBent
            now_x = RadiusBent * np.sin(radian)
            now_z = RadiusBent * (1 - np.cos(radian))
            x[i] = StartBent + now_x
            z[i] = now_z
        else:
            print(i)
            x[i] = x[i - 1] + np.cos(Angle / 180 * np.pi) * length
            z[i] = z[i - 1] + np.sin(Angle / 180 * np.pi) * length


    ax1 = self.ui.OtherFig.figure.add_subplot(1, 1, 1, label='plot3D',projection='3d')
    print('000000')
    ax1.plot3D(x[:StartBent_point], y[:StartBent_point], z[:StartBent_point], color='b')
    ax1.plot3D(x[StartBent_point:EndBent_points]
               , y[StartBent_point:EndBent_points], z[StartBent_point:EndBent_points], color='r')
    print(x[StartBent_point:EndBent_points]
               , y[StartBent_point:EndBent_points], z[StartBent_point:EndBent_points])
    ax1.plot3D(x[EndBent_points:], y[EndBent_points:], z[EndBent_points:], color='b')
    print(x[StartBent_point:EndBent_points]
          , y[StartBent_point:EndBent_points], z[StartBent_point:EndBent_points])
    print('text???')
    ax1.text(x[StartBent_point], y[StartBent_point], z[StartBent_point], 'Region 0 - Blue', 'x')
    print('text  1111')
    ax1.text(x[EndBent_points], y[EndBent_points], z[EndBent_points], 'Region 1 - Red', 'x')
    print('text  2222')
    ax1.text(x[-1], y[-1], z[-1], 'Region 2 - Blue', 'x')
    print('111111')
    # print('x',x)
    # print('y', y)
    # print('z', z)
    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    print(np.min(z), np.max(z))
    try:
        print(math.floor((np.min(x) * 1000)) / 1000, math.ceil((np.max(x) * 1000)) / 1000)
        print(math.floor((np.min(y) * 1000)) / 1000, math.ceil((np.max(y) * 1000)) / 1000)
        print(math.floor((np.min(z) * 1000)) / 1000, math.ceil((np.max(z) * 1000)) / 1000)
    except:
        pass

    # ax1.set_xlim((math.floor((np.min(x) * 10)) / 10, math.ceil((np.max(x) * 10)) / 10))
    # ax1.set_ylim((math.floor(np.min(y) * 10) / 10, math.ceil(np.max(y) * 10) / 10))
    # ax1.set_zlim((math.floor(np.min(z) * 10) / 10, math.ceil(np.max(z) * 10) / 10))
    print('222222')
    ax1.set_xlabel('x axis', fontsize=fontsize)
    ax1.set_ylabel('y axis', fontsize=fontsize)
    ax1.set_zlabel('z axis', fontsize=fontsize)
    ax1.set_title('WholePipe-CentreLine', fontsize=fontsize)
    print('3333333')
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    print('4444444')
    # ax1.invert_yaxis()
    # print('11113')
    # ax1.set_xticks(fontproperties='Times New Roman')
    # ax1.set_yticks(fontproperties='Times New Roman')
    elev = 0
    azim = -90
    ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    ax1.view_init(elev=elev, azim=azim)
    self.ui.OtherFig.figure.tight_layout()
    # self.ui.Results.setText('%s绘图完成')#%self.workbook.columns[index].split('\n')[0]
    self.ui.Results.setText('Completed')  # %self.workbook.columns[index].split('\n')[0]
    # ax1.show()
    self.ui.OtherFig.figure.canvas.draw()

    print('****'*3)