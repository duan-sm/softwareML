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

def rotate_point(x, y, z, angle_degrees):
    '''
    Transformation of coordinate axes
    :param x: Original X-axis data
    :param y: Original Y-axis data
    :param z: Original Z-axis data
    :param angle_degrees: Rotation Angle
    :return: New data
    '''
    # Convert angles to radians
    angle_rad = np.radians(angle_degrees)

    # Defined rotation matrix
    #     rotation_matrix = np.array([
    #         [np.cos(angle_rad), -np.sin(angle_rad), 0],
    #         [np.sin(angle_rad), np.cos(angle_rad), 0],
    #         [0, 0, 1]
    #     ])
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

    # Construct a coordinate point vector
    original_point = np.array([x, y, z])

    # Matrix multiplication is performed to obtain the rotated coordinate points
    rotated_point = np.dot(rotation_matrix, original_point)

    return rotated_point


def DataInputBtn(self):
    '''
    It will open the file selection screen, read the selected file and display it.
    '''
    print('DataInputBtn|'*10)
    # Re-read the data and clear the previous contents
    self.ui.NondimenBtn.setEnabled(True)
    self.ui.DimenComBtn.setEnabled(True)
    self.ui.NondimenBtn.setEnabled(True)
    self.ui.FeatureFig.figure.clear()  # Clear Figure
    self.ui.FeatureFig.figure.clf()
    # self.ui.FeatureFig.figure.cla()
    self.ui.OtherFig.figure.clear()  # Clear Figure
    self.ui.OtherFig.figure.clf()
    # self.ui.OtherFig.figure.cla()
    self.ui.FeatureFig2.figure.clear()  # Clear Figure
    self.ui.FeatureFig2.figure.clf()
    # self.ui.FeatureFig2.figure.cla()
    self.ui.Fig3.figure.clear()  # Clear Figure
    self.ui.Fig3.figure.clf()
    # self.ui.Fig3.figure.cla()
    self.ui.LossFig.figure.clear()  # Clear Figure
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
        self.ui.WellName.setPlaceholderText('Please input FileName')
        self.ui.Results.setText('Enter the folder name to determine the path for saving subsequent files.')
        return 0
    # if not os.path.isdir(self.Wellname):
    #     os.mkdir(self.Wellname)
    # Example Create a data save path
    if not os.path.exists(r'./%s'%self.Wellname):
        os.makedirs(r'./%s'%self.Wellname)
    # if not os.path.isdir(self.Wellname):
    #     os.mkdir(self.Wellname)
    # Open the Select file window
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
            # Open csv file
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
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='lsx': # Open xlsx file
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
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='xls':# Open xls file
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
                self.ui.Results.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            else:
                self.ui.Results.setText('The file format should be: xls, csv, xlsx' % (fname.split('/')[-1]))
            # Get the value of the entire row and column (array)
            columns = [i.split('\n')[0] for i in self.workbook.columns]
            print(columns)
            rows = len(self.workbook)
            workbook_np = self.workbook.values
            # print(sheet1.nrows)
            self.ui.InputDataTableView.setRowCount(rows)
            self.ui.InputDataTableView.setColumnCount(len(columns))
            # Set the header in the window
            for i in range(len(columns)):
                # print(i)
                headerItem = QTableWidgetItem(columns[i])
                font = headerItem.font()
                ##  font.setBold(True)
                font.setPointSize(18)
                headerItem.setFont(font)
                headerItem.setForeground(QBrush(Qt.red))  # Foreground color, text color
                self.ui.InputDataTableView.setHorizontalHeaderItem(i, headerItem)

            self.ui.InputDataTableView.resizeRowsToContents()
            self.ui.InputDataTableView.resizeColumnsToContents()
            # Initializing can select drawn features
            for num in range(2,5):
                eval('self.ui.SelectFeature%d.clear()' % num)  # Clear chart
                eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # Clear chart

            # # Displays the data in a window
            for i in range(rows):
                rowslist = workbook_np[i,:]  # Get the content of each row in Excel
                # print(rowslist)
                for j in range(len(rowslist)):
                    # Adding rows in tablewidget
                    row = self.ui.InputDataTableView.rowCount()
                    self.ui.InputDataTableView.insertRow(row)
                    # Write data into tablewidget
                    newItem = QTableWidgetItem(str(rowslist[j]))
                    self.ui.InputDataTableView.setItem(i, j, newItem)
            self.ui.InputDataTableView.setAlternatingRowColors(True)
            print(self.size_)
            print(self.workbook.columns)
    except:
        self.ui.Results.setText('Something is wrong')


def merge(self):
    '''
    Early function, can be ignored
    '''
    self.Wellname = self.ui.WellName.text()
    if self.Wellname == '例：呼111' or self.Wellname == '请输入井号' or self.Wellname == '' or len(
            self.Wellname) < 1:
        self.ui.WellName.setPlaceholderText('Please input FileName')
        self.ui.Results.setText('Enter the folder name to determine the path for saving subsequent files.')
        return 0
    if not os.path.isdir(self.Wellname):
        os.mkdir(self.Wellname)

    try:
        self.size_ = 1
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
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            elif allFileName[0][-3:] == 'lsx':
                self.filetype = '.xlsx'
                data = pd.read_excel(fname+'\\'+allFileName[0], sheet_name=0)
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            elif allFileName[0][-3:] == 'xls':
                self.filetype = '.xlsx'
                data = pd.read_excel(fname+'\\'+allFileName[0], sheet_name=0)
                self.ui.Results.setText('Open the file:%s. Successful' % allFileName[0])
            else:
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
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                elif allFileName[fileIndex][-3:] == 'lsx':
                    self.filetype = '.xlsx'
                    data1 = pd.read_excel(fname+'\\'+allFileName[fileIndex], sheet_name=0)
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                elif allFileName[fileIndex][-3:] == 'xls':
                    self.filetype = '.xlsx'
                    data1 = pd.read_excel(fname+'\\'+allFileName[fileIndex], sheet_name=0)
                    self.ui.Results.setText('Open the file:%s. Successful' % allFileName[fileIndex])
                else:
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
        self.ui.Results.setText('Error: file')
        self.size_ = 0

def btnDelCols(self):
    '''
    Delete empty column 
    '''
    try:
        row0, col0 = self.workbook.shape
        self.workbook = self.workbook.dropna(axis=1, how='all')
        row1, col1 = self.workbook.shape
        cha = col0 - col1
        self.ui.delColsNum.setText(str(cha))
        self.ui.Results.setText('Complete deletion of empty columns, the number is:%d' % cha)
    except:
        self.ui.Results.setText('Please input the data and click Delete')

def btnDelRows(self):
    '''
        Delete empty lines 
    '''
    try:
        row0, col0 = self.workbook.shape
        self.workbook = self.workbook.dropna(axis=0, how='any')
        row1, col1 = self.workbook.shape
        cha = row0 - row1
        self.ui.delRowsNum.setText(str(cha))
        self.ui.Results.setText('Complete deletion of empty lines, the number is:%d' % cha)
    except:
        self.ui.Results.setText('Please input the data and click Delete')

def btnSaveData(self):
    '''
    The modified data is saved 
    '''
    try:
        # Determine data type 
        if self.filetype == '.csv':
            print('file style is .csv')
            self.workbook.to_csv(os.getcwd() + '\\' + self.Wellname + '\\' + '%s-0NoneNull.csv' % self.Wellname
                                 , encoding='gb18030')
            self.ui.Results.setText('Save file: %s-0NoneNull' % self.Wellname)
        else:
            print('file style is .xlsx')
            self.workbook.to_excel(os.getcwd() + '\\' + self.Wellname + '\\' + '%s-0NoneNull.xlsx' % self.Wellname
                                   )
            self.ui.Results.setText('Save file: %s-0NoneNull' % self.Wellname)


        self.ui.InputDataTableView.clearContents()
        columns = [i.split('\n')[0] for i in self.workbook.columns]
        workbook_np = self.workbook.values
        rows, cols = workbook_np.shape
        print('rows，cols')
        print(rows, cols)
        # Set header 
        self.ui.InputDataTableView.setRowCount(rows)
        self.ui.InputDataTableView.setColumnCount(len(columns))
        for i in range(len(columns)):
            print('columns=', i, columns[i])
            headerItem = QTableWidgetItem(columns[i])
            font = headerItem.font()
            ##  font.setBold(True)
            font.setPointSize(18)
            headerItem.setFont(font)
            headerItem.setForeground(QBrush(Qt.red))  # Foreground color, i.e. text color
            self.ui.InputDataTableView.setHorizontalHeaderItem(i, headerItem)

        self.ui.InputDataTableView.resizeRowsToContents()
        self.ui.InputDataTableView.resizeColumnsToContents()
        # Initializing can select drawn features 
        for num in range(2, 5):
            eval('self.ui.SelectFeature%d.clear()' % num)  # clear fig
            eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # clear fig

        for i in range(rows):
            rowslist = workbook_np[i, :]
            # print(rowslist)
            for j in range(len(rowslist)):
                # Add rows to the tablewidget
                row = self.ui.InputDataTableView.rowCount()
                self.ui.InputDataTableView.insertRow(row)
                # Write data to the tablewidget
                newItem = QTableWidgetItem(str(rowslist[j]))
                self.ui.InputDataTableView.setItem(i, j, newItem)
        self.ui.InputDataTableView.setAlternatingRowColors(True)
        self.ui.Results.setText('Delete empty values of the table has been saved')
    except:
        self.ui.Results.setText('Please input the data and then click Save')


def DrawFig(self):
    '''
        Plot slice
        Draws a slice of the selected location (as determined by the input page data).
    '''
    bwith = 1
    fontsize=13
    self.ui.FeatureFig.figure.clear()  # Clear chart clear fig

    # index = self.ui.SelectFeature1.currentIndex()
    # print('绘图 begin')
    # Read interface parameters
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
    # Ensure that the parameter range is correct
    if Region == 0:
        if number <= StartBent and number>=0:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 0 is wrong. Please input the number again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton
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
            defaultBtn = QMessageBox.NoButton
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
            defaultBtn = QMessageBox.NoButton
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    # split different region

    # Zone division
    data = self.workbook.copy(deep=True)
    centre_point_x = StartBent
    centre_point_y = 0
    centre_point_z = RadiusBent
    region0_index = np.where(data.values[:, 0] < StartBent)[0]
    region12_index = np.where(data.values[:, 0] >= StartBent)[0]
    region0_data = data.iloc[region0_index, :]
    region12_data = data.iloc[region12_index, :]
    region0_data_xyz = region0_data.iloc[:, :3].values.copy()
    region12_data_xyz = region12_data.iloc[:, :3].values.copy()
    # Adjust origin
    region12_data_xyz[:, 0] = region12_data_xyz[:, 0] - centre_point_x
    region12_data_xyz[:, 2] = region12_data_xyz[:, 2] - centre_point_z

    # Transformation coordinate
    r = np.sqrt(region12_data_xyz[:, 0] ** 2 + region12_data_xyz[:, 1] ** 2 + region12_data_xyz[:, 2] ** 2)
    theta = np.arccos(region12_data_xyz[:, 2] / r) / np.pi * 180
    phi = np.arctan(region12_data_xyz[:, 1], region12_data_xyz[:, 0]) / np.pi * 180
    region12_data['r'] = r
    region12_data['theta'] = theta  # After adjusting the coordinate system, the Angle decreases from 180
    region12_data['phi'] = phi
    print(np.sort(list(set(theta))))
    # Ignore bug about region 1:
    # region 1 Since only two points on a slice are accurate = theta (these two points are located at the center of the circle and the center line)
    # , the theta of other points is not accurate = 180, so the slice cannot be found
    # This also causes problems when dividing region1 and region2
    region1_index = np.where(theta > 180 - Angle)[0]
    region2_index = np.where(theta <= 180 - Angle)[0]
    region1_data = region12_data.iloc[region1_index, :]
    region2_data = region12_data.iloc[region2_index, :]
    region1_data_xyz = region1_data.iloc[:, :3].values
    # Adjust origin
    centre_point_x2 = StartBent + RadiusBent * np.sin(Angle / 180 * np.pi)
    centre_point_y2 = 0
    centre_point_z2 = RadiusBent * (1 - np.cos(Angle / 180 * np.pi))
    region2_data_xyz = region2_data.iloc[:, :3].values.copy()
    region2_data_xyz[:, 0] = region2_data_xyz[:, 0] - centre_point_x2
    region2_data_xyz[:, 2] = region2_data_xyz[:, 2] - centre_point_z2
    new_point = rotate_point(region2_data_xyz[:, 0], region2_data_xyz[:, 1], region2_data_xyz[:, 2], Angle)
    region2_data['x'] = new_point.T[:, 0]
    region2_data['y'] = new_point.T[:, 1]
    region2_data['z'] = new_point.T[:, 2]
    # Calculate the points for each slice
    '''
    Note: Most of the code below is region-specific, so it is only fine-tuned
    Different pressure /stress results in different data retention, resulting in different variables p/stress being used.
    All else being equal, comments are only partial
    '''
    if number<=StartBent:
        # Region 0
        print('number<=StartBent')
        # Retrieve the location by length
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
            p_y = p.sort_values(by='Points_1', ascending=True).values # Arrange the data in ascending order of y coordinates
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < 0: # Underslice data
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1])) # concatenate data
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :] # Rearrange data
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
        # Region 1
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
    else: # Region 2
        print('else')
        print('number, (StartBent + np.pi * RadiusBent * Angle / 180)')
        print(number, StartBent, np.pi, RadiusBent, Angle , 180)
        remain_len = number - (StartBent + np.pi * RadiusBent * Angle / 180)
        print('remain_len',remain_len)
        # z_real = RadiusBent + DCircle / 2
        o1 = np.where(region2_data['x'] < remain_len+length)[0]
        o2 = np.where(region2_data['x'] >= remain_len)[0]
        print('shape',self.workbook.values.shape)
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        original_position_index = region2_data.iloc[original_position_index,:].index
        print(original_position_index)
        data_ori = self.workbook.loc[original_position_index, :]
        x_y_z = data_ori.iloc[:, [0, 1, 2]].values
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        z_special = centre_point_z2 + remain_len * np.sin(Angle / 180 * np.pi)
        elev = 90
        azim = 0
        # Style = pressure / stress
        if Style == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < z_special:
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
                if stress_y[i, 2] < z_special:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            print('2' * 10)
            stress = stress_y[new_i, :]
            print('3' * 10)

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
    self.ui.CurveFig.figure.clear()  # clear fig
    print('test')
    ax1 = self.ui.CurveFig.figure.add_subplot(1, 1, 1, label='plot3D')

    # Data point ordering
    if Style==0:
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
    elif Style==1:
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
    '''
    Draw the center curve of the pipe 绘制管道中心曲线
    '''
    print('DrawWholePipe')
    print('1'*20)
    bwith = 1
    fontsize=13
    self.ui.OtherFig.figure.clear()  # Clear chart clear fig
    print('2' * 20)
    # index = self.ui.SelectFeature1.currentIndex()
    # Read interface input
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
    # Determine whether the plot area and the input data range are reasonable
    if Region == 0:
        if number <= StartBent and number >= 0:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 0 is wrong. Please input the number again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # Default button
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
            defaultBtn = QMessageBox.NoButton  # Default button
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
            defaultBtn = QMessageBox.NoButton  # Default button
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    # Calculate the key parameters of the pipeline
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
        # Calculate the coordinates of each point
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

    # Draw a picture of the pipe center line
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
    ax1.text(x[StartBent_point], y[StartBent_point], z[StartBent_point], 'Region 0 - Blue', 'x') # Annotate key points on the diagram
    ax1.text(x[EndBent_points], y[EndBent_points], z[EndBent_points], 'Region 1 - Red', 'x') # Annotate key points on the diagram
    ax1.text(x[-1], y[-1], z[-1], 'Region 2 - Blue', 'x') # Annotate key points on the diagram
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