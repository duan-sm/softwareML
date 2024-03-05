#!/usr/bin/env python
# coding=utf-8

"""
@Time: 2/12/2024 3:05 PM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Test.py
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt
from tensorflow.python.keras.models import load_model

def LoadParaBtn(self):
    print('DataInputBtn|'*10)

    fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.'
                                           , 'Data file(*.csv *.xlsx *.xls)'
                                           )
    self.ui.ParaTable.clear()
    print('2|' * 10)
    index_col = 0
    try:
        if fname:
            self.ui.ParaTable.clearContents()
            print('fname', fname)
            # 打开文件
            if fname[-3:]=='csv':
                filetype = '.csv'
                self.paraData = pd.read_csv(fname
                                       ,encoding='gb18030'
                                       )
                # print(workbook.iloc[:,0].values)
                # workbook_first_col1 = workbook.iloc[:, 0].values == np.arange(len(workbook))
                # workbook_first_col2 = workbook.iloc[:,0].values == np.arange(1,len(workbook)+1)
                # if sum(workbook_first_col1)==len(workbook_first_col1):
                #     workbook = workbook.iloc[:,1:]
                # elif sum(workbook_first_col2)==len(workbook_first_col2):
                #     workbook = workbook.iloc[:, 1:]
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='lsx':
                print('3|' * 10)
                filetype = '.xlsx'
                self.paraData = pd.read_excel(fname,index_col=index_col)
                # print('4|' * 10)
                # workbook_first_col1 = workbook.iloc[:, 0].values == np.arange(len(workbook))
                # workbook_first_col2 = workbook.iloc[:, 0].values == np.arange(1, len(workbook) + 1)
                # print('5|' * 10)
                # if sum(workbook_first_col1) == len(workbook_first_col1):
                #     workbook = workbook.iloc[:, 1:]
                # elif sum(workbook_first_col2) == len(workbook_first_col2):
                #     workbook = workbook.iloc[:, 1:]
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:]=='xls':
                print('3|' * 10)
                filetype = '.xlsx'
                self.paraData = pd.read_excel(fname,skiprows=1,sheet_name=0)
                # print('4|' * 10)
                # workbook = workbook.dropna(axis=0, how='all')
                # col_o = workbook.columns
                # col_c = []
                # for i in range(len(col_o)):
                #     str_ = ''
                #     new_c = col_o[i].split('\n')
                #     if len(new_c) > 1:
                #         new_1 = str_.join(new_c[:-1])
                #         new_2 = new_c[-1]
                #         new_c = new_1 + '\n' + new_2
                #         col_c.append(new_c)
                #     else:
                #         col_c.append(new_c[0])
                # print('5|' * 10)
                # workbook.columns = col_c
                # self.ui.Results.setText('打开文件:%s，成功' % (fname.split('/')[-1]))
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            else:
                # self.ui.Results.setText('打开文件格式为%s，不满足要求'%(fname.split('.')[1]))
                self.ui.Results4.setText('The file format should be: xls, csv, xlsx' % (fname.split('/')[-1]))
            # 获取整行和整列的值（数组）
            columns = [i.split('\n')[0] for i in self.paraData.columns]
            print(columns)
            rows = len(self.paraData)
            workbook_np = self.paraData.values
            # print(sheet1.nrows)
            self.ui.ParaTable.setRowCount(rows)
            self.ui.ParaTable.setColumnCount(len(columns))
            for i in range(len(columns)):
                # print(i)
                headerItem = QTableWidgetItem(columns[i])
                font = headerItem.font()
                ##  font.setBold(True)
                font.setPointSize(18)
                headerItem.setFont(font)
                headerItem.setForeground(QBrush(Qt.red))  # 前景色，即文字颜色
                self.ui.ParaTable.setHorizontalHeaderItem(i, headerItem)

            self.ui.ParaTable.resizeRowsToContents()
            self.ui.ParaTable.resizeColumnsToContents()

            # 在table中显示数据
            for i in range(rows):
                rowslist = workbook_np[i,:]  # 获取excel每行内容
                # print(rowslist)
                for j in range(len(rowslist)):
                    # 在tablewidget中添加行
                    row = self.ui.ParaTable.rowCount()
                    self.ui.ParaTable.insertRow(row)
                    # 把数据写入tablewidget中
                    newItem = QTableWidgetItem(str(rowslist[j]))
                    self.ui.ParaTable.setItem(i, j, newItem)
            self.ui.ParaTable.setAlternatingRowColors(True)
            print(self.size_)
            print(self.paraData.columns)
    except:
        self.ui.Results.setText('Something is wrong')



def LoadTestData(self):

    fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.'
                                           , 'Data file(*.csv *.xlsx *.xls)')
    try:
        if fname:
            print('fname', fname)
            # 打开文件
            if fname[-3:] == 'csv':
                filetype = '.csv'
                self.testworkbook = pd.read_csv(fname, encoding='gb18030')
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:] == 'lsx':
                print('3|' * 10)
                filetype = '.xlsx'
                self.testworkbook = pd.read_excel(fname, index_col=0)
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:] == 'xls':
                print('3|' * 10)
                filetype = '.xlsx'
                self.testworkbook = pd.read_excel(fname, skiprows=1, sheet_name=0)
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            else:
                # self.ui.Results.setText('打开文件格式为%s，不满足要求'%(fname.split('.')[1]))
                self.ui.Results4.setText('The file format should be: xls, csv, xlsx' % (fname.split('/')[-1]))
        # dlgTitle = "Tips"
        # strInfo = ("You should input the Z value for the center point of the circle. "
        #            "Otherwise, the figure will display an error..")
        # defaultBtn = QMessageBox.NoButton  # 缺省按钮
        # result = QMessageBox.question(self, dlgTitle, strInfo,
        #                               QMessageBox.Yes,
        #                               defaultBtn)
    except:
        self.ui.Results.setText('There are some mistakes about the data file.')

def Test(self):
    print('1'*15)
    m = load_model(self.model)
    data_cols = self.testworkbook.columns
    ori_test_data = self.testworkbook.values.copy()
    DimensionlessPara = self.paraData.values.copy()
    print('2' * 15)
    DimensionlessType_index = self.ui.DimensionlessType.currentIndex() #
    if DimensionlessType_index == 0:
        test_data = (ori_test_data-DimensionlessPara[0,:])/np.sqrt(DimensionlessPara[1,:])
    else:
        test_data = (ori_test_data - DimensionlessPara[1, :]) / (DimensionlessPara[0, :] - DimensionlessPara[1, :])
    print('3' * 15)
    test_data = test_data[:,:4]
    StrePres_t = self.ui.StrePres_t.currentIndex()
    # style = self.ui.Style1.currentIndex()
    if StrePres_t == 0:
        y_index = np.where('p'==data_cols)[0]
        print('style=0')
        print(data_cols)
        print(y_index)
    else:
        y_index1 = np.where('wallShearStress_0' == data_cols)[0]
        y_index2 = np.where('wallShearStress_1'==data_cols)[0]
        y_index3 = np.where('wallShearStress_2' == data_cols)[0]
        y_index = np.concatenate((y_index1,y_index2,y_index3))
        print('style=1')
        print(data_cols)
        print(y_index1,y_index2,y_index3,y_index)
    print('4' * 15)
    print('test_data.shape', test_data.shape)
    results = m(test_data)
    print('4.5' * 15)
    if DimensionlessType_index == 0:
        print('DimensionlessType_index=%d'%DimensionlessType_index)
        print('results.shape, DimensionlessPara.shape')
        print(results.shape, DimensionlessPara.shape)
        print('y_index',y_index)
        print('DimensionlessPara[1,y_index]',DimensionlessPara[1,y_index].shape)
        results_inv = results * np.sqrt(DimensionlessPara[1,y_index]) + DimensionlessPara[0,y_index]
    else:
        print('DimensionlessType_index=%d' % DimensionlessType_index)
        print('results.shape, DimensionlessPara.shape')
        print(results.shape, DimensionlessPara.shape)
        print('DimensionlessPara[1,y_index]', DimensionlessPara[1, y_index].shape)
        print('y_index', y_index)
        results_inv = results * (DimensionlessPara[0, y_index] - DimensionlessPara[1, y_index]) + DimensionlessPara[1, y_index]
    print('5' * 15)
    ori_test_data[:,y_index] = results_inv
    results_ = pd.DataFrame(ori_test_data, columns=data_cols)
    results_.to_csv(r'E:\Software-Duan\res\result\results_new.csv')
    print('6' * 15)
    if StrePres_t == 0:
        self.testworkbook['prediction_p'] = results_inv
    else:
        self.testworkbook['prediction_stress0'] = results_inv[:,0]
        self.testworkbook['prediction_stress1'] = results_inv[:,1]
        self.testworkbook['prediction_stress2'] = results_inv[:,2]
    ########################################################################################
    ########################################################################################
    ########################################################################################
def DrawSliceT(self):
    print('DrawSliceT')
    fontsize = 13
    bwith = 1
    self.ui.TestFig.figure.clear()  # 清除图表

    # index = self.ui.SelectFeature1.currentIndex()
    print('绘图 begin')
    Length_t = float(self.ui.Length_t.text())
    Angle_t = float(self.ui.Angle_t.text())
    StartBent_t = float(self.ui.StartBent_t.text())
    RadiusBent_t = float(self.ui.RadiusBent_t.text())
    DCircle_t = float(self.ui.DCircle_t.text())
    Number_t = float(self.ui.Number_t.text())
    AllLength_t = float(self.ui.AllLength_t.text())
    Region_t = self.ui.Region_t.currentIndex()
    StrePres_t = self.ui.StrePres_t.currentIndex()
    print('read data is finished')
    
    # print('Length_t,Angle_t,StartBent_t',Length_t,Angle_t,StartBent_t)
    # print('RadiusBent_t', RadiusBent_t)
    # print('DCircle_t', DCircle_t)
    
    # StartBent_t = 240in/6.096m
    # Angle_t=90
    # DCircle_t = 30in/0.762m
    # RadiusBent_t = 60in/1.524m

    # judge region
    if Region_t == 0:
        if Number_t <= StartBent_t and Number_t >= 0:
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
    elif Region_t == 1:
        if Number_t <= 90 and Number_t >= 0:
            Number_t = StartBent_t + np.pi * RadiusBent_t * Number_t / 180
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 1 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region_t == 2:
        if Number_t <= AllLength_t and Number_t >= StartBent_t + np.pi * RadiusBent_t * Angle_t / 180:
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

    # find position
    if Number_t <= StartBent_t:
        print('number<=StartBent_t')
        o1 = np.where(self.testworkbook.values[:, 0] < Number_t + Length_t)[0]
        o2 = np.where(self.testworkbook.values[:, 0] >= Number_t)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.iloc[:, [0, 1, 2]].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 30
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'prediction_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
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
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2'
                                      , 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2'
                                      , 'prediction_stress0', 'prediction_stress1', 'prediction_stress2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
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
    elif Number_t <= (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180):
        print('number<=(StartBent_t+np.pi*RadiusBent_t*Angle_t/180)')
        remain_len = Number_t - StartBent_t
        sita = remain_len / (np.pi * RadiusBent_t)  #
        centre_x = StartBent_t + RadiusBent_t * np.sin(sita)
        centre_y = 0
        centre_z = RadiusBent_t * (1 - np.cos(sita))
        x_min = centre_x - DCircle_t * np.sin(sita) - Length_t / 2
        x_max = centre_x + DCircle_t * np.sin(sita) + Length_t / 2
        z_min = centre_z - DCircle_t * np.cos(sita) - Length_t / 2
        z_max = centre_z + DCircle_t * np.cos(sita) + Length_t / 2

        o1 = np.where(self.testworkbook.values[:, 0] <= x_max)[0]
        o2 = np.where(self.testworkbook.values[:, 0] >= x_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index1 = []
        for point in o1:
            if point in o2:
                original_position_index1.append(point)
        o1 = np.where(self.testworkbook.values[:, 2] <= z_max)[0]
        o2 = np.where(self.testworkbook.values[:, 2] >= z_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index2 = []
        for point in o1:
            if point in o2:
                original_position_index2.append(point)
        original_position_index = []
        for point in original_position_index1:
            if point in original_position_index2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        data = data_ori.iloc[:, [0, 1, 2]].values.copy()
        # position_index = []
        # for aim_point in range(len(data_ori)):
        #     DCircle_2 = ((data[aim_point, 0] - centre_x) ** 2 + (data[aim_point, 1] - centre_y) ** 2
        #      + (data[aim_point, 2] - centre_z) ** 2)
        #     if (DCircle_2-DCircle_t**2)<0.02**2:
        #         position_index.append(aim_point)
        # data_ori = data_ori.iloc[position_index, :]
        x_y_z = data
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        # y_ = DCircle_t * np.sin(omega)
        # x_ = DCircle_t * np.cos(omega) * np.sin(sita) #x +-
        # z_ = DCircle_t * np.cos(omega) * np.cos(sita)  # z +-
        elev = 45
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'prediction_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
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
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2'
                                         , 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2'
                                         , 'prediction_stress0', 'prediction_stress1', 'prediction_stress2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
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
    else:
        print('else')
        print('number, (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180)')
        print(Number_t, StartBent_t, np.pi, RadiusBent_t, Angle_t, 180)
        remain_len = Number_t - (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180)
        print('remain_len', remain_len)
        z_centre = RadiusBent_t + remain_len
        print('z', z_centre)
        o1 = np.where(self.testworkbook.values[:, 2] < z_centre + Length_t)[0]
        o2 = np.where(self.testworkbook.values[:, 2] >= z_centre)[0]
        print('shape', self.testworkbook.values.shape)
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.iloc[:, [0, 1, 2]].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 90
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'prediction_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 0] < StartBent_t + RadiusBent_t:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2'
                                         , 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2'
                                         , 'prediction_stress0', 'prediction_stress1', 'prediction_stress2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 0] < StartBent_t + RadiusBent_t:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]
        z_special = z_centre
    columns = [i.split('\n')[0] for i in self.testworkbook.columns]
    print('Length_t', Length_t)
    print('number', Number_t)
    print('test')
    ax1 = self.ui.TestFig.figure.add_subplot(1, 1, 1, label='plot3D')
    if StrePres_t == 0:
        if Region_t == 0 or Region_t == 1:
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
            ax1.scatter(x_axis, p[:, 3], c='r', s=1, label='Pressure CFD')
            ax1.scatter(x_axis, p[:, 4], c='r', s=1, label='Pressure Prediction')
        elif Region_t == 2:
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 0] < StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 0] < StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] > 0)[0]
            diyi_2 = np.where(p[:, 0] > StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            diyi_1 = np.where(p[:, 1] < 0)[0]
            diyi_2 = np.where(p[:, 0] > StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))

            print('000000')
            ax1.scatter(x_axis, p[:, 3], c='r', s=1, label='Pressure CFD')
            ax1.scatter(x_axis, p[:, 4], c='r', s=1, label='Pressure Prediction')
    elif StrePres_t == 1:
        if Region_t == 0 or Region_t == 1:
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
            stress_magnitude = np.sqrt(stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)
            prediction_stress_magnitude = np.sqrt(stress[:, 6] ** 2 + stress[:, 7] ** 2 + stress[:, 8] ** 2)
            ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='CFD wallShearStress_magnitude')
            ax1.scatter(x_axis, prediction_stress_magnitude, c='r', s=1, label='Prediction Stress')
            print('*'*100)
            print(x_axis)
            print(stress_magnitude)
            print('*' * 100)
            print(prediction_stress_magnitude)
            print('*' * 100)
        elif Region_t == 2:
            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 0] < StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis1 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 0] < StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis2 = 180 - np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] > 0)[0]
            diyi_2 = np.where(stress[:, 0] > StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis3 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            diyi_1 = np.where(stress[:, 1] < 0)[0]
            diyi_2 = np.where(stress[:, 0] > StartBent_t + RadiusBent_t)[0]
            index = []
            for i in diyi_1:
                if i in diyi_2:
                    index.append(i)
            x_axis4 = 180 + np.arccos(stress[index, 1] / np.max(stress[:, 1])) / np.pi * 180
            x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))
            stress_magnitude = np.sqrt(stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)
            prediction_stress_magnitude = np.sqrt(stress[:, 6] ** 2 + stress[:, 7] ** 2 + stress[:, 8] ** 2)
            ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='CFD wallShearStress_magnitude')
            ax1.scatter(x_axis, prediction_stress_magnitude, c='r', s=1, label='Prediction Stress')
    print('111111')
    print('x', x_axis)
    ax1.set_xlabel('x axis (Angle_t)', fontsize=fontsize)

    if StrePres_t == 0:
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
    ax1.legend(loc="best", edgecolor='white', facecolor=None, framealpha=0)
    self.ui.TestFig.figure.tight_layout()
    self.ui.Results4.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    self.ui.TestFig.figure.canvas.draw()

    print('****' * 3)

    # ax1 = self.ui.TestFig.figure.add_subplot(1, 1, 1, label='plot3D')
    # print('111111')
    # ax1.set_xlabel('x axis (point number)', fontsize=fontsize)
    # if style == 0:
    #     ax1.plot(ori_test_data[:, y_index], c='b', label='pressure')
    #     ax1.set_title('Pressure', fontsize=fontsize)
    #     ax1.set_ylabel('y axis (Pressure)', fontsize=fontsize)
    # else:
    #     ax1.plot(ori_test_data[:, y_index1], c='b', label='wallShearStress_0')
    #     ax1.plot(ori_test_data[:, y_index2], c='r', label='wallShearStress_1')
    #     ax1.plot(ori_test_data[:, y_index3], c='k', label='wallShearStress_2')
    #     ax1.set_title('Stress', fontsize=fontsize)
    #     ax1.set_ylabel('y axis (Stress)', fontsize=fontsize)
    # print('3333333')
    # ax1.spines['bottom'].set_linewidth(bwith)
    # ax1.spines['left'].set_linewidth(bwith)
    # ax1.spines['top'].set_linewidth(bwith)
    # ax1.spines['right'].set_linewidth(bwith)
    # print('4444444')
    # # ax1.invert_yaxis()
    # # print('11113')
    # # ax1.set_xticks(fontproperties='Times New Roman')
    # # ax1.set_yticks(fontproperties='Times New Roman')
    #
    # ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    # ax1.legend(loc="best", edgecolor='white', facecolor=None, framealpha =0)
    # self.ui.TestFig.figure.tight_layout()
    # self.ui.Results4.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    # # self.ui.Results.setText('%s绘图完成')#%self.workbook.columns[index].split('\n')[0]
    # # ax1.show()
    # self.ui.TestFig.figure.canvas.draw()
    #
    #
    #
    #
    #
    #
