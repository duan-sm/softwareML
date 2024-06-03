#!/usr/bin/env python
# coding=utf-8

"""
@Time: 11/1/2023 11:38 AM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Outlier.py
@Software: PyCharm
"""

from outlier_function import *
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush
import pandas as pd
import os


def SaveDataBtn(self):
    try:
        self.ui.InputDataTableView.clear()
        # 获取整行和整列的值（数组）
        self.workbook = self.workbook2
        columns = [i.split('\n')[0] for i in self.workbook.columns]
        print(columns,len(columns))
        rows = len(self.workbook)
        workbook_np = self.workbook.values
        # print(sheet1.nrows)
        self.ui.InputDataTableView.setRowCount(rows)
        self.ui.InputDataTableView.setColumnCount(len(columns))
        for i in range(len(columns)):
            print('len(columns)', i)
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
        for num in range(2, 5):
            eval('self.ui.SelectFeature%d.clear()' % num)  # clear fig
            eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # clear fig
        print('*'*20,np.arange(rows),'*'*20)
        # 在table中显示数据
        for i in range(rows):
            print('range(rows)', i)
            rowslist = workbook_np[i, :]  # 获取excel每行内容
            # print(rowslist)
            for j in range(len(rowslist)):
                # 在tablewidget中添加行
                row = self.ui.InputDataTableView.rowCount()
                self.ui.InputDataTableView.insertRow(row)
                # 把数据写入tablewidget中
                newItem = QTableWidgetItem(str(rowslist[j]))
                self.ui.InputDataTableView.setItem(i, j, newItem)
        # print('setAlternatingRowColors(True)')
        self.ui.InputDataTableView.setAlternatingRowColors(True)
        # self.ui.DataReplaceBtn.setEnabled(False)
        # self.ui.DataReplaceBtn2.setEnabled(False)
        # self.ui.Results2.setText('保存异常值修正后数据完成')
        # self.ui.Results.setText('保存异常值修正后数据完成')
        # print('Results2.setText')
        self.ui.Results2.setText('Save the data after outlier correction is completed')
        # print('Results.setText')
        self.ui.Results.setText('Save the data after outlier correction is completed')
    except:
        # self.ui.Results2.setText('请输入数据后进行保存')
        self.ui.Results2.setText('Please input the data and save it')

def Outlier(self
              , original_data
              , path):
    '''
    The modified 3 sigma is used to determine outliers for each feature in the data.
    :param original_data: The DataFrame that stores the logging data 
    :param path: Save the analysis DataFrame after the analysis is complete 
    :return: Result after the outlier is deleted 
    '''
    print('Outlier is processing')
    data = original_data.copy()
    # Outlier processing and visualization 
    col = data.columns
    i_index = []
    self.index_ = []
    for i in range(len(col)):
        print('%d = %s'%(i, col[i]))
        _ = threesigma(data[col[i]]
                       , detail=False
                       , results_s='a'
                       , n=3) # 3 Sigma outlier screening 
        out_index = _[-1]
        i_index.append(i)
        # All abnormal indexes are stored 
        self.index_.append(out_index)
    print('The initial outlier calculation is complete') # The initial outlier calculation is complete 
    print(' ' * 5 + '*' * 5)

    # The number of outliers is counted 
    for i in range(len(self.index_)):
        print(i_index[i], col[i_index[i]])
        len_data = len(data)
        if len(self.index_[i]) == 0:
            self.ui.ResultsText1.append('\n')
            self.ui.ResultsText1.append('%s No outliers' % col[i_index[i]].split('\n')[0])
            continue
        if len(self.index_[i]):
            self.ui.ResultsText1.append('\n')
            self.ui.ResultsText1.append(' ' * 5 + '*' * 5)
            self.ui.ResultsText1.append(
                'The number of outliers in %s is: %d' % (col[i_index[i]].split('\n')[0], len(self.index_[i])))
    data_copy_ = np.array(data.values)

    # Print the number of outliers in ResultsText1 
    for i in range(len(self.index_)):
        if len(self.index_[i]) == 0:
            continue
        self.ui.ResultsText1.append('\n')
        self.ui.ResultsText1.append(' ' * 5 + '*' * 5)
        self.ui.ResultsText1.append(
            'Serial number %d feature: %s, changes in the number of outliers:' % (i, col[i_index[i]].split('\n')[0]))
        # Use the find_discontinue_data function to find and correct for continuous outliers
        _, in_ = find_discontinue_data(self.index_[i], data_long=10, ind_=True)
        num = 0
        adjust_index = []
        for j in in_:
            num += j[1] - j[0] + 1
            adjust_index.extend(self.index_[i][j[0]:j[1] + 1])
        self.ui.ResultsText1.append('The original quantity %d ' % (len(self.index_[i])))
        self.index_[i] = adjust_index
        self.ui.ResultsText1.append('becomes %d' % (num))

    # Count how many points there are 
    list_ = self.index_[0]
    for i in range(1, len(self.index_)):
        list_ = np.concatenate((list_, self.index_[i]))
    # self.ui.ResultsText1.append('-' * 5 + ' 结论 ' + '-' * 5)
    self.ui.ResultsText1.append('-' * 5 + ' CONCLUSION ' + '-' * 5)
    self.ui.ResultsText1.append('Total number of outliers data: %d' % len(list_))
    list_ = np.sort(list(set(list_)))
    self.ui.ResultsText1.append('Total number of outliers after correction: %d' % len(np.sort(list(set(list_)))))
    ab_point2 = find_same_diff_num(list_, np.arange(len(data)))
    if path:
        data.iloc[ab_point2[1], :].to_csv(path + r'\DataAfterOutliers.csv'
                                          , encoding='gb18030')
    else:
        data.iloc[ab_point2[1], :].to_csv('...//DataAfterOutliers.csv'
                                          , encoding='gb18030')

    return data


def OutlierDraw(self
                  , index
                  , path):
    '''
    Plots values and outliers for selected features.
    :param self: class
    :param index: ignore
    :param path: Plots values and outliers for selected features.
    :return: none
    '''
    bwith = 1
    fontsize = 13
    print('Start drawing 开始绘图')
    self.ui.StartWD.setEnabled(False)
    self.ui.EndWD.setEnabled(False)
    for vvv in ['FeatureFig2']:
    # for vvv in ['FeatureFig2', 'OtherFig']:
        eval('self.ui.%s.figure.clear()' % vvv)  # clear fig
    index = self.ui.SelectFeature2.currentIndex()
    print('plot(绘图)index=%d' % index)

    columns = [j.split('\n')[0] for j in self.workbook.columns]
    # xaxis = np.where('井深' == np.array(columns))[0][0]
    fea = self.workbook.columns[index].split('\n')[0]

    print('self.ui.StartWD.text(), self.ui.EndWD.text()')
    print(self.ui.StartWD.text(), self.ui.EndWD.text())
    SMD = float(self.ui.StartWD.text())
    EMD = float(self.ui.EndWD.text())
    if EMD < 1e-1:
        truncted = False
        print('No truncted')
    else:
        truncted = True
        print('truncted')

    col = self.workbook.columns
    data_copy_ = np.array(self.workbook.values)
    judge_var_index = index

    print('fea=', fea)
    # jingshen = self.workbook.iloc[:, xaxis].values
    # tezheng = self.workbook.iloc[:, index].values

    fea_index = np.arange(len(self.workbook))
    # if truncted:
    #     fea_index1 = np.where(jingshen > SMD)[0]
    #     fea_index[fea_index1] += 1
    #     fea_index2 = np.where(jingshen < EMD)[0]
    #     fea_index[fea_index2] += 1
    #     fea_index = np.where(fea_index > 1)[0]
    # else:
    #     fea_index = np.arange(len(jingshen))
    #     SMD = np.min(jingshen)
    #     EMD = np.max(jingshen)

    print('self.index_')
    for i in range(len(self.index_)):
        print(self.index_[i])
        print(sum(self.index_[i]))
        print('='*20)
    ab_point = find_same_diff_num(self.index_[index], fea_index)# Look for the same and different numbers in the two lists
    # print('*' * 20)
    count_point = np.zeros(len(self.workbook))
    # print('*'*20)
    # Whether to draw outliers 是否绘制异常点
    if len(ab_point[0]) > 0:
        count_point[ab_point[0]] = 1
        print('*' * 20)
        self.ui.DataOutlierDraw1.setCheckable(True)
    else:
        print('-' * 20)
        self.ui.DataOutlierDraw1.setCheckable(False)

    for vvv in ['FeatureFig2']: # plot figure for outlier
        ax1 = eval('self.ui.%s.figure.add_subplot(1, 1, 1, label=fea)' % vvv)
        print('self.ui.DataOutlierDraw1.isChecked()')
        print(self.ui.DataOutlierDraw1.isChecked())
        if self.ui.DataOutlierDraw1.isChecked():
            ax1.plot(self.workbook.iloc[fea_index, index].values, np.arange(len(fea_index)), 'r-',
                     label=fea)
            ax1.scatter(self.workbook.iloc[ab_point[0], index].values, ab_point[0],
                        c='b', label='Outliers')
            self.ui.Results2.setText('%s: index = %d, number of outlier=%d' % (
                col[judge_var_index].split('\n')[0], index, len(ab_point[0])))

        else:
            self.ui.Results2.setText('%s: index = %d, number of outlier=%d' % (
                col[judge_var_index].split('\n')[0], index, len(ab_point[0])))
            ax1.plot(self.workbook.iloc[fea_index, index].values, np.arange(len(fea_index)), 'r-',
                     label=fea)
        ax1.legend()
        ax1.set_xlabel(fea, fontsize=fontsize)
        ax1.set_ylabel('Number', fontsize=fontsize)
        ax1.set_title(fea, fontsize=fontsize)
        ax1.spines['bottom'].set_linewidth(bwith)
        ax1.spines['left'].set_linewidth(bwith)
        ax1.spines['top'].set_linewidth(bwith)
        ax1.spines['right'].set_linewidth(bwith)
        # ax1.invert_yaxis()
        ax1.tick_params(width=bwith, length=bwith * 2, labelsize=fontsize, direction='in')
        eval('self.ui.%s.figure.tight_layout()' % vvv)
        eval('self.ui.%s.figure.canvas.draw()' % vvv)
    self.ui.Results2.setText('%s draw is completed' % self.workbook.columns[index].split('\n')[0])
    print('ab_point')
    ab_point = find_same_diff_num(self.index_[index], np.arange(len(self.workbook)))
    print('count_point')
    count_point = np.zeros(len(self.workbook))
    print('len(ab_point[0]) > 0')
    if len(ab_point[0]) > 0:
        count_point[ab_point[0]] = 1
    print('draw_curve')
    print(data_copy_[:, judge_var_index])
    print(np.array(count_point))
    print(col[judge_var_index], )
    print(self.ui.DataOutlierDraw1.isChecked())
    print('%s Modified outliers(修正后的异常点)' % col[judge_var_index].split('\n')[0])
    print(path)
    # draw_curve(data_copy_[:, judge_var_index]
    #            , np.array(count_point)
    #            , other=self.ui.DataOutlierDraw1.isChecked()
    #            , option='abnormal_scatter'
    #            , point_s=6
    #            , xfigsize=(8, 6)
    #            , y_label=col[judge_var_index]
    #            , fontsize=15
    #            , xdpi=300
    #            , x_point=15
    #            , bwith=1.5
    #            , label=col[judge_var_index]
    #            , x_label='井深$\mathrm{(m)}$'
    #            , x_xticks=self.workbook[qwer].values
    #            , fig_name='%s修正后的异常点' % col[judge_var_index].split('\n')[0]
    #            , path=path
    #            )
    print('DataOutlierDraw1')
    self.ui.DataOutlierDraw1.setCheckable(True)