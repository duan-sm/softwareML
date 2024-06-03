#!/usr/bin/env python
# coding=utf-8

"""
@Time: 11/1/2023 12:17 PM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _FeatureAnalysis.py
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import numpy as np
from FunctionDataPreprocessing import chi2_contingency, mutualInfo
from PyQt5.QtWidgets import QTableWidgetItem,QDialog
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt
from myDimenDialog import QmyDimenDialog
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PyQt5.QtWidgets import QMessageBox
import seaborn as sns

def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0 means to find the mean by column. If list is input, axis=1
def PCA(data
        , limited_score=0.999
        , path=None):
    '''
    PCA dimension reduction was performed
    :param data: Enter pure data in the DataFrame format      输入纯数据，要求格式为DataFrame
    :param limited_score: Lower limit of cumulative contribution (should not be less than the contribution of the first principal component)     累计贡献度的下限（不应该少于第一主成分的贡献度）
    :param path: path to save the image and analysis results     保存图片和分析结果的路径
    :return:
    finalData: Principal component analysis results and cumulative contribution
    featValue: Sorts the feature values and outputs the descending order:
    gx: Contribution of eigenvalues:
    lg: Cumulative contribution of eigenvalues:
    '''
    df = data.copy()
    average = meanX(df)

    # View the number of columns and rows
    m, n = np.shape(df)
    print('columns(列数):%d,rows(行数)%d' % (m, n))

    # Mean value matrix
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    # decentralization
    data_adjust = df - avgs
    # Covariance matrix
    covX = np.cov(data_adjust.T)  # Calculate the covariance matrix
    # print(covX)

    # The eigenvalues and eigenvectors of the covariance matrix are calculated
    featValue, featVec = np.linalg.eig(covX)  # The eigenvalues and eigenvectors of the covariance matrix are solved
    # print(featValue, featVec)
    # Sort the eigenvalues and output descending order
    featValue = sorted(featValue)[::-1]

    # Draw scatter plots and line plots
    # Draw scatter plots and line plots with the same data
    plt.scatter(range(1, df.shape[1] + 1), featValue)
    plt.plot(range(1, df.shape[1] + 1), featValue)

    # Displays the title of the graph and the name of the xy axis
    # Ignore
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")

    plt.grid()  # Display grid
    plt.show()  # Display graphics

    # Find the contribution degree of the eigenvalue
    gx = featValue / np.sum(featValue)

    # Find the cumulative contribution degree of eigenvalues
    lg = np.cumsum(gx)

    # Selected principal component
    k = [i for i in range(len(lg)) if lg[i] < limited_score]
    k = list(k)
    print(k)

    # The eigenvector matrix corresponding to the principal component is selected
    selectVec = np.matrix(featVec.T[k]).T
    selectVe = selectVec * (-1)
    # print(selectVec)

    # Principal component score
    finalData = np.dot(data_adjust, selectVec)

    # Mapping heat map
    _ = selectVe.shape
    plt.figure(figsize=(4, 3 * _[0] / _[1]), dpi=300)
    ax = sns.heatmap(selectVec, annot=True, cmap='bwr'
                     , fmt=".4f"  # Format the numbers in the output graph, that is, keep the decimal places, etc 格式化输出图中数字，即保留小数位数等
                     , annot_kws={'size': 8, 'weight': 'normal', 'color': '#253D24'},  # Numeric property Settings, such as size, pound value, color 数字属性设置，例如字号、磅值、颜色
                     # mask=np.triu(np.ones_like(selectVec, dtype=np.bool)),  # Displays part of the diagram below the diagonal line 显示对脚线下面部分图
                     square=True, linewidths=.5,  # Each square frame is displayed, and the width of the outer frame is set 每个方格外框显示，外框宽度设置
                     cbar_kws={"shrink": .5})
    font2 = {
        'family': 'Times New Roman'
        , 'color': "black"
        , 'weight': 'normal'
        , 'size': 10
    }
    # Set the Y-axis font size
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    plt.title("Factor Analysis", fontdict=font2)

    # Set the Y-axis label
    plt.ylabel("Sepal Width", fontdict=font2)
    if path:
        plt.savefig(path+r'\Principal component analysis eigenvector matrix.jpg'
                   ,bbox_inches="tight")
        pd.DataFrame(finalData).to_csv(path+r'\Principal component analysis results.csv'
                                       , encoding='gb18030')
    else:
        plt.savefig(r'...\Principal component analysis eigenvector matrix.jpg'
                    , bbox_inches="tight")
        pd.DataFrame(finalData).to_csv(r'...\Principal component analysis results.csv'
                                       , encoding='gb18030')

    plt.show()
    # time.sleep(10)
    plt.close()
    return [finalData,featValue,gx,lg,selectVec]

def chi2Btn(self):
    '''
    Chi-square test is used to analyze the relationship between features.
    '''
    index = self.ui.SelectFeature3.currentIndex()
    print('*index' * 10)
    data_cols_sim = np.array([i.split('\n')[0] for i in self.workbook.columns])
    print('data_cols_sim=', data_cols_sim)
    for i in range(len(data_cols_sim)):
        if i == index:
            continue
        print('i=',i)
        try:
            print('chi2_contingency')
            kf = chi2_contingency([self.workbook.iloc[:, i].values, self.workbook.iloc[:, index].values])
            print('kf=',kf)
            self.ui.ResultsText2.append('The chi-square test results of feature *%s* and feature *%s* are:'
                                        % (data_cols_sim[i], data_cols_sim[index]))
            self.ui.ResultsText2.append('Chi-square value=%.4f, P value=%.4f, degrees of freedom=%i'
                                        % (kf[0], kf[1], kf[2]))
            self.ui.ResultsText2.append('\n')
        except:
            print('except')
            self.ui.ResultsText2.append('COME TO NOTHING: All values in "observed" must be nonnegative.')
    self.ui.ResultsText2.append(' ' * 10 + '-' * 10 + ' ' * 10)


def mutualInforBtn(self):
    '''
    The relationship between features is analyzed by mutual information method.
    '''
    index = self.ui.SelectFeature3.currentIndex()
    data_cols_sim = np.array([i.split('\n')[0] for i in self.workbook.columns])
    for i in range(len(data_cols_sim)):
        mutual_info = mutualInfo(self.workbook.iloc[:, i].values, self.workbook.iloc[:, index].values)
        # '特征*%s*与特征*%s*的互信息结果为：%.4f'
        self.ui.ResultsText2.append('The mutual information result of feature *%s* and feature *%s* is: %.4f'
                                    % (data_cols_sim[i], data_cols_sim[index], mutual_info))
    self.ui.ResultsText2.append(' ' * 10 + '-' * 10 + ' ' * 10)


def DimenComBtn(self):
    '''
    The principal component analysis is used to process the data and reduce the data dimension. (Processed data will lose its physical meaning)
    '''
    self.ui.DimenComBtn.setEnabled(False)
    self.ui.NondimenBtn.setEnabled(True)
    self.ui.ResultsText2.append('Because it is a regression question, this button is not enable.')
    return 0
    columns = np.array([i.split('\n')[0] for i in self.workbook.columns])
    if (self._DimDialog == None):  # No dialog box created 
        self._DimDialog = QmyDimenDialog(self, column=columns)
    res = self._DimDialog.exec()
    if (res == QDialog.Accepted):
        print('*1'*10)
        index = self._DimDialog.index
        if index == -1:
            self.ui.ResultsText2.append('Please re-select')
            print('*2' *10)
        else:
            # Processed data will lose its physical meaning. So we need save some features.
            self.all_y = self.workbook.iloc[:, -4:-1].values
            self.all_y_index = self.workbook.columns[-4:-1]
            new_index = np.where(np.arange(len(columns)) != index)[0]
            print('*3' * 10)
            for num in range(2, 5):
                print('num=%d'%num)
                eval('self.ui.SelectFeature%d.clear()' % num)  # clear fig
                eval('self.ui.SelectFeature%d.addItems(columns[new_index])' % num)  # clear fig
            print('*4' * 10)
            self.workbook = self.workbook.iloc[:, new_index]
            limited_socre = self.ui.DimenLImitedScore.text()
            limited_socre = float(limited_socre)
            print('*5' * 10)
            [self.workbook3, featValue, gx, lg, selectVec] = PCA(self.workbook
                                                                 , limited_score=limited_socre
                                                                 , path=os.getcwd() + '\\' + self.Wellname)
            # '对协方差阵的特征值进行排序并输出(降序):'
            self.ui.ResultsText2.append('The eigenvalues of the covariance matrix are in descending order:')
            self.ui.ResultsText2.append(str(featValue))
            # '特征值的贡献度: '
            self.ui.ResultsText2.append('Contribution of eigenvalues: ')
            self.ui.ResultsText2.append(str(gx))
            # '特征值的累计贡献度:'
            self.ui.ResultsText2.append('Cumulative contribution of eigenvalues:')
            self.ui.ResultsText2.append(str(lg))
            self.ui.ResultsText2.append(' ' * 10 + '*' * 10)
            for vvv in ['Fig3']:  # Plot figure of dimensionality reduction eigenvalue
                eval('self.ui.%s.figure.clear()' % vvv)
                ax1 = eval(
                    'self.ui.%s.figure.add_subplot(1, 1, 1, label="Dimensionality reduction eigenvalue results")' % vvv)
                im = ax1.imshow(selectVec
                                # , cmap='autumn_r'
                                , cmap='hot'
                                )
                im_bar = eval('self.ui.%s.figure.colorbar(im)' % vvv)
                fontsize = 18
                bwith = 1.5
                ax1.set_title('Dimensionality reduction analysis', fontsize=fontsize)
                ax1.spines['bottom'].set_linewidth(bwith)
                ax1.spines['left'].set_linewidth(bwith)
                ax1.spines['top'].set_linewidth(bwith)
                ax1.spines['right'].set_linewidth(bwith)
                ax1.set_xticks(np.arange(selectVec.shape[1]))
                ax1.set_xticklabels(np.arange(selectVec.shape[1]), fontsize=fontsize)
                ax1.tick_params(width=bwith, length=bwith * 2, labelsize=fontsize)

                eval('self.ui.%s.figure.tight_layout()' % vvv)
    else:
        self.ui.ResultsText2.append('Please re-select')
        self.ui.ResultsText2.append('Dimensionality reduction analysis is completed')

def SaveData(self):
    '''
    The corrected data presentation will be modified
    '''
    if self.state == 2:
        # Correlation coefficient
        self.workbook = self.workbook_
    else:
        # PCA, NondimenBtn
        self.workbook = self.workbook3
    columns = [i.split('\n')[0] for i in self.workbook.columns]
    print(columns)
    rows = len(self.workbook)
    workbook_np = self.workbook.values
    # print(sheet1.nrows)
    self.ui.InputDataTableView.setRowCount(rows)
    self.ui.InputDataTableView.setColumnCount(len(columns))
    # Show the corrected data in the QTableWidget
    for i in range(len(columns)): # Table Title
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
    for num in range(2, 5):
        eval('self.ui.SelectFeature%d.clear()' % num)  # Clear Feature 
        eval('self.ui.SelectFeature%d.addItems(columns)' % num)  # Add Feature 

    # Displays the data in the table 在table中显示数据
    for i in range(rows):
        rowslist = workbook_np[i, :]  # Get each line of excel content 
        # print(rowslist)
        for j in range(len(rowslist)):
            # Add rows to the tablewidget 
            row = self.ui.InputDataTableView.rowCount()
            self.ui.InputDataTableView.insertRow(row)
            # Write data to the tablewidget 
            newItem = QTableWidgetItem(str(rowslist[j]))
            self.ui.InputDataTableView.setItem(i, j, newItem)
    self.ui.InputDataTableView.setAlternatingRowColors(True)
    # self.ui.DataReplaceBtn.setEnabled(False)
    # self.ui.DataReplaceBtn2.setEnabled(False)
    if self.state == 0:
        self.ui.ResultsText2.append('Use dimensionality reduction data')
        self.ui.Results2.setText('Use dimensionality reduction data')
        self.ui.Results.setText('Use dimensionality reduction data')
    elif self.state == 1:
        self.ui.ResultsText2.append('Use normalized data')
        self.ui.Results2.setText('Use normalized data')
        self.ui.Results.setText('Use normalized data')
    elif self.state == 2:
        self.ui.ResultsText2.append('Use correlation data')
        self.ui.Results2.setText('Use correlation data')
        self.ui.Results.setText('Use correlation data')


def CorrDraw(self
               , path):
    '''
    Pearson correlation coefficient and Spearman correlation coefficient are used to calculate the correlation coefficient between each feature.
    '''
    self.state = 2
    bwith = 1
    fontsize = 13
    for vvv in ['Fig3']:
    # for vvv in ['Fig3', 'OtherFig']:
        eval('self.ui.%s.figure.clear()' % vvv)  # clear fig
    index = self.ui.SelectFeature3.currentIndex()
    print('draw(绘图)index=%d' % index)

    data_columns = np.array([i.split('\n')[0] for i in self.workbook.columns])
    fea = self.workbook.columns[index].split('\n')[0]

    limited_corr = float(self.ui.CorrLimitedScore.text())
    limited_var = float(self.ui.CorrLimitedVar.text())
    print('limited_corr, limited_var')
    print(limited_corr, limited_var)
    np_data = self.workbook.values
    # Standardize the data
    Intermediate_data = []
    standard_var = []
    print('*' * 20, np_data.shape, '*' * 20)
    for kind in range(np_data.shape[1]):
        # print('kind=' ,kind)
        data_col = (np_data[:, kind] - np.mean(np_data[:, kind])) / (np.std(np_data[:, kind]))
        Intermediate_data.append(data_col)
        standard_var.append(np.std(np_data[:, kind]))
        # print('*' * 20)
    np_data_STD = np.array(Intermediate_data).T
    self.ui.ResultsText2.append('Effective features with variance greater than %.2f are:' % limited_var)
    standard_var = np.array(standard_var)

    vaild_cols = data_columns[standard_var > limited_var]
    # print('vaild_cols:',vaild_cols)
    self.ui.ResultsText2.append(str(vaild_cols))
    self.ui.ResultsText2.append('The remaining features are:')
    self.ui.ResultsText2.append(str(data_columns[standard_var <= limited_var]))
    vaild_np_data_STD = np_data_STD[:, np.where(standard_var > limited_var)[0]]
    vaild_np_data_STD = pd.DataFrame(vaild_np_data_STD, columns=vaild_cols)
    corr1 = vaild_np_data_STD.corr(method='pearson')
    corr1_0 = corr1[corr1 > limited_corr]
    corr1_1 = corr1[corr1 < -limited_corr]
    corr1_0 = corr1_0.fillna(0)
    corr1_1 = corr1_1.fillna(0)
    corr_1 = corr1_0 + corr1_1

    corr3 = vaild_np_data_STD.corr(method='spearman')
    corr1_0 = corr3[corr3 > limited_corr]
    corr1_1 = corr3[corr3 < -limited_corr]
    corr1_0 = corr1_0.fillna(0)
    corr1_1 = corr1_1.fillna(0)
    corr_3 = corr1_0 + corr1_1
    corr_ = corr_1 + corr_3  # corr_2+
    corr_ = corr_.fillna(0)
    # plot a correlation figure
    for vvv in ['Fig3']:
    # for vvv in ['Fig3', 'OtherFig']:
        # self.ui.%s.figure.clear() # clear fig
        ax1 = eval('self.ui.%s.figure.add_subplot(1, 1, 1, label=fea)' % vvv)
        im = ax1.imshow(corr_
                        # , cmap='autumn_r'
                        , cmap='hot'
                        )
        im_bar = eval('self.ui.%s.figure.colorbar(im)' % vvv)
        ax1.set_title('Correlation Analysis', fontsize=fontsize)
        ax1.spines['bottom'].set_linewidth(bwith)
        ax1.spines['left'].set_linewidth(bwith)
        ax1.spines['top'].set_linewidth(bwith)
        ax1.spines['right'].set_linewidth(bwith)
        # print(np.arange(len(vaild_cols)), vaild_cols)
        ax1.set_xticks(np.arange(len(vaild_cols)))
        ax1.set_xticklabels(vaild_cols, rotation=90, fontsize=fontsize)
        ax1.set_yticks(np.arange(len(vaild_cols)))
        ax1.set_yticklabels(vaild_cols, fontsize=fontsize)
        ax1.tick_params(width=bwith, length=bwith * 2, labelsize=fontsize)

        eval('self.ui.%s.figure.tight_layout()' % vvv)

    if path:
        self.ui.Fig3.figure.savefig(path + r'\CorrelationCoefficient.jpg'
                                    , bbox_inches="tight")
        corr_.to_csv(path + r'\CorrelationCoefficient.csv'
                     , encoding='gb18030')
    else:
        self.ui.Fig3.figure.savefig(r'...\CorrelationCoefficient.jpg'
                                    , bbox_inches="tight")
        corr_.to_csv(r'...\CorrelationCoefficient.csv'
                     , encoding='gb18030')
    self.ui.ResultsText2.append('The drawing is completed, save the CorrelationCoefficient.jpg')
    self.workbook_ = self.workbook[vaild_cols]

    return corr_



def NondimenBtn(self, original_data):
    '''
    数据无量纲化
    The data is processed without dimension. It can be divided into standardization method and maximum and minimum method
    :param original_data: The DataFrame that stores the logging data
    :param s_m: Dimensionless methods, including "M" max-min normalization, "S" standardization
    :param path: Save the analysis path after the analysis is complete
    :return: The result after the data dimension
    '''
    index = self.ui.DimensionlessType.currentIndex()
    fangfa = ['Standard', 'MaxMin'] # different mode
    if index == 0:
        s_m = 'S' # Standard
        i = 0
    else:
        s_m = 'M' # MaxMin
        i = 1

    s_m = s_m
    path = os.getcwd() + '\\' + self.Wellname
    # original_data = self.workbook.values
    # print('data type: type(data)', type(original_data))
    if isinstance(original_data, pd.core.frame.DataFrame):
        print('DataFrame')
        input_data = original_data.values
        input_data_cols = original_data.columns
    else:
        print('np.ndarray')
    Intermediate_data = []
    # Determine whether to determine the label feature
    if isinstance(self.all_y, type(None)):
        print('all data is normalization')
        if s_m == 'M':
            print('s_m == M')
            max_min = MinMaxScaler().fit(input_data)
            self.data_max_ = max_min.data_max_
            self.data_min_ = max_min.data_min_
            input_data = max_min.transform(input_data)
            print('self.data_max_, self.data_min_')
            print(self.data_max_, self.data_min_)
        else:
            print('s_m == S')
            standard = StandardScaler().fit(input_data)
            self.mean_ = standard.mean_
            self.var_ = standard.var_
            input_data = standard.transform(input_data)
            print('self.mean_, self.var_')
            print(self.mean_, self.var_)
        # print('NondimenBtn finished')
    else:
        # print('Have Y')
        dlgTitle = "Waring Error"
        strInfo = ("Please normalize the data before dividing the label "
                   "(please re-enter the imported data run, otherwise an error occurs)")
        defaultBtn = QMessageBox.NoButton  # Default button
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.tabWidget.setTabEnabled(3, False)
    if isinstance(original_data, pd.core.frame.DataFrame):
        # print('Start save data')
        input_data = pd.DataFrame(input_data,
                                  columns=input_data_cols)
        # print('input_data')
        # print('./%s/无量纲后的数据.csv')
        input_data.to_csv('./%s/Data after no dimension.csv'%self.Wellname
                          , encoding='gb18030')
        # print('Save complete')
    self.ui.NondimenBtn.setEnabled(False)
    return input_data