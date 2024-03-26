#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 10:53
# @Author  : Shiming Duan
# @Email   : 1124682706@qq.com
# @File    : software_IntelligenceModel.py
# @Software: PyCharm
import sys
import matplotlib as mpl
from PyQt5.QtGui import QFont, QBrush, QPalette
from PyQt5.QtCore import pyqtSlot, QDir, Qt
from IntelligentModelInterface import Ui_Interface #Ui_Interface
from QDataLabel_select import Ui_DataLabel_select
from FunctionDataPreprocessing import *
from _ReadData import *
from _Outlier import *
from _FeatureAnalysis import *
from _Model import *
from _Test import *

'''
打包方法：
找到pyinstaller.exe的路径：C:\\Users\\LSD\\AppData\\Roaming\\Python\\Python39\\Scripts
利用cmd进入此路径,输入
pyinstaller -F -w -i G:\\F盘\\1师兄任务\\2022春季学期\\15呼图壁\\0929\\呼图壁源代码\\tu.ico  G:\\F盘\\1师兄任务\\2022春季学期\\15呼图壁\\0929\\呼图壁源代码\\software_IntelligenceModel.py
完成
https://blog.csdn.net/m0_47682721/article/details/124008653
尚未尝试
https://blog.csdn.net/flyskymood/article/details/123668136
'''
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        '''
        initializaiton
        Define the parameters of the window
        '''
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_Interface()
        self.ui.setupUi(self)
        # Set plot parameters
        mpl.rcParams['font.sans-serif'] = ['Times New Roman']
        mpl.rcParams['font.size'] = 9  # 显示汉字
        mpl.rcParams['axes.unicode_minus'] = False  # 减号unicode编
        self.setCentralWidget(self.ui.tabWidget)
        self.ui.tabWidget.setCurrentIndex(0)
        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)
        self.ui.Results.setPalette(pe)
        self.ui.Results2.setPalette(pe)

        # DimensionlessType = ['Standard', 'MaxMin']
        # self.ui.DimensionlessType.addItems(DimensionlessType)
        self.size_ = 0 # A variable defined earlier that represents the number of times the modified data has been saved 早期定义的变量，表示保存修改后的数据次数
        self._DimDialog=None # A variable that records whether dimensionality has been reduced 记录是否进行了降维处理的变量
        self.all_y = None # Save the output of the model 保存模型的输出
        self.nondim = 'No' #A variable that records whether dimensionless processing is performed 记录是否进行了无量纲化处理的变量
        self.all_y_index = None # Saves the index of the output of the model 保存模型的输出的索引
        self.state = -1 # Record the status of the data 记录数据的状态

        # Add a value to the interface 在界面上添加数值
        self.ui.Style0.addItems(['Pressure'])
        self.ui.Style0.addItems(['Stress'])
        self.ui.Style1.addItems(['Pressure'])
        self.ui.Style1.addItems(['Stress'])
        self.ui.Region.addItems(['0'])
        self.ui.Region.addItems(['1'])
        self.ui.Region.addItems(['2'])
        # self.ui.Region_t.addItems(['0'])
        # self.ui.Region_t.addItems(['1'])
        # self.ui.Region_t.addItems(['2'])
        # self.ui.StrePres_t.addItems(['Pressure'])
        # self.ui.StrePres_t.addItems(['Stress'])
        self.ui.Test.setEnabled(False)
        self.ui.DrawSliceT.setEnabled(False)
        self.ui.ModelTypeSelect.setCurrentIndex(0)
        self.ui.Model_tab.setCurrentIndex(0)
    # page changed after the prompt page改变后提示
    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self):
        print('on_tabWidget_currentChanged')
        initial(self=self)


    '''
    The function in the page of 'ReadData' interface
    It including:
    on_DataInputBtn_clicked
    on_merge_clicked
    on_btnDelCols_clicked
    '''
    # Input data 输入数据
    @pyqtSlot(bool)
    def on_DataInputBtn_clicked(self):
        print('44444adfa dfasdf ')
        # Read data 读取数据
        DataInputBtn(self=self)
        # Shows the position of the maximum pressure/stress 显示最大压力/应力的位置
        try:
            print('111')
            cols = self.workbook.columns
            values = self.workbook.dropna(axis=0, how='any').values
            if 'p' in cols:
                cols_index = np.where('p'==cols)[0]
            else:
                cols_index = np.where('wallShearStress_Magnitude'==cols)[0]
            print('222')
            print('cols_index',cols_index)
            val = values[:,cols_index]
            max_val = np.max(val)
            print('max_val',max_val)
            index = np.where(max_val==val)[0][0]
            print('index', index)
            print(val.shape)
            x = values[index, 0]
            y = values[index, 1]
            z = values[index, 2]
            print('x,y,z', x,y,z)
            print('333')
            if 'p' in cols:
                self.ui.max_.setText('MaxPressure: %.4f'%max_val)
                self.ui.max_x.setText('X: %.4f' % x)
                self.ui.max_y.setText('Y: %.4f' % y)
                self.ui.max_z.setText('Z: %.4f' % z)
                self.ui.Style0.setCurrentIndex(0)
            else:
                self.ui.max_.setText('MaxStress: %.4f' % max_val)
                self.ui.max_x.setText('X: %.4f' % x)
                self.ui.max_y.setText('Y: %.4f' % y)
                self.ui.max_z.setText('Z: %.4f' % z)
                self.ui.Style0.setCurrentIndex(1)
            print('444')
        except:
            self.ui.Results.setText('Please input the data before drawing')


    @pyqtSlot(bool)
    def on_merge_clicked(self):
        """
        # 数据合并
        Early function, can be ignored
        """
        merge(self=self)



    @pyqtSlot(bool)
    def on_btnDelCols_clicked(self):
        '''
        # Delete empty column 删除列
        :return: none
        '''
        dlgTitle = "Tips"
        strInfo = ("This processing will directly change the data content."
                   "The 'SaveData' button is used to save the result after deleting the null value and update the table data below.")
        defaultBtn = QMessageBox.NoButton  # Default button 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)

        btnDelCols(self=self)



    @pyqtSlot(bool)
    def on_btnDelRows_clicked(self):
        '''
        # Delete empty rows 删除行
        :return: none
        '''
        dlgTitle = "Tips"
        strInfo = ("This processing will directly change the data content."
                   "The 'SaveData' button is used to save the result after deleting the null value and update the table data below.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        btnDelRows(self=self)



    @pyqtSlot(bool)
    def on_btnSaveData_clicked(self):
        '''
        # Save data 保存数据
        :return: none
        '''
        btnSaveData(self=self)

    # 选择绘制特征
    '''
    ?????????
    '''
    # @pyqtSlot(str)
    # def on_SelectFeature1_currentIndexChanged(self, curText):
    #     print(curText)
    #     index = self.ui.SelectFeature1.currentIndex()
    #     # self.ui.Results.setText('当前选中绘图变量：%s，索引是%d'%(curText,index))
    #     self.ui.Results.setText('Currently selected variable: %s, index is %d' % (curText, index))



    # @pyqtSlot(bool)
    # def on_OutlierDeaklBtn_clicked(self):
    #     self.ui.tabWidget.setCurrentIndex(1)
    #
    # @pyqtSlot(bool)
    # def on_DataCorrelationBtn_clicked(self):
    #     self.ui.tabWidget.setCurrentIndex(2)


    # Plot slice 绘图
    @pyqtSlot(bool)
    def on_DrawSlice_clicked(self):
        try:
            DrawFig(self)
            # self.ui.FeatureFig.figure.canvas.draw()
            # self.ui.FeatureFig.figure.draw()
            # self.ui.FeatureFig.figure.show()
        except:
            # self.ui.Results.setText('请输入数据后在进行绘图')
            self.ui.Results.setText('Please input the data before drawing')

    # Draw the center curve of the pipe 绘制管道中心曲线
    @pyqtSlot(bool)
    def on_WholePipe_clicked(self):
        try:
            DrawWholePipe(self)
            # self.ui.FeatureFig.figure.canvas.draw()
            # self.ui.FeatureFig.figure.draw()
            # self.ui.FeatureFig.figure.show()
        except:
            # self.ui.Results.setText('请输入数据后在进行绘图')
            self.ui.Results.setText('Please input the data before drawing')

    '''
    Outlier_interface
    '''
    # Display select the feature to be drawn 显示选择将要绘制的特征
    @pyqtSlot(str)
    def on_SelectFeature2_currentIndexChanged(self, curText):
        print(curText)
        index = self.ui.SelectFeature2.currentIndex()
        # self.ui.Results2.setText('当前选中绘图变量：%s，索引是%d' % (curText, index))
        self.ui.Results2.setText('Currently selected drawing variable: %s, index is %d' % (curText, index))
        try:
            num = len(self.index_[index])
            self.ui.OutlierNumber.setText(str(num))
        except:
            self.ui.OutlierNumber.setText('0')
            # self.ui.Results2.setText('请首先计算筛选异常值')
            self.ui.Results2.setText('Please calculate filter outliers first')

    # Set drawing outliers 绘制异常点
    @pyqtSlot()
    def on_DataOutlierDraw1_clicked(self):
        self.ui.DataOutlierDraw1.setChecked(True)
        self.ui.DataOutlierDraw2.setChecked(False)

    # Set not to draw outliers 不绘制异常点
    @pyqtSlot()
    def on_DataOutlierDraw2_clicked(self):
        self.ui.DataOutlierDraw1.setChecked(False)
        self.ui.DataOutlierDraw2.setChecked(True)

    # Return to page1 返回page1
    @pyqtSlot(bool)
    def on_Return1_clicked(self):
        self.ui.tabWidget.setCurrentIndex(0)

    #Calculated outlier 计算异常点
    @pyqtSlot(bool)
    def on_computeOutlier_clicked(self):
        try:
            dlgTitle = "Tips"
            strInfo = ("This processing will not directly change the data content"
                       ", please click 'SaveData' on this page to confirm the deletion of outliers.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            # print('Outlier???')
            # print('self.workbook')
            # print(self.workbook,os.getcwd() + '\\' + self.Wellname)
            self.workbook2 = Outlier(self, self.workbook
                                            , path=os.getcwd() + '\\' + self.Wellname)
            # self.ui.Results2.setText('异常值计算完成，具体结果可查看：处理异常值后的数据.csv')
            self.ui.Results2.setText('Outliers is completed, the results can be viewed: DataAfterOutliers.csv')
        except:
            # self.ui.Results2.setText('请输入数据后再进行计算筛选')
            self.ui.Results2.setText('Please input the data before calculating and filtering')

    @pyqtSlot(bool)
    def on_OutlierDrawBtn_clicked(self):
        try:
            index = self.ui.SelectFeature2.currentIndex()
            # print('点击绘图')
            print('Click draw a figure')
            OutlierDraw(self, index, path=os.getcwd() + '\\' + self.Wellname)
        except:
            # self.ui.Results2.setText('请输入数据后再进行绘图')
            self.ui.Results2.setText('Please input data before drawing')

    @pyqtSlot(bool)
    def on_SaveDataBtn_clicked(self):
        SaveDataBtn(self=self)

    @pyqtSlot(int)
    def on_Style0_currentIndexChanged(self):
        try:
            print('on_Style0_currentChanged')
            index = self.ui.Style0.currentIndex()
            print('index=',index)
            self.ui.Style1.setCurrentIndex(index)
            self.ui.StrePres_t.setCurrentIndex(index)
            print(self.ui.Style1.currentIndex())
        except:
            print('error about index changed')
    '''
    FeatureAnalysis_interface
    '''
    @pyqtSlot(bool)
    def on_Return2_clicked(self):
        self.ui.tabWidget.setCurrentIndex(0)

    @pyqtSlot(str)
    def on_SelectFeature3_currentIndexChanged(self, curText):
        print('  ' * 7 + '*' * 5)
        try:
            print(curText)
            index = self.ui.SelectFeature3.currentIndex()
            # self.ui.ResultsText2.append('当前选中绘图变量：%s，索引是%d' % (curText, index))
            self.ui.ResultsText2.append('Currently selected variable: %s, index is %d' % (curText, index))
        except:
            # self.ui.ResultsText2.append('请首先输入数据')
            self.ui.ResultsText2.append('Please input data first')

    @pyqtSlot(str)
    def on_DimensionlessType_currentIndexChanged(self, curText):
        print('  ' * 7 + '*' * 5)
        print(curText)
        index = self.ui.DimensionlessType.currentIndex()
        if index == 0:
            type_ = 'Standard'
        else:
            type_ = 'MaxMin'
        # self.ui.ResultsText2.append('当前选中归一化方法为：%s，索引是%d' % (type_, index))
        self.ui.ResultsText2.append('Normalization method is: %s, and the index is %d' % (type_, index))


    @pyqtSlot(bool)
    def on_CorrComBtn_clicked(self):
        dlgTitle = "Tips"
        strInfo = ("This processing will not directly change the data content"
                   ", please click 'SaveData' on this page to confirm the deletion of outliers.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        try:
            # for vvv in ['Fig3','OtherFig']:
            for vvv in ['Fig3']:
                eval('self.ui.%s.figure.clear()'%vvv)  # 清除图表
            CorrDraw(self, path=os.getcwd()+'\\'+self.Wellname)
            index = self.ui.SelectFeature3.currentIndex()
            print('绘图index=%d' % index)
        except:
            # self.ui.ResultsText2.append('请首先输入数据，保证数据全为数字')
            self.ui.ResultsText2.append('Please input the data first, make sure the data is all numbers')

    @pyqtSlot(bool)
    def on_chi2Btn_clicked(self):
        try:
            chi2Btn(self=self)
        except:
            # self.ui.ResultsText2.append('请首先输入数据，保证数据全为数字')
            self.ui.ResultsText2.append('Please input the data first, make sure the data is all numbers')

    @pyqtSlot(bool)
    def on_mutualInforBtn_clicked(self):
        try:
            mutualInforBtn(self=self)
        except:
            # self.ui.ResultsText2.append('请首先输入数据，保证数据全为数字')
            self.ui.ResultsText2.append('Please input the data first, make sure the data is all numbers')

    @pyqtSlot(bool)
    def on_DimenComBtn_clicked(self):
        self.state = 0
        self.ui.NondimenBtn.setEnabled(False)
        dlgTitle = "DimensionReductionConfirmation"
        strInfo = ("You will lose the meaning of each column after dimension reduction. Are you sure?"
                   "But the data will not directly overwrite the original data until you click on 'SvaeData' on this page.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                      defaultBtn)
        if (result == QMessageBox.Yes):
            try:
                DimenComBtn(self)
            except:
                # self.ui.ResultsText2.append('请输入数据后再点击降维/降维贡献度阈值为0-1')
                self.ui.ResultsText2.append(
                    'Please input the data and then click dimensionality reduction (threshold is 0-1)')
        elif (result == QMessageBox.No):
            self.ui.ResultsText2.append("DimensionReductionConfirmation: No is selected")
        elif (result == QMessageBox.Cancel):
            self.ui.ResultsText2.append("DimensionReductionConfirmation: Cancel is selected")
        else:
            self.ui.ResultsText2.append("DimensionReductionConfirmation: Nothing is selected")
        # elif (result == QMessageBox.No):
        #     self.ui.ResultsText2.append("Question消息框: No 被选择")
        # elif (result == QMessageBox.Cancel):
        #     self.ui.ResultsText2.append("Question消息框: Cancel 被选择")
        # else:
        #     self.ui.ResultsText2.append("Question消息框: 无选择")


    @pyqtSlot(bool)
    def on_NondimenBtn_clicked(self):
        self.ui.DimenComBtn.setEnabled(False)
        self.state = 1
        dlgTitle = "Tips"
        strInfo = ("This processing will not directly change the data "
                   ", please click 'SaveData' on this page to confirm the deletion of outliers.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        try:
            self.workbook3 = NondimenBtn(self, self.workbook)
            # self.ui.ResultsText2.append('选择归一化方法为：%s，完成'%fangfa[i])
            self.ui.ResultsText2.append('Select the normalization method as: %s, completed' % self.ui.DimensionlessType.currentText())
        except:
            # self.ui.ResultsText2.append('请输入数据后再点击归一化/选择归一化方法')
            self.ui.ResultsText2.append('Please input the data before normalizing')

    @pyqtSlot(bool)
    def on_SaveData_clicked(self):
        try:
            SaveData(self=self)
        except:
            self.ui.ResultsText2.append('Please process the data first and then click Save')
            # self.ui.ResultsText2.append('请首先处理数据后再点击保存')

    '''
    Model_interface
    '''

    @pyqtSlot(str)
    def on_SelectFeature4_currentIndexChanged(self, curText):
        print('SelectFeature4 clicked')
        print(curText)


    @pyqtSlot(str)
    def on_ModelTypeSelect_currentIndexChanged(self, curText):
        print('clicked')
        print(curText)
        try:
            index = self.ui.ModelTypeSelect.currentIndex()
            print('ModelTypeSelect index=',index)
            self.ui.Model_tab.setCurrentIndex(index)
            print('Model_tab index=', self.ui.Model_tab.CurrentIndex())
        except:
            self.ui.Results3.setText('Error')

    @pyqtSlot(str)
    def on_Style1_currentIndexChanged(self, curText):
        print('clicked')
        print(curText)
        try:
            index = self.ui.Style1.currentIndex()
            self.ui.Style0.setCurrentIndex(index)
            self.ui.StrePres_t.setCurrentIndex(index)
            columns = np.array([i.split('\n')[0] for i in self.workbook.columns])
            if index == 1:
                self.ui.VariableOutput.clear()
                cols = ''
                for i in columns[-4:-1]:
                    cols = cols + str(i) + ','
                # self.ui.VariableOutput.setText(cols)
                self.ui.VariableOutput.setText('stress0, stress1, stress2')
                print('columns[-4:-1]', columns[-4:-1])
            else:
                self.ui.VariableOutput.clear()
                self.ui.VariableOutput.setText('p')

            print('ModelTypeSelect index=', index)
            self.ui.Model_tab.setCurrentIndex(index)
            print('Model_tab index=', self.ui.Model_tab.CurrentIndex())
        except:
            self.ui.Results3.setText('Error')

    @pyqtSlot(int)
    def on_Model_tab_currentChanged(self):
        index = self.ui.Model_tab.currentIndex()
        # print('ModelTypeSelect index=',index)
        self.ui.ModelTypeSelect.setCurrentIndex(index)

    @pyqtSlot(bool)
    def on_Initialize_clicked(self):
        Initialize(self=self)

    @pyqtSlot(bool)
    def on_ModelCompute_clicked(self):
        try:
            Compute(self=self)
            self.ui.Results3.setText('Compute is finished')
        except:
            self.ui.Results3.setText('Something is wrong! Please change the parameters')

    '''
    Test_interface
    '''
    # LoadParaBtn
    @pyqtSlot(bool)
    def on_LoadParaBtn_clicked(self):
        try:
            dlgTitle = "Tips"
            strInfo = ("You should input the ...para.csv file in the folder of res-model after model training.")
            defaultBtn = QMessageBox.NoButton  # 缺省按钮
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            LoadParaBtn(self=self)
        except:
            self.ui.Results4.setText('Please input parameters of data')

    # LoadTestData
    @pyqtSlot(bool)
    def on_LoadTestData_clicked(self):
        try:
            LoadTestData(self=self)
        except:
            self.ui.Results4.setText('Please input test data')

    # SelectModel
    @pyqtSlot(bool)
    def on_SelectM_clicked(self):
        '''
        Select the file window. Select the model to be tested.
        选择文件窗口.选择被测试的模型。
        '''
        try:
            self.model, _ = QFileDialog.getOpenFileName(self, 'Open file', '.'
                                                   , 'Data file(*.h5 *.m)')
            self.ui.Test.setEnabled(True)
        except:
            self.ui.Results4.setText('Please select the test model')

    # Test
    @pyqtSlot(bool)
    def on_Test_clicked(self):
        try:
            self.PositionZ = float(self.ui.PositionZ.text())
        except:
            self.PositionZ = -999
        try:
            Test(self=self)
            self.ui.DrawSliceT.setEnabled(True)
        except:
            self.ui.Results4.setText('There are some mistakes about testing')

    @pyqtSlot(int)
    def on_StrePres_t_currentIndexChanged(self):
        try:
            index = self.ui.StrePres_t.currentIndex()
            self.ui.Style1.setCurrentIndex(index)
            self.ui.Style0.setCurrentIndex(index)
        except:
            print('error about index changed')

    @pyqtSlot(bool)
    def on_DrawSliceT_clicked(self):
        print('123456')
        try:
            print('on_DrawSliceT_clicked')
            DrawSliceT(self)
        except:
            print('DrawSliceT is error')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
