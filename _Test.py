#!/usr/bin/env python
# coding=utf-8

"""
@Time: 2025/9/19 10:18 AM
@Author: Fatimah Ahmadi Godini
@Email: Ahmadi.ths@gmail.com
@File: _Test.py
@Software: Visual Studio Code

@Time: 2/12/2024 3:05 PM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Test.py
@Software: PyCharm
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import metrics


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
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0,1,0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

    # Construct a coordinate point vector
    original_point = np.array([x, y, z])
    # Matrix multiplication is performed to obtain the rotated coordinate points
    rotated_point = np.dot(rotation_matrix, original_point)

    return rotated_point


def LoadParaBtn(self):
    '''
    Select the file window. Select model normalization parameters.
    '''
    print('DataInputBtn|' * 10)

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
            self.para_index = "S" if "standard" in fname else "M"
            # open file
            if fname[-3:] == 'csv':
                filetype = '.csv'
                self.paraData = pd.read_csv(fname
                                            , encoding='gb18030'
                                            , index_col=0)
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:] == 'lsx':
                print('3|' * 10)
                filetype = '.xlsx'
                self.paraData = pd.read_excel(fname, index_col=index_col)
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            elif fname[-3:] == 'xls':
                print('3|' * 10)
                filetype = '.xlsx'
                self.paraData = pd.read_excel(fname, skiprows=1, sheet_name=0)
                self.ui.Results4.setText('Open the file:%s. Successful' % (fname.split('/')[-1]))
            else:
                self.ui.Results4.setText('The file format should be: xls, csv, xlsx' % (fname.split('/')[-1]))
            # Get values for entire rows and columns (arrays)
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
                headerItem.setForeground(QBrush(Qt.red))  # Foreground color
                self.ui.ParaTable.setHorizontalHeaderItem(i, headerItem)

            self.ui.ParaTable.resizeRowsToContents()
            self.ui.ParaTable.resizeColumnsToContents()

            # Displays the data in the table
            for i in range(rows):
                rowslist = workbook_np[i, :]  # Get the content of each row in Excel
                # print(rowslist)
                for j in range(len(rowslist)):
                    # Add rows to the tablewidget 
                    row = self.ui.ParaTable.rowCount()
                    self.ui.ParaTable.insertRow(row)
                    # Write data to the tablewidget 
                    newItem = QTableWidgetItem(str(rowslist[j]))
                    self.ui.ParaTable.setItem(i, j, newItem)
            self.ui.ParaTable.setAlternatingRowColors(True)
            print(self.size_)
            print(self.paraData.columns)
    except:
        self.ui.Results.setText('Something is wrong')


def LoadTestData(self):
    '''
    Select the file window. Select the data to test the model against.
    '''
    fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.'
                                           , 'Data file(*.csv *.xlsx *.xls)')
    try:
        if fname:
            print('fname', fname)
            # open file
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
    except:
        self.ui.Results.setText('There are some mistakes about the data file.')


def Test(self):
    '''
    Test button.
    When the test data is imported and the normalized parameters and corresponding model are selected,
     the second button can be used to test the model.
    '''
    
    input_cols = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude']
    # Load the model under test
    name = os.path.splitext(os.path.basename(self.model))[0]
    model_name = name.split('+')[0]
    m = load_model(self.model)
    # Extract test data
    ori_test_data = self.testworkbook.loc[:, input_cols].to_numpy()

    # Test data normalization
    if hasattr(self, 'paraData') and self.paraData is not None:
        # print("columns: ", self.paraData.columns)
        # print("Index values:", self.paraData.index.tolist())
        if self.para_index == "S":
            print('Standard_para')
            mean = self.paraData.loc['mean', input_cols].to_numpy()
            var = self.paraData.loc['var', input_cols].to_numpy()
            test_data = (ori_test_data - mean) / np.sqrt(var)
        
        elif self.para_index == "M":
            print('MaxMin')
            max_ = self.paraData.loc['max', input_cols].to_numpy()
            min_ = self.paraData.loc['min', input_cols].to_numpy()
            test_data = (ori_test_data - min_) / (max_ - min_) 
        
    else:  # No normalization applied
        print('No Normalization')
        test_data = ori_test_data.copy()

    StrePres_t = self.ui.StrePres_t.currentIndex()
    # style = self.ui.Style1.currentIndex()
    if StrePres_t == 0:
        cols = ['p']
        print('style=0')
    else:
        cols = ['NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']
        print('style=1')

    if model_name in ['MLP', 'PINN_MLP']:
        test_input = test_data
        results_inv = m.predict(test_input)
        # results_inv = results_inv if isinstance(results_inv, np.ndarray) else results_inv.numpy()
    elif model_name in ['CNN', 'PINN_CNN', 'LSTM', 'PINN_LSTM']:
        print(f"Creating sequences for {model_name}")
        seq_len = 128
        stride = 64
        num_points = len(test_data)
        num_targets = len(cols)
        pred_sum = np.zeros((num_points, num_targets))
        pred_count = np.zeros((num_points, 1))

        # Slide window to create sequences
        for start in range(0, num_points - seq_len + 1, stride):
            end = start + seq_len
            seq_input = test_data[start:end]
            print("seq_input.shape", seq_input.shape)
            seq_input = np.expand_dims(seq_input, axis=0)  # batch dimension
            seq_pred = m.predict(seq_input)
            seq_pred = seq_pred[0]  # remove batch dimension
            pred_sum[start:end] += seq_pred
            pred_count[start:end] += 1

        # Handle last window if it doesn't align perfectly
        if end < num_points:
            start = num_points - seq_len
            end = num_points
            seq_input = test_data[start:end]
            seq_input = np.expand_dims(seq_input, axis=0)
            seq_pred = m.predict(seq_input)[0]
            pred_sum[start:end] += seq_pred
            pred_count[start:end] += 1

        # Average overlapping predictions
        results_inv = pred_sum / pred_count

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # The prediction results are reverse-normalized
    if hasattr(self, 'paraData') and self.paraData is not None:
        if self.para_index == "S":
            mean = self.paraData.loc['mean', cols].to_numpy()
            var = self.paraData.loc['var', cols].to_numpy()
            results_inv = results_inv * np.sqrt(var) + mean
        
        elif self.para_index == "M":
            max_ = self.paraData.loc['max', cols].to_numpy()
            min_ = self.paraData.loc['min', cols].to_numpy()
            results_inv = results_inv * (max_ - min_) + min_
    
    # Save original test data + predicted results
    results_df = self.testworkbook.copy()
    print("results_inv.shape", results_inv.shape)
    print("results_df.shape", results_df.shape)
    for i, col in enumerate(cols):
        results_df_col = 'Predicted_' + col
        results_df[results_df_col] = results_inv[:, i] if results_inv.ndim > 1 else results_inv

        self.testworkbook[results_df_col] = results_inv[:, i]
    
    # Save to CSV
    os.makedirs(os.path.join(os.getcwd(), 'res', 'result'), exist_ok=True)
    try:
        results_df.to_csv(os.path.join(os.getcwd(), 'res', 'result', 'results_new.csv'), index=False)
    except Exception as e:
        print("Error saving CSV:", e)

    
    # Calculate metrics
    test_cols = ['Predicted_' + col for col in cols]
    y_pred = results_df[test_cols]
    y_test = results_df[cols]

    if StrePres_t == 0:
        bp_mse = metrics.MeanSquaredError()(y_test, y_pred).numpy()
        bp_mae = metrics.MeanAbsoluteError()(y_test, y_pred).numpy()
        bp_rmse = np.sqrt(bp_mse)

        print('Test Results (Pressure Prediction):')
        print(f'MSE = {bp_mse:.5f}')
        print(f'MAE = {bp_mae:.5f}')
        print(f'RMSE = {bp_rmse:.5f}')
        
        metrics_df = pd.DataFrame([{
            'Model': name,
            'Component': 'Pressure',
            'MSE': float(bp_mse),
            'MAE': float(bp_mae),
            'RMSE': float(bp_rmse)
        }])
        metrics_df.to_csv(f'./res/result/{name}_test_metrics.csv', index=False)

    else:
        metrics_data = []
        for i in range(y_test.shape[1]):
            y_true_col = y_test.iloc[:, i].to_numpy()
            y_pred_col = y_pred.iloc[:, i].to_numpy()

            mse = metrics.MeanSquaredError()(y_true_col, y_pred_col).numpy()
            mae = metrics.MeanAbsoluteError()(y_true_col, y_pred_col).numpy()
            rmse = np.sqrt(mse)

            print(f'Test Results for Component {i}:')
            print(f'  MSE = {mse:.5f}')
            print(f'  MAE = {mae:.5f}')
            print(f'  RMSE = {rmse:.5f}')
            
            metrics_data.append({
                'Model': name,
                'Component': f'Shear_{i}',
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'./res/result/{name}_test_metrics.csv', index=False)

    from _Model import compute_max_ind
    comparison_pressure = compute_max_ind(results_df[input_cols + cols + test_cols], StrePres_t, name+'_test')
    print("Comparison Data done")

    ########################################################################################
    ########################################################################################
    ########################################################################################


def DrawSliceT(self):
    '''
    Draw the simulation results of slices and the prediction results of intelligent models.
    See TestDataPara framework for slice location selection.
    """
    All of the code in this section appears in _ReadData.py, so it's not commented on in detail
    """
    '''
    print('DrawSliceT')
    fontsize = 13
    bwith = 1
    self.ui.TestFig.figure.clear()  # clear fig

    # index = self.ui.SelectFeature1.currentIndex()
    print('DrawSlice begin')
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
            defaultBtn = QMessageBox.NoButton  # Default Button
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
            defaultBtn = QMessageBox.NoButton  # Default Button
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
            defaultBtn = QMessageBox.NoButton  # Default Button
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    # Divide the area
    data = self.testworkbook.copy(deep=True)
    centre_point_x = StartBent_t
    centre_point_y = 0
    centre_point_z = RadiusBent_t
    region0_index = np.where(data['Points_0'].values < StartBent_t)[0]
    region12_index = np.where(data['Points_0'].values >= StartBent_t)[0]
    region0_data = data.iloc[region0_index, :]
    region12_data = data.iloc[region12_index, :]
    region0_data_xyz = region0_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    region12_data_xyz = region12_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    # Adjust the origin
    region12_data_xyz[:, 0] = region12_data_xyz[:, 0] - centre_point_x
    region12_data_xyz[:, 2] = region12_data_xyz[:, 2] - centre_point_z
    
    # Convert coordinates
    r = np.sqrt(region12_data_xyz[:, 0] ** 2 + region12_data_xyz[:, 1] ** 2 + region12_data_xyz[:, 2] ** 2)
    theta = np.arccos(region12_data_xyz[:, 2] / r) / np.pi * 180
    phi = np.arctan2(region12_data_xyz[:, 1], region12_data_xyz[:, 0]) / np.pi * 180

    region12_data = region12_data.copy().assign(r=r, theta=theta, phi=phi)
    print('theta: ', np.sort(list(set(theta))))
    # There is a bug in dividing the region
    # region 1 Since only two points on a slice are accurate = theta (these two points are located at the center of the circle and the center line)
    # , the theta of other points is not accurate = 180, so the slice cannot be found
    # This also causes problems when dividing region1 and region2
    region1_index = np.where(theta > 180 - Angle_t)[0]
    region2_index = np.where(theta <= 180 - Angle_t)[0]
    region1_data = region12_data.iloc[region1_index, :]
    region2_data = region12_data.iloc[region2_index, :]
    region1_data_xyz = region1_data.iloc[:, :3].values
    # Adjust the origin
    centre_point_x2 = StartBent_t + RadiusBent_t * np.sin(Angle_t / 180 * np.pi)
    centre_point_y2 = 0
    centre_point_z2 = RadiusBent_t * (1 - np.cos(Angle_t / 180 * np.pi))
    region2_data_xyz = region2_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    region2_data_xyz[:, 0] -= centre_point_x2
    region2_data_xyz[:, 2] -= centre_point_z2
    new_point = rotate_point(region2_data_xyz[:, 0], region2_data_xyz[:, 1], region2_data_xyz[:, 2], Angle_t)
    region2_data = region2_data.copy()
    region2_data['x'] = new_point.T[:, 0]
    region2_data['y'] = new_point.T[:, 1]
    region2_data['z'] = new_point.T[:, 2]

    # find position
    if Number_t <= StartBent_t:
        print('number<=StartBent_t')
        o1 = np.where(self.testworkbook['Points_0'].values < Number_t + Length_t)[0]
        o2 = np.where(self.testworkbook['Points_0'].values >= Number_t)[0]

        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 30
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]

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

        elif StrePres_t == 1:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
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
        z_special = np.mean(p[:,2])
        print("z_special = ", z_special)
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

        o1 = np.where(self.testworkbook['Points_0'].values <= x_max)[0]
        o2 = np.where(self.testworkbook['Points_0'].values >= x_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index1 = []
        for point in o1:
            if point in o2:
                original_position_index1.append(point)
        o1 = np.where(self.testworkbook['Points_2'].values <= z_max)[0]
        o2 = np.where(self.testworkbook['Points_2'].values >= z_min)[0]
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
        data = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
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
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]
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
        elif StrePres_t == 1:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
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
        z_special = centre_point_z2 + remain_len * np.sin(Angle_t / 180 * np.pi)
        o1 = np.where(region2_data['x'] < remain_len + Length_t)[0]
        o2 = np.where(region2_data['x'] >= remain_len)[0]
        print('shape', self.testworkbook.values.shape)
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        original_position_index = region2_data.iloc[original_position_index, :].index
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 90
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]
            print(p)
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
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
            print(p)
        elif StrePres_t == 1:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < z_special:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]

    columns = [i.split('\n')[0] for i in self.testworkbook.columns]
    print('Length_t', Length_t)
    print('number', Number_t)
    print('test')
    ax1 = self.ui.TestFig.figure.add_subplot(1, 1, 1, label='plot3D')
    print('*1'*10)
    def safe_arccos(arr):
        arr = np.clip(arr / np.max(np.abs(arr)), -1.0, 1.0)  # avoid out-of-bounds
        return np.arccos(arr) / np.pi * 180
    
    if StrePres_t == 0:
        print('p 2 '*10)
        print(p.shape)
        p = p[:, 1:].astype(float)
        mask1 = (p[:,1] < 0) & (p[:,2] < z_special)
        mask2 = (p[:,1] > 0) & (p[:,2] < z_special)
        mask3 = (p[:,1] > 0) & (p[:,2] > z_special)
        mask4 = (p[:,1] < 0) & (p[:,2] > z_special)
        print(p[mask1, 1])
        print(p[mask2, 1])
        print(p[mask3, 1])
        print(p[mask4, 1])
        x_axis1 = 180 - safe_arccos(p[mask1, 1])
        x_axis2 = 180 - safe_arccos(p[mask2, 1])
        x_axis3 = 180 + safe_arccos(p[mask3, 1])
        x_axis4 = 180 + safe_arccos(p[mask4, 1])

        print('000000')
        ax1.scatter(x_axis, p[:, 3], c='b', s=1, label='Pressure CFD')
        ax1.scatter(x_axis, p[:, 4], c='r', s=1, label='Pressure Prediction')
    
    elif StrePres_t == 1:
        mask1 = (stress['Points_1'].values < 0) & (stress['Points_2'].values < z_special)
        mask2 = (stress['Points_1'].values > 0) & (stress['Points_2'].values < z_special)
        mask3 = (stress['Points_1'].values > 0) & (stress['Points_2'].values > z_special)
        mask4 = (stress['Points_1'].values < 0) & (stress['Points_2'].values > z_special)

        x_axis1 = 180 - safe_arccos(stress.loc[mask1, 'Points_1'].values)
        x_axis2 = 180 - safe_arccos(stress.loc[mask2, 'Points_1'].values)
        x_axis3 = 180 + safe_arccos(stress.loc[mask3, 'Points_1'].values)
        x_axis4 = 180 + safe_arccos(stress.loc[mask4, 'Points_1'].values)

        x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))
        
        stress_magnitude = np.sqrt(
            stress['wallShearStress_0']**2 +
            stress['wallShearStress_1']**2 +
            stress['wallShearStress_2']**2
        )
        prediction_stress_magnitude = np.sqrt(
            stress['Predicted_wallShearStress_0']**2 +
            stress['Predicted_wallShearStress_1']**2 +
            stress['Predicted_wallShearStress_2']**2
        )
        ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='CFD wallShearStress_magnitude')
        ax1.scatter(x_axis, prediction_stress_magnitude, c='r', s=1, label='Prediction Stress')
        print('*' * 100)
        print(x_axis)
        print(stress_magnitude)
        print('*' * 100)
        print(prediction_stress_magnitude)
        print('*' * 100)

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
