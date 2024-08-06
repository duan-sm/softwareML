#!/usr/bin/env python
# coding=utf-8

"""
@Time: 4/5/2024 9:51 PM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Regression.py
@Software: PyCharm
"""

import os
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import sympy as sy


def SelectFolder(self):
    '''
    Select folders for data formula regression
    The folder name must be Variable_Radius_Degree
    :param self:
    :return: the path of Folder
    '''
    # print('1'*10)
    cb_count = self.ui.VariableList.count()
    # print('cb_count', cb_count)
    if cb_count==0:
        pass
    else:
        self.ui.VariableList.clear()
    # print('2' * 10)
    folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
    files = os.listdir(folder_path)
    # print('3' * 10)
    i = 0
    data = pd.read_csv(folder_path + '\\' + files[i])
    columns = data.columns
    # print('4' * 10)
    self.ui.VariableList.addItems(columns)
    # print('5' * 10)

    files = os.listdir(folder_path)
    file_name_split_ = files[0].split('.')[0]
    file_name_split = file_name_split_.split('_')
    index_ = np.arange(len(file_name_split))[:-1]
    str_ = ''
    for i in range(len(index_)):
        str_ = str_ + str(index_[i])
    self.ui.InputPara.setText(str_)
    # radius = int(file_name_split[1][0])
    # degree = int(file_name_split[2])
    return folder_path


def Regression(self, folder_path):
    '''
    Select folders for data formula regression
    The folder name must be Variable_Radius_Degree
    The Degree of the regression polynomial is determined by the interface degree
    :param self: class
    :return: Polynomials about variables
    '''
    # print('Regression')
    # folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
    print('folder_path',folder_path)
    Degree = int(self.ui.Degree.text())
    print('Degree',Degree)
    files = os.listdir(folder_path)
    variable_name = self.ui.VariableList.currentText()

    para_index_ = self.ui.InputPara.text()
    para_index = []
    for _ in para_index_:
        para_index.append(int(_))
    print('para_index_', para_index_)
    InputParaName = np.array(['Radius', 'Angle'])

    positions = []
    p_maxs = []
    radiuses = []
    degrees = []
    all_data = []
    # read data
    for i in range(len(files)):
        file_name_split_ = files[i].split('.')[0]
        file_name_split = file_name_split_.split('_')
        all_data_ = []
        for _ in range(1,len(file_name_split)):
            if _ == 1:
                x = int(file_name_split[1][:-1])
            elif _ == 2:
                x = int(file_name_split[_])/180*np.pi
            else:
                x = int(file_name_split[_])
            all_data_.append(x)
        # radius = int(file_name_split[1][:-1])
        # degree = int(file_name_split[2])
        # radiuses.append(radius)
        # degrees.append(degree)

        print('index=', i, 'file name=', files[i])
        data = pd.read_csv(folder_path + '\\' + files[i])
        # p = data['p'].values
        p = data[variable_name].values
        print(variable_name)
        p_max = np.max(p)
        p_max_index = np.where(p_max == p)[0]
        position = data.iloc[p_max_index, :3].values
        p_maxs.append(p_max)
        # print('position', position)
        positions.append(position[0])
        all_data.append([p_max,all_data_, position[0]])
        print('*' * 10)
    print('make new folder')
    if os.path.exists(r'E:\Software-Duan\Regression_res'):
        pass
    else:
        os.mkdir(r'E:\Software-Duan\Regression_res')
    # print('all_data', all_data)
    data_save = pd.DataFrame(all_data,columns=[variable_name+'_max', 'AllPara','position'])
    # print('data_save',data_save)
    data_save.to_csv(r'E:\Software-Duan\Regression_res\originInputData.csv',encoding='gb18030')
    print('Save originInputData')
    p_maxs_ = np.array(p_maxs).reshape((len(p_maxs), 1))
    InputPara = np.array([all_data[i][1] for i in range(len(all_data))])[:,para_index]
    InputParaN = InputParaName[para_index]
    num_data = len(InputPara)
    data = np.concatenate((p_maxs_, InputPara), axis=1)

    x = data[:, 1:]
    y = np.abs(data[:, 0])
    poly_reg = PolynomialFeatures(degree=Degree)  # Quadratic polynomial
    X_ploy = poly_reg.fit_transform(x) # Generate polynomial data
    # max parameter
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_ploy, y)
    predict_y = lin_reg_2.predict(X_ploy)
    MSE = np.mean((predict_y-y)**2)
    RMSE = np.sqrt(np.mean((predict_y-y)**2))
    R2 = 1 - ((predict_y - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()
    # MAPE = np.mean((predict_y-y)/y)
    MAPE = np.mean(np.divide((predict_y - y), y, out=np.zeros_like((predict_y - y)), where=y != 0))
    score = lin_reg_2.score(X_ploy, y)  ## The model evaluation that comes with sklearn has the same logic as R2_1
    print('3' * 10)
    var_n = x.shape[1]
    var_str = []
    for i in range(1, var_n + 1):
        exec("x%d = sy.symbols('x_%d')" % (i, i))
        var_str.append('x%d' % i)
    print('4' * 10)
    feature_names_out = poly_reg.get_feature_names_out(var_str)
    print('feature_names_out',feature_names_out)
    print('lin_reg_2.coef_', lin_reg_2.coef_)
    equation = sy.Float(lin_reg_2.intercept_, 4)# + sy.Float(lin_reg_2.coef_[0], 4)
    for i in range(len(feature_names_out)):
        var = feature_names_out[i].replace('^', '**').replace(' ', '*')
        equation = equation + sy.Float(lin_reg_2.coef_[i], 4) * eval(var)
    len_limited = 30
    equation_str = str(equation).replace('**', '^')
    equation_str = equation_str.replace('_', '')
    equation_str_len = len(equation_str)
    epoch = equation_str_len // len_limited + 1
    first_line = ''
    for tt in range(len(para_index)):
        first_line = first_line + 'x%d: %s '%(tt, InputParaName[para_index[tt]])
    equation_str_new = first_line  + '\n\n'
    # equation_str_new = 'x1:Radius   x2:Angle' + '\n\n'
    equation_str_new = equation_str_new + '%smax='%variable_name + '\n' + equation_str[:len_limited] + '\n'
    for i in range(1, epoch):
        equation_str_new += equation_str[i * len_limited:(i + 1) * len_limited] + '\n'
    equation_str_new_ = equation_str_new + '\n' + 'MSE=%.3f,RMSE=%.3f,R2=%.3f,MAPE=%.3f' % (MSE, RMSE, R2, MAPE)
    self.ui.Eqution.setText(equation_str_new_)

    # draw Fig.
    bwith = 1
    fontsize = 13
    print('Regression 0 ' * 3)
    self.ui.plotR.figure.clear()  # Clear chart clear fig
    ax = self.ui.plotR.figure.add_subplot(1, 1, 1, label='plot3D')
    data_plot = np.concatenate((data, predict_y.reshape((len(p_maxs), 1))), axis=1)
    radius = data_plot[:, 1]
    num = np.sort(list(set(radius)))
    print('num', num)
    label = ['o', '^', '<', '.', 'x', '+', 's', 'd']
    c_max = 0
    print('Regression 1 ' * 3)
    for i in range(len(num)):
        exec('num_%d = np.where(radius==num[%d])[0]' % (i, i))
        data_new = eval(f'data_plot[num_{i}]')  # [raduis,angle,values,values_e]
        x = data_new[:, 3]
        y = data_new[:, 0]
        c = np.array(data_new[:, 2]/np.pi*180 / 10, dtype=int)
        if c_max < np.max(c) + 1:
            c_max = np.max(c) + 1
        p = ax.scatter(x, y, marker=label[i], c=c, label=f'Radius={num[i]}')
    cbar = ax.figure.colorbar(p, ax=ax)
    cbar.ax.set_ylabel('Angle', rotation=-90, va="bottom", fontdict={'family': 'Times New Roman'})
    print('Regression 2 ' * 3)
    for size in cbar.ax.get_yticklabels():
        size.set_fontname('Times New Roman')
    cbar.ax.set_yticks(np.arange(c_max))
    cbar.ax.set_yticklabels(np.round(np.arange(c_max) * 10/180*np.pi, decimals=2))
    (x_d, x_u) = ax.get_xbound()
    (y_d, y_u) = ax.get_ybound()
    print('Regression 3 ' * 3)
    ax.plot([x_d, x_u], [y_d, y_u], c='k')
    ax.legend()
    self.ui.plotR.figure.tight_layout()
    self.ui.plotR.figure.canvas.draw()
    print('Regression 4 ' * 3)

    # compute x, y, z
    x = data[:, 1:]
    y = np.array(positions)[:, 0]
    lin_reg_x = linear_model.LinearRegression()
    lin_reg_x.fit(X_ploy, y)
    predict_y_x = lin_reg_2.predict(X_ploy)
    MSE_x = np.mean((predict_y_x - y) ** 2)
    RMSE_x = np.sqrt(np.mean((predict_y_x - y) ** 2))
    if ((y.mean() - y) ** 2).sum() == 0:
        R2_x = ((predict_y_x - y) ** 2).sum()
    else:
        R2_x = 1 - ((predict_y_x - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()
    print('8*'*5)
    ttt = predict_y_x - y
    print(ttt, y)
    MAPE_x = np.mean(np.divide(ttt, y, out=np.zeros_like(ttt), where=y!=0))
    print('MAPE_x', MAPE_x)
    var_n = x.shape[1]
    var_str = []
    for i in range(1, var_n + 1):
        exec("x%d = sy.symbols('x_%d')" % (i, i))
        var_str.append('x%d' % i)
    feature_names_out = poly_reg.get_feature_names_out(var_str)
    equationx = sy.Float(lin_reg_x.intercept_, 4)
    for i in range(len(feature_names_out)):
        var = feature_names_out[i].replace('^', '**').replace(' ', '*')
        equationx = equationx + sy.Float(lin_reg_x.coef_[i], 4) * eval(var)

    x = data[:, 1:]
    y = np.array(positions)[:, 1]
    lin_reg_y = linear_model.LinearRegression()
    lin_reg_y.fit(X_ploy, y)
    predict_y_y = lin_reg_2.predict(X_ploy)
    MSE_y = np.mean((predict_y_y - y) ** 2)
    RMSE_y = np.sqrt(np.mean((predict_y_y - y) ** 2))
    if ((y.mean() - y) ** 2).sum()==0:
        R2_y = ((predict_y_y - y) ** 2).sum()
    else:
        R2_y = 1 - ((predict_y_y - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()
    print('9*' * 5)
    MAPE_y = np.mean(np.divide((predict_y_y - y), y, out=np.zeros_like((predict_y_y - y)), where=y != 0))
    var_n = x.shape[1]
    var_str = []
    for i in range(1, var_n + 1):
        exec("x%d = sy.symbols('x_%d')" % (i, i))
        var_str.append('x%d' % i)
    feature_names_out = poly_reg.get_feature_names_out(var_str)
    equationy = sy.Float(lin_reg_y.intercept_, 4)
    for i in range(len(feature_names_out)):
        var = feature_names_out[i].replace('^', '**').replace(' ', '*')
        equationy = equationy + sy.Float(lin_reg_y.coef_[i], 4) * eval(var)

    x = data[:, 1:]
    y = np.array(positions)[:, 2]
    lin_reg_z = linear_model.LinearRegression()
    lin_reg_z.fit(X_ploy, y)
    predict_y_z = lin_reg_2.predict(X_ploy)
    MSE_z = np.mean((predict_y_z - y) ** 2)
    RMSE_z = np.sqrt(np.mean((predict_y_z - y) ** 2))
    if ((y.mean() - y) ** 2).sum():
        R2_z = ((predict_y_z - y) ** 2).sum()
    else:
        R2_z = 1 - ((predict_y_z - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()
    print('10*' * 5)
    MAPE_z = np.mean(np.divide((predict_y_z - y), y, out=np.zeros_like((predict_y_z - y)), where=y != 0))
    # MAPE_z = np.mean((predict_y_z - y) / y)
    var_n = x.shape[1]
    var_str = []
    for i in range(1, var_n + 1):
        exec("x%d = sy.symbols('x_%d')" % (i, i))
        var_str.append('x%d' % i)
    feature_names_out = poly_reg.get_feature_names_out(var_str)
    equationz = sy.Float(lin_reg_z.intercept_, 4)
    for i in range(len(feature_names_out)):
        var = feature_names_out[i].replace('^', '**').replace(' ', '*')
        equationz = equationz + sy.Float(lin_reg_z.coef_[i], 4) * eval(var)
    print(equation)
    print('------------')
    print(equationx)
    print(equationy)
    print(equationz)

    # x
    equation_str = str(equationx).replace('**', '^')
    equation_str = equation_str.replace('_', '')
    equation_str_len = len(equation_str)
    epoch = equation_str_len // len_limited + 1
    equation_str_new = 'xmax=' + '\n' + equation_str[:len_limited] + '\n'
    for i in range(1, epoch):
        equation_str_new += equation_str[i * len_limited:(i + 1) * len_limited] + '\n'
    equation_str_new_x = equation_str_new + '\n' + 'MSE_x=%.3f,RMSE_x=%.3f,R2_x=%.3f,MAPE_x=%.3f' % (MSE_x, RMSE_x, R2_x, MAPE_x)

    equation_str = str(equationy).replace('**', '^')
    equation_str = equation_str.replace('_', '')
    equation_str_len = len(equation_str)
    epoch = equation_str_len//len_limited+1
    equation_str_new = 'ymax='+ '\n' +equation_str[:len_limited] + '\n'
    for i in range(1,epoch):
        equation_str_new += equation_str[i * len_limited:(i + 1) * len_limited] + '\n'
    equation_str_new_y = equation_str_new + '\n' + 'MSE_y=%.3f,RMSE_y=%.3f,R2_y=%.3f,MAPE_y=%.3f'%(MSE_y, RMSE_y, R2_y, MAPE_y)

    equation_str = str(equationz).replace('**', '^')
    equation_str = equation_str.replace('_', '')
    equation_str_len = len(equation_str)
    epoch = equation_str_len//len_limited+1
    equation_str_new = 'zmax='+ '\n' +equation_str[:len_limited] + '\n'
    for i in range(1,epoch):
        equation_str_new += equation_str[i * len_limited:(i + 1) * len_limited] + '\n'
    equation_str_new_z = equation_str_new + '\n' + 'MSE_z=%.3f,RMSE_z=%.3f,R2_z=%.3f,MAPE_z=%.3f'%(MSE_z, RMSE_z, R2_z, MAPE_z)

    print('ready to save equation')
    with open(r'E:\Software-Duan\Regression_res\Regression_equation.txt', 'w') as Regression_equation:
        Regression_equation.write(equation_str_new_ + '\n\n\n' + equation_str_new_x + '\n\n\n' + equation_str_new_y + '\n\n\n' + equation_str_new_z)
    print('save equation')


    return equation, [equationx, equationy, equationz] #

def ComputeR(self, equation, equation_):
    '''
    Calculate the maximum value of the variable under the new radius and Angle by the regression polynomial
    :param self: class
    :param equation: Polynomial formulas for radius and Angle
    :return: Variable maximum
    '''
    print('ComputeR')
    # for i in range(1, 2 + 1):
    #     exec("x%d = sy.symbols('x_%d')" % (i, i))
    # var_n = x.shape[1]
    # for i in range(1, var_n + 1):
    #     exec("x%d = sy.symbols('x_%d')" % (i, i))
    Radius = float(self.ui.Radius.text())
    Angle = float(self.ui.Angle_2.text())
    print('Radius=',Radius,'Angle=',Angle)
    print(equation)
    p_max = np.abs(equation.subs({'x_1': Radius, 'x_2': Angle}))
    x_max = equation_[0].subs({'x_1': Radius, 'x_2': Angle})
    y_max = equation_[1].subs({'x_1': Radius, 'x_2': Angle})
    z_max = equation_[2].subs({'x_1': Radius, 'x_2': Angle})
    print('p_max',p_max)
    self.ui.ResultR.setText('Maximum Pressure='+str(p_max)+'\n (%.2f, %.2f, %.2f)'%(x_max, y_max, z_max)) # +',(%.2f, %.2f, %.2f)'%(x_max, y_max, z_max)
    print(equation)
    print(p_max)
    return p_max
