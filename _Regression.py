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

def Regression(self):
    '''
    Select folders for data formula regression
    The folder name must be Variable_Radius_Degree
    The Degree of the regression polynomial is determined by the interface degree
    :param self: class
    :return: Polynomials about variables
    '''
    print('Regression')
    folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
    print('folder_path',folder_path)
    Degree = int(self.ui.Degree.text())
    print('Degree',Degree)
    files = os.listdir(folder_path)
    positions = []
    p_maxs = []
    radiuses = []
    degrees = []
    # read data
    for i in range(len(files)):
        file_name_split_ = files[i].split('.')[0]
        file_name_split = file_name_split_.split('_')
        radius = int(file_name_split[1][0])
        degree = int(file_name_split[2])
        radiuses.append(radius)
        degrees.append(degree)

        print('index=', i, 'file name=', files[i])
        data = pd.read_csv(folder_path + '\\' + files[i])
        p = data['p'].values
        p_max = np.max(p)
        p_max_index = np.where(p_max == p)[0]
        position = data.iloc[p_max_index, :3].values
        p_maxs.append(p_max)
        positions.append(position[0])
        print('*' * 10)
    p_maxs_ = np.array(p_maxs).reshape((len(p_maxs), 1))
    radiuses_ = np.array(radiuses).reshape((len(radiuses), 1))
    degrees_ = np.array(degrees).reshape((len(degrees), 1))
    data = np.concatenate((radiuses_, degrees_, p_maxs_), axis=1)
    x = data[:, :2]
    y = data[:, 2]
    poly_reg = PolynomialFeatures(degree=Degree)  # Quadratic polynomial
    X_ploy = poly_reg.fit_transform(x) # Generate polynomial data
    # max parameter
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_ploy, y)
    predict_y = lin_reg_2.predict(X_ploy)
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

    y = np.array(positions)[:, 0]
    lin_reg_x = linear_model.LinearRegression()
    lin_reg_x.fit(X_ploy, y)
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

    y = np.array(positions)[:, 1]
    lin_reg_y = linear_model.LinearRegression()
    lin_reg_y.fit(X_ploy, y)
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

    y = np.array(positions)[:, 2]
    lin_reg_z = linear_model.LinearRegression()
    lin_reg_z.fit(X_ploy, y)
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
    len_limited = 30
    equation_str = str(equation).replace('**', '^')
    equation_str = equation_str.replace('_', '')
    equation_str_len = len(equation_str)
    epoch = equation_str_len//len_limited+1
    equation_str_new = 'pmax='+ '\n' +equation_str[:len_limited] + '\n'
    for i in range(1,epoch):
        equation_str_new += equation_str[i * len_limited:(i + 1) * len_limited] + '\n'
    print('print',equation_str_new)
    self.ui.Eqution.setText(equation_str_new)
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
    p_max = equation.subs({'x_1': Radius, 'x_2': Angle})
    x_max = equation_[0].subs({'x_1': Radius, 'x_2': Angle})
    y_max = equation_[1].subs({'x_1': Radius, 'x_2': Angle})
    z_max = equation_[2].subs({'x_1': Radius, 'x_2': Angle})
    print('p_max',p_max)
    self.ui.ResultR.setText('Maximum Pressure='+str(p_max)+'\n (%.2f, %.2f, %.2f)'%(x_max, y_max, z_max)) # +',(%.2f, %.2f, %.2f)'%(x_max, y_max, z_max)
    print(equation)
    print(p_max)
    return p_max
