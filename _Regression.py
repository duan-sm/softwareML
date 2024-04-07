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
    print('1'*10)
    p_maxs_ = np.array(p_maxs).reshape((len(p_maxs), 1))
    radiuses_ = np.array(radiuses).reshape((len(radiuses), 1))
    degrees_ = np.array(degrees).reshape((len(degrees), 1))
    data = np.concatenate((radiuses_, degrees_, p_maxs_), axis=1)
    x = data[:, :2]
    y = data[:, 2]
    print('2' * 10)
    poly_reg = PolynomialFeatures(degree=Degree)  # 二次多项式
    X_ploy = poly_reg.fit_transform(x)
    print('2.1' * 10)
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_ploy, y)
    print('2.2' * 10)
    predict_y = lin_reg_2.predict(X_ploy)
    score = lin_reg_2.score(X_ploy, y)  ##sklearn中自带的模型评估，与R2_1逻辑相同
    print('3' * 10)
    var_n = x.shape[1]
    var_str = []
    for i in range(1, var_n + 1):
        exec("x%d = sy.symbols('x_%d')" % (i, i))
        var_str.append('x%d' % i)
    print('4' * 10)
    feature_names_out = poly_reg.get_feature_names_out(var_str)
    equation = sy.Float(lin_reg_2.intercept_, 4)
    for i in range(len(feature_names_out)):
        var = feature_names_out[i].replace('^', '**').replace(' ', '*')
        equation = equation + sy.Float(lin_reg_2.coef_[i], 4) * eval(var)

    print(equation)
    self.ui.Eqution.setText(str(equation).replace('**', '^'))
    print('The End')
    return equation

def ComputeR(self, equation):
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
    print('p_max',p_max)
    self.ui.ResultR.setText('Maximum Pressure='+str(p_max))
    print(equation)
    print(p_max)
    return p_max
