#!/usr/bin/env python
# coding=utf-8

"""
@Time: 2/8/2024 11:27 AM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: test_new_data.py
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
import os

data = pd.read_csv(r'E:\Software-Duan\test\Shmming_TrialData.csv')
data = data.dropna(how='any').reset_index(drop=True)
# train_data = data[['Points_0', 'Points_1', 'Points_2']]
train_data = data[['Points_0', 'Points_1', 'Points_2', 'pMean']].values
train_values = data[['wallShearStress_0','wallShearStress_1', 'wallShearStress_2']].values
normalization = pd.read_csv(r'E:\Software-Duan\res\model\BP+10_3+relu+0.2+0.001+20+standard_para.csv')
normalization = normalization.values
x_mean = normalization[0,:4]
x_var = normalization[1,:4]
y_mean = normalization[0,4:7]
y_var = normalization[1,4:7]
x = (train_data-x_mean)/np.sqrt(x_var)
m = load_model(r'E:\Software-Duan\res\model\BP+10_3+relu+0.2+0.001+20.h5')
pred = m(x)
pred2 = pred*np.sqrt(y_var)+y_mean
print(train_values)
print(pred2)

