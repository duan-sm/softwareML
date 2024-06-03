#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 19:38
# @Author  : dsm
# @File    : outlier_function.py
# @Software: win10  python3.8.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## 3sigma principle
def threesigma(data
               , n=3
               , results_s='p'
               , detail=False):
    '''
    data：Represents a time series, including two columns of time and value | 表示时间序列，包括时间和数值两列；
    n：Represents several times the standard deviation | 表示几倍的标准差
    results_s:Output result form, | 输出结果形式，
            "a" on behalf of all, including results, results_x, index, mintime, maxtime, outlier, outlier_x, outlier_index
            They represent: filtered results, result time index, result sequence index,
            Start year, end time, out-of-range value, time index of out-of-range data, and sequential index of out-of-range data
            "p" stands for parts, including results, results_x, and index
            [results, results_x, index]
    '''
    data_x = data.index.tolist()  ##Get the time of the time series
    # print (data_x)
    # print ("**********",j)
    mintime = data_x[0]  ##Gets the starting point of the time series
    maxtime = data_x[-1]  ##Gets the ending point of the time series

    data_y = data.values.tolist()  ##Gets the time series value
    ymean = np.mean(data_y)  ##Find the time series average
    ystd = np.std(data_y)  ##Find the standard deviation of time series
    down = ymean - n * ystd  ##Compute lower bound
    up = ymean + n * ystd  ##Compute upper bound

    outlier = []  # Save the outlier
    outlier_x = []
    outlier_index = []
    results = []  # The results after screening
    results_x = []
    index = []
    for i in range(0, len(data_y)):
        if (data_y[i] < down) | (data_y[i] > up):
            outlier.append(data_y[i])
            outlier_x.append(data_x[i])
            outlier_index.append(i)
        else:
            results.append(data_y[i])
            results_x.append(data_x[i])
            index.append(i)
    if detail:
        # Test function uses code, prints variables, has no effect
        print('The name of the variable is：', data.name)
        print('Calculate the lower bound down, calculate the upper bound up:', down, up)
        print('The lower bound of the data is down, and the upper bound of the data is up:', np.min(data_y), np.max(data_y))
        print('The original data volume is: %d, the current data volume is: %d' % (len(data), len(index)))
        print('The amount of data deleted is,', len(outlier_index))
        print('*' * 50)
    if results_s == 'a':
        return [results, results_x, index, down, up, outlier, outlier_x, outlier_index]
    elif results_s == 'p':
        return [results, results_x, index]


def find_same_diff_num(num1, num2):
    '''
    Look for the same and different numbers in the two lists
    :param num1: list 1
    :param num2: list 2
    :return: The same number; Different numbers
    '''
    num1 = np.sort(list(set(num1)))
    num2 = np.sort(list(set(num2)))
    same = []
    i = 0
    j = 0
    while i < len(num1):
        #         print(i,j)
        #         print('-'*10)
        while j < len(num2):
            if num1[i] == num2[j]:
                same.append(num1[i])
                j += 1
                break
            elif num1[i] < num2[j]:
                break
            else:
                j += 1
                continue
        i += 1
    num = np.concatenate((np.array(num1), np.array(num2)))
    num = np.sort(list(set(num)))
    #     print(num)
    #     print(np.sum(num == num2))
    diff = []
    i = 0
    j = 0
    while i < len(same):
        #         print(i,j)
        #         print('-'*10)
        #         print(same[i])
        while j < len(num):
            if same[i] == num[j]:
                j += 1
                break
            elif same[i] < num[j]:
                diff.append(same[i])
                break
            else:
                diff.append(num[j])
                j += 1
                continue
        i += 1
    #         if i == len(num):
    while j < len(num):
        diff.append(num[j])
        j += 1
    return [np.array(same), np.array(diff)]


def find_continue_data(a
                       , data_long=50
                       , lag_point=None
                       , ind_=False):
    '''
    Input one-dimensional data, look for consecutive strings of data in the data, and return the beginning and end numbers|
    输入一维数据，寻找数据中连续的数据串，返回开始和结束的数字
    :param a: data | 数据
    :param data_long: Limit the length of continuous data, up to a certain length is regarded as continuous number   限制连续数据的长度，达到一定长度视为连续的数
    :param lag_point: Sets the continuous length for discontinuous points   代表对于不连续的点的，设定连续长度
    For example, when lag_point=4, 1 and 5 are considered consecutive        比如当lag_point=4时，认为1和5为连续的
    :param ind_: Whether to return the index of the corresponding value  是否返回对应值的索引
    :return: ss：A value representing each successive segment        代表 每一段连续的值
    ss_i：An index that represents each successive value         代表 每一段连续的值的索引

    sss：Merge the ss of all data     将所有数据的ss合并
    sss_i：Merge ss_i of all data       将所有数据的ss_i合并
    '''

    ss = []
    sss = []

    ss_i = []
    sss_i = []
    # Judge the first point 判断第一个点
    if lag_point == None:
        for i0 in range(0, len(a)):
            # Determine whether a continuous point exists 判断是否存在连续点
            if i0 == len(a) - 1:
                if ind_ == False:
                    return sss
                else:
                    return sss, sss_i
            if a[i0] == a[i0 + 1] - 1:
                ss.append(a[i0])
                ss_i.append(i0)
                break

    else:
        for i0 in range(0, len(a)):
            # Determine whether a continuous point exists 判断是否存在连续点
            if i0 == len(a) - 1:
                if ind_ == False:
                    return sss
                else:
                    return sss, sss_i
            if a[i0] >= a[i0 + 1] - lag_point:
                ss.append(a[i0])
                ss_i.append(i0)
                break
    #     print(ss, i0)
    # Determine subsequent points 判断后续的点
    for i in range(i0 + 1, len(a) - 1):
        #         print(i)
        if lag_point == None:
            if a[i] != a[i - 1] + 1 and a[i] != a[i + 1] - 1:
                ss = []
                ss_i = []
                #                 ss.append(a[i])
                continue
            if a[i] != a[i - 1] + 1:
                #         print(a[i], a[i-1])
                ss.append(a[i])
                ss_i.append(i)
            if a[i] != a[i + 1] - 1:
                ss.append(a[i])
                ss_i.append(i)
            elif i + 1 == len(a) - 1:
                ss.append(a[i + 1])
                ss_i.append(i + 1)
            if len(ss) >= 2:
                if ss[-1] - ss[0] > data_long:
                    sss.append(ss)
                    sss_i.append(ss_i)
                ss = []
                ss_i = []
        else:
            # a[i]<= a[i-1]+lag_point|a[i]>=a[i+1]-lag_point:
            # 范围print(a[i+1]-lag_point,a[i],a[i-1]+lag_point)
            if a[i] > a[i - 1] + lag_point and a[i] < a[i + 1] - lag_point:
                ss = []
                ss_i = []
                continue
            if a[i] > a[i - 1] + lag_point:
                #         print(a[i], a[i-1])
                ss.append(a[i])
                ss_i.append(i)
            if a[i] < a[i + 1] - lag_point:
                ss.append(a[i])
                ss_i.append(i)
            elif i + 1 == len(a) - 1:
                ss.append(a[i + 1])
                ss_i.append(i + 1)
            if len(ss) >= 2:
                if ss[-1] - ss[0] > data_long:
                    sss.append(ss)
                    sss_i.append(ss_i)
                ss = []
                ss_i = []
    if ind_ == False:
        return sss
    else:
        return sss, sss_i


# 发现不连续点
def find_discontinue_data(a
                          , data_long=50
                          , lag_point=None
                          , ind_=False):
    '''
    Input one-dimensional data, look for consecutive strings of data in the data, and return the beginning and end numbers
    :param a: data | 数据
    :param data_long: Limit the length of continuous data, up to a certain length is regarded as continuous number | 限制连续数据的长度，达到一定长度视为连续的数
    :param lag_point: Sets the continuous length for discontinuous points | 代表对于不连续的点的，设定连续长度
    For example, when lag_point=4, 1 and 5 are considered consecutive | 比如当lag_point=4时，认为1和5为连续的
    :param ind_: Whether to return the index of the corresponding value | 是否返回对应值的索引
    :return: ss：Discontinuous index | 不连续的索引
            sss：The value of a discontinuous index | 不连续的索引的值
    '''
    # Use the find_continue_data function to find continuous data
    _, index_list = find_continue_data(a=a
                                       , data_long=data_long
                                       , lag_point=lag_point
                                       , ind_=True)
    # Record index
    ss = []
    # Record the value of index
    sss = []
    # The above indicates that there are no continuous points, then all points are discontinuous points
    if len(index_list) == 0:
        for i in range(len(a)):
            ss.append([i, i])
        i = 0
        for i in range(len(ss)):
            a0 = ss[i][0]
            a1 = ss[i][1]
            sss.append([a[a0], a[a1]])
        if ind_ == False:
            return sss
        else:
            return sss, ss
    else:
        if index_list[0][0] == 0:
            if index_list[0][1] == len(a) - 1:
                # print('连续的索引为：', index_list, '；数据长度为：', len(a))
                # print('因此此数据没有不连续的异常点')
                pass
            else:
                i = 0
                for i in range(len(index_list) - 1):
                    s0 = index_list[i][1] + 1
                    s1 = index_list[i + 1][0] - 1
                    if s1 >= s0:
                        ss.append([s0, s1])
                print(len(index_list), i)
                if len(index_list) == 1:
                    ss.append([index_list[0][1] + 1, len(a) - 1])
                else:
                    if index_list[i + 1][1] != len(a) - 1:
                        ss.append([index_list[i + 1][1] + 1, len(a) - 1])
        else:
            i = 0
            ss.append([0, index_list[0][0] - 1])
            for i in range(1, len(index_list)):
                s0 = index_list[i - 1][1] + 1
                s1 = index_list[i][0] - 1
                if s1 >= s0:
                    ss.append([s0, s1])
            if index_list[i][1] != len(a) - 1:
                ss.append([index_list[i][1] + 1, len(a) - 1])
        i = 0
        for i in range(len(ss)):
            a0 = ss[i][0]
            a1 = ss[i][1]
            sss.append([a[a0], a[a1]])
        if ind_ == False:
            return sss
        else:
            return sss, ss