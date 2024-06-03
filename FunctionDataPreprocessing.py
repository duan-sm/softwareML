#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 19:42
# @Author  : dsm
# @File    : FunctionDataPreprocessing.py
# @Software: win10  python3.8.3
import numpy as np
import pandas as pd
from outlier_function import threesigma, find_discontinue_data, find_same_diff_num
from scipy.stats import chi2_contingency



def mutualInfo(X, Y):
    '''
    互信息计算
    Mutual information computing
    :param X: computes a variable, narray
    :param Y: computes another variable, narray
    :return: mutual information value
    '''
    #     X = np.asarray(data.iloc[:, 0])
    #     Y = np.asarray(data.iloc[:, 1])
    # Use a dictionary to count the number of occurrences of each x element
    d_x = dict()  # Dictionary of x
    for x in X:
        if x in d_x:
            d_x[x] += 1
        else:
            d_x[x] = 1
    # Calculate the probability of occurrence of each element
    p_x = dict()
    for x in d_x.keys():
        p_x[x] = d_x[x] / X.size

    # Use a dictionary to count the number of occurrences of each y element
    d_y = dict()  # Dictionary of y
    for y in Y:
        if y in d_y:
            d_y[y] += 1
        else:
            d_y[y] = 1
    # Calculate the probability of occurrence of each element
    p_y = dict()
    for y in d_y.keys():
        p_y[y] = d_y[y] / Y.size

    # Use a dictionary to count the number of occurrences of each (x,y) element
    d_xy = dict()  #  Dictionary of x
    for i in range(X.size):
        if (X[i], Y[i]) in d_xy:
            d_xy[X[i], Y[i]] += 1
        else:
            d_xy[X[i], Y[i]] = 1
    # Calculate the probability of occurrence of each element
    p_xy = dict()
    for xy in d_xy.keys():
        p_xy[xy] = d_xy[xy] / X.size
    # print(d_x, d_y, d_xy)
    # print(p_x, p_y, p_xy)

    # Initialize the mutual information value to 0
    mi = 0
    for xy in p_xy.keys():
        mi += p_xy[xy] * np.log(p_xy[xy] / (p_x[xy[0]] * p_y[xy[1]]))
    # print(mi)

    return mi


def chi2_mutualinfo(data
                    , feature='钻时'):
    '''
    Chi-square test and mutual information calculation are performed for a feature
    :param data: DataFrame for storing log data
    :param data: indicates the feature name
    :return: Chi-square value and mutual information value
    '''
    record = []
    data_cols_sim = np.array([_col.split('\n')[0] for _col in data.columns])
    DT_index = np.where(feature == data_cols_sim)[0][0]
    for i in range(len(data_cols_sim)):
        kf = chi2_contingency([data.iloc[:, i].values, data.iloc[:, DT_index].values])
        mutual_info = mutualInfo(data.iloc[:, i].values, data.iloc[:, DT_index].values)
        print(' ' * 10 + '\033[1;31m The chi-square test results of feature %s and feature %s are: \033[0m' % (
        data_cols_sim[i], data_cols_sim[DT_index]))
        print('Chi-square value =%.4f, P-value =%.4f, degrees of freedom =%i,'
              ' corresponding theoretical values of the same dimension as the original data array' % kf)
        print(' ' * 10 + '\033[1;34m The mutual information results of feature %s and feature %s are as follows: %0.4f \033[0m' % (
            data_cols_sim[i], data_cols_sim[DT_index], mutual_info))
        record.append([kf[0], mutual_info])
        print(' ' * 10 + '-' * 10 + ' ' * 10)
    record = np.array(record)
    return record

# Ignore this function
def data_fusion(data
                , data1
                , rock_type):
    """
    Ignore this function
    The lithology and raw data are fused
    :param data: DataFrame for storing log data
    :param data1: A DataFrame that stores the lithology corresponding to the start and stop depth
    :param rock_type: lithology type
    :return: indicates the merged data
    """

    data['岩性'] = [[]] * len(data)
    data.reset_index(inplace=True,drop=True)
    rock_type = np.array(rock_type)
    WD = data.iloc[:, 0].values
    data = data.astype('object')
    for i in range(len(data1)):
        sta = data1.iloc[i, 0]
        end = data1.iloc[i, 1]
        ind = np.where((WD >= sta) * (WD < end))[0]
        rock_encoding_num = np.where(data1.iloc[i, 2] == rock_type)[0][0]
        rock_encoding = np.zeros(len(rock_type))
        rock_encoding[rock_encoding_num] = 1
        try:
            data.loc[ind, '岩性'] = [rock_encoding_num] * len(ind)
            # print('ind, rock_encoding_num')
            # print(ind, rock_encoding_num)
            # print('np.where(data1.iloc[i, 2] == rock_type)[0]')
            # print(np.where(data1.iloc[i, 2] == rock_type)[0])
        except:
            # ignore
            print('出错',ind)
            print(data.loc[ind, '岩性'])
            print([rock_encoding_num] * len(ind))
            print('岩性:',data1.iloc[i, 2])
            print('未编号的位置包括以下，其数量为：', len(ind))
            print(ind)
    return data

# Ignore this function
def data_validity(original_data
                  , bit_size_path=r'G:\F盘\1师兄任务\2022春季学期\15呼图壁\0709数据处理\钻头尺寸.xlsx'
                  , size=0
                  , path=0
                  , inplace=True):
    """
    Ignore this function
    Analyze the data for accuracy
    :param original_data: specifies the DataFrame that stores well log data
    :param path: indicates the DataFrame for saving the analysis result after the analysis is complete
    :param path:  Whether to replace the original data
    :return: none
    """

    data = original_data.copy()
    if size == 0:
        data['钻头尺寸'] = 1
        data.reset_index(inplace=True,drop=True)
        WD = data.iloc[:, 0].values
        print('???',bit_size_path[-3:])
        if bit_size_path[-3:] == 'csv':
            print('csv文件')
            chicun = pd.read_csv(bit_size_path, encoding='gb18030')
        elif bit_size_path[-3:] == 'lsx':
            print(bit_size_path)
            print('xlsx文件')
            chicun = pd.read_excel(bit_size_path, skiprows=1)
        chicun = chicun.dropna(axis=0, how='all')
        shendu = chicun.iloc[:, 1].values
        index = []
        print('*'*10)
        for i in range(len(shendu)):
            re = shendu[i].split('~')
            if i == 0:
                index.append([int(re[0]), int(re[-1])])
            else:
                index.append([int(index[-1][-1] + 1), int(re[1])])
            print(index)
        print('*-' * 10)
        for i in range(len(index)):
            sta = index[i][0]
            end = index[i][1]
            print((WD >= sta) * (WD < end))
            ind = np.where((WD >= sta) * (WD < end))[0]
            print(ind)
            print(data)
            print(data.loc[ind, '钻头尺寸'])
            print(chicun.iloc[i, -1])
            data.loc[ind, '钻头尺寸'] = float(chicun.iloc[i, -1])
        print('0000000')
        ROP = np.array(60 / data['钻时\nmin/m'].values, dtype=np.float16)
        RPM = np.array(data['转盘转速\nrpm'].values, dtype=np.float16)
        WOB = np.array(data['钻压\nkN'].values / 9.81, dtype=np.float16)
        Dh = np.array(data['钻头尺寸'].values, dtype=np.float16)
        ECD = np.array(data['出口密度\ng/cm3'].values, dtype=np.float16)
        H = np.array(data['井深\nm'].values, dtype=np.float16)
        print('1111111')
        dc = np.log10(0.0547 * ROP / RPM) / np.log10(0.671 * WOB / Dh) / ECD
        sigma = (25.4 * WOB ** 0.5 * RPM ** 0.25 / (Dh * ROP ** 0.25) + 0.028 * (7 - 0.001 * H)) ** 2
        data['dc-'] = dc
        data['sigma-'] = sigma
        rrr = data[['dc', 'dc-', 'sigma', 'sigma-']].values
        del_ind = list(set(np.where(rrr != rrr)[0]))
        print('22222222')
        data.drop(index=del_ind, inplace=True)
        data['dc相对误差%'] = (data['dc'] - data['dc-']) / data['dc'] * 100
        data['sigma相对误差%'] = (data['sigma'] - data['sigma-']) / data['sigma'] * 100
        if path:
            data.to_csv(path + '\\' + '数据准确性分析结果.csv'
                        , encoding='gb18030')
        print("=" * 10 + "\033[1;31m 数据有效性验证分析结果如下 \033[0m" + "=" * 10)
        print(data[['dc', 'dc-', 'sigma', 'sigma-']].describe())
        if inplace:
            limited_1 = np.percentile(data['dc相对误差%'].values,99)
            limited_2 = np.percentile(data['sigma相对误差%'].values, 99)
            data = data[data['dc相对误差%']<limited_1]
            data = data[data['sigma相对误差%'] < limited_2]
        else:
            data=original_data
    else:
        ROP = data['钻速'].values
        RPM = np.array(data['转速'].values, dtype=np.float16)
        WOB = np.array(data['钻压'].values / 9.81, dtype=np.float16)
        Dh = np.array(data['钻头直径'].values, dtype=np.float16)
        ECD = np.array(data['出口钻井液密度'].values, dtype=np.float16)
        H = np.array(data['井深'].values, dtype=np.float16)
        print('1111111')
        sigma = (25.4 * WOB ** 0.5 * RPM ** 0.25 / (Dh * ROP ** 0.25) + 0.028 * (7 - 0.001 * H)) ** 2
        data['sigma-'] = sigma
        rrr = data[['sigma指数', 'sigma-']].values
        del_ind = list(set(np.where(rrr != rrr)[0]))
        print('22222222')
        data.drop(index=del_ind, inplace=True)
        data['sigma相对误差%'] = (data['sigma指数'] - data['sigma-']) / data['sigma指数'] * 100
        if path:
            data.to_csv(path + '\\' + '数据准确性分析结果.csv'
                        , encoding='gb18030')
        print("=" * 10 + "\033[1;31m 数据有效性验证分析结果如下 \033[0m" + "=" * 10)
        print(data[['sigma指数', 'sigma-']].describe())
        if inplace:
            limited_2 = np.percentile(data['sigma相对误差%'].values, 99)
            data = data[data['sigma相对误差%'] < limited_2]
        else:
            data=original_data
    return data



'''
# Ignore this function
if __name__ == '__main__':
    rock_type = np.array(['浅灰色泥岩', '浅灰色泥质粉砂岩', '浅灰色石膏质粉砂岩', '浅灰色细砂岩', '深灰色泥岩', '深灰色砂砾岩',
                          '深灰色砂砾石', '深灰色砂质泥岩', '深灰色砾石', '深灰色粉砂质泥岩', '灰白色含砾泥质粉砂岩',
                          '灰白色石膏质粉砂岩', '灰白色砂砾岩', '灰色含砾泥质粉砂岩', '灰色含砾泥质细砂岩', '灰色泥岩', '灰色泥质砾岩',
                          '灰色泥质粉砂岩', '灰色石膏质粉砂岩', '灰色砂砾岩', '灰色砂砾石', '灰色砂质泥岩', '灰色粉砂质泥岩',
                          '灰色细砂岩', '灰褐色含砾泥质粉砂岩', '灰褐色泥岩', '灰褐色泥质粉砂岩', '灰褐色砂质泥岩', '灰褐色粉砂质泥岩',
                          '灰黄色泥质粉砂岩', '红褐色泥岩', '红褐色泥质粉砂岩', '红褐色石膏质泥岩', '红褐色石膏质粉砂岩',
                          '红褐色砂质泥岩', '红褐色粉砂质泥岩', '绿灰色泥岩', '绿灰色泥质粉砂岩', '绿灰色粉砂质泥岩',
                          '褐灰色含砾泥质粉砂岩', '褐灰色含砾泥质细砂岩', '褐灰色泥岩', '褐灰色泥质粉砂岩', '褐灰色泥质细砂岩',
                          '褐灰色砂砾岩', '褐灰色砂质泥岩', '褐灰色粉砂质泥岩', '褐灰色细砂岩', '褐色泥岩', '褐色粉砂质泥岩',
                          '黄灰色含砾泥岩', '黄灰色含砾泥质粉砂岩', '黄灰色含砾泥质细砂岩', '黄灰色泥岩', '黄灰色泥质砾岩',
                          '黄灰色泥质粉砂岩', '黄灰色砂砾岩', '黄灰色砂砾石', '黄灰色粉砂质泥岩'], dtype=object)
    file_saving_path = r'G:\F盘\1师兄任务\2022春季学期\15呼图壁\0929\0'
    # 读取数据
    file_dir1 = r'G:\F盘\1师兄任务\2022春季学期\15呼图壁\0709数据处理\呼101-102(1)\呼101钻参及岩屑录井.xls'
    data = pd.read_excel(file_dir1, sheet_name=0, skiprows=1)
    data = data.dropna(axis=0, how='all')
    col_o = data.columns
    col_c = []
    for i in range(len(col_o)):
        str_ = ''
        new_c = col_o[i].split('\n')
        if len(new_c) > 1:
            new_1 = str_.join(new_c[:-1])
            new_2 = new_c[-1]
            new_c = new_1 + '\n' + new_2
            col_c.append(new_c)
        else:
            col_c.append(new_c[0])
    data.columns = col_c
    # 数据有效性分析
    data = data_validity(data
                         , bit_size_path=r'G:\F盘\1师兄任务\2022春季学期\15呼图壁\0709数据处理\钻头尺寸.xlsx'
                         , path=file_saving_path
                         , inplace=False)

    # 数据可视化与预处理
    # 可视化函数
    factor = [
        '转盘转速\nrpm'
    ]
    for i in range(len(factor)):
        draw_cross_curves(data
                          , x_=data['井深\nm'].values
                          , factor=[factor[i]]
                          , fontsize=15
                          , xfigsize=(8, 6)
                          , xdpi=300
                          , bwith=1.5
                          , point_num=10
                          , mode='auto'
                          , x_label=r'井深$\mathrm{(m)}$'
                          , y_label=r'%s$\mathrm{(%s)}$' % (factor[i].split('\n')[0], factor[i].split('\n')[1])
                          , truncate=False
                          , label=[factor[i].split('\n')[0]]
                          , c_=['b', 'r', 'k', 'g']
                          , lins=['-', '-']
                          , picture_name=factor[i].split('\n')[0]
                          , path=file_saving_path
                          , trend=False
                          # , s_m='M'
                          )
    # 异常值处理与可视化
    data = outlier(data
                   , path=file_saving_path)

    # 方差选择与相关系数
    draw_corr(data
              , limited_corr=0.6
              , limited_var=0.01
              , path=file_saving_path)

    # 卡方与互信息检验
    chi2_mutualinfo_res = chi2_mutualinfo(data
                                          , feature='钻时')
    print('\033[1;47m  \033[1;31m数据卡方与互信息检验结束\033[0m  \033[0m')
    # 降维
    PCA_res = PCA(data
                  , limited_score=0.999
                  , path=file_saving_path)
    print('\033[1;47m  \033[1;31m数据降维成功\033[0m  \033[0m')
    # 归一化
    data = dimensionless(data
                         , s_m='S'
                         , path=file_saving_path)
    print('\033[1;47m  \033[1;31m数据归一化成功\033[0m  \033[0m')
    # 数据融合
    data1 = pd.read_excel(file_dir1, sheet_name=1)
    data1.columns = ['0', '1', '2']
    X = data1.iloc[:, -1].values
    br = 0
    for i in range(len(X)):
        if X[i] in rock_type:
            pass
        else:
            print('请在数据库中增加岩石种类：%s' % X[i])
            print('数据融合失败')
            break
    data = data_fusion(data, data1, rock_type)
    data.to_csv(file_saving_path + r'\融合后的数据.csv'
                , encoding='gb18030')
    print('\033[1;47m  \033[1;31m数据融合成功\033[0m  \033[0m')
'''
