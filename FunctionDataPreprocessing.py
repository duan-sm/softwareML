#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 19:42
# @Author  : dsm
# @File    : FunctionDataPreprocessing.py
# @Software: win10  python3.8.3
import numpy as np
import pandas as pd
from outlier_function import threesigma, find_discontinue_data, find_same_diff_num
from plotting_function import draw_curve, draw_cross_curves, draw_corr, PCA
from scipy.stats import chi2_contingency



def mutualInfo(X, Y):
    '''
    互信息计算
    :param X: 计算的一个变量，narray
    :param Y: 计算的另一个变量，narray
    :return: 互信息值
    '''
    #     X = np.asarray(data.iloc[:, 0])
    #     Y = np.asarray(data.iloc[:, 1])
    # 使用字典统计每一个x元素出现的次数
    d_x = dict()  # x的字典
    for x in X:
        if x in d_x:
            d_x[x] += 1
        else:
            d_x[x] = 1
    # 计算每个元素出现的概率
    p_x = dict()
    for x in d_x.keys():
        p_x[x] = d_x[x] / X.size

    # 使用字典统计每一个y元素出现的次数
    d_y = dict()  # y的字典
    for y in Y:
        if y in d_y:
            d_y[y] += 1
        else:
            d_y[y] = 1
    # 计算每个元素出现的概率
    p_y = dict()
    for y in d_y.keys():
        p_y[y] = d_y[y] / Y.size

    # 使用字典统计每一个(x,y)元素出现的次数
    d_xy = dict()  # x的字典
    for i in range(X.size):
        if (X[i], Y[i]) in d_xy:
            d_xy[X[i], Y[i]] += 1
        else:
            d_xy[X[i], Y[i]] = 1
    # 计算每个元素出现的概率
    p_xy = dict()
    for xy in d_xy.keys():
        p_xy[xy] = d_xy[xy] / X.size
    # print(d_x, d_y, d_xy)
    # print(p_x, p_y, p_xy)

    # 初始化互信息值为0
    mi = 0
    for xy in p_xy.keys():
        mi += p_xy[xy] * np.log(p_xy[xy] / (p_x[xy[0]] * p_y[xy[1]]))
    # print(mi)

    return mi


def chi2_mutualinfo(data
                    , feature='钻时'):
    '''
    对某个特征进行卡方检验和互信息计算
    :param data: 存储录井数据的DataFrame
    :param data: 特征名称
    :return: 卡方值和互信息值
    '''
    record = []
    data_cols_sim = np.array([_col.split('\n')[0] for _col in data.columns])
    DT_index = np.where(feature == data_cols_sim)[0][0]
    for i in range(len(data_cols_sim)):
        kf = chi2_contingency([data.iloc[:, i].values, data.iloc[:, DT_index].values])
        mutual_info = mutualInfo(data.iloc[:, i].values, data.iloc[:, DT_index].values)
        print(' ' * 10 + '\033[1;31m 特征%s与特征%s的卡方检验结果为：\033[0m' % (data_cols_sim[i], data_cols_sim[DT_index]))
        print('卡方值=%.4f, P值=%.4f, 自由度=%i, 与原数据数组同维度的对应理论值=%s' % kf)
        print(' ' * 10 + '\033[1;34m 特征%s与特征%s的互信息结果为：%0.4f \033[0m' % (
        data_cols_sim[i], data_cols_sim[DT_index], mutual_info))
        record.append([kf[0], mutual_info])
        print(' ' * 10 + '-' * 10 + ' ' * 10)
    record = np.array(record)
    return record


def data_fusion(data
                , data1
                , rock_type):
    """
    对岩性和原始数据进行融合
    :param data: 存储录井数据的DataFrame
    :param data1: 存储起止井深对应的岩性的DataFrame
    :param rock_type: 岩性种类
    :return: 融合后的数据
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
            print('出错',ind)
            print(data.loc[ind, '岩性'])
            print([rock_encoding_num] * len(ind))
            print('岩性:',data1.iloc[i, 2])
            print('未编号的位置包括以下，其数量为：', len(ind))
            print(ind)
    return data


def data_validity(original_data
                  , bit_size_path=r'G:\F盘\1师兄任务\2022春季学期\15呼图壁\0709数据处理\钻头尺寸.xlsx'
                  , size=0
                  , path=0
                  , inplace=True):
    """
    对数据进行准确性分析
    :param original_data: 存储录井数据的DataFrame
    :param path: 分析结束后保存分析结果DataFrame
    :param path: 是否替换原始数据
    :return: 无
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



def outlier(original_data
            , path):
    '''
    异常值处理
    :param original_data: 存储录井数据的DataFrame
    :param path: 分析结束后保存分析结果DataFrame
    :return: 删除异常值后的结果
    '''
    data = original_data.copy()
    # 异常值处理与可视化
    col = data.columns
    i_index = []
    index_ = []
    for i in range(len(col)):
        _ = threesigma(data[col[i]]
                       , detail=False
                       , results_s='a'
                       , n=3)
        out_index = _[-1]
        i_index.append(i)
        # 所有的异常索引存储
        index_.append(out_index)
        print(col[i], len(out_index))
    print('*' * 50)
    print('存在异常值的特征数量为%d' % len(index_))
    print('*' * 50)

    # 对异常点数量进行统计
    for i in range(len(index_)):
        print(i_index[i], col[i_index[i]])
        len_data = len(data)
        if len(index_[i]) == 0:
            print('\033[1;34m %s无异常值 \033[0m' % col[i_index[i]].split('\n')[0])
            continue
        if len(index_[i]) < len_data * 0.01:
            print('-' * 50)
            print('\033[1;31m %s的异常值数量为：%d \033[0m' % (col[i_index[i]].split('\n')[0], len(index_[i])))
            print('-' * 50)
    data_copy_ = np.array(data.values)

    for i in range(len(index_)):
        judge_var_index = i
        if len(index_[i]) == 0:
            print('%s的顺序为%d, 无异常值' % (col[judge_var_index].split('\n')[0], i))
            continue
        ab_point = find_same_diff_num(index_[i], np.arange(len(data)))
        count_point = np.zeros(len(data))
        count_point[ab_point[0]] = 1
        print('*' * 10)
        print('%s的顺序为%d' % (col[judge_var_index].split('\n')[0], i))
        draw_curve(data_copy_[:, judge_var_index]
                   , np.array(count_point)
                   , other=True
                   , option='abnormal_scatter'
                   , point_s=6
                   , xfigsize=(8, 6)
                   , y_label=col[judge_var_index]
                   , fontsize=15
                   , xdpi=300
                   , x_point=15
                   , bwith=1.5
                   , label=col[judge_var_index]
                   , x_label='井深$\mathrm{(m)}$'
                   , x_xticks=data['井深\nm'].values
                   , ignore=True
                   , yinv=False
                   , fig_name='%s未修正前的异常点' % col[judge_var_index].split('\n')[0]
                   , path=path
                   )

    for i in range(len(index_)):
        if len(index_[i]) == 0:
            continue
        print('第', i, '个特征异常值数量变化情况')
        print('原来数量', len(index_[i]))
        _, in_ = find_discontinue_data(index_[i], data_long=10, ind_=True)
        num = 0
        adjust_index = []
        for j in in_:
            num += j[1] - j[0] + 1
            adjust_index.extend(index_[i][j[0]:j[1] + 1])
        index_[i] = adjust_index
        print('后来数量', num)
        print('*' * 20)

    data_copy_ = np.array(data.values)
    for i in range(len(index_)):
        judge_var_index = i
        if len(index_[i]) == 0:
            print('%s的顺序为%d, 无异常值' % (col[judge_var_index].split('\n')[0], i))
            print(i)
            continue
        ab_point = find_same_diff_num(index_[i], np.arange(len(data)))
        count_point = np.zeros(len(data))
        count_point[ab_point[0]] = 1
        print('*' * 10)
        print('%s的顺序为%d' % (col[judge_var_index].split('\n')[0], i))
        print(len(ab_point[0]), ab_point[0])
        draw_curve(data_copy_[:, judge_var_index]
                   , np.array(count_point)
                   , other=True
                   , option='abnormal_scatter'
                   , point_s=6
                   , xfigsize=(8, 6)
                   , y_label=col[judge_var_index]
                   , fontsize=15
                   , xdpi=300
                   , x_point=15
                   , bwith=1.5
                   , label=col[judge_var_index]
                   , x_label='井深$\mathrm{(m)}$'
                   , x_xticks=data['井深\nm'].values
                   , fig_name='%s修正后的异常点' % col[judge_var_index].split('\n')[0]
                   , path=path
                   )
    # 统计一共有多少个点
    list_ = index_[0]
    for i in range(1, len(index_)):
        list_ = np.concatenate((list_, index_[i]))
    print('数据量为', len(data))
    list_ = np.sort(list(set(list_)))
    print('修正后异常值数量', len(np.sort(list(set(list_)))))
    ab_point2 = find_same_diff_num(list_, np.arange(len(data)))
    if path:
        data.iloc[ab_point2[1], :].to_csv(path + r'\处理异常值后的数据.csv'
                                          , encoding='gb18030')
    else:
        data.iloc[ab_point2[1], :].to_csv('...//处理异常值后的数据.csv'
                                          , encoding='gb18030')
    return data




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
