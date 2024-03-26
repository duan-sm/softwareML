#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 16:12
# @Author  : dsm
# @File    : plotting_function.py
# @Software: win10  python3.8.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
# import time
import seaborn as sns
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号的问题



# 对多个变量绘制曲线
def draw_cross_curves(data, x_=None, factor=None, picture_name=None
                      , mode=None, trend=False, times=3, point_num=40
                      , truncate=False, start_time=None, end_time=None
                      , path=None, fontsize=5, bwith=3, xfigsize=(10, 8)
                      , xdpi=200, s_m='M', all_time=True, yinv=False
                      , x_label='时间', y_label='测深/m', c_=['b', 'r', 'k']
                      , linw=2, lins=['-', '-'], label=''):
    '''
    Ignore
    绘制变量曲线
    :param data: 数据
    :param x_: x坐标轴刻度值
    :param factor: 画在一张图上的曲线名
    :param picture_name: 保存图片时的图片名（决定是否保存图片）
    :param mode: 确定是自动调节归一化/不归一化   可以写auto
    :param trend: 是否进行归一化绘图
    :param times: 当不同变量最大值相差为 多少倍时 进行归一化绘图
    :param point_num: 横坐标刻度值数量
    :param truncate: 是否对数据进行截断
    :param start_time: 绘制图片的数据开始时间
    :param end_time: 绘制图片的数据结束时间
    :param path: 保存图片的路径
    :param fontsize: 字体大小
    :param bwith: 边框线条宽度
    :param xfigsize: 图片大小
    :param xdpi: 图片清晰度
    :param s_m: 归一化方式S标准化，M最大最小
    :param all_time: 横坐标是否显示全部时间
    :param yinv: 纵坐标是否反转
    :param x_label: x坐标轴的标签
    :param y_label: y坐标轴的标签
    :param c_: 曲线的颜色
    :param linw: 绘制的曲线的宽度
    :param lins: 绘制的曲线的类型
    :param label: 绘制的曲线的比标签
    :return: 无
    '''

    input_data = data.copy()
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        #     "font.serif": ['SimSun'],
        "font.serif": ['SimHei'],
    }
    rcParams.update(config)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False
    if factor:
        max_min = input_data.loc[:, factor].max()
        # print(max_min)
    if len(factor) == 1:
        mode = False
    if mode == 'auto':
        if max_min.min() == 0:
            if max_min.max() > times:
                print(max_min)
                print('变量最大值相差过大，进行归一化处理绘图')
                trend = True
        elif max_min.max() / max_min.min() > times:
            print(max_min)
            print('变量最大值相差过大，进行归一化处理绘图')
            trend = True
    if x_ is not None:
        x = x_
    else:
        if '时间' in input_data.columns:
            x = input_data['时间'].values
        else:
            print('数据中无时间列，请增加或指定每个点的时间（即输入x_）')
            return 0

    if truncate:
        input_data['时间'] = pd.to_datetime(x)
        input_data.set_index('时间', inplace=True)
        input_data = input_data.loc[start_time:end_time, :]
        x = input_data.index
    if trend:
        try:
            print('进行归一化')
            input_data_cols = input_data.columns
            print(input_data.values.shape)
            input_data = input_data.values
            Intermediate_data = []
            for kind in range(input_data.shape[1]):
                print(input_data[:, kind])
                if s_m == 'M':
                    data_col = (input_data[:, kind] - np.min(input_data[:, kind])) / (
                                np.max(input_data[:, kind]) - np.min(input_data[:, kind]))
                    Intermediate_data.append(data_col)
                else:
                    data_col = (input_data[:, kind] - np.mean(input_data[:, kind])) / (np.std(input_data[:, kind]))
                    Intermediate_data.append(data_col)
            input_data = np.array(Intermediate_data).T
            input_data = pd.DataFrame(input_data,
                                      columns=input_data_cols)
            print('完成归一化')
        except Exception as e:
            print('出错', e)
            print('input_data的时间已删除，无法确定x轴')
            input_data['时间'] = input_data.index

    plt.figure(figsize=xfigsize, dpi=xdpi)

    # 获取横坐标
    x_change = x.astype(str)
    if all_time != True:
        x_change = np.array([i.split(' ')[1] for i in x_change])
        plt.xticks(fontproperties='Times New Roman', size=fontsize
                   , rotation=45)
    if trend:
        pass
    else:
        input_data_cols = input_data.columns
        input_data = pd.DataFrame(input_data
                                  , columns=input_data_cols)
    compare_depth = input_data[factor]
    # 绘制井深和钻头位置图片
    for i in range(len(compare_depth.columns)):
        print('*' * 40 + factor[i] + '*' * 40)
        y = compare_depth.iloc[:, i]
        if trend:
            if label:
                plt.plot(y, label=label[i], color=c_[i], lw=linw, ls=lins[i])
            else:
                plt.plot(y, label=factor[i], color=c_[i], lw=linw, ls=lins[i])
        else:
            if label:
                plt.plot(y.values, label=label[i], color=c_[i], lw=linw, ls=lins[i])
            else:
                plt.plot(y.values, label=factor[i], color=c_[i], lw=linw, ls=lins[i])

    ax = plt.gca()

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    # 横坐标显示坐标数量

    all_point_num = len(x_change)
    if all_point_num > point_num:
        row_point_num = int(all_point_num / point_num)
        plt.xticks(range(0, all_point_num, row_point_num)
                   , x_change[::row_point_num]
                   , rotation=45
                   , fontsize=fontsize
                   , fontproperties='Times New Roman')
    else:
        plt.xticks(fontsize=fontsize
                   , rotation=45
                   , fontproperties='Times New Roman')

    font1 = {
        'family': 'SimHei'
        , 'weight': 'normal'
        , 'size': fontsize
    }
    plt.legend(loc="best", prop=font1, edgecolor='white', framealpha=0)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    # 设置边框大小
    ax = plt.gca()  # 获取边框
    if yinv:
        ax.invert_yaxis()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.tick_params(width=bwith, length=bwith + 1, direction='in')
    if picture_name:
        if path:
            plt.savefig(path + '//' + picture_name + '.jpg', bbox_inches="tight")
        else:
            plt.savefig('...//' + picture_name + '.jpg', bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    # time.sleep(1)
    plt.close()


# 判断
def draw_curve(data
               , K_data=None
               , other=None
               , option=None
               , fig_name=None
               , path=None
               , title=None
               , label='nothing'
               , x_label='井深'
               , y_label='训练次数'
               , x_xticks=None
               , x_point=40
               , state=None
               , xfigsize=(14, 8)
               , xdpi=120
               , fontsize=20
               , bwith=3
               , truncate=False
               , start_t=None
               , end_t=None
               , ignore=False
               , yinv=False
               , point_s=5
               , cls=['b', 'g', 'red', 'c', 'm', 'k', 'coral'
            , 'darkkhaki', 'darkmagenta', 'springgreen',
                      'steelblue', 'tan', 'teal', 'firebrick']
               ):
    '''
    Ignore
    :param data: 数据绘制曲线图
    :param K_data:绘制other中的数据
    :param other: 是否绘制其他图片
    :param option: 绘制哪种，可选，目前只有'var_trend_scatter','state_scatter'
    :param fig_name: 保存名为fig_name的图片
    :param path: 保存fig_name图片的路径
    :param title: 图片的title
    :param label: 绘制直线的线名
    :param x_label: data曲线图的x_label
    :param y_label: data曲线图的y_label
    :param x_xticks:改变x坐标值
    :param x_point:在x轴上面显示的点的数量，默认40
    :param state:对于K_data的标签
    :param xfigsize: 图片的figsize
    :param xdpi: 图片的dpi
    :param fontsize: 图片横纵坐标数字，标签的字体大小
    :param bwith: data曲线图的线宽
    :param truncate:截断时间段
    :param start_t:开始截断点
    :param end_t:结束截断点
    :param ignore:忽略x轴坐标的年份
    :return: 无
    '''
    print('config')
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        #     "font.serif": ['SimSun'],
        "font.serif": ['SimHei'],
    }
    rcParams.update(config)
    plt.figure(figsize=xfigsize, dpi=xdpi)
    print('title')
    if title is not None:
        plt.title(title, fontsize=fontsize)
    print('truncate')
    if truncate:
        print('np.min(time), np.max(time)')
        print(np.min(x_xticks), np.max(x_xticks))

        s = pd.DataFrame(range(len(x_xticks)), pd.to_datetime(x_xticks), columns=['time'])
        index = s.loc[start_t:end_t, :]['time'].values
        x_xticks = s.loc[start_t:end_t, :].index
        data = data[index]
    print('len(data.shape)')
    if len(data.shape) > 1:
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=label[i])
    else:
        plt.plot(data, label=label)
    print('other')
    if other:
        if option == 'var_trend_scatter':
            if truncate:
                K_data = K_data[index]
            x_1 = np.where(K_data > 0)
            x0 = np.where(K_data == 0)
            x1 = np.where(K_data < 0)
            #                   记号形状       颜色      点的大小    设置标签
            plt.scatter(x_1, data[x_1], color='red', label='增大', s=point_s)
            plt.scatter(x0, data[x0], color='b', label='不变', s=point_s)
            plt.scatter(x1, data[x1], color='g', label='减小', s=point_s)
        if option == 'abnormal_scatter':
            print('truncate')
            if truncate:
                K_data = K_data[index]
            x_1 = np.where(K_data > 0)
            print('plt.scatter')
            plt.scatter(x_1, data[x_1], color='red', label='异常点', s=point_s)
        if option == 'state_scatter':
            K_data = K_data[:, index]
            if len(state) != K_data.shape[0]:
                print('num error: len(state) != K_data.shape[0]')
                return 'error'
            for i in range(len(state)):
                #                 print(i)
                x0 = np.where(K_data[i] > 0)

                plt.scatter(x0, data[x0], color=cls[i], label=state[i], s=point_s)
    print('plt.xlabel')
    # 设置横纵标签字体大小
    plt.xlabel(x_label, fontsize=fontsize, color='k', fontproperties='SimHei')
    plt.ylabel(y_label, fontsize=fontsize, color='k', fontproperties='SimHei')

    print('x_xticks')
    # 设置横纵坐标数字
    if x_xticks is not None:
        x_step = len(x_xticks) // x_point
        if x_step == 0:
            x_step = 2
        plt.xticks(np.arange(0, len(x_xticks), x_step), x_xticks[::x_step], \
                   fontproperties='Times New Roman', size=fontsize
                   , rotation=45)
    else:
        plt.xticks(fontproperties='Times New Roman', size=fontsize)
    print('yticks')
    plt.yticks(fontproperties='Times New Roman', size=fontsize)

    # 设置图例
    plt.tick_params(labelsize=fontsize)
    font1 = {
        'family': 'SimHei'
        , 'weight': 'normal'
        , 'size': fontsize - 5

    }
    plt.legend(loc="best", prop=font1, framealpha=0)

    # 设置边框大小
    ax = plt.gca()  # 获取边框
    if yinv:
        ax.invert_yaxis()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.tick_params(width=bwith, length=bwith * 2, direction='in')
    print('fig_name')
    if fig_name is not None:
        if path == None:
            print('please input the path of saving pictures')
        else:
            plt.savefig(path + '//' + fig_name + '.jpg', bbox_inches="tight")
    plt.tight_layout()
    print('plt.show')
    plt.show()
    # time.sleep(1)
    plt.close()


def draw_corr(data
              , limited_corr=0.6
              , limited_var=0.01
              , draw_all = False
              , path=None):
    '''
    Ignore
    绘制相关系数
    :param data: 输入纯数据，要求格式为DataFrame
    :param limited_corr: 绘制的相关系数阈值限制，大于阈值则显示
    :param limited_var: 绘制相关系数的方差大小限制，大于阈值则显示
    :param draw_all: 是否不考虑上述限制，绘制全部相关系数
    :param path: 图片保存的路径
    :return: 无
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False
    limited_corr = 0
    limited_var = 0.01
    draw_all = False
    data_columns = np.array([c.split('\n')[0] for c in data.columns])
    np_data = data.values
    # 对数据进行标准化
    Intermediate_data = []
    standard_var = []
    for kind in range(np_data.shape[1]):
        data_col = (np_data[:, kind] - np.mean(np_data[:, kind])) / (np.std(np_data[:, kind]))
        Intermediate_data.append(data_col)
        standard_var.append(np.std(np_data[:, kind]))
    np_data_STD = np.array(Intermediate_data).T
    standard_var = np.array(standard_var)
    vaild_cols = data_columns[standard_var > limited_var]
    print('方差大于%f的有效特征为：' % limited_var)
    print(vaild_cols)
    print('剩余特征为：', data_columns[standard_var <= limited_var])

    if draw_all == True:
        vaild_np_data_STD = pd.DataFrame(np_data, columns=data_columns)
    else:
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
    corr_ = corr1[corr_ != 0] + corr3[corr_ != 0]
    corr_ = corr_.fillna(0)
    m = np.max(np.abs(corr_.values))
    plt.figure(figsize=(8, 8), dpi=300)
    sns.heatmap(data=corr_,
                vmax=m,
                vmin=-m,
                #             cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
                #             annot=True,#图中数字文本显示
                fmt=".2f",  # 格式化输出图中数字，即保留小数位数等
                annot_kws={'size': 8, 'weight': 'normal', 'color': '#253D24'},  # 数字属性设置，例如字号、磅值、颜色
                mask=np.triu(np.ones_like(corr1, dtype=np.bool)),  # 显示对脚线下面部分图
                square=True, linewidths=.5,  # 每个方格外框显示，外框宽度设置
                cbar_kws={"shrink": .5}
                )
    if path:
        plt.savefig(path+r'\相关系数.jpg'
                   ,bbox_inches="tight")
    else:
        plt.savefig(r'...\相关系数.jpg'
                    , bbox_inches="tight")
    plt.show()
    # time.sleep(1)
    plt.close()
    return 0


def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示依照列来求均值。假设输入list,则axis=1

def PCA(data
        , limited_score=0.999
        , path=None):
    '''
    进行PCA降维
    PCA dimension reduction was performed
    :param data: Enter pure data in the DataFrame format      输入纯数据，要求格式为DataFrame
    :param limited_score: Lower limit of cumulative contribution (should not be less than the contribution of the first principal component)     累计贡献度的下限（不应该少于第一主成分的贡献度）
    :param path: path to save the image and analysis results     保存图片和分析结果的路径
    :return:
    finalData: Principal component analysis results and cumulative contribution
    featValue: Sorts the feature values and outputs the descending order:
    gx: Contribution of eigenvalues:
    lg: Cumulative contribution of eigenvalues:

    finalData:主成分分析结果和累计贡献度
    featValue:对特征值进行排序并输出 降序:
    gx:特征值的贡献度:
    lg:特征值的累计贡献度:
    '''
    df = data.copy()
    average = meanX(df)

    # View the number of columns and rows  查看列数和行数
    m, n = np.shape(df)
    print('columns(列数):%d,rows(行数)%d' % (m, n))

    # Mean value matrix 均值矩阵
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    # decentralization 去中心化
    data_adjust = df - avgs
    # Covariance matrix 协方差阵
    covX = np.cov(data_adjust.T)  # Calculate the covariance matrix 计算协方差矩阵
    # print(covX)

    # The eigenvalues and eigenvectors of the covariance matrix are calculated 计算协方差阵的特征值和特征向量
    featValue, featVec = np.linalg.eig(covX)  # The eigenvalues and eigenvectors of the covariance matrix are solved 求解协方差矩阵的特征值和特征向量
    # print(featValue, featVec)
    # Sort the eigenvalues and output descending order 对特征值进行排序并输出 降序
    featValue = sorted(featValue)[::-1]

    # Draw scatter plots and line plots 绘制散点图和折线图
    # Draw scatter plots and line plots with the same data 同样的数据绘制散点图和折线图
    plt.scatter(range(1, df.shape[1] + 1), featValue)
    plt.plot(range(1, df.shape[1] + 1), featValue)

    # Displays the title of the graph and the name of the xy axis      显示图的标题和xy轴的名字
    # Ignore 最好使用英文，中文可能乱码
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")

    plt.grid()  # Display grid 显示网格
    plt.show()  # Display graphics 显示图形

    # Find the contribution degree of the eigenvalue 求特征值的贡献度
    gx = featValue / np.sum(featValue)

    # Find the cumulative contribution degree of eigenvalues 求特征值的累计贡献度
    lg = np.cumsum(gx)

    # Selected principal component 选出主成分
    k = [i for i in range(len(lg)) if lg[i] < limited_score]
    k = list(k)
    print(k)

    # The eigenvector matrix corresponding to the principal component is selected 选出主成分对应的特征向量矩阵
    selectVec = np.matrix(featVec.T[k]).T
    selectVe = selectVec * (-1)
    # print(selectVec)

    # Principal component score 主成分得分
    finalData = np.dot(data_adjust, selectVec)

    # Mapping heat map 绘制热力图
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
    # Set the Y-axis font size 设置y轴字体大小
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    plt.title("Factor Analysis", fontdict=font2)

    # Set the Y-axis label 设置y轴标签
    plt.ylabel("Sepal Width", fontdict=font2)
    if path:
        plt.savefig(path+r'\Principal component analysis eigenvector matrix-主成分分析特征向量矩阵.jpg'
                   ,bbox_inches="tight")
        pd.DataFrame(finalData).to_csv(path+r'\Principal component analysis results-主成分分析结果.csv'
                                       , encoding='gb18030')
    else:
        plt.savefig(r'...\Principal component analysis eigenvector matrix-主成分分析特征向量矩阵.jpg'
                    , bbox_inches="tight")
        pd.DataFrame(finalData).to_csv(r'...\Principal component analysis results-主成分分析结果.csv'
                                       , encoding='gb18030')

    plt.show()
    # time.sleep(10)
    plt.close()
    return [finalData,featValue,gx,lg,selectVec]