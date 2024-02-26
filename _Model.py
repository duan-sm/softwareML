#!/usr/bin/env python
# coding=utf-8

"""
@Time: 11/1/2023 4:28 PM
@Author: Shiming Duan
@Email: 1124682706@qq.com
@File: _Model.py
@Software: PyCharm
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import QMessageBox
from myLabelDialog import QmyLabelDialog
from PyQt5.QtWidgets import QDialog
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import metrics, Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v1 import Adam, SGD, RMSprop, Adadelta, Adagrad
from tensorflow.python.keras.callbacks import ModelCheckpoint

def initial(self):
    try:
        index = self.ui.tabWidget.currentIndex()
        print('index=',index)
        if index == 3:
            print('111')


            print('222')

            self.ui.Results3.setText('Label has been selected')
            for num in range(2, 5):
                print('num=',num)
                eval('self.ui.SelectFeature%d.clear()' % num)  # 清除图表
                eval('self.ui.SelectFeature%d.addItems(columns[:-4])' % num)  # 清除图表
            '''
            if isinstance(self.all_y, type(None)):
                if (self._DimDialog == None):  # 未创建对话框
                    self._DimDialog = QmyLabelDialog(self, column=columns)
                res = self._DimDialog.exec()
                if (res == QDialog.Accepted):
                    labelIndex = self._DimDialog.index
                    if labelIndex == -1:
                        self.ui.tabWidget.setCurrentIndex(0)
                        self.ui.Results.setText('Please re-select label')
                        self.ui.ResultsText2.append('Please re-select label')
                    else:
                        self.all_y = self.workbook.iloc[:, labelIndex].values
                        self.all_y_index = labelIndex
                        new_index = np.where(np.arange(len(columns)) != labelIndex)[0]
                        self.workbook = self.workbook.iloc[:, new_index]
                        self.ui.VariableOutput.setText(columns[labelIndex])
                        self.ui.Results3.setText('Label has been selected')
                        for num in range(1, 5):
                            eval('self.ui.SelectFeature%d.clear()' % num)  # 清除图表
                            eval('self.ui.SelectFeature%d.addItems(columns[new_index])' % num)  # 清除图表
                else:
                    self.ui.tabWidget.setCurrentIndex(0)
                    self.ui.Results.setText('Please re-select label')
                    self.ui.ResultsText2.append('Please re-select label')

            else:
                self.ui.ResultsText2.append('Label is existed')
            '''
    except:
        self.ui.Results.setText('Please input the data')
        self.ui.Results2.setText('Please input the data')
        self.ui.Results3.setText('Please input the data')
        self.ui.ResultsText1.append('Please input the data')
        self.ui.ResultsText2.append('Please input the data')

def Initialize(self):
    if self.ui.ModelTypeSelect.currentIndex() == 0:
        self.ui.ANN_layers.setText('100')
        self.ui.ANN_dropout.setText('0.2')
        self.ui.ANN_activation.setCurrentIndex(0)
        self.ui.ANN_optimizer.setCurrentIndex(0)
        self.ui.ANN_lr.setText('0.001')
        self.ui.ANN_loss.setCurrentIndex(0)
        self.ui.ANN_epochs.setText('2000')
    elif self.ui.ModelTypeSelect.currentIndex() == 1:
        self.ui.LSTM_layers.setText('100')
        self.ui.LSTM_dropout.setText('0.2')
        self.ui.LSTM_optimizer.setCurrentIndex(0)
        self.ui.LSTM_lr.setText('0.001')
        self.ui.LSTM_loss.setCurrentIndex(0)
        self.ui.LSTM_epochs.setText('500')
    else:
        pass

def get_optimizer(optimizer_name, lr):
    if optimizer_name == 'adam':
        return tf.keras.optimizers.Adam(lr)
    elif optimizer_name == 'adadelta':
        return tf.keras.optimizers.Adadelta(lr)
    elif optimizer_name == 'adagrad':
        return tf.keras.optimizers.Adagrad(lr)
    elif optimizer_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr)
    elif optimizer_name == 'sgd':
        return tf.keras.optimizers.SGD(lr)

def get_loss(loss_name, y_true, y_pre):
    if loss_name == 'mse':
        loss_ = tf.reduce_mean(tf.losses.mse(y_true, y_pre))
        return loss_
    elif loss_name == 'mae':
        loss_ = tf.reduce_mean(tf.losses.mean_absolute_error(y_true, y_pre))
        return loss_
    elif loss_name == 'mape':
        loss_ = tf.reduce_mean(tf.losses.mape(y_true, y_pre))
        return loss_
    elif loss_name == 'mean squard logarithmic':
        loss_ = tf.reduce_mean(tf.losses.mean_squared_logarithmic_error(y_true, y_pre))
        return loss_
    elif loss_name == 'binary crossentropy':
        loss_ = tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pre))
        return loss_

def draw_LossFig(self
                 , train_loss
                 , valid_loss):
    bwith = 1
    fontsize = 13
    # print('|1'*10)
    self.ui.LossFig.figure.clear()  # 清除图表
    font_ = {
        'family': 'Times New Roman'
        , 'weight': 'normal'
        , 'color': 'k'
        , 'size': fontsize
    }
    ax1 = self.ui.LossFig.figure.add_subplot(1, 1, 1, label='loss figure')
    # print('|1.5' * 10)
    ax1.plot(np.arange(len(train_loss)), train_loss, 'b-', label='train loss', linewidth=bwith)
    ax1.plot(np.arange(len(valid_loss)), valid_loss, 'r-', label='valid loss', linewidth=bwith)
    # print('|1.8' * 10)
    ax1.set_xlabel('Train Times', fontdict=font_)
    ax1.set_ylabel('Loss Value', fontdict=font_)
    ax1.set_title('Loss Visualization', fontsize=fontsize)
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    # print('|2' * 10)
    font1 = {
        'family': 'Times New Roman'
        , 'weight': 'normal'
        , 'size': fontsize
    }
    ax1.legend(loc="best", prop=font1, framealpha =0)
    points_num = 10
    # print('|3' * 10)
    x1_label = ax1.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    # print('|4' * 10)
    # points_num_gaps = len(train_loss)//points_num
    # new_xticks = np.arange(0,len(train_loss),points_num_gaps)
    # new_xticklabels = np.arange(0,len(train_loss),points_num_gaps)
    # print('|4' * 10)
    # ax1.set_xticks(new_xticks)
    # ax1.set_xticklabels(new_xticklabels, fontproperties='Times New Roman', fontsize=fontsize)

    # print('|5' * 10)
    ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    # print('|5.5' * 10)
    self.ui.LossFig.figure.tight_layout()
    # print('|6' * 10)
    self.ui.Results3.setText('loss绘图完成')
    # ax1.show()
    print('draw_LossFig|' * 10)
    self.ui.LossFig.figure.canvas.draw()
    # print('****' * 3)
    return 0


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
# workbook = pd.read_excel(r'E:/Software-Duan/dist/test_data.xlsx',index_col=0)
# print(workbook)
def timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=128,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None,):
    '''
    transfer data to time array
    '''
    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(
            "`sampling_rate` must be higher than 0. Received: "
            f"sampling_rate={sampling_rate}"
        )
    if sampling_rate >= len(data):
        raise ValueError(
            "`sampling_rate` must be lower than the length of the "
            f"data. Received: sampling_rate={sampling_rate}, for data "
            f"of length {len(data)}"
        )
    if sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be higher than 0. Received: "
            f"sequence_stride={sequence_stride}"
        )
    if sequence_stride >= len(data):
        raise ValueError(
            "`sequence_stride` must be lower than the length of the "
            f"data. Received: sequence_stride={sequence_stride}, for "
            f"data of length {len(data)}"
        )

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory
    # usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        target_ds = sequences_from_indices(
            targets, indices, start_index+sequence_length-1, end_index+sequence_length-1
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


def MyBP(self, x_train, y_train, x_test, y_test
         , units=[100], dropout=0.2, activation='relu'
         , loss='mape', optimizer='adam', epochs=100
         , lr=0.01, batch_num=3000, style=1):
    """
    建立三层BP神经网络模型。
    :param units:隐藏层神经元数，默认10。
    :param activations:激活函数，默认‘relu’。
    :param dropout:随机失活比例，默认0.2。
    :param lr:学习率，默认0.01。
    :return:模型。
    """
    self.ui.LossFig.figure.clear()
    Dimen_index = self.ui.DimensionlessType.currentIndex()
    if Dimen_index == 0:
        s_m = 'S'
    else:
        s_m = 'M'

    if self.state != 0 and self.state != 1:
        dlgTitle = "Tips"
        strInfo = ("Please normalize or reduce dimension.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        self.ui.tabWidget.setCurrentIndex(2)
        self.ui.ResultsText2.append('Please normalize or reduce dimension')
        return 0
    # if self.state == 1:
    #     NondimenBtn(self, y_test)
    print('*2' * 10)
    if not os.path.exists('./res/model'):
        os.makedirs('./res/model')
    if not os.path.exists('./res/history'):
        os.makedirs('./res/history')
    if not os.path.exists('./res/result'):
        os.makedirs('./res/result')
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    if len(x_train)<batch_num:
        batch_size = len(x_train)
    else:
        batch_size = len(x_train)//batch_num
    net_ = str(units[0])
    print('*3' * 10)
    model = Sequential()
    print('x_train.shape',x_train.shape)
    model.add(Dense(units=units[0], input_shape=(None, x_train.shape[1])
                    , activation=activation, kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dropout(rate=dropout))  # 防止过拟合
    for unit in range(1,len(units)):
        model.add(Dense(units=units[unit], activation=activation, kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dropout(rate=dropout))  # 防止过拟合
        net_ = net_+'_'+str(units[unit])
    if style == 0:
        model.add(Dense(units=1, activation='sigmoid'))
        net_ = net_ + '_' + str(1)
    else:
        model.add(Dense(units=3, activation='sigmoid'))
        net_ = net_ + '_' + str(3)
    model.summary()
    filepath = './res/model/BP+{}+{}+{}+{}+{}.h5'.format(net_, activation, dropout, lr,
                                                         epochs)  # 保存最优权重.units+dropout+lr+act+epochs
    if s_m == 'S':
        # if style == 0:
        #     pd.DataFrame([self.mean_, self.var_],
        #                  columns=['x', 'y', 'z', 'mean', 'p', 'stress1', 'stress2', 'no']).to_csv(
        #         './res/model/BP+{}+{}+{}+{}+{}+standard_para.csv'.format(net_, activation, dropout, lr,
        #                                                                  epochs), index=False)

        pd.DataFrame([self.mean_, self.var_], columns=self.workbook.columns, index=['mean', 'var']).to_csv(
                './res/model/BP+{}+{}+{}+{}+{}+standard_para.csv'.format(net_, activation, dropout, lr,
                                                             epochs), index=False)
    elif s_m == 'M':
        pd.DataFrame([self.data_max_, self.data_min_], columns=self.workbook.columns, index=['max', 'min']).to_csv(
            './res/model/BP+{}+{}+{}+{}+{}+maxmin_para.csv'.format(net_, activation, dropout, lr,
                                                         epochs), index=False)
    '''
    style: 1
    '''

    COUNT = 0
    train_loss_bs = []
    train_loss_sum = []
    valid_loss_sum = []
    min_valid_loss = 10e8
    point_num = 1
    opt = get_optimizer(optimizer, lr)
    for i in range(epochs):
        print('-----')
        for bs in range(batch_size):
            with tf.GradientTape() as tape:
                predict = model(x_train[bs::batch_size])
                train_loss = get_loss(loss_name=loss, y_true=y_train[bs::batch_size], y_pre=predict)
            grads = tape.gradient(train_loss, model.trainable_variables)
            train_loss_bs.append(train_loss.numpy())
            opt.apply_gradients(zip(grads, model.trainable_variables))
        # 测试集
            #             model.save(r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式'+'\\'+str(units)+'\\'+str(i)+'.h5')
        train_loss_sum.append(np.mean(train_loss_bs))
        out = model(x_valid)
        valid_loss = get_loss(loss_name=loss, y_true=y_valid, y_pre=out) # tf.reduce_mean(tf.losses.mape(y_valid, out))
        valid_loss_sum.append(valid_loss.numpy())
        if i % 10 == 0:
            print('The epoch of training is:%d' % i)
            print('loss_value:%.3f，valid_loss:%.3f' % (np.mean(train_loss_bs), valid_loss.numpy()))
        if valid_loss < min_valid_loss:
            COUNT = 0
            min_valid_loss = valid_loss
            model.save(filepath)
        else:
            COUNT += 1
        print('*' * 50)
        if COUNT > 30:
            # self.ui.Results3.setText('Training %d times,'
            #                          'the effect of the model did not change for 30 times, training ia stopped'%i)
            break
    draw_LossFig(self, np.array(train_loss_sum), np.array(valid_loss_sum))
    history = np.array([train_loss_sum, valid_loss_sum]).T
    pd.DataFrame(history, columns=['train_loss', 'valid_loss']).to_csv(
            './res/history/BP+{}+{}+{}+{}+{}.csv'.format(net_, activation, dropout, lr,
                                                             epochs), index=False)

    print('net_=', net_)
    '''
    style: 2
    
    # print('net_' ,net_)
    # model.compile(loss=loss, optimizer=optimizer, metrics=loss)
    # model.summary()
    # # print('*5' * 10)
    # print(net_, activation, dropout, lr, epochs)
    # filepath = './res/model/BP+{}+{}+{}+{}+{}.h5'.format(net_, activation, dropout, lr,
    #                                                          epochs)  # 保存最优权重.units+dropout+lr+act+epochs
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    # bp_history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
    #                        callbacks=callbacks_list, verbose=0)
    # # print('*6' * 10)
    # pd.DataFrame(bp_history.history).to_csv(
    #     './res/history/BP+{}+{}+{}+{}+{}.csv'.format(net_, activation, dropout, lr,
    #                                                      epochs), index=False)
    '''
    # print('*7' * 10)
    # 导入模型进行测试
    # start = t.perf_counter()
    model_saved = load_model(filepath)
    y_pred = model_saved.predict(x_test)
    # end = t.perf_counter()
    y_pred = np.array(y_pred).flatten()
    print(y_pred)
    # print('V最大值:', np.max(data['V']))
    # print('V最小值:', np.min(data['V']))
    # y_pred_it = inverse_transform(y_pred, np.max(data['V']), np.min(data['V']))
    # y_test_it = inverse_transform(y_test, np.max(data['V']), np.min(data['V']))

    bp_mse = metrics.mean_squared_error(y_test, y_pred)
    bp_mae = metrics.mean_absolute_error(y_test, y_pred)
    bp_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # bp_mre = relative_error(y_test, y_pred)
    # print('*8' * 10)
    df = pd.DataFrame()
    print('test result:')
    print('bp_mse=%.3f, bp_mae=%.3f, bp_rmse=%.3f'%(bp_mse, bp_mae, bp_rmse))
    df['V_t'] = y_test
    method = self.ui.DimensionlessType.currentIndex()
    if method==0:
        y_pred = y_pred*((self.data_max_-self.data_min_))+self.data_min_
    else:
        y_pred = y_pred * self.var_ + self.mean_
    df['V_p'] = y_pred
    df.to_csv('./res/result/BP_result.csv', index=False)
    # print('*9' * 10)
    return model



def MyLSTM(self, dataset_train, dataset_valid, dataset_test
         , units=[100], dropout=0.2
         , loss='mape', optimizer='adam', epochs=100
         , lr=0.01, batch_num=3000,style=1):
    """
    建立三层BP神经网络模型。
    :param units:隐藏层神经元数，默认10。
    :param activations:激活函数，默认‘relu’。
    :param dropout:随机失活比例，默认0.2。
    :param lr:学习率，默认0.01。
    :return:模型。
    """
    print('RUN myLSTM function  ')
    self.ui.LossFig.figure.clear()
    Dimen_index = self.ui.DimensionlessType.currentIndex()
    if Dimen_index == 0:
        s_m = 'S'
    else:
        s_m = 'M'

    if self.state != 0 and self.state != 1:
        dlgTitle = "Tips"
        strInfo = ("Please normalize or reduce dimension.")
        defaultBtn = QMessageBox.NoButton  # 缺省按钮
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        self.ui.tabWidget.setCurrentIndex(2)
        self.ui.ResultsText2.append('Please normalize or reduce dimension')
        return 0
    # if self.state == 1:
    #     NondimenBtn(self, y_test)
    # print('*2' * 10)
    print('myLSTM 1 '*10)
    if not os.path.exists('./res/model'):
        os.makedirs('./res/model')
    if not os.path.exists('./res/history'):
        os.makedirs('./res/history')
    if not os.path.exists('./res/result'):
        os.makedirs('./res/result')
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    for batch in dataset_train.take(1):
        inputs, targets = batch
    # print(inputs, targets)
    net_ = str(units[0])
    # print('*3' * 10)
    print('myLSTM 2 ' * 10)
    model = Sequential()
    if len(units)<=1:
        # print('len(units)<1')
        # print(units)
        # for j in units:
        #     print(j)
        #     print(type(j))
        # print('inputs.shape')
        # print(inputs.shape)
        # print('dropout')
        # print(dropout)
        # print(units[0], type(units[0]), (inputs.shape[1], inputs.shape[2]),dropout)
        model.add(LSTM(units=units[0], input_shape=(inputs.shape[1], inputs.shape[2]), dropout=dropout
                        , kernel_initializer='random_uniform', bias_initializer='zeros'))
        # print('model.add LSTM')
    else:
        print('len(units)>1')
        print(units)
        model.add(LSTM(units=units[0], input_shape=(inputs.shape[1], inputs.shape[2]), dropout=dropout
                       , kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True))
        for unit in range(1, len(units)-1):
            model.add(LSTM(units=units[unit], dropout=dropout
                       , kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True))
            net_ = net_ + '_' + str(units[unit])
        model.add(LSTM(units=units[-1], dropout=dropout
                       , kernel_initializer='random_uniform', bias_initializer='zeros'))
        net_ = net_ + '_' + str(units[-1])
    if style == 0:
        model.add(Dense(units=1, activation='sigmoid'))
        net_ = net_ + '_' + str(1)
    else:
        model.add(Dense(units=3, activation='sigmoid'))
        net_ = net_ + '_' + str(3)
    # print('model.add Dense')
    model.summary()
    # print('model.summary()')
    print('myLSTM 3 ' * 10)
    filepath = './res/model/LSTM+{}+{}+{}+{}.h5'.format(net_, dropout, lr, epochs)  # 保存最优权重.units+dropout+lr+act+epochs
    if s_m == 'S':
        pd.DataFrame([self.mean_, self.var_], columns=self.workbook.columns, index=['mean', 'var']).to_csv(
            './res/model/LSTM+{}+{}+{}+{}+standard_para.csv'.format(net_, dropout, lr,
                                                                     epochs), index=False)
    elif s_m == 'M':
        pd.DataFrame([self.data_max_, self.data_min_], columns=self.workbook.columns, index=['max', 'min']).to_csv(
            './res/model/LSTM+{}+{}+{}+{}+maxmin_para.csv'.format(net_, dropout, lr,
                                                                   epochs), index=False)
    '''
    style: 1
    '''
    print('myLSTM 4 ' * 10)
    COUNT = 0
    train_loss_bs = []
    train_loss_sum = []
    valid_loss_sum = []
    min_valid_loss = 10e8
    point_num = 1
    opt = get_optimizer(optimizer, lr)
    for i in range(epochs):
        print('epoch=%d'%i)
        for bs in range(batch_num):
            print('bs=',bs)
            print('batch_num=', batch_num)
            for batch in dataset_train.take(bs + 1):
                inputs, targets = batch
            with tf.GradientTape() as tape:
                predict = model(inputs)
                train_loss =get_loss(loss_name=loss, y_true=targets, y_pre=predict)
            grads = tape.gradient(train_loss, model.trainable_variables)
            train_loss_bs.append(train_loss.numpy())
            opt.apply_gradients(zip(grads, model.trainable_variables))
        print('start compute')
        # 测试集
        #             model.save(r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式'+'\\'+str(units)+'\\'+str(i)+'.h5')
        train_loss_sum.append(np.mean(train_loss_bs))
        print('train_loss_sum.append')
        for batch in dataset_valid.take(1):
            inputs, targets = batch
        print('inputs, targets = batch')
        out = model(inputs)
        print('out = model(inputs)')
        valid_loss = get_loss(loss_name=loss, y_true=targets, y_pre=out)  # tf.reduce_mean(tf.losses.mape(y_valid, out))
        print('valid_loss')
        valid_loss_sum.append(valid_loss.numpy())
        if i % 10 == 0:
            print('The epoch of training is:%d' % i)
            print('loss_value:%.3f，valid_loss:%.3f' % (np.mean(train_loss_bs), valid_loss.numpy()))
        if valid_loss < min_valid_loss:
            COUNT = 0
            min_valid_loss = valid_loss
            model.save(filepath)
        else:
            COUNT += 1
        print('*' * 50)
        if COUNT > 30:
            # self.ui.Results3.setText('Training %d times,'
            #                          'the effect of the model did not change for 30 times, training ia stopped'%i)
            break
    draw_LossFig(self, np.array(train_loss_sum), np.array(valid_loss_sum))
    history = np.array([train_loss_sum, valid_loss_sum]).T
    pd.DataFrame(history, columns=['train_loss', 'valid_loss']).to_csv(
        './res/history/LSTM+{}+{}+{}+{}.csv'.format(net_, dropout, lr,epochs), index=False)

    print('net_=', net_)
    print('myLSTM 5 ' * 10)
    # print('*7' * 10)
    # 导入模型进行测试
    # start = t.perf_counter()
    for batch in dataset_test.take(1):
        inputs, targets = batch
    model_saved = load_model(filepath)
    y_pred = model_saved.predict(inputs)
    # end = t.perf_counter()
    y_pred = np.array(y_pred).flatten()
    print(y_pred)
    # print('V最大值:', np.max(data['V']))
    # print('V最小值:', np.min(data['V']))
    # y_pred_it = inverse_transform(y_pred, np.max(data['V']), np.min(data['V']))
    # y_test_it = inverse_transform(y_test, np.max(data['V']), np.min(data['V']))

    bp_mse = metrics.mean_squared_error(targets, y_pred)
    bp_mae = metrics.mean_absolute_error(targets, y_pred)
    bp_rmse = np.sqrt(metrics.mean_squared_error(targets, y_pred))
    # bp_mre = relative_error(y_test, y_pred)
    # print('*8' * 10)
    df = pd.DataFrame()
    print('test result:')
    print('bp_mse=%.3f, bp_mae=%.3f, bp_rmse=%.3f' % (bp_mse, bp_mae, bp_rmse))
    print('myLSTM 6 ' * 10)
    method = self.ui.DimensionlessType.currentIndex()
    if method == 0:
        y_pred = y_pred * ((self.data_max_ - self.data_min_)) + self.data_min_
        targets = targets * ((self.data_max_ - self.data_min_)) + self.data_min_
    else:
        y_pred = y_pred * self.var_ + self.mean_
        targets = targets * self.var_ + self.mean_
    df['V_t'] = targets
    df['V_p'] = y_pred
    df.to_csv('./res/result/LSTM_result.csv', index=False)
    print('myLSTM 7 ' * 10)
    # print('*9' * 10)
    return model


def Compute(self):
    activation_dict = {
        0:'relu'
        ,1:'tanh'
        ,2:'sigmoid'
        ,3:'softmax'
    }
    optimizer_dict = {
        0: 'adam'
        , 1: 'adadelta'
        , 2: 'adagrad'
        , 3: 'rmsprop'
        , 4:'sgd'
    }
    loss_dict = {
        0: 'mse'
        , 1: 'mae'
        , 2: 'mape'
        , 3: 'mean squard logarithmic'
        , 4: 'binary crossentropy'
    }
    index = self.ui.ModelTypeSelect.currentIndex()
    if index==0:
        try:
            units = []
            layers = self.ui.ANN_layers.text()
            dropout = float(self.ui.ANN_dropout.text())
            activation_index = self.ui.ANN_activation.currentIndex()
            optimizer_index = self.ui.ANN_optimizer.currentIndex()
            lr = float(self.ui.ANN_lr.text())
            loss_index = self.ui.ANN_loss.currentIndex()
            epochs = int(self.ui.ANN_epochs.text())
            pre_units = layers.split('-')
            for i in range(len(pre_units)):
                units.append(int(pre_units[i]))
            self.ui.Results3.setText('Parameter input is correct, the model is working')
            units = np.array(units)
            if units[-1]==3:
                units = units[:-1]
            activation = activation_dict[activation_index]
            optimizer = optimizer_dict[optimizer_index]
            loss = loss_dict[loss_index]

            print('units, dropout, activation, optimizer, lr, loss, epochs')
            print(units, dropout, activation, optimizer, lr, loss, epochs)
        except:
            self.ui.Results3.setText('Please input the correct model parameters')
            return 0
    elif index==1:
        try:
            units = []
            layers = self.ui.LSTM_layers.text()
            dropout = float(self.ui.LSTM_dropout.text())
            optimizer_index = self.ui.LSTM_optimizer.currentIndex()
            lr = float(self.ui.LSTM_lr.text())
            loss_index = self.ui.LSTM_loss.currentIndex()
            epochs = int(self.ui.LSTM_epochs.text())
            pre_units = layers.split('-')
            for i in range(len(pre_units)):
                units.append(int(pre_units[i]))
            self.ui.Results3.setText('Parameter input is correct, the model is working')
            units = np.array(units)
            if units[-1]==3:
                units = units[:-1]

            optimizer = optimizer_dict[optimizer_index]
            loss = loss_dict[loss_index]

            print('units, dropout, optimizer, lr, loss, epochs')
            print(units, dropout, optimizer, lr, loss, epochs)
        except:
            self.ui.Results3.setText('Please input the correct model parameters')
            return 0
    else:
        self.ui.Results3.setText('Something is wrong ???')

    style = self.ui.Style1.currentIndex()
    try:
        if index==0:
            if style == 0:
                all_data = self.workbook.values
                x_data = all_data[:, :4]
                y_data = self.workbook.loc[:, 'p'].values.reshape((len(all_data),1))
            else:
                all_data = self.workbook.values
                x_data = all_data[:,:4]
                y_data = all_data[:,4:-1]
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data
                                                                , test_size=0.33, random_state=42, shuffle=True)
            print('*1'*10)
            MyBP(self, x_train, y_train, x_test, y_test
                 , units=units, dropout=dropout, activation=activation
                 , loss=loss, optimizer=optimizer, epochs=epochs
                 , lr=lr, batch_num=3000,style=style)
            # the batch_num is different from LSTM
            # this means the size of batch
        elif index == 1:
            print('Run LSTM model')
            if style == 0:
                all_data = self.workbook.values
                x_data = all_data[:, :4]
                y_data = self.workbook.loc[:, 'p'].values.reshape((len(all_data),1))
            else:
                all_data = self.workbook.values
                x_data = all_data[:, :4]
                y_data = all_data[:, 4:-1]
            x_train = x_data[:-5000]
            y_train = y_data[:-5000]
            x_test = x_data[-5000:]
            y_test = y_data[-5000:]
            sequence_length = 10
            sequence_stride = 1
            step = 1
            batch_size = 2000

            # the batch_num is different from BP
            # this means the number of batch
            if len(y_train) % batch_size == 0:
                batch_num = len(y_train) // batch_size
            else:
                batch_num = len(y_train) // batch_size + 1
            print('y_train=',len(y_train))
            print('batch_num=', batch_num)
            dataset_train = timeseries_dataset_from_array(
                x_train,
                y_train,
                sequence_length=sequence_length,  # 一批采集120次
                sequence_stride=sequence_stride,
                sampling_rate=step,  # 6步采集一次，即 60分钟采集一次
                batch_size=batch_size,
            )
            dataset_valid = timeseries_dataset_from_array(
                x_train,
                y_train,
                sequence_length=sequence_length,  # 一批采集120次
                sequence_stride=sequence_stride,
                sampling_rate=step,  # 6步采集一次，即 60分钟采集一次
                batch_size=len(y_train) + 1,
            )
            dataset_test = timeseries_dataset_from_array(
                x_test,
                y_test,
                sequence_length=sequence_length,  # 一批采集120次
                sequence_stride=sequence_stride,
                sampling_rate=step,  # 6步采集一次，即 60分钟采集一次
                batch_size=len(y_test) + 1,
            )
            MyLSTM(self, dataset_train, dataset_valid, dataset_test
                 , units=units, dropout=dropout
                 , loss=loss, optimizer=optimizer, epochs=epochs
                 , lr=lr, batch_num=batch_num,style=style)
            print('*2' * 10)
    except:
        self.ui.Results3.setText('Something is wrong, you should have a change')




r'''
random.seed(10)
tf.random.set_seed(10)
np.random.seed(10)
# 循环次数 测试点
point_num = 5
epochs = 3001
batch_num = 50000
batch_sizes = len(y_train) // batch_num
units_comb = np.array([2, 3, 5, 8])
# units_comb = np.array([10])
# units_comb = np.array([50,100,200,300,500,800,1000])
units_comb_valid_loss_sum = []
units_comb_loss_sum = []
record_times = []
for units in units_comb:
    print('-' * 50)
    print('神经元数量为', units)
    print('-' * 50)
    t1 = time.time()
    min_valid_loss = 10e8
    valid_loss_sum = []
    loss_sum = []
    m = 0
    m = Sequential([layers.Dense(units, activation='relu')
                       , tf.keras.layers.Dropout(0.2)
                       , layers.Dense(1)])
    m.build(input_shape=(None, x_train.shape[1]))
    m.summary()
    COUNT = 0
    optimizer = optimizers.Adam(0.001)
    for i in range(epochs):
        #         if i%100 == 0:
        #             print('训练%d次'%i)
        for bs in range(batch_sizes):
            with tf.GradientTape() as tape:
                predict = m(x_train[bs::batch_sizes])
                loss = tf.reduce_mean(tf.losses.mape(y_train[bs::batch_sizes], predict))
            grads = tape.gradient(loss, m.trainable_variables)
            loss_sum.append(loss.numpy())
            optimizer.apply_gradients(zip(grads, m.trainable_variables))
        # 测试集
        if i % point_num == 0:
            #             m.save(r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式'+'\\'+str(units)+'\\'+str(i)+'.h5')
            print('训练次数为:%d' % i)
            out = m(x_valid)
            valid_loss = tf.reduce_mean(tf.losses.mape(y_valid, out))
            valid_loss_sum.append(valid_loss.numpy())
            print('loss为:%.3f，valid_loss为:%.3f' % (loss, valid_loss))
            t2 = time.time()
            record_times.append(t2 - t1)
            if valid_loss < min_valid_loss:
                COUNT = 0
                min_valid_loss = valid_loss
                m.save(
                    r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式' + '\\' + str(units) + '\\' + str(
                        i) + '-best.h5')
            else:
                COUNT += 1
                m.save(
                    r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式' + '\\' + str(units) + '\\' + str(
                        i) + '.h5')

            print('耗时:', t2 - t1)
            print('*' * 50)
        if COUNT > 30:
            break

    units_comb_loss_sum.append(loss_sum)
    units_comb_valid_loss_sum.append(valid_loss_sum)

    loss__ = pd.DataFrame(units_comb_valid_loss_sum, index=units_comb[:np.where(units == units_comb)[0][0] + 1])
    loss__.to_csv(
        r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式\归一化神经元%d_验证损失值.csv' % units
        )
    loss__ = pd.DataFrame(units_comb_loss_sum, index=units_comb[:np.where(units == units_comb)[0][0] + 1])
    loss__.to_csv(
        r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式\归一化神经元%d_训练损失值.csv' % units
        )
#     ttt = np.array(record_times).reshape(np.where(units==units_comb)[0][0]+1,-1)
#     ttt = pd.DataFrame(ttt, index=units_comb[:np.where(units==units_comb)[0][0]+1])
#     ttt.to_csv(r'G:\F盘\0论文\压力信号小波分析约束下的气侵预警模型\出口流量公式\归一化神经元%d_训练时间.csv'%units
#                 )
print('结束')


'''