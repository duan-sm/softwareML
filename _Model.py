#!/usr/bin/env python
# coding=utf-8

"""
@Time: 2025/4/2 11:14 AM
@Author: Fatimah Ahmadi Godini
@Email: Ahmadi.ths@gmail.com
@File: _Model.py
@Software: Visual Studio Code

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
from tensorflow.python.keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v1 import Adam, SGD, RMSprop, Adadelta, Adagrad
from tensorflow.python.keras.callbacks import ModelCheckpoint
import vtk
import time


def initial(self):
    '''
    useless
    '''
    try:
        # Print the current page index 
        index = self.ui.tabWidget.currentIndex()
        print('index=', index)
        if index == 3:
            print('111')
            print('222')

            self.ui.Results3.setText('Label has been selected')
            for num in range(2, 5):
                print('num=', num)
                eval('self.ui.SelectFeature%d.clear()' % num)  # clear fig
                eval('self.ui.SelectFeature%d.addItems(columns[:-4])' % num)  # clear fig
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
                            eval('self.ui.SelectFeature%d.clear()' % num)  # clear fig
                            eval('self.ui.SelectFeature%d.addItems(columns[new_index])' % num)  # clear fig
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
    '''
    Since there are many parameters of the intelligent model, the parameters of the model can be
     quickly initialized through this function (the initial parameters are only for reference and
     are not necessarily applicable to the current data).
    '''
    if self.ui.ModelTypeSelect.currentIndex() == 0:
        self.ui.ANN_layers.setText('32')
        self.ui.ANN_dropout.setText('0.3')
        self.ui.ANN_activation.setCurrentIndex(0)
        self.ui.ANN_optimizer.setCurrentIndex(0)
        self.ui.ANN_lr.setText('0.005')
        self.ui.ANN_loss.setCurrentIndex(0)
        self.ui.ANN_epochs.setText('1000')
    elif self.ui.ModelTypeSelect.currentIndex() == 1:
        self.ui.LSTM_layers.setText('32')
        self.ui.LSTM_dropout.setText('0.3')
        self.ui.LSTM_optimizer.setCurrentIndex(0)
        self.ui.LSTM_lr.setText('0.005')
        self.ui.LSTM_loss.setCurrentIndex(0)
        self.ui.LSTM_epochs.setText('500')
    elif self.ui.ModelTypeSelect.currentIndex() == 2:
        self.ui.CNN_blocks.setText('32')
        self.ui.CNN_kernel_size.setText('3')
        self.ui.CNN_pool_size.setText('2')
        self.ui.CNN_strides.setText('1')
        self.ui.CNN_dropout.setText('0.3')
        self.ui.CNN_optimizer.setCurrentIndex(0)
        self.ui.CNN_lr.setText('0.005')
        self.ui.CNN_loss.setCurrentIndex(0)
        self.ui.CNN_epochs.setText('500')
    else:
        pass
    if self.ui.pinn.isChecked():
        self.ui.mu.setText('0.001')
        self.ui.physics_weight.setText('0.01')
    else:
        pass


def get_optimizer(optimizer_name, lr):
    '''
    Return to the optimizer based on the interface selection
    :param optimizer_name: Optimizer name:adam adadelta adagrad rmsprop sgd
    :param lr: Learning rate
    :return: optimizer
    '''
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
    '''
    According to the selection of the interface, select the loss value calculation method to return the loss value
    :param loss_name: Loss function
    :param y_true: label
    :param y_pre: Predicted value
    :return: Loss value
    '''
    loss_name = loss_name.lower()

    if loss_name == 'mse':
        return tf.keras.losses.MeanSquaredError()(y_true, y_pre)
    elif loss_name == 'mae':
        return tf.keras.losses.MeanAbsoluteError()(y_true, y_pre)
    elif loss_name == 'mape':
        return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pre)
    elif loss_name in ['msle', 'mean squared logarithmic']:
        return tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pre)
    elif loss_name in ['binary_crossentropy', 'bce']:
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pre)
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}")


def draw_LossFig(self, train_loss, valid_loss, name, pinn_loss=None):
    '''
    Plot the loss value curves for training, validation, and optional PINN loss.
    :param train_loss: List of training loss values per epoch
    :param valid_loss: List of validation loss values per epoch
    :param name: Filename suffix for saving the plot
    :param pinn_loss: (Optional) List of PINN loss values per epoch
    '''
    try:
        bwith = 3
        fontsize = 22
        self.ui.LossFig.figure.clear()  # Clear chart clear fig
        font_ = {
            'family': 'Times New Roman'
            , 'weight': 'normal'
            , 'color': 'k'
            , 'size': fontsize
        }
        ax1 = self.ui.LossFig.figure.add_subplot(1, 1, 1, label='loss figure')
        ax1.plot(np.arange(len(train_loss)), train_loss, 'b-', label='train loss', linewidth=bwith)
        ax1.plot(np.arange(len(valid_loss)), valid_loss, 'r-', label='valid loss', linewidth=bwith)

        if pinn_loss is not None:
            ax1.plot(np.arange(len(pinn_loss)), pinn_loss, 'g--', label='PINN Loss', linewidth=bwith)

        ax1.set_xlabel('Train Times', fontdict=font_)
        ax1.set_ylabel('Loss Value', fontdict=font_)
        ax1.set_title('Loss Visualization', fontsize=fontsize)
        for spine in ax1.spines.values():
            spine.set_linewidth(bwith)

        font1 = {
            'family': 'Times New Roman'
            , 'weight': 'normal'
            , 'size': fontsize
        }
        ax1.legend(loc="best", prop=font1, framealpha=0)
        x1_label = ax1.get_xticklabels()
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        # points_num = 10
        # print('|4' * 10)
        # points_num_gaps = len(train_loss)//points_num
        # new_xticks = np.arange(0,len(train_loss),points_num_gaps)
        # new_xticklabels = np.arange(0,len(train_loss),points_num_gaps)
        # print('|4' * 10)
        # ax1.set_xticks(new_xticks)
        # ax1.set_xticklabels(new_xticklabels, fontproperties='Times New Roman', fontsize=fontsize)
        ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
        self.ui.LossFig.figure.tight_layout()
        self.ui.Results3.setText('loss drawing completion')
        # ax1.show()
        self.ui.LossFig.figure.canvas.draw()
        print('**** LossFig ****')
        os.makedirs('./res/result', exist_ok=True)
        self.ui.LossFig.figure.savefig(f'./res/result/{name}_lossfig.png')
    except Exception as e:
        print('Error in draw_LossFig:', e)


def sequences_from_indices(array, indices_ds, start_index, end_index):
    '''
    Obtain sequence index
    :param array: dataset
    :param indices_ds: For each initial window position, generates indices of the window elements
    :param start_index: an integer that indicates the time index from which the time window is created. The default value is 0, which means that it starts with the first timestamp.
    :param end_index: an integer that indicates the time at which the index ends the creation time window. The default is None, which means until the end of the last timestamp.
    :return: dataset
    '''
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
        return_all_targets=False,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None):
    """
    Convert non-time series to time series data with targets for all time steps in each sequence.
    :param data: Represents x data, each of which is called a timestep.
    :param targets: indicates the y label. If you do not process labels and only process data, pass targets=None.
    :param sequence_length: The length of an output sequence, that is, how many timesteps there are.
    :param sequence_stride: The beginning of each sequence is separated by several timesteps.
    For stride s, output samples would start at index data[i], data[i + s], data[i + 2 * s], etc.
    :param sampling_rate: Sampling frequency of timestep in a sequence.
    For rate r, timesteps data[i], data[i + r], … data[i + sequence_length] are used for create a sample sequence.
    :param batch_size: Because tf.data.Dataset is returned, batch is set.
    :param return_all_targets: Boolean flag to decide whether to use targets for all time steps in each sequence (True) or only for the last step (False).
    :param shuffle: Boolean value: indicates whether to shuffle the generated data set (random rearrangement). The default is False.
    :param seed: Random number seed.
    :param start_index: an integer that indicates the time index from which the time window is created. The default value is 0, which means that it starts with the first timestamp.
    :param end_index: an integer that indicates the time at which the index ends the creation time window. The default is None, which means until the end of the last timestamp.
    :return: Time series data set.
    """
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

    # Determine the number of sequences
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

    # Create datasets for input data and indices
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
        # Modify targets to include all time steps
        if return_all_targets:
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
            target_ds = sequences_from_indices(
                targets, indices, start_index, end_index
            )

        else:
            indices = tf.data.Dataset.zip(
                (tf.data.Dataset.range(len(start_positions)), positions_ds)
            ).map(
                lambda i, positions: positions[i],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            target_ds = sequences_from_indices(
                targets, indices, start_index + sequence_length - 1, end_index + sequence_length - 1
            )

        dataset = tf.data.Dataset.zip((dataset, target_ds))

    # Apply prefetching and batching
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


def reverse_normalization(self, data, s_m, style):
    data = np.asarray(data)
    if style == 0:
        cols = ['p']
    elif style == 1:
        cols = ['NormalStress_0', 'NormalStress_1', 'NormalStress_2', 
                'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']
    
    if s_m == 'S':  # Standardization
        # Extract mean and var for the selected columns
        mean = self.standard_para.loc['mean', cols].to_numpy()
        var = self.standard_para.loc['var', cols].to_numpy()
        denorm_data = data * np.sqrt(var) + mean
    
    elif s_m == 'M':  # Max-Min
        max_ = self.maxmin_para.loc['max', cols].to_numpy()
        min_ = self.maxmin_para.loc['min', cols].to_numpy()
        denorm_data = data * (max_ - min_) + min_
    
    else:  # No normalization applied
        denorm_data = data.copy()
    
    return denorm_data


def compute_gradient_vtk(variable, x, y, z):
    points = vtk.vtkPoints()
    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    # Add dummy cells (necessary for unstructured grid)
    for i in range(len(x)):
        grid.InsertNextCell(vtk.VTK_VERTEX, 1, [i])

    variable_array = vtk.vtkDoubleArray()
    variable_array.SetName("variable")
    variable_array.SetNumberOfComponents(1)
    variable_array.SetNumberOfTuples(len(variable))

    for i in range(len(variable)):
        variable_array.SetValue(i, variable[i])

    grid.GetPointData().AddArray(variable_array)
    grid.GetPointData().SetScalars(variable_array)  # Ensure proper setting

    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(grid)
    gradient_filter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "variable")
    gradient_filter.SetResultArrayName("variableGradient")
    gradient_filter.Update()

    gradient_grid = gradient_filter.GetOutput()
    variable_gradient = gradient_grid.GetPointData().GetArray("variableGradient")

    if variable_gradient is None:
        raise ValueError("Gradient computation failed. 'variableGradient' array is missing.")

    d_dx = np.zeros(len(variable))
    d_dy = np.zeros(len(variable))
    d_dz = np.zeros(len(variable))

    for i in range(len(variable)):
        gradient = variable_gradient.GetTuple(i)
        d_dx[i], d_dy[i], d_dz[i] = gradient[0], gradient[1], gradient[2]

    return d_dx, d_dy, d_dz


def Navier_Stokes_Residual(spatial_gradu_data, predicted_pressure, spatial_data=None, mu=0.001):
    """
    Args:
    - x, y, z: Numpy arrays containing the spatial coordinates.
    - gradient_velocity: A numpy array with the first derivatives of the velocity components
      (u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z).
    - predicted_pressure: Predicted pressure field (numpy array).
    - mu: Dynamic viscosity of the fluid (default is 0.001).

    Returns:
    - pinn_loss: The total physics-informed loss (scalar).
    """
    if np.any(spatial_data):
        x = spatial_data[:, 0, 0]
        y = spatial_data[:, 0, 1]
        z = spatial_data[:, 0, 2]
        gradient_velocity = spatial_gradu_data[:, 0, :]
    else:
        x = spatial_gradu_data[:, 0, 0]
        y = spatial_gradu_data[:, 0, 1]
        z = spatial_gradu_data[:, 0, 2]
        gradient_velocity = spatial_gradu_data[:, 0, 4:]

    # Extract velocity gradients (these are already provided)
    u_x, v_x, w_x = gradient_velocity[:, 0], gradient_velocity[:, 1], gradient_velocity[:, 2]
    u_y, v_y, w_y = gradient_velocity[:, 3], gradient_velocity[:, 4], gradient_velocity[:, 5]
    u_z, v_z, w_z = gradient_velocity[:, 6], gradient_velocity[:, 7], gradient_velocity[:, 8]

    u_xx, v_xx, w_xx = gradient_velocity[:, 9], gradient_velocity[:, 10], gradient_velocity[:, 11]
    u_yy, v_yy, w_yy = gradient_velocity[:, 12], gradient_velocity[:, 13], gradient_velocity[:, 14]
    u_zz, v_zz, w_zz = gradient_velocity[:, 15], gradient_velocity[:, 16], gradient_velocity[:, 17]

    p = predicted_pressure[:, 0]

    # Compute the derivatives
    p_x, p_y, p_z = compute_gradient_vtk(p, x, y, z)

    # Calculate Navier-Stokes residuals
    ns_residual_x = mu * (u_xx + u_yy + u_zz) - p_x
    ns_residual_y = mu * (v_xx + v_yy + v_zz) - p_y
    ns_residual_z = mu * (w_xx + w_yy + w_zz) - p_z

    # Continuity equation residual (mass conservation)
    continuity_residual = u_x + v_y + w_z

    # Convert residuals to tensors for TensorFlow operations
    ns_residual_x = tf.convert_to_tensor(ns_residual_x, dtype=tf.float32)
    ns_residual_y = tf.convert_to_tensor(ns_residual_y, dtype=tf.float32)
    ns_residual_z = tf.convert_to_tensor(ns_residual_z, dtype=tf.float32)
    continuity_residual = tf.convert_to_tensor(continuity_residual, dtype=tf.float32)

    # physics-informed loss (sum of squares of residuals)
    momentum_loss = tf.reduce_mean(tf.square(ns_residual_x)) + tf.reduce_mean(
        tf.square(ns_residual_y)) + tf.reduce_mean(tf.square(ns_residual_z))
    continuity_loss = tf.reduce_mean(tf.square(continuity_residual))
    alpha = 1
    beta = 0
    pinn_loss = alpha * momentum_loss + beta * continuity_loss

    return pinn_loss


def Stress_Tensor_Residual(spatial_gradu_p_data, predicted_stress, spatial_data=None, mu=0.001):
    """
    Args:
    - x, y, z: Numpy arrays containing the spatial coordinates.
    - gradient_velocity: A numpy array with the first derivatives of the velocity components
      (u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z).
    - predicted_pressure: Predicted pressure field (numpy array).
    - mu: Dynamic viscosity of the fluid (default is 0.001).

    Returns:
    - pinn_loss: The total physics-informed loss (scalar).
    """

    if np.any(spatial_data):
        x = spatial_data[:, 0, 0]
        y = spatial_data[:, 0, 1]
        z = spatial_data[:, 0, 2]
        gradient_velocity = spatial_gradu_p_data[:, 0, :]
    else:
        x = spatial_gradu_p_data[:, 0, 0]
        y = spatial_gradu_p_data[:, 0, 1]
        z = spatial_gradu_p_data[:, 0, 2]
        gradient_velocity = spatial_gradu_p_data[:, 0, 4:]

    # Extract velocity gradients (these are already provided)
    u_x, v_x, w_x = gradient_velocity[:, 0], gradient_velocity[:, 1], gradient_velocity[:, 2]
    u_y, v_y, w_y = gradient_velocity[:, 3], gradient_velocity[:, 4], gradient_velocity[:, 5]
    u_z, v_z, w_z = gradient_velocity[:, 6], gradient_velocity[:, 7], gradient_velocity[:, 8]

    u_xx, v_xx, w_xx = gradient_velocity[:, 9], gradient_velocity[:, 10], gradient_velocity[:, 11]
    u_yy, v_yy, w_yy = gradient_velocity[:, 12], gradient_velocity[:, 13], gradient_velocity[:, 14]
    u_zz, v_zz, w_zz = gradient_velocity[:, 15], gradient_velocity[:, 16], gradient_velocity[:, 17]

    p = gradient_velocity[:, -1]

    # Stress tensor components
    sigma_xx = predicted_stress[:, 0]
    sigma_yy = predicted_stress[:, 1]
    sigma_zz = predicted_stress[:, 2]
    sigma_xy = predicted_stress[:, 3]
    sigma_xz = predicted_stress[:, 4]
    sigma_yz = predicted_stress[:, 5]

    # Compute the derivatives
    p_x, p_y, p_z = compute_gradient_vtk(p, x, y, z)

    sigma_xx_x = compute_gradient_vtk(sigma_xx, x, y, z)[0]
    sigma_yy_y = compute_gradient_vtk(sigma_yy, x, y, z)[1]
    sigma_zz_z = compute_gradient_vtk(sigma_zz, x, y, z)[2]
    sigma_xy_x, sigma_xy_y, sigma_xy_z = compute_gradient_vtk(sigma_xy, x, y, z)
    sigma_xz_x, sigma_xz_y, sigma_xz_z = compute_gradient_vtk(sigma_xz, x, y, z)
    sigma_yz_x, sigma_yz_y, sigma_yz_z = compute_gradient_vtk(sigma_yz, x, y, z)

    # Navier-Stokes residuals (simplified for steady-state, incompressible flow)
    residual_ns_x = -p_x + sigma_xx_x + sigma_xy_y + sigma_xz_z
    residual_ns_y = -p_y + sigma_yy_y + sigma_xy_x + sigma_yz_z
    residual_ns_z = -p_z + sigma_zz_z + sigma_xz_x + sigma_yz_y

    # Normal stress residuals
    residual_normal_x = sigma_xx + p - 2 * mu * u_x
    residual_normal_y = sigma_yy + p - 2 * mu * v_y
    residual_normal_z = sigma_zz + p - 2 * mu * w_z

    # Shear stress residuals
    residual_shear_xy = sigma_xy - mu * (u_y + v_x)
    residual_shear_xz = sigma_xz - mu * (u_z + w_x)
    residual_shear_yz = sigma_yz - mu * (v_z + w_y)

    # Convert to tensors
    tensors = [residual_ns_x, residual_ns_y, residual_ns_z,
               residual_normal_x, residual_normal_y, residual_normal_z,
               residual_shear_xy, residual_shear_xz, residual_shear_yz]
    tensors = [tf.convert_to_tensor(r, dtype=tf.float32) for r in tensors]

    # Compute loss
    alpha, beta, gamma = 0.5, 0.25, 0.25

    momentum_loss = sum(tf.reduce_mean(tf.square(r)) for r in tensors[:3])
    normal_stress_loss = sum(tf.reduce_mean(tf.square(r)) for r in tensors[3:6])
    shear_stress_loss = sum(tf.reduce_mean(tf.square(r)) for r in tensors[6:])

    pinn_loss = (alpha * momentum_loss + beta * normal_stress_loss + gamma * shear_stress_loss)

    return pinn_loss


def compute_max_ind(output, style, name):
    if style == 0:
        var_list = ['p']
    else:
        var_list = ['NormalStress_0', 'NormalStress_1', 'NormalStress_2', 
                    'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

    max_var_data = []
    comparison_data_list = []

    for var in var_list:
        # Find the maximum var and corresponding x, y, z coordinates
        max_actual_var_index = output[var].idxmax()
        max_actual_var_row = output.iloc[max_actual_var_index]
        
        max_var_data.append({
            'type': 'Actual ' + var,
            'Max_value': max_actual_var_row[var],
            'Points_0': max_actual_var_row['Points_0'],
            'Points_1': max_actual_var_row['Points_1'],
            'Points_2': max_actual_var_row['Points_2']
        })
        
        # Find the maximum predicted var and its corresponding x, y, z coordinates
        max_pred_var_index = output['Predicted_' + var].idxmax()
        max_pred_var_row = output.iloc[max_pred_var_index]

        max_var_data.append({
            'type': 'Predicted ' + var,
            'Max_value': max_pred_var_row['Predicted_' + var],
            'Points_0': max_pred_var_row['Points_0'],
            'Points_1': max_pred_var_row['Points_1'],
            'Points_2': max_pred_var_row['Points_2']
        })

        # Compute the absolute difference between actual and predicted var
        var_difference = abs(max_actual_var_row[var] - max_pred_var_row['Predicted_' + var])
        
        # Add the comparison data
        comparison_data = {
            'type': var + ' Difference',
            'differences': var_difference,
            'Points_0_actual': max_actual_var_row['Points_0'],
            'Points_1_actual': max_actual_var_row['Points_1'],
            'Points_2_actual': max_actual_var_row['Points_2'],
            'Points_0_pred': max_pred_var_row['Points_0'],
            'Points_1_pred': max_pred_var_row['Points_1'],
            'Points_2_pred': max_pred_var_row['Points_2']
        }

        comparison_data_list.append(comparison_data)

    max_var_df = pd.DataFrame(max_var_data)
    print(max_var_df)

    os.makedirs('./res/result', exist_ok=True)
    output_csv = f'./res/result/{name}_max_results.csv'
    max_var_df.to_csv(output_csv, mode='a', header=not pd.io.common.file_exists(output_csv), index=False)

    for comparison_data in comparison_data_list:
        print(comparison_data)
    comparison_df = pd.DataFrame(comparison_data_list)
    comparison_df.to_csv(output_csv, mode='a', header=not pd.io.common.file_exists(output_csv), index=False)
    
    return comparison_data_list


def MyBP(self, x_train, y_train, x_test, y_test
         , units=[100], dropout=0.2, activation='relu'
         , loss='mape', optimizer='adam', epochs=100
         , lr=0.01, batch_num=3000, style=1):
    '''
    A multi-layer BP neural network model is established.
    :param self: class
    :param x_train: Model input train data
    :param y_train: Model input train data label
    :param x_test: Model input test data
    :param y_test: Model input test data label
    :param units: Network structure and neurons
    :param dropout: dropout
    :param activation: Activation function
    :param loss: Loss function
    :param optimizer: optimizer
    :param epochs: Training times
    :param lr: Learning rate
    :param batch_num:Batch training quantity
    :param style: Pressure-0/stress-1
    :return: Trained model
    '''

    self.ui.LossFig.figure.clear()
    Dimen_index = self.ui.DimensionlessType.currentIndex()
    if Dimen_index == 0:
        s_m = 'S' # Standard
    else:
        s_m = 'M' # MaxMin

    if self.state != 0 and self.state != 1:
        dlgTitle = "Tips"
        strInfo = ("Please normalize or reduce dimension.")
        defaultBtn = QMessageBox.NoButton  # Default button
        result = QMessageBox.question(self, dlgTitle, strInfo,
                                      QMessageBox.Yes,
                                      defaultBtn)
        self.ui.tabWidget.setCurrentIndex(2)
        self.ui.ResultsText2.append('Please normalize or reduce dimension')
        return 0
    # if self.state == 1:
    #     NondimenBtn(self, y_test)
    print('*2' * 10)
    # Create a folder to save the model and results
    if not os.path.exists('./res/model'):
        os.makedirs('./res/model')
    if not os.path.exists('./res/history'):
        os.makedirs('./res/history')
    if not os.path.exists('./res/result'):
        os.makedirs('./res/result')
    # Split dataset for training and validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    if len(x_train) < batch_num:
        batch_size = len(x_train)
    else:
        batch_size = len(x_train) // batch_num
    net_ = str(units[0])
    print('*3' * 10)
    # Build model
    model = Sequential()
    print('x_train.shape', x_train.shape)
    model.add(Dense(units=units[0], input_shape=(None, x_train.shape[1])
                    , activation=activation, kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dropout(rate=dropout))  # Prevent overfitting
    for unit in range(1, len(units)):
        model.add(Dense(units=units[unit], activation=activation, kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dropout(rate=dropout))  # Prevent overfitting
        net_ = net_ + '_' + str(units[unit])
    if style == 0:
        model.add(Dense(units=1, activation='sigmoid'))
        net_ = net_ + '_' + str(1)
    else:
        model.add(Dense(units=3, activation='sigmoid'))
        net_ = net_ + '_' + str(3)
    model.summary() # Show the structure of model
    filepath = './res/model/BP+{}+{}+{}+{}+{}.h5'.format(net_, activation, dropout, lr,
                                                         epochs)  # 保存最优权重.units+dropout+lr+act+epochs
    if s_m == 'S':
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
    train_loss_bs = [] # record train loss for every batch
    train_loss_sum = [] # record train loss of sum
    valid_loss_sum = [] # record valid loss
    min_valid_loss = 10e8
    point_num = 1
    opt = get_optimizer(optimizer, lr)
    for i in range(epochs):
        print('-----')
        for bs in range(batch_size):
            with tf.GradientTape() as tape: # compute gradient
                predict = model(x_train[bs::batch_size])
                train_loss = get_loss(loss_name=loss, y_true=y_train[bs::batch_size], y_pre=predict)
            grads = tape.gradient(train_loss, model.trainable_variables)
            train_loss_bs.append(train_loss.numpy())
            opt.apply_gradients(zip(grads, model.trainable_variables))
        # test set
        train_loss_sum.append(np.mean(train_loss_bs))
        out = model(x_valid) # Validate the model using validation data sets
        valid_loss = get_loss(loss_name=loss, y_true=y_valid, y_pre=out)  # tf.reduce_mean(tf.losses.mape(y_valid, out))
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
    # start = t.perf_counter()
    model_saved = load_model(filepath) # Save model
    y_pred = model_saved.predict(x_test)
    # end = t.perf_counter()
    y_pred = np.array(y_pred).flatten()
    print(y_pred)
    # y_pred_it = inverse_transform(y_pred, np.max(data['V']), np.min(data['V']))
    # y_test_it = inverse_transform(y_test, np.max(data['V']), np.min(data['V']))

    bp_mse = metrics.mean_squared_error(y_test, y_pred)
    bp_mae = metrics.mean_absolute_error(y_test, y_pred)
    bp_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # bp_mre = relative_error(y_test, y_pred)
    # print('*8' * 10)
    df = pd.DataFrame()
    print('test result:')
    print('bp_mse=%.3f, bp_mae=%.3f, bp_rmse=%.3f' % (bp_mse, bp_mae, bp_rmse))
    df['V_t'] = y_test
    method = self.ui.DimensionlessType.currentIndex()
    if method == 0:
        y_pred = y_pred * ((self.data_max_ - self.data_min_)) + self.data_min_
    else:
        y_pred = y_pred * self.var_ + self.mean_
    df['V_p'] = y_pred
    df.to_csv('./res/result/BP_result.csv', index=False)
    # print('*9' * 10)
    return model


def MyLSTM(self, dataset_train, dataset_valid, dataset_test
           , units=[100], dropout=0.2
           , loss='mape', optimizer='adam', epochs=100
           , lr=0.01, batch_num=3000, style=1):
    """
    The overall process is the same as MyBP function.
    The LSTM neural network model is established.
    :param units:Number of hidden layer neurons, default 100.
    :param activations:Activate function, default 'relu'.
    :param dropout:Random inactivation ratio, default 0.2.
    :param lr:Learning rate, default is 0.01.
    :return:Model.模型。
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
    print('myLSTM 1 ' * 10)
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
    if len(units) <= 1:
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
        for unit in range(1, len(units) - 1):
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
        print('epoch=%d' % i)
        for bs in range(batch_num):
            print('bs=', bs)
            print('batch_num=', batch_num)
            for batch in dataset_train.take(bs + 1):
                inputs, targets = batch
            with tf.GradientTape() as tape:
                predict = model(inputs)
                train_loss = get_loss(loss_name=loss, y_true=targets, y_pre=predict)
            grads = tape.gradient(train_loss, model.trainable_variables)
            train_loss_bs.append(train_loss.numpy())
            opt.apply_gradients(zip(grads, model.trainable_variables))
        print('start compute')
        # test set
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
        './res/history/LSTM+{}+{}+{}+{}.csv'.format(net_, dropout, lr, epochs), index=False)

    print('net_=', net_)
    print('myLSTM 5 ' * 10)
    # print('*7' * 10)
    # Import the model for testing
    # start = t.perf_counter()
    for batch in dataset_test.take(1):
        inputs, targets = batch
    model_saved = load_model(filepath)
    y_pred = model_saved.predict(inputs)
    # end = t.perf_counter()
    y_pred = np.array(y_pred).flatten()
    print(y_pred)

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


def My_PINN_MLP(self, x_train, y_train, x_valid, y_valid, x_test, y_test,
               units=[32, 32], dropout=0.1, activation='tanh',
               loss='mse', optimizer='adam', epochs=50,
               lr=0.001, batch_num=2048, style=0, s_m=None,
               mu=0.001, physics_weight=0.01, pinn=False):
    """
    Feedforward neural network with optional Physics-Informed Neural Network (PINN) capability.

    Parameters:
    - train_data, valid_data, test_data: DataFrames containing features and targets.
    - units: List[int], number of neurons per hidden layer.
    - dropout: float, dropout rate.
    - activation: str, activation function.
    - loss: str, loss function name.
    - optimizer: str, optimizer name.
    - epochs: int, number of training epochs.
    - lr: float, learning rate.
    - batch_num: int, batch size.
    - style: int, 0 for pressure prediction, 1 for stress prediction.
    - mu: float, dynamic viscosity for PINN loss.
    - physics_weight: float, weighting factor for the physics-based loss term.
    - pinn: bool, enable physics-informed training.

    Returns:
    - Trained model.
    """

    model_name = 'PINN_MLP' if pinn else 'MLP'
    print(f"\n==== RUNNING {model_name} FUNCTION ====\n")
    self.ui.LossFig.figure.clear()
    
    if len(x_train) < batch_num:
        batch_size = len(x_train)
    else:
        batch_size = len(x_train) // batch_num

    inputs = x_train[:, :4]
    grad_u = x_train[:, 4:]
    net_ = ''
    print('========== Model Definition ==========')

    # Build model
    model = Sequential()
    # Input layer
    model.add(Input(shape=(inputs.shape[1],)))

    # Hidden layers with Dropout
    for unit in units:
        model.add(
            Dense(units=unit, activation=activation, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(rate=dropout))
        net_ += f"_{unit}"

    # Output layer
    output_units = 1 if style == 0 else 6
    model.add(Dense(units=output_units, activation='sigmoid'))  # 'linear' for Regression and Standardized (mean 0, std 1), 'sigmoid' for Normalized to [0,1] range and Classification (2 classes), 'softmax' for Classification (multi-class)
    net_ += f"_{output_units}"

    model.summary()

    # Compile model
    opt = get_optimizer(optimizer, lr)
    # model.compile(optimizer=opt, loss=loss)

    name = f"{model_name}+{net_}+{activation}+{dropout}+{lr}+{batch_size}+{epochs}"
    modelpath = f'./res/model/{name}.h5'

    if s_m == 'S':
        self.standard_para.to_csv(f'./res/model/{name}+standard_para.csv', index=True)
    elif s_m == 'M':
        self.maxmin_para.to_csv(f'./res/model/{name}+maxmin_para.csv', index=True)

    early_stop_counter = 0
    train_loss_sum = []
    valid_loss_sum = []
    pinn_loss_sum = []
    min_valid_loss = float("inf")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss_bs = []
        pinn_loss_bs = []
        
        for bs in range(batch_size):
            print(f"Epoch {epoch}/{epochs}")
            print(f"Batch {bs}/{batch_size}")

            with tf.GradientTape() as tape:
                x_batch = inputs[bs::batch_size]
                grad_u_batch = grad_u[bs::batch_size]
                y_batch = y_train[bs::batch_size]

                predict = model(x_batch)
                train_loss = get_loss(loss_name=loss, y_true=y_batch, y_pre=predict)
                print('train loss: ', train_loss)

                if pinn:
                    predict_denorm = reverse_normalization(self, predict, s_m, style)
                    pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                        grad_u_batch[:, np.newaxis, :], predict_denorm, spatial_data=x_batch[:, np.newaxis, :], mu=mu)
                    
                    pinn_loss_value = pinn_loss.numpy() if tf.is_tensor(pinn_loss) else pinn_loss
                    pinn_loss_bs.append(pinn_loss_value)
                    print('Physics loss: ', pinn_loss)
                    
                    total_loss = train_loss + physics_weight * pinn_loss
                else:
                    total_loss = train_loss
                print('Total loss: ', total_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_bs.append(total_loss.numpy())

        train_loss_sum.append(np.mean(train_loss_bs))
        if pinn and pinn_loss_bs:
            pinn_loss_sum.append(np.mean(pinn_loss_bs))

        # Validation
        val_inputs = x_valid[:, :4]
        predictions = model(val_inputs)
        valid_loss = get_loss(loss_name=loss, y_true=y_valid, y_pre=predictions)
        print('valid_loss: ', valid_loss)
        valid_loss_sum.append(valid_loss.numpy())

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            early_stop_counter = 0
            model.save(modelpath)
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss_sum[-1]:.5f} | Valid Loss: {valid_loss.numpy():.5f}")

        if early_stop_counter > 30:
            print("Early stopping triggered.")
            break

        print('*' * 50)

    draw_LossFig(self, np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)
    
    history = pd.DataFrame({
        'train_loss': train_loss_sum,
        'valid_loss': valid_loss_sum,
        'pinn_loss': pinn_loss_sum if pinn else [0.0] * len(train_loss_sum)
        })
    history.to_csv(f'./res/history/{name}.csv', index=False)

    if pinn and pinn_loss_sum:
        pinn_mse = metrics.MeanSquaredError()([0.0] * len(pinn_loss_sum), pinn_loss_sum).numpy()
        print("PINN Loss MSE:", pinn_mse)

    """
    # Load model & history file
    modelpath = 'res/model/MLP+32_1+relu+0.3+0.001+50.h5'

    filename = os.path.splitext(os.path.basename(modelpath))[0]
    model_name, net_, activation, dropout, lr, batch_size, epochs = filename.split('+')
    name = f"{model_name}+{net_}+{activation}+{dropout}+{lr}+{batch_size}+{epochs}"

    df = pd.read_csv('res/history/MLP+32_1+relu+0.3+0.001+50.csv')

    train_loss_sum = np.array(df['train_loss'])
    valid_loss_sum = np.array(df['valid_loss'])

    draw_LossFig(train_loss_sum, valid_loss_sum, name)
    """
    
    end_time = time.time()
    train_time = end_time - start_time

    train_characteristics = {
        'Model': name,
        'Train_Time_sec': round(train_time, 2),
        'Final_Train_Loss': train_loss_sum[-1],
        'Final_Valid_Loss': valid_loss_sum[-1],
        'Epochs_Run': len(train_loss_sum)
    }
    pd.DataFrame([train_characteristics]).to_csv(f'./res/history/{name}_training_summary.csv', index=False)

    # Evaluation
    test_inputs = x_test[:, :4]
    model_saved = load_model(modelpath)
    y_predict = model_saved.predict(test_inputs)
    y_pred = reverse_normalization(self, y_predict, s_m, style)

    print(name)
    if style == 0:
        y_pred_flat = y_pred.flatten()
        y_test_flat = y_test.flatten()

        bp_mse = metrics.MeanSquaredError()(y_test_flat, y_pred_flat)
        bp_mae = metrics.MeanAbsoluteError()(y_test_flat, y_pred_flat)
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
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame({
            'V_t': y_test_flat,
            'V_p': y_pred_flat
        })
        df.to_csv(f'./res/result/{name}_result.csv', index=False)
        output_data = np.hstack((
            test_inputs,
            y_test_flat.reshape(-1, 1),
            y_pred_flat.reshape(-1, 1)
        ))

        output_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude', 'p', 'Predicted_p']


    else:
        metrics_data = []
        for i in range(y_test.shape[1]):
            y_true_col = y_test[:, i]
            y_pred_col = y_pred[:, i]

            mse = metrics.MeanSquaredError()(y_true_col, y_pred_col)
            mae = metrics.MeanAbsoluteError()(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)

            print(f'Test Results for Component {i}:')
            print(f'  MSE = {mse:.5f}')
            print(f'  MAE = {mae:.5f}')
            print(f'  RMSE = {rmse:.5f}')
            
            metrics_data.append({
                'Model': name,
                'Component': f'Stress_{i}',
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame()
        for i in range(y_test.shape[1]):
            df[f'V_t_{i}'] = y_test[:, i]
            df[f'V_p_{i}'] = y_pred[:, i]
        df.to_csv(f'./res/result/{name}_result.csv', index=False)

        output_data = np.hstack((
            test_inputs,
            y_test,
            y_pred
        ))

        output_columns = [
            'Points_0', 'Points_1', 'Points_2', 'Points_Magnitude',
            'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
            'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
            'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
            'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2'
        ]

    output_df = pd.DataFrame(output_data, columns=output_columns)
    output_df.to_csv(f'./res/result/{name}_prediction.csv', index=False)

    comparison_pressure = compute_max_ind(output_df, style, name)

    return model


def My_PINN_LSTM(self, dataset_train, dataset_valid, dataset_test,
                 units=[32, 32], dropout=0.1,
                 loss='mse', optimizer='adam', epochs=50,
                 lr=0.001, batch_num =464, style=0, s_m=None,
                 mu=0.001, physics_weight=0.01, pinn=False):
    """
    LSTM model with optional physics-informed Neural Network (PINN).
    Parameters:
    - dataset_train: ndarray, Training dataset containing input features and target values.
    - dataset_valid: ndarray, Validation dataset for monitoring model performance during training.
    - dataset_test: ndarray, Test dataset used to evaluate the model after training.
    - units: List[int], number of neurons per hidden layer.
    - dropout: float, Dropout rate to prevent overfitting.
    - loss: str, Loss function for optimization.
    - optimizer: str, Optimization algorithm.
    - epochs: int, Number of training iterations.
    - lr: float, Learning rate for the optimizer.
    - style: int, 0 for pressure prediction, 1 for stress prediction.
    - mu: float, Dynamic viscosity, used in physics-informed loss calculations.
    - physics_weight: float, Weighting factor for the physics-based loss term.
    - pinn: bool, If True, includes physics-informed loss constraints.
    Returns:
    - Trained LSTM model.
    """

    # Model name
    model_name = 'PINN_LSTM' if pinn else 'LSTM'
    print(f"\n==== RUNNING {model_name} FUNCTION ====\n")
    self.ui.LossFig.figure.clear()
    
    for batch in dataset_train.take(1):
        sample_inputs, _ = batch
        inputs = sample_inputs[:, :, :4]

    net_ = str(units[0])
    print('========== Model Definition ==========')
    model = Sequential()
    for idx, unit in enumerate(units):
        return_seq = idx < len(units) - 1

        if idx == 0:
            model.add(LSTM(units=unit,
                           input_shape=(inputs.shape[1], inputs.shape[2]),
                           dropout=dropout, return_sequences=return_seq,
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
        else:
            model.add(LSTM(units=unit,
                           dropout=dropout, return_sequences=return_seq,
                           kernel_initializer='random_uniform', bias_initializer='zeros'))

        if idx > 0:
            net_ += f"_{unit}"

    # Output layer
    output_units = 1 if style == 0 else 6
    model.add(Dense(units=output_units, activation='sigmoid'))
    net_ += f"_{output_units}"

    model.summary()

    # Compile model
    opt = get_optimizer(optimizer, lr)
    # model.compile(optimizer=optimizer, loss=loss)

    name = f"{model_name}+{net_}+{dropout}+{lr}+{batch_num}+{epochs}"
    modelpath = f'./res/model/{name}.h5'
    
    if s_m == 'S':
        self.standard_para.to_csv(f'./res/model/{name}+standard_para.csv', index=True)
    elif s_m == 'M':
        self.maxmin_para.to_csv(f'./res/model/{name}+maxmin_para.csv', index=True)
    
    early_stop_counter = 0
    train_loss_sum = []
    valid_loss_sum = []
    pinn_loss_sum = []
    min_valid_loss = float("inf")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss_bs = []
        pinn_loss_bs = []

        for bs in range(batch_num):
            print(f'Epoch {epoch}/{epochs}')
            print(f"Batch {bs}/{batch_num}")
            for batch in dataset_train.take(bs + 1):
                inputs_batch, targets = batch
                inputs = inputs_batch[:, :, :4]
                grad_u = inputs_batch[:, :, 4:]

            with tf.GradientTape() as tape:
                predict = model(inputs)

                train_loss = get_loss(loss_name=loss, y_true=targets, y_pre=predict)
                print('train loss: ', train_loss)

                if pinn:
                    predict_denorm = reverse_normalization(self, predict, s_m, style)
                    pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                        grad_u, predict_denorm, spatial_data=inputs, mu=mu)
                        
                    pinn_loss_value = pinn_loss.numpy() if tf.is_tensor(pinn_loss) else pinn_loss
                    pinn_loss_bs.append(pinn_loss_value)    
                    print('Physics loss: ', pinn_loss)
                    
                    total_loss = train_loss + physics_weight * pinn_loss
                else:
                    total_loss = train_loss
                print('Total loss: ', total_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_bs.append(total_loss.numpy())

        train_loss_sum.append(np.mean(train_loss_bs))
        if pinn and pinn_loss_bs:
            pinn_loss_sum.append(np.mean(pinn_loss_bs))

        # Validation
        for batch in dataset_valid.take(1):
            inputs, targets = batch
            val_inputs = inputs[:, :, :4]
        predictions = model(val_inputs)
        valid_loss = get_loss(loss_name=loss, y_true=targets, y_pre=predictions).numpy()
        print('valid_loss: ', valid_loss)
        valid_loss_sum.append(valid_loss)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            early_stop_counter = 0
            model.save(modelpath)
        else:
            early_stop_counter += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss_sum[-1]:.5f} | Valid Loss: {valid_loss:.5f}")

        if early_stop_counter > 30:
            print("Early stopping triggered.")
            break
        print('*' * 50)
    
    draw_LossFig(self, np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)

    history = pd.DataFrame({
        'train_loss': train_loss_sum,
        'valid_loss': valid_loss_sum,
        'pinn_loss': pinn_loss_sum if pinn else [0.0] * len(train_loss_sum)
        })
    history.to_csv(f'./res/history/{name}.csv', index=False)

    if pinn and pinn_loss_sum:
        pinn_mse = metrics.MeanSquaredError()([0.0] * len(pinn_loss_sum), pinn_loss_sum).numpy()
        print("PINN Loss MSE:", pinn_mse)

    """
    # Load model & history file
    modelpath = './res/model/LSTM+32_64_1+0.1+0.001+100.h5'

    filename = os.path.splitext(os.path.basename(modelpath))[0]
    model_name, net_, dropout, lr, batch_num, epochs = filename.split('+')
    name = f"{model_name}+{net_}+{dropout}+{lr}+{batch_num}+{epochs}"

    df = pd.read_csv('./res/history/LSTM+32_64_1+0.1+0.001+100.csv')

    train_loss_sum = np.array(df['train_loss'])
    valid_loss_sum = np.array(df['valid_loss'])

    draw_LossFig(train_loss_sum, valid_loss_sum, name)
    """

    end_time = time.time()
    train_time = end_time - start_time

    train_characteristics = {
        'Model': name,
        'Train_Time_sec': round(train_time, 2),
        'Final_Train_Loss': train_loss_sum[-1],
        'Final_Valid_Loss': valid_loss_sum[-1],
        'Epochs_Run': len(train_loss_sum)
    }
    pd.DataFrame([train_characteristics]).to_csv(f'./res/history/{name}_training_summary.csv', index=False)

    # Evaluation
    for batch in dataset_test.take(1):
        inputs, targets = batch
        test_inputs = inputs[:, :, :4]
    model_saved = load_model(modelpath)
    y_predict = model_saved.predict(test_inputs)
    y_pred = reverse_normalization(self, y_predict, s_m, style)

    print(name)
    if style == 0:
        y_pred_flat = y_pred.flatten()
        y_test_flat = targets.numpy().flatten()

        bp_mse = metrics.MeanSquaredError()(y_test_flat, y_pred_flat)
        bp_mae = metrics.MeanAbsoluteError()(y_test_flat, y_pred_flat)
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
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame({
            'V_t': y_test_flat,
            'V_p': y_pred_flat
        })
        df.to_csv(f'./res/result/{name}_result.csv', index=False)
        output_data = np.hstack((
            test_inputs[:, -1, :].numpy(),
            y_test_flat[:, None],
            y_pred_flat[:, None]
        ))

        output_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude', 'p', 'Predicted_p']

    else:
        y_test = targets.numpy()
        metrics_data = []
        for i in range(y_test.shape[1]):
            y_true_col = y_test[:, i]
            y_pred_col = y_pred[:, i]

            mse = metrics.MeanSquaredError()(y_true_col, y_pred_col)
            mae = metrics.MeanAbsoluteError()(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)

            print(f'Test Results for Component {i}:')
            print(f'  MSE = {mse:.5f}')
            print(f'  MAE = {mae:.5f}')
            print(f'  RMSE = {rmse:.5f}')
            
            metrics_data.append({
                'Model': name,
                'Component': f'Stress_{i}',
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame()
        for i in range(y_test.shape[1]):
            df[f'V_t_{i}'] = y_test[:, i]
            df[f'V_p_{i}'] = y_pred[:, i]
        df.to_csv(f'./res/result/{name}_result.csv', index=False)

        output_data = np.hstack((
            test_inputs[:, -1, :].numpy(),
            y_test,
            y_pred
        ))

        output_columns = [
            'Points_0', 'Points_1', 'Points_2', 'Points_Magnitude',
            'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
            'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
            'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
            'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2'
        ]

    output_df = pd.DataFrame(output_data, columns=output_columns)
    output_df.to_csv(f'./res/result/{name}_prediction.csv', index=False)

    comparison_pressure = compute_max_ind(output_df, style, name)

    return model


def My_PINN_CNN(self, dataset_train, dataset_valid, dataset_test,
                filters=[32], dropout=0.3,
                kernel_size=3, pool_size=2, strides=1,
                loss='mse', optimizer='adam', epochs=100,
                lr=0.001, batch_num=464, style=0, s_m=None,
                mu=0.001, physics_weight=0.01, pinn=False):
    """
    Convolutional Neural Network (CNN) model with optional physics-informed Neural Network (PINN).
    Parameters:
    - dataset_train: ndarray, Training dataset containing input features and target values.
    - dataset_valid: ndarray, Validation dataset for monitoring model performance during training.
    - dataset_test: ndarray, Test dataset used to evaluate the model after training.
    - filters: list of int, Defines the number of filters in each convolutional block.
    - dropout: float, Dropout rate to prevent overfitting.
    - kernel_size: int, Size of the convolutional kernel.
    - pool_size: int, Pool size for max pooling layers.
    - strides: int, Stride value for convolutional block.
    - loss: str, Loss function for optimization.
    - optimizer: str, Optimization algorithm.
    - epochs: int, Number of training iterations.
    - lr: float, Learning rate for the optimizer.
    - style: int, 0 for pressure prediction, 1 for stress prediction.
    - mu: float, Dynamic viscosity, used in physics-informed loss calculations.
    - physics_weight: float, Weighting factor for the physics-based loss term.
    - pinn: bool, If True, includes physics-informed loss constraints.
    Returns:
    - Trained CNN model.
    """

    # Model name
    model_name = 'PINN_CNN' if pinn else 'CNN'
    print(f"\n==== RUNNING {model_name} FUNCTION ====\n")
    self.ui.LossFig.figure.clear()
    
    for batch in dataset_train.take(1):
        print("Training batch shape:", batch[0].shape, batch[1].shape)
        sample_inputs, _ = batch
        sample_inputs = sample_inputs[:, :, :4]
    net_ = str(filters[0])

    print('========== Model Definition ==========')
    model = Sequential()
    for idx, filter_count in enumerate(filters):
        if idx == 0:
            model.add(Conv1D(filters=filter_count, kernel_size=kernel_size, strides=strides,
                             activation='relu', padding='same', input_shape=(sample_inputs.shape[1], sample_inputs.shape[2])))
        else:
            model.add(Conv1D(filters=filter_count, kernel_size=kernel_size, strides=strides,
                             activation='relu', padding='same'))

        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(dropout))

        if idx > 0:
            net_ += f"_{filter_count}"

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer(s)
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))

    # Output layer
    output_units = 1 if style == 0 else 6
    model.add(Dense(units=output_units))
    net_ += f"_{output_units}_{kernel_size}_{pool_size}_{strides}"

    model.summary()

    # Compile model
    opt = get_optimizer(optimizer, lr)
    # model.compile(optimizer=opt, loss=loss)

    name = f"{model_name}+{net_}+{dropout}+{lr}+{batch_num}+{epochs}"
    print('Model name:', name)
    modelpath = f'./res/model/{name}.h5'
    
    if s_m == 'S':
        self.standard_para.to_csv(f'./res/model/{name}+standard_para.csv', index=True)
    elif s_m == 'M':
        self.maxmin_para.to_csv(f'./res/model/{name}+maxmin_para.csv', index=True)
    
    early_stop_counter = 0
    train_loss_sum = []
    valid_loss_sum = []
    pinn_loss_sum = []
    min_valid_loss = float("inf")
    print("Batch number:", batch_num)

    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss_bs = []
        pinn_loss_bs = []
        
        for bs in range(batch_num):
            print(f'Epoch {epoch}/{epochs}')
            print(f"Batch {bs}/{batch_num}")

            for batch in dataset_train.take(bs + 1):
                inputs_batch, targets = batch
                inputs = inputs_batch[:, :, :4]
                grad_u = inputs_batch[:, :, 4:]

            with tf.GradientTape() as tape:
                predict = model(inputs)

                train_loss = get_loss(loss_name=loss, y_true=targets, y_pre=predict)
                print('train loss: ', train_loss)

                if pinn:
                    predict_denorm = reverse_normalization(self, predict, s_m, style)
                    pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                        grad_u, predict_denorm, spatial_data=inputs, mu=mu)
                    
                    pinn_loss_value = pinn_loss.numpy() if tf.is_tensor(pinn_loss) else pinn_loss
                    pinn_loss_bs.append(pinn_loss_value)
                    print('Physics loss: ', pinn_loss)
                    
                    total_loss = train_loss + physics_weight * pinn_loss
                else:
                    total_loss = train_loss
                    
                print('Total loss: ', total_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_bs.append(total_loss.numpy())

        train_loss_sum.append(np.mean(train_loss_bs))
        if pinn and pinn_loss_bs:
            pinn_loss_sum.append(np.mean(pinn_loss_bs))

        # Validation
        for batch in dataset_valid.take(1):
            inputs, targets = batch
            val_inputs = inputs[:, :, :4]
        predictions = model(val_inputs)
        valid_loss = get_loss(loss_name=loss, y_true=targets, y_pre=predictions).numpy()
        print('valid_loss: ', valid_loss)
        valid_loss_sum.append(valid_loss)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            early_stop_counter = 0
            model.save(modelpath)
        else:
            early_stop_counter += 1

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss_sum[-1]:.5f} | Valid Loss: {valid_loss:.5f}")

        if early_stop_counter > 30:
            print("Early stopping triggered.")
            break
        print('*' * 50)
        
    draw_LossFig(self, np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)
    
    history = pd.DataFrame({
        'train_loss': train_loss_sum,
        'valid_loss': valid_loss_sum,
        'pinn_loss': pinn_loss_sum if pinn else [0.0]*len(train_loss_sum)})
    history.to_csv(f'./res/history/{name}.csv', index=False)

    if pinn and pinn_loss_sum:
        pinn_mse = metrics.MeanSquaredError()([0.0] * len(pinn_loss_sum), pinn_loss_sum).numpy()
        print("PINN Loss MSE:", pinn_mse)
    
    """
    # Load model & history file
    modelpath = './res/model/PINN_CNN+32_1_64_1+0.3+0.001+100.h5'

    filename = os.path.splitext(os.path.basename(modelpath))[0]
    model_name, net_, dropout, lr, batch_num, epochs = filename.split('+')
    name = f"{model_name}+{net_}+{dropout}+{lr}+{batch_num}+{epochs}"

    df = pd.read_csv('./res/history/PINN_CNN+32_1_64_1+0.3+0.001+100.csv')

    train_loss_sum = np.array(df['train_loss'])
    valid_loss_sum = np.array(df['valid_loss'])

    draw_LossFig(train_loss_sum, valid_loss_sum, name)
    """

    end_time = time.time()
    train_time = end_time - start_time

    train_characteristics = {
        'Model': name,
        'Train_Time_sec': round(train_time, 2),
        'Final_Train_Loss': train_loss_sum[-1],
        'Final_Valid_Loss': valid_loss_sum[-1],
        'Epochs_Run': len(train_loss_sum)
    }
    pd.DataFrame([train_characteristics]).to_csv(f'./res/history/{name}_training_summary.csv', index=False)

    # Evaluation
    for batch in dataset_test.take(1):
        inputs, targets = batch
        test_inputs = inputs[:, :, :4]

    model_saved = load_model(modelpath)
    y_predict = model_saved.predict(test_inputs)
    y_pred = reverse_normalization(self, y_predict, s_m, style)

    print(name)
    if style == 0:
        y_pred_flat = y_pred.flatten()
        y_test_flat = targets.numpy().flatten()

        bp_mse = metrics.MeanSquaredError()(y_test_flat, y_pred_flat)
        bp_mae = metrics.MeanAbsoluteError()(y_test_flat, y_pred_flat)
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
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame({
            'V_t': y_test_flat,
            'V_p': y_pred_flat
        })
        df.to_csv(f'./res/result/{name}_result.csv', index=False)
        output_data = np.hstack((
            test_inputs[:, -1, :].numpy(),
            y_test_flat[:, None],
            y_pred_flat[:, None]
        ))

        output_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude', 'p', 'Predicted_p']

    else:
        y_test = targets.numpy()
        metrics_data = []
        for i in range(y_test.shape[1]):
            y_true_col = y_test[:, i]
            y_pred_col = y_pred[:, i]

            mse = metrics.MeanSquaredError()(y_true_col, y_pred_col)
            mae = metrics.MeanAbsoluteError()(y_true_col, y_pred_col)
            rmse = np.sqrt(mse)

            print(f'Test Results for Component {i}:')
            print(f'  MSE = {mse:.5f}')
            print(f'  MAE = {mae:.5f}')
            print(f'  RMSE = {rmse:.5f}')
            
            metrics_data.append({
                'Model': name,
                'Component': f'Stress_{i}',
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'./res/result/{name}_metrics.csv', index=False)

        df = pd.DataFrame()
        for i in range(y_test.shape[1]):
            df[f'V_t_{i}'] = y_test[:, i]
            df[f'V_p_{i}'] = y_pred[:, i]
        df.to_csv(f'./res/result/{name}_result.csv', index=False)

        output_data = np.hstack((
            test_inputs[:, -1, :].numpy(),
            y_test,
            y_pred
        ))

        output_columns = [
            'Points_0', 'Points_1', 'Points_2', 'Points_Magnitude',
            'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
            'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
            'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
            'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2'
        ]

    output_df = pd.DataFrame(output_data, columns=output_columns)
    output_df.to_csv(f'./res/result/{name}_prediction.csv', index=False)

    comparison_pressure = compute_max_ind(output_df, style, name)

    return model


def Compute(self):
    '''
    The data is divided into validation set and training set by 7:3.
    The model is trained and verified according to the set parameters.
    '''
    activation_dict = {
        0: 'relu'
        , 1: 'tanh'
        , 2: 'sigmoid'
        , 3: 'softmax'
    }
    optimizer_dict = {
        0: 'adam'
        , 1: 'adadelta'
        , 2: 'adagrad'
        , 3: 'rmsprop'
        , 4: 'sgd'
    }
    loss_dict = {
        0: 'mse'
        , 1: 'mae'
        , 2: 'mape'
        , 3: 'mean squard logarithmic'
        , 4: 'binary crossentropy'
    }

    index = self.ui.ModelTypeSelect.currentIndex()
    if index == 0:
        try:
            # Read interface parameters
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
            if units[-1] == 3:
                units = units[:-1]
            activation = activation_dict[activation_index]
            optimizer = optimizer_dict[optimizer_index]
            loss = loss_dict[loss_index]

            print('units, dropout, activation, optimizer, lr, loss, epochs')
            print(units, dropout, activation, optimizer, lr, loss, epochs)
        except:
            self.ui.Results3.setText('Please input the correct model parameters')
            return 0
    elif index == 1:
        try:
            # Read interface parameters
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
            if units[-1] == 3:
                units = units[:-1]

            optimizer = optimizer_dict[optimizer_index]
            loss = loss_dict[loss_index]

            print('units, dropout, optimizer, lr, loss, epochs')
            print(units, dropout, optimizer, lr, loss, epochs)
        except:
            self.ui.Results3.setText('Please input the correct model parameters')
            return 0
    elif index == 2:
        try:
            # Read interface parameters
            filters = []
            blocks = self.ui.CNN_blocks.text()
            kernel_size = int(self.ui.CNN_kernel_size.text())
            pool_size = int(self.ui.CNN_pool_size.text())
            strides = int(self.ui.CNN_strides.text())
            dropout = float(self.ui.CNN_dropout.text())
            optimizer_index = self.ui.CNN_optimizer.currentIndex()
            lr = float(self.ui.CNN_lr.text())
            loss_index = self.ui.CNN_loss.currentIndex()
            epochs = int(self.ui.CNN_epochs.text())
            pre_filters = blocks.split('-')
            for i in range(len(pre_filters)):
                filters.append(int(pre_filters[i]))
            self.ui.Results3.setText('Parameter input is correct, the model is working')
            filters = np.array(filters)

            optimizer = optimizer_dict[optimizer_index]
            loss = loss_dict[loss_index]

            print('filters, dropout, optimizer, lr, loss, epochs')
            print(filters, dropout, optimizer, lr, loss, epochs)
        except:
            self.ui.Results3.setText('Please input the correct model parameters')
            return 0
    else:
        self.ui.Results3.setText('Something is wrong ???')

    try:
        os.makedirs('./res/data_process', exist_ok=True)
        os.makedirs('./res/model', exist_ok=True)
        os.makedirs('./res/history', exist_ok=True)
        os.makedirs('./res/result', exist_ok=True)

        style = self.ui.Style1.currentIndex()
        all_data = self.workbook
        Dimen_index = self.ui.DimensionlessType.currentIndex()
        if Dimen_index == 0:
            s_m = 'S'
        elif Dimen_index == 1:
            s_m = 'M'

        if self.state not in (0, 1):
            dlgTitle = "Confirmation"
            strInfo = ("Are you sure you want to continue without normalization?")
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
            if result == QMessageBox.Yes:
                s_m = None   # continue without normalization
            else:
                self.ui.tabWidget.setCurrentIndex(2)  # go to normalization tab
                self.ui.ResultsText2.append('Please normalize or reduce dimension')
                return 0
            
        print("------- START -------")

        # Specify the feature columns, target column, and gradient columns
        target_column = 'p' if style == 0 else ['NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                                'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

        if self.ui.pinn.isChecked():
            pinn = True
            mu = float(self.ui.mu.text())
            physics_weight = float(self.ui.physics_weight.text())
            feature_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude',
                               'U_grad_0', 'U_grad_1', 'U_grad_2',
                               'U_grad_3', 'U_grad_4', 'U_grad_5',
                               'U_grad_6', 'U_grad_7', 'U_grad_8',
                                'U_gradgrad_0', 'U_gradgrad_10', 'U_gradgrad_20',
                                'U_gradgrad_3', 'U_gradgrad_13', 'U_gradgrad_23',
                                'U_gradgrad_6', 'U_gradgrad_16', 'U_gradgrad_26']
            
            if style == 1:
                feature_columns += ['p']
        else:
            pinn = False
            mu = None
            physics_weight = None
            feature_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude']

        # Extract features, target, and gradient data from the dataset
        x_data = all_data[feature_columns].values.astype(np.float32)
        if style == 0:
            y_data = all_data[target_column].values.reshape(-1, 1).astype(np.float32)
        else:
            y_data = all_data[target_column].values.astype(np.float32)
        
        # Check for NaN values
        nan_mask_x = ~np.isnan(x_data).any(axis=1)
        nan_mask_y = ~np.isnan(y_data).any(axis=1)
        nan_mask = nan_mask_x & nan_mask_y
        x_data = x_data[nan_mask]
        y_data = y_data[nan_mask]

        # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42, shuffle=True)
        # Define split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        # Compute split indices
        total_samples = len(x_data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        # Split the data
        x_train = x_data[:train_end]
        y_train = y_data[:train_end]
        x_val = x_data[train_end:val_end]
        y_val = y_data[train_end:val_end]
        x_test = x_data[val_end:]
        y_test = y_data[val_end:]
        pinn_name = 'PINN_' if pinn else ''

        if index == 0:
            print("Run {}MLP model".format(pinn_name))

            # build model
            My_PINN_MLP(self, x_train, y_train, x_val, y_val, x_test, y_test,
                        units=units, dropout=dropout, activation=activation,
                        loss=loss, optimizer=optimizer, epochs=epochs,
                        lr=lr, batch_num=2048, style=style, s_m=s_m,
                        mu=mu, physics_weight=physics_weight, pinn=pinn)
            # the batch_num is different from LSTM
            # this means the size of batch
            print('*1' * 10)
        elif index == 1:
            print(f"Run {pinn_name}LSTM model")

            sequence_length = 128
            sequence_stride = 64
            step = 1
            batch_size = 512

            effective_seq_count = (len(x_train) - sequence_length * step) // sequence_stride + 1

            if effective_seq_count % batch_size == 0:
                batch_num = effective_seq_count // batch_size
            else:
                batch_num = effective_seq_count // batch_size + 1

            print(f'batch_num: {batch_num}')

            # Split data for LSTM
            dataset_train = timeseries_dataset_from_array(
                x_train,
                y_train,
                sequence_length=sequence_length,  # Exp: One batch was collected 120 times
                sequence_stride=sequence_stride,
                sampling_rate=step,  # Exp:Collect once in 6 steps, that is, once in 60 minutes
                batch_size=batch_size,
            )
            dataset_valid = timeseries_dataset_from_array(
                x_val,
                y_val,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_val)
            )
            dataset_test = timeseries_dataset_from_array(
                x_test,
                y_test,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_test)
            )
            # Build model
            My_PINN_LSTM(self, dataset_train, dataset_valid, dataset_test,
                         units=units, dropout=dropout,
                         loss=loss, optimizer=optimizer, epochs=epochs,
                         lr=lr, batch_num=batch_num, style=style, s_m=s_m,
                         mu=mu, physics_weight=physics_weight, pinn=pinn)
            print('*2' * 10)
        elif index == 2:
            print(f"Run {pinn_name}CNN model")

            sequence_length = 128
            sequence_stride = 64
            step = 1
            batch_size = 512
            
            effective_seq_count = (len(x_train) - sequence_length * step) // sequence_stride + 1

            if effective_seq_count % batch_size == 0:
                batch_num = effective_seq_count // batch_size
            else:
                batch_num = effective_seq_count // batch_size + 1

            print(f'batch_num: {batch_num}')

            # Split data for CNN
            dataset_train = timeseries_dataset_from_array(
                x_train,
                y_train,
                sequence_length=sequence_length,  # Exp: One batch was collected 120 times
                sequence_stride=sequence_stride,
                sampling_rate=step,  # Exp:Collect once in 6 steps, that is, once in 60 minutes
                batch_size=batch_size,
            )
            dataset_valid = timeseries_dataset_from_array(
                x_val,
                y_val,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_val)
            )
            dataset_test = timeseries_dataset_from_array(
                x_test,
                y_test,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_test)
            )
            # Build model
            My_PINN_CNN(self, dataset_train, dataset_valid, dataset_test,
                        filters=filters, dropout=dropout,
                        kernel_size=kernel_size, pool_size=pool_size, strides=strides,
                        loss=loss, optimizer=optimizer, epochs=epochs,
                        lr=lr, batch_num=batch_num, style=style, s_m=s_m,
                        mu=mu, physics_weight=physics_weight, pinn=pinn)
            print('*3' * 10)
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