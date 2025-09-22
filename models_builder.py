#!/usr/bin/env python
# coding=utf-8

"""
@Time: 2025/8/14 5:23 PM
@Author: Fatimah Ahmadi Godini
@Email: Ahmadi.ths@gmail.com
@File: _models_builder.py
@Software: Visual Studio Code
"""

import os.path
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
import time
import traceback
import vtk
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import metrics, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint


print("--------- Script is running ---------")
# Data Preprocessing Functions
def normalize_data(original_data, style=0, method='minmax', normalized_path=None, scaler=None):
    """
    Normalize selected columns in a DataFrame and add them as new columns.
    If a scaler is provided, uses it to transform; otherwise fits a new one.

    :param original_data: pandas DataFrame with input data.
    :param style: 0 -> normalize p, 1 -> normalize wallShearStress.
    :param method: 'standard' (mean=0, std=1) or 'minmax' (0-1 scaling).
    :param normalized_path: Directory path to save the scaler and normalized CSV (if fitting a new scaler).
    :param scaler: Optional pre-fitted scaler to apply.
    :return: Tuple (normalized DataFrame, scaler)
    """
    normalize_features = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude']
    if style == 0:
        normalize_features += ['p']
    elif style == 1:
        normalize_features += ['NormalStress_0', 'NormalStress_1', 'NormalStress_2', 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

    normalized_data = original_data.copy()

    if scaler is None:
        # Fit a new scaler on the training data
        if method.lower() == 'minmax':
            scaler = MinMaxScaler()
            print("Using Min-Max Scaling (0~1 normalization).")
        elif method.lower() == 'standard':
            scaler = StandardScaler()
            print("Using Standardization (mean=0, std=1).")
        else:
            raise ValueError("method must be 'standard' or 'minmax'.")

        scaler.fit(original_data[normalize_features])

        # Save the fitted scaler
        if normalized_path:
            normalized_dir = os.path.dirname(normalized_path)
            joblib.dump(scaler, os.path.join(normalized_dir, 'p_stress_scaler.pkl'))
    else:
        print("Using pre-fitted scaler for normalization.")

    # Apply transformation
    scaled_values = scaler.transform(original_data[normalize_features])

    # Add new normalized columns
    for i, col in enumerate(normalize_features):
        normalized_data[col + '_norm'] = scaled_values[:, i]

    # Optionally save the normalized data
    if normalized_path:
        normalized_data.to_csv(normalized_path, encoding='gb18030', index=False)

    return normalized_data, scaler


def denormalize_data(normalized_values, scaler, style=0):
    """
    Denormalize the predictions or targets from normalized values back to original scale.

    :param normalized_values: np.array or shape-matched list of normalized values.
    :param scaler: the fitted sklearn scaler used for normalization.
    :param style: 0 -> pressure, 1 -> wall shear stress.
    :return: denormalized values.
    """
    # Columns order must match normalization order
    normalize_features = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude']
    if style == 0:
        normalize_features += ['p']
    elif style == 1:
        normalize_features += ['NormalStress_0', 'NormalStress_1', 'NormalStress_2', 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

    # Create a dummy array to hold the normalized values in the full feature space
    full_normalized = np.zeros((normalized_values.shape[0], len(normalize_features)))

    # Insert normalized targets into correct positions
    if style == 0:
        full_normalized[:, -1] = normalized_values.flatten()
    else:
        full_normalized[:, -6:] = normalized_values

    # Apply inverse transform
    denormalized_full = scaler.inverse_transform(full_normalized)

    # Return only the target columns
    if style == 0:
        return denormalized_full[:, -1].reshape(-1, 1)
    else:
        return denormalized_full[:, -6:]


def sequences_from_indices(array, indices_ds, start_index, end_index):
    '''
    获得序列索引
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

# Utility Functions
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


def draw_LossFig(train_loss, valid_loss, name, pinn_loss=None):
    '''
    Plot the loss value curves for training, validation, and optional PINN loss.
    :param train_loss: List of training loss values per epoch
    :param valid_loss: List of validation loss values per epoch
    :param name: Filename suffix for saving the plot
    :param pinn_loss: (Optional) List of PINN loss values per epoch
    '''
    
    bwith = 3
    fontsize = 22

    plt.figure(figsize=(10, 6))
    font_ = {
        'family': 'Times New Roman'
        , 'weight': 'normal'
        , 'color': 'k'
        , 'size': fontsize
    }

    plt.plot(np.arange(len(train_loss)), train_loss, 'b-', label='train loss', linewidth=bwith)
    plt.plot(np.arange(len(valid_loss)), valid_loss, 'r-', label='valid loss', linewidth=bwith)

    if pinn_loss is not None:
        plt.plot(np.arange(len(pinn_loss)), pinn_loss, 'g--', label='PINN Loss', linewidth=bwith)


    plt.xlabel('Train Times', fontdict=font_)
    plt.ylabel('Loss Value', fontdict=font_)
    plt.title('Loss Visualization', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.legend(loc="best", prop={'family': 'Times New Roman', 'weight': 'normal', 'size': fontsize}, framealpha=0)
    plt.tight_layout()

    os.makedirs('./res/result', exist_ok=True)
    plt.savefig(f'./res/result/{name}_lossfig.png')
    plt.close()
    return 0


def compute_max_ind(output, style, name):
    if style == 0:
        var_list = ['p']
    else:
        var_list = ['NormalStress_0', 'NormalStress_1', 'NormalStress_2', 'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

    max_var_data = []
    comparison_data_list = []

    for var in var_list:
        # Find the maximum var and corresponding x, y, z coordinates
        max_actual_var_index = output[var].idxmax()
        max_actual_var_row = output.iloc[max_actual_var_index]

        max_var_data.append({
            'type': 'Actual ' + var,
            'pressure': max_actual_var_row[var],
            'Points_0': max_actual_var_row['Points_0'],
            'Points_1': max_actual_var_row['Points_1'],
            'Points_2': max_actual_var_row['Points_2']
        })

        # Find the maximum predicted var and its corresponding x, y, z coordinates
        max_pred_var_index = output['Predicted_' + var].idxmax()
        max_pred_var_row = output.iloc[max_pred_var_index]

        max_var_data.append({
            'type': 'Predicted ' + var,
            'pressure': max_pred_var_row['Predicted_' + var],
            'Points_0': max_pred_var_row['Points_0'],
            'Points_1': max_pred_var_row['Points_1'],
            'Points_2': max_pred_var_row['Points_2']
        })

        # Compute the absolute difference between actual and predicted var
        var_difference = abs(max_actual_var_row[var] - max_pred_var_row['Predicted_' + var])

        # Add the comparison data
        comparison_data = {
            'type': var + ' Difference',
            'pressure_difference': var_difference,
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


# Visualization Functions
def draw_3d_dataframe(df, x_col='Points_0', y_col='Points_1', z_col='Points_2', 
                      color_col=None, title="3D Data Visualization"):
    """
    Draws a 3D scatter plot from a pandas DataFrame using all available points.
    :param df: pandas DataFrame
    :param x_col, y_col, z_col: names of the columns to use for 3D coordinates
    :param color_col: column name for coloring the points (e.g., pressure)
    :param title: plot title
    """
    xs = df[x_col]
    ys = df[y_col]
    zs = df[z_col]
    c = df[color_col] if color_col else 'b'

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=c, cmap='viridis', s=5, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    if color_col:
        cb = plt.colorbar(sc)
        cb.set_label(color_col)

    plt.tight_layout()
    plt.show()


def draw_3d_points(x, title="3D Point Cloud", color='blue'):
    """
    Draws a 3D scatter of (x, y, z) locations using all available points.
    :param x: ndarray of shape (n_samples, n_features)
    :param title: Plot title
    :param color: Color of the points
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract spatial coordinates
    xs, ys, zs = x[:, 0], x[:, 1], x[:, 2]

    ax.scatter(xs, ys, zs, s=5, c=color, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def draw_input_dataset(inputs):
    """
    Draw 3D spatial points from a batch of time series input.
    Assumes the first 3 features in each sequence are X, Y, Z spatial coordinates.
    """
    if not isinstance(inputs, np.ndarray):
        inputs = inputs.numpy()

    # Use the last timestep for each sequence to get one point per sample
    xyz_points = inputs[:, -1, :3]  # shape: (batch_size, 3)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2],
               c='blue', s=10, alpha=0.6)

    ax.set_title("3D Points from Batch")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


# Physics-Informed Loss Functions
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


# Neural Network Model Functions
def My_PINN_MLP(x_train, y_train, x_valid, y_valid, x_test, y_test,
               units=[32, 32], dropout=0.1, activation='tanh',
               loss='mse', optimizer='adam', epochs=50,
               lr=0.001, batch_num=2048, style=0, scaler=None,
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
                    if scaler:
                        predict_denorm = denormalize_data(predict.numpy(), scaler, style=style)
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u_batch[:, np.newaxis, :], predict_denorm, mu=mu)
                    else:
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u_batch[:, np.newaxis, :], predict, spatial_data=x_batch[:, np.newaxis, :], mu=mu)
                    
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

    draw_LossFig(np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)
    
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
    if scaler:
        y_pred = denormalize_data(y_predict, scaler, style=style)
    else:
        y_pred = y_predict

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
                'Component': f'Shear_{i}',
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


def My_PINN_LSTM(dataset_train, dataset_valid, dataset_test,
                 units=[32, 32], dropout=0.1,
                 loss='mse', optimizer='adam', epochs=50,
                 lr=0.001, batch_num =464, style=0, scaler=None,
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
    model.add(Dense(units=output_units, activation='linear'))  # 'sigmoid'
    net_ += f"_{output_units}"

    model.summary()

    # Compile model
    opt = get_optimizer(optimizer, lr)
    # model.compile(optimizer=optimizer, loss=loss)

    name = f"{model_name}+{net_}+{dropout}+{lr}+{batch_num}+{epochs}"
    modelpath = f'./res/model/{name}.h5'

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
                    if scaler:
                        predict_denorm = denormalize_data(predict.numpy(), scaler, style=style)
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u, predict_denorm, mu=mu)
                    else:
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u, predict, spatial_data=inputs, mu=mu)
                        
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
    
    draw_LossFig(np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)

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

    if scaler:
        y_pred = denormalize_data(y_predict, scaler, style=style)
    else:
        y_pred = y_predict

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
                'Component': f'Shear_{i}',
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

    return model_saved


def My_PINN_CNN(dataset_train, dataset_valid, dataset_test,
                filters=[32], dropout=0.3,
                kernel_size=3, pool_size=2, strides=1,
                loss='mse', optimizer='adam', epochs=100,
                lr=0.001, batch_num=464, style=0, scaler=None,
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
    modelpath = f'./res/model/{name}.h5'

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
                    if scaler:
                        predict_denorm = denormalize_data(predict.numpy(), scaler, style=style)
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u, predict_denorm, mu=mu)
                    else:
                        pinn_loss = (Navier_Stokes_Residual if style == 0 else Stress_Tensor_Residual)(
                            grad_u, predict, spatial_data=inputs, mu=mu)
                    
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
        
    draw_LossFig(np.array(train_loss_sum), np.array(valid_loss_sum), name, pinn_loss_sum if pinn else None)
    
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

    if scaler:
        y_pred = denormalize_data(y_predict, scaler, style=style)
    else:
        y_pred = y_predict

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
                'Component': f'Shear_{i}',
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

    return model_saved


# Training Pipeline Function
def Compute(datapath="./wall_csv/",
            units=[32], epochs = 50,
            style=0, pinn=False, model_ind=2,
            kernel_size = 3, pool_size = 2, strides = 1,
            activation = 'tanh', batch_size = 3000,
            method=None, shuffledata=True, splitdata=True,
            dropout = 0.1, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu=0.001, physics_weight=0.01):
    '''
    The data is divided into training, validation and test datasets by 92:4:4.
    '''
    """
    os.makedirs('./res/data_process', exist_ok=True)
    os.makedirs('./res/model', exist_ok=True)
    os.makedirs('./res/history', exist_ok=True)
    os.makedirs('./res/result', exist_ok=True)
    """
    print("Start Computation")
    os.makedirs('./res/data_process', exist_ok=True)
    os.makedirs('./res/model', exist_ok=True)
    os.makedirs('./res/history', exist_ok=True)
    os.makedirs('./res/result', exist_ok=True)

    # split data and save split_files
    def split_data_save_txt(csv_files_path, split_file_path):
        
        # Check if datapath is a .zip file or a directory containing .csv files
        if csv_files_path.endswith('.zip'):
            import zipfile
            import tempfile

            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(csv_files_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            csv_files = glob.glob(os.path.join(temp_dir, '*.csv'))

        elif os.path.isdir(csv_files_path):
            csv_files = glob.glob(os.path.join(csv_files_path, '*.csv'))
        else:
            print(f"Error: {csv_files_path} is neither a valid directory nor a zip file.")
            return

        print(f"Found {len(csv_files)} CSV files in {csv_files_path}")
        
        if shuffledata:
            random.seed(42)
            random.shuffle(csv_files)

            train_end = int(0.80 * len(csv_files))
            val_end = train_end + int(0.12 * len(csv_files))

            train_files = csv_files[:train_end]
            valid_files = csv_files[train_end:val_end]
            test_files = csv_files[val_end:]

        else:
            train_files = csv_files[:-5]
            valid_files = csv_files[-5:-2]
            print(f"Valid files: {valid_files}")
            test_files = csv_files[-2:]
            print(f"Test files: {test_files}")

        
        # Save the filenames to a text file
        with open(split_file_path, 'w') as f:
            f.write("Training files:\n")
            for file in train_files:
                f.write(f"{file}\n")
            f.write("\nValidation files:\n")
            for file in valid_files:
                f.write(f"{file}\n")
            f.write("\nTest files:\n")
            for file in test_files:
                f.write(f"{file}\n")

        return train_files, valid_files, test_files
    # load split_files
    def load_split_files(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        train_files = []
        valid_files = []
        test_files = []

        current_section = None
        for line in lines:
            line = line.strip()
            if line == "Training files:":
                current_section = 'train'
            elif line == "Validation files:":
                current_section = 'valid'
            elif line == "Test files:":
                current_section = 'test'
            elif line == "":
                continue
            else:
                if current_section == 'train':
                    train_files.append(line)
                elif current_section == 'valid':
                    valid_files.append(line)
                elif current_section == 'test':
                    test_files.append(line)

        return train_files, valid_files, test_files

    if splitdata:
        train_files, valid_files, test_files = split_data_save_txt(datapath, './res/data_process/split_files.txt')
    else:
        train_files, valid_files, test_files = load_split_files('./res/data_process/split_files.txt')
        """
        train_files = [
            "./wall_csv/3D_80.csv", "./wall_csv/3D_40.csv", "./wall_csv/3D_10.csv",
            "./wall_csv/4D_30.csv", "./wall_csv/4D_20.csv", "./wall_csv/2D_60.csv",
            "./wall_csv/2D_50.csv", "./wall_csv/3D_20.csv", "./wall_csv/3D_70.csv",
            "./wall_csv/4D_50.csv", "./wall_csv/3D_30.csv", "./wall_csv/4D_0.csv",
            "./wall_csv/2D_10.csv", "./wall_csv/3D_60.csv", "./wall_csv/4D_60.csv",
            "./wall_csv/3D_50.csv", "./wall_csv/2D_20.csv", "./wall_csv/4D_70.csv",
            "./wall_csv/2D_40.csv", "./wall_csv/4D_80.csv"
        ]

        valid_files = [
            "./wall_csv/2D_80.csv", "./wall_csv/3D_0.csv", "./wall_csv/2D_0.csv"
        ]

        test_files = [
            "./wall_csv/2D_30.csv", "./wall_csv/4D_40.csv"
        ]
        """
    
    # Load data from the CSV files
    def load_csv(file_list):
        data = []
        for file in file_list:
            try:
                df = pd.read_csv(file)
                data.append(df)
                print(f'{file}: len_data = {len(df)}')
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return pd.concat(data, ignore_index=True)
    print("Loading CSVs...")
    train_origin_data = load_csv(train_files)
    valid_origin_data = load_csv(valid_files)
    test_origin_data = load_csv(test_files)

    target_column_norm = 'p_norm' if style == 0 else ['wallShearStress_0_norm', 'wallShearStress_1_norm', 'wallShearStress_2_norm']
    test_target_column = 'p' if style == 0 else ['wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2']

    if method:
        # Fit only on train set
        train_data, scaler = normalize_data(train_origin_data,style=style, method=method, normalized_path='./res/data_process/normalized_train_data.csv')

        # Apply to valid/test using the same scaler
        valid_data, _ = normalize_data(valid_origin_data, style=style, method=method, normalized_path='./res/data_process/normalized_valid_data.csv', scaler=scaler)
        test_data, _ = normalize_data(test_origin_data, style=style, method=method, normalized_path='./res/data_process/normalized_test_data.csv', scaler=scaler)

        input_columns = ['Points_0_norm', 'Points_1_norm', 'Points_2_norm', 'Points_Magnitude_norm']
        gradu_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude',
                         'U_grad_0', 'U_grad_1', 'U_grad_2',
                         'U_grad_3', 'U_grad_4', 'U_grad_5',
                         'U_grad_6', 'U_grad_7', 'U_grad_8',
                         'U_gradgrad_0', 'U_gradgrad_10', 'U_gradgrad_20',
                         'U_gradgrad_3', 'U_gradgrad_13', 'U_gradgrad_23',
                         'U_gradgrad_6', 'U_gradgrad_16', 'U_gradgrad_26']
        target_column = target_column_norm
    else:
        scaler = None
        train_data = train_origin_data
        valid_data = valid_origin_data
        test_data = test_origin_data

        input_columns = ['Points_0', 'Points_1', 'Points_2', 'Points_Magnitude']
        gradu_columns = ['U_grad_0', 'U_grad_1', 'U_grad_2',
                         'U_grad_3', 'U_grad_4', 'U_grad_5',
                         'U_grad_6', 'U_grad_7', 'U_grad_8',
                         'U_gradgrad_0', 'U_gradgrad_10', 'U_gradgrad_20',
                         'U_gradgrad_3', 'U_gradgrad_13', 'U_gradgrad_23',
                         'U_gradgrad_6', 'U_gradgrad_16', 'U_gradgrad_26']
        target_column = test_target_column
    if style == 1:
        gradu_columns += ['p']
    feature_columns = input_columns + gradu_columns if pinn else input_columns

    print("Train data shape:", train_data.shape)
    print("Validation data shape:", valid_data.shape)
    print("Test data shape:", test_data.shape)
    print(test_data.head())

    x_train = train_data[feature_columns].values.astype(np.float32)
    x_valid = valid_data[feature_columns].values.astype(np.float32)
    x_test = test_data[feature_columns].values.astype(np.float32)

    y_train = train_data[target_column].values.astype(np.float32)
    y_valid = valid_data[target_column].values.astype(np.float32)
    y_test = test_data[test_target_column].values.astype(np.float32)

    if style == 0:
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
    else:
        def compute_normal_stress(data, mu):
            u_x = data['U_grad_0'].values.astype(np.float32)
            v_y = data['U_grad_4'].values.astype(np.float32)
            w_z = data['U_grad_8'].values.astype(np.float32)
            p   = data['p'].values.astype(np.float32)

            normalstress_0 = -p + 2 * mu * u_x
            normalstress_1 = -p + 2 * mu * v_y
            normalstress_2 = -p + 2 * mu * w_z

            return np.stack([normalstress_0, normalstress_1, normalstress_2], axis=1)

        normalstress_train = compute_normal_stress(train_data, mu)
        normalstress_valid = compute_normal_stress(valid_data, mu)
        normalstress_test = compute_normal_stress(test_data, mu)

        y_train = np.concatenate([normalstress_train, y_train], axis=1)
        y_valid = np.concatenate([normalstress_valid, y_valid], axis=1)
        y_test = np.concatenate([normalstress_test, y_test], axis=1)

    pinn_name = 'PINN_' if pinn else ''
    try:
        if model_ind == 0:
            print("Run {}MLP model".format(pinn_name))

            My_PINN_MLP(x_train, y_train, x_valid, y_valid, x_test, y_test,
                       units=units, dropout=dropout, activation=activation,
                       loss=loss, optimizer=optimizer, epochs=epochs,
                       lr=lr, batch_num=batch_size, style=style, scaler=scaler,
                       mu=mu, physics_weight=physics_weight, pinn=pinn)

            print('*' * 20)

        elif model_ind == 1:
            print(f"Run {pinn_name}LSTM model")

            sequence_length = 128
            sequence_stride = 64
            step = 1

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
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=batch_size,
            )
            dataset_valid = timeseries_dataset_from_array(
                x_valid,
                y_valid,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_valid)
            )
            dataset_test = timeseries_dataset_from_array(
                x_test,
                y_test,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_test)
            )

            My_PINN_LSTM(dataset_train, dataset_valid, dataset_test,
                         units=units, dropout=dropout,
                         loss=loss, optimizer=optimizer, epochs=epochs,
                         lr=lr, batch_num=batch_num, style=style, scaler=scaler,
                         mu=mu, physics_weight=physics_weight, pinn=pinn)

            print('*' * 20)

        elif model_ind == 2:
            print(f"Run {pinn_name}CNN model")

            sequence_length = 128
            sequence_stride = 64
            step = 1

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
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=batch_size,
            )
            dataset_valid = timeseries_dataset_from_array(
                x_valid,
                y_valid,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_valid)
            )
            dataset_test = timeseries_dataset_from_array(
                x_test,
                y_test,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                sampling_rate=step,
                batch_size=len(x_test)
            )

            """
            draw_3d_dataframe(train_data, title="Train Data Distribution")
            draw_3d_dataframe(valid_data, title="Validation Data Distribution")
            draw_3d_dataframe(test_data, title="Test Data Distribution")
            
            draw_3d_points(x_train, title="x_train", color='blue')
            draw_3d_points(x_valid, title="x_valid", color='green')
            draw_3d_points(x_test, title="x_test", color='red')
            
            for epoch in range(epochs):
                for bs in range(batch_num):
                    for batch in dataset_train.take(bs + 1):
                        inputs, _ = batch
                    draw_input_dataset(inputs[:, :, :4])
            
            for batch in dataset_valid.take(1):
                inputs, _ = batch
            draw_input_dataset(inputs[:, :, :4])

            for batch in dataset_test.take(1):
                inputs, _ = batch
            draw_input_dataset(inputs[:, :, :4])
            """
            # Build model
            My_PINN_CNN(dataset_train, dataset_valid, dataset_test,
                        filters=units, dropout=dropout,
                        kernel_size=kernel_size, pool_size=pool_size, strides=strides,
                        loss=loss, optimizer=optimizer, epochs=epochs,
                        lr=lr, batch_num=batch_num, style=style, scaler=scaler,
                        mu=mu, physics_weight=physics_weight, pinn=pinn)
            
            print('*' * 20)

    except Exception as e:
        print(f"Error encountered: {e}")
        traceback.print_exc()


# Run the models
"""
style = 0 for p, 1 for s
model_ind = 0 MLP, 1 LSTM, 2 CNN
method = 'minmax', 'standard', None
"""

# for example for PINN-CNN:
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = True, model_ind = 2,
            kernel_size = 3, pool_size = 2, strides = 1,
            batch_size = 512,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 1)


# Pressure prediction
"""
# MLP  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = False, model_ind = 0,
            activation = 'tanh', batch_size = 256,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)

    
# PINN-MLP  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = True, model_ind = 0,
            activation = 'tanh', batch_size = 256,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)

# LSTM  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = False, model_ind = 1,
            batch_size = 256,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)           

# PINN-LSTM  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = True, model_ind = 1,
            batch_size = 256,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)

# CNN  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = False, model_ind = 2,
            kernel_size = 3, pool_size = 2, strides = 1,
            batch_size = 256,
            method = None, shuffledata = True, splitdata=True,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)

# PINN-CNN  # OK
Compute(datapath="./wall_csv/",
            units = [32, 64], epochs = 200,
            style = 0, pinn = True, model_ind = 2,
            kernel_size = 3, pool_size = 2, strides = 1,
            batch_size = 256,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.001,
            mu = 0.001, physics_weight = 0.01)
"""

# Shear stress prediction
'''
# CNN
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = False, model_ind = 2,
            kernel_size = 3, pool_size = 2, strides = 1,
            batch_size = 512,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 0.1)

# PINN-CNN
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = True, model_ind = 2,
            kernel_size = 3, pool_size = 2, strides = 1,
            batch_size = 512,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 1)


# LSTM
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = False, model_ind = 1,
            batch_size = 512,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 0.1)

# PINN-LSTM
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = True, model_ind = 1,
            batch_size = 512,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 1)


# MLP
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = False, model_ind = 0,
            activation = 'tanh', batch_size = 2048,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 0.1)

# PINN-MLP
Compute(datapath="./wall_csv/",
            units = [32, 64, 32], epochs = 200,
            style = 1, pinn = True, model_ind = 0,
            activation = 'tanh', batch_size = 2048,
            method = None, shuffledata = True, splitdata=False,
            dropout = 0.3, optimizer = 'adam', loss = 'mse', lr = 0.005,
            mu = 0.001, physics_weight = 1)
'''

def DrawSliceT(self):
    '''
    Draw the simulation results of slices and the prediction results of intelligent models.
    See TestDataPara framework for slice location selection.
    """
    All of the code in this section appears in _ReadData.py, so it's not commented on in detail
    """
    '''
    print('DrawSliceT')
    fontsize = 13
    bwith = 1
    self.ui.TestFig.figure.clear()  # clear fig

    # index = self.ui.SelectFeature1.currentIndex()
    print('DrawSlice begin')
    Length_t = float(self.ui.Length_t.text())
    Angle_t = float(self.ui.Angle_t.text())
    StartBent_t = float(self.ui.StartBent_t.text())
    RadiusBent_t = float(self.ui.RadiusBent_t.text())
    DCircle_t = float(self.ui.DCircle_t.text())
    Number_t = float(self.ui.Number_t.text())
    AllLength_t = float(self.ui.AllLength_t.text())
    Region_t = self.ui.Region_t.currentIndex()
    StrePres_t = self.ui.StrePres_t.currentIndex()
    print('read data is finished')

    # print('Length_t,Angle_t,StartBent_t',Length_t,Angle_t,StartBent_t)
    # print('RadiusBent_t', RadiusBent_t)
    # print('DCircle_t', DCircle_t)

    # StartBent_t = 240in/6.096m
    # Angle_t=90
    # DCircle_t = 30in/0.762m
    # RadiusBent_t = 60in/1.524m

    # judge region
    if Region_t == 0:
        if Number_t <= StartBent_t and Number_t >= 0:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 0 is wrong. Please input the number again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # Default Button
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region_t == 1:
        if Number_t <= 90 and Number_t >= 0:
            Number_t = StartBent_t + np.pi * RadiusBent_t * Number_t / 180
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 1 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # Default Button
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    elif Region_t == 2:
        if Number_t <= AllLength_t and Number_t >= StartBent_t + np.pi * RadiusBent_t * Angle_t / 180:
            pass
        else:
            dlgTitle = "Tips"
            strInfo = ("The range of region 2 is wrong. Please input the angle again."
                       "Click 'WholePipe-CentreLine', you will get the right range.")
            defaultBtn = QMessageBox.NoButton  # Default Button
            result = QMessageBox.question(self, dlgTitle, strInfo,
                                          QMessageBox.Yes,
                                          defaultBtn)
            return 0
    # Divide the area
    data = self.testworkbook.copy(deep=True)
    print('data', data)
    centre_point_x = StartBent_t
    centre_point_y = 0
    centre_point_z = RadiusBent_t
    region0_index = np.where(data['Points_0'].values < StartBent_t)[0]
    region12_index = np.where(data['Points_0'].values >= StartBent_t)[0]
    region0_data = data.iloc[region0_index, :]
    region12_data = data.iloc[region12_index, :]
    region0_data_xyz = region0_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    region12_data_xyz = region12_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    # Adjust the origin
    region12_data_xyz[:, 0] = region12_data_xyz[:, 0] - centre_point_x
    region12_data_xyz[:, 2] = region12_data_xyz[:, 2] - centre_point_z
    
    # Convert coordinates
    r = np.sqrt(region12_data_xyz[:, 0] ** 2 + region12_data_xyz[:, 1] ** 2 + region12_data_xyz[:, 2] ** 2)
    theta = np.arccos(region12_data_xyz[:, 2] / r) / np.pi * 180
    phi = np.arctan2(region12_data_xyz[:, 1], region12_data_xyz[:, 0]) / np.pi * 180

    region12_data = region12_data.copy().assign(r=r, theta=theta, phi=phi)
    print('theta: ', np.sort(list(set(theta))))
    # There is a bug in dividing the region
    # region 1 Since only two points on a slice are accurate = theta (these two points are located at the center of the circle and the center line)
    # , the theta of other points is not accurate = 180, so the slice cannot be found
    # This also causes problems when dividing region1 and region2
    region1_index = np.where(theta > 180 - Angle_t)[0]
    region2_index = np.where(theta <= 180 - Angle_t)[0]
    region1_data = region12_data.iloc[region1_index, :]
    region2_data = region12_data.iloc[region2_index, :]
    region1_data_xyz = region1_data.iloc[:, :3].values
    # Adjust the origin
    centre_point_x2 = StartBent_t + RadiusBent_t * np.sin(Angle_t / 180 * np.pi)
    centre_point_y2 = 0
    centre_point_z2 = RadiusBent_t * (1 - np.cos(Angle_t / 180 * np.pi))
    region2_data_xyz = region2_data.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
    region2_data_xyz[:, 0] -= centre_point_x2
    region2_data_xyz[:, 2] -= centre_point_z2
    new_point = rotate_point(region2_data_xyz[:, 0], region2_data_xyz[:, 1], region2_data_xyz[:, 2], Angle_t)
    region2_data = region2_data.copy()
    region2_data['x'] = new_point.T[:, 0]
    region2_data['y'] = new_point.T[:, 1]
    region2_data['z'] = new_point.T[:, 2]

    # find position
    if Number_t <= StartBent_t:
        print('number<=StartBent_t')
        o1 = np.where(self.testworkbook['Points_0'].values < Number_t + Length_t)[0]
        o2 = np.where(self.testworkbook['Points_0'].values >= Number_t)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 30
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < 0:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < 0:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]
        z_special = 0
    elif Number_t <= (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180):
        print('number<=(StartBent_t+np.pi*RadiusBent_t*Angle_t/180)')
        remain_len = Number_t - StartBent_t
        sita = remain_len / (np.pi * RadiusBent_t)  #
        centre_x = StartBent_t + RadiusBent_t * np.sin(sita)
        centre_y = 0
        centre_z = RadiusBent_t * (1 - np.cos(sita))
        x_min = centre_x - DCircle_t * np.sin(sita) - Length_t / 2
        x_max = centre_x + DCircle_t * np.sin(sita) + Length_t / 2
        z_min = centre_z - DCircle_t * np.cos(sita) - Length_t / 2
        z_max = centre_z + DCircle_t * np.cos(sita) + Length_t / 2

        o1 = np.where(self.testworkbook['Points_0'].values <= x_max)[0]
        o2 = np.where(self.testworkbook['Points_0'].values >= x_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index1 = []
        for point in o1:
            if point in o2:
                original_position_index1.append(point)
        o1 = np.where(self.testworkbook['Points_2'].values <= z_max)[0]
        o2 = np.where(self.testworkbook['Points_2'].values >= z_min)[0]
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index2 = []
        for point in o1:
            if point in o2:
                original_position_index2.append(point)
        original_position_index = []
        for point in original_position_index1:
            if point in original_position_index2:
                original_position_index.append(point)
        data_ori = self.testworkbook.iloc[original_position_index, :]
        data = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
        # position_index = []
        # for aim_point in range(len(data_ori)):
        #     DCircle_2 = ((data[aim_point, 0] - centre_x) ** 2 + (data[aim_point, 1] - centre_y) ** 2
        #      + (data[aim_point, 2] - centre_z) ** 2)
        #     if (DCircle_2-DCircle_t**2)<0.02**2:
        #         position_index.append(aim_point)
        # data_ori = data_ori.iloc[position_index, :]
        x_y_z = data
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        # y_ = DCircle_t * np.sin(omega)
        # x_ = DCircle_t * np.cos(omega) * np.sin(sita) #x +-
        # z_ = DCircle_t * np.cos(omega) * np.cos(sita)  # z +-
        elev = 45
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < centre_z:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < centre_z:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]
        z_special = centre_z
    else:
        print('else')
        print('number, (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180)')
        print(Number_t, StartBent_t, np.pi, RadiusBent_t, Angle_t, 180)
        remain_len = Number_t - (StartBent_t + np.pi * RadiusBent_t * Angle_t / 180)
        print('remain_len', remain_len)
        z_special = centre_point_z2 + remain_len * np.sin(Angle_t / 180 * np.pi)
        o1 = np.where(region2_data['x'] < remain_len + Length_t)[0]
        o2 = np.where(region2_data['x'] >= remain_len)[0]
        print('shape', self.testworkbook.values.shape)
        # print('min=%.3f,max=%.3f' % (o2, o1))
        original_position_index = []
        for point in o1:
            if point in o2:
                original_position_index.append(point)
        original_position_index = region2_data.iloc[original_position_index, :].index
        data_ori = self.testworkbook.iloc[original_position_index, :]
        x_y_z = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2']].values.copy()
        x = x_y_z[:, 0]
        y = x_y_z[:, 1]
        z = x_y_z[:, 2]
        elev = 90
        azim = 0

        if StrePres_t == 0:
            p = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2', 'p', 'Predicted_p']]
            p_y = p.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(p_y)):
                if p_y[i, 2] < z_special:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            p = p_y[new_i, :]
        else:
            stress = data_ori.loc[:, ['Points_0', 'Points_1', 'Points_2',
                                      'NormalStress_0', 'NormalStress_1', 'NormalStress_2',
                                      'wallShearStress_0', 'wallShearStress_1', 'wallShearStress_2',
                                      'Predicted_NormalStress_0', 'Predicted_NormalStress_1', 'Predicted_NormalStress_2',
                                      'Predicted_wallShearStress_0', 'Predicted_wallShearStress_1', 'Predicted_wallShearStress_2']]
            stress_y = stress.sort_values(by='Points_1', ascending=True).values.copy()
            new_index = []
            new_index_inv = []
            for i in range(len(stress_y)):
                if stress_y[i, 2] < z_special:
                    new_index.append(i)
                else:
                    new_index_inv.append(i)
            new_i = np.concatenate((new_index, new_index_inv[::-1]))
            new_i = np.array(new_i, dtype=int)
            stress = stress_y[new_i, :]

    columns = [i.split('\n')[0] for i in self.testworkbook.columns]
    print('Length_t', Length_t)
    print('number', Number_t)
    print('test')
    ax1 = self.ui.TestFig.figure.add_subplot(1, 1, 1, label='plot3D')
    if StrePres_t == 0:
        diyi_1 = np.where(p[:, 1] < 0)[0]
        diyi_2 = np.where(p[:, 2] < z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis1 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
        diyi_1 = np.where(p[:, 1] > 0)[0]
        diyi_2 = np.where(p[:, 2] < z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis2 = 180 - np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
        diyi_1 = np.where(p[:, 1] > 0)[0]
        diyi_2 = np.where(p[:, 2] > z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis3 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
        diyi_1 = np.where(p[:, 1] < 0)[0]
        diyi_2 = np.where(p[:, 2] > z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis4 = 180 + np.arccos(p[index, 1] / np.max(p[:, 1])) / np.pi * 180
        x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))

        print('000000')
        ax1.scatter(x_axis, p[:, 3], c='b', s=1, label='Pressure CFD')
        ax1.scatter(x_axis, p[:, 4], c='r', s=1, label='Pressure Prediction')
    elif StrePres_t == 1:
        diyi_1 = np.where(stress['Points_1'].values < 0)[0]
        diyi_2 = np.where(stress['Points_2'].values < z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis1 = 180 - np.arccos(stress.loc[index, 'Points_1'] / stress['Points_1'].max()) / np.pi * 180
        diyi_1 = np.where(stress['Points_1'].values > 0)[0]
        diyi_2 = np.where(stress['Points_2'].values < z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis2 = 180 - np.arccos(stress.loc[index, 'Points_1'] / stress['Points_1'].max()) / np.pi * 180
        diyi_1 = np.where(stress['Points_1'].values > 0)[0]
        diyi_2 = np.where(stress['Points_2'].values > z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis3 = 180 + np.arccos(stress.loc[index, 'Points_1'] / stress['Points_1'].max()) / np.pi * 180
        diyi_1 = np.where(stress['Points_1'].values < 0)[0]
        diyi_2 = np.where(stress['Points_2'].values > z_special)[0]
        index = []
        for i in diyi_1:
            if i in diyi_2:
                index.append(i)
        x_axis4 = 180 + np.arccos(stress.loc[index, 'Points_1'] / stress['Points_1'].max()) / np.pi * 180
        x_axis = np.concatenate((x_axis1, x_axis2, x_axis3, x_axis4))
        stress_magnitude = np.sqrt(
            stress['wallShearStress_0']**2 +
            stress['wallShearStress_1']**2 +
            stress['wallShearStress_2']**2
        )
        prediction_stress_magnitude = np.sqrt(
            stress['Predicted_wallShearStress_0']**2 +
            stress['Predicted_wallShearStress_1']**2 +
            stress['Predicted_wallShearStress_2']**2
        )
        ax1.scatter(x_axis, stress_magnitude, c='b', s=1, label='CFD wallShearStress_magnitude')
        ax1.scatter(x_axis, prediction_stress_magnitude, c='r', s=1, label='Prediction Stress')
        print('*' * 100)
        print(x_axis)
        print(stress_magnitude)
        print('*' * 100)
        print(prediction_stress_magnitude)
        print('*' * 100)

    print('111111')
    print('x', x_axis)
    ax1.set_xlabel('x axis (Angle_t)', fontsize=fontsize)

    if StrePres_t == 0:
        ax1.set_title('Pressure', fontsize=fontsize)
        ax1.set_ylabel('y axis (Pressure)', fontsize=fontsize)
    else:
        ax1.set_title('Stress', fontsize=fontsize)
        ax1.set_ylabel('y axis (Stress)', fontsize=fontsize)
    print('3333333')
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    print('4444444')
    # ax1.invert_yaxis()
    # print('11113')
    # ax1.set_xticks(fontproperties='Times New Roman')
    # ax1.set_yticks(fontproperties='Times New Roman')

    ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    ax1.legend(loc="best", edgecolor='white', facecolor=None, framealpha=0)
    self.ui.TestFig.figure.tight_layout()
    self.ui.Results4.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    self.ui.TestFig.figure.canvas.draw()

    print('****' * 3)

    # ax1 = self.ui.TestFig.figure.add_subplot(1, 1, 1, label='plot3D')
    # print('111111')
    # ax1.set_xlabel('x axis (point number)', fontsize=fontsize)
    # if style == 0:
    #     ax1.plot(ori_test_data[:, y_index], c='b', label='pressure')
    #     ax1.set_title('Pressure', fontsize=fontsize)
    #     ax1.set_ylabel('y axis (Pressure)', fontsize=fontsize)
    # else:
    #     ax1.plot(ori_test_data[:, y_index1], c='b', label='wallShearStress_0')
    #     ax1.plot(ori_test_data[:, y_index2], c='r', label='wallShearStress_1')
    #     ax1.plot(ori_test_data[:, y_index3], c='k', label='wallShearStress_2')
    #     ax1.set_title('Stress', fontsize=fontsize)
    #     ax1.set_ylabel('y axis (Stress)', fontsize=fontsize)
    # print('3333333')
    # ax1.spines['bottom'].set_linewidth(bwith)
    # ax1.spines['left'].set_linewidth(bwith)
    # ax1.spines['top'].set_linewidth(bwith)
    # ax1.spines['right'].set_linewidth(bwith)
    # print('4444444')
    # # ax1.invert_yaxis()
    # # print('11113')
    # # ax1.set_xticks(fontproperties='Times New Roman')
    # # ax1.set_yticks(fontproperties='Times New Roman')
    #
    # ax1.tick_params(width=bwith, length=bwith, labelsize=fontsize, direction='in')
    # ax1.legend(loc="best", edgecolor='white', facecolor=None, framealpha =0)
    # self.ui.TestFig.figure.tight_layout()
    # self.ui.Results4.setText('Figure is completed')  # %self.workbook.columns[index].split('\n')[0]
    # # self.ui.Results.setText('%s绘图完成')#%self.workbook.columns[index].split('\n')[0]
    # # ax1.show()
    # self.ui.TestFig.figure.canvas.draw()
    #
    #
    #
    #
    #
    #

