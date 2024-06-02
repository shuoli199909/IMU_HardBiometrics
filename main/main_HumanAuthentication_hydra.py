"""
Main function of the human authentication pipeline. (HYDRA)
"""

# Author: Shuo Li
# Date: 2024/01/09

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
warnings.filterwarnings('ignore')  # Ignore warnings.
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data
from hydra import Hydra, SparseScaler


def main_HumanAuthentication_hydra(Params):
    """
    Parameters 
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """
    # Two conditions. One for training, the other for testing.
    df_result = pd.DataFrame(columns=['condition', 'label_test_gt', 'label_test_pred'])
    print('One for training, the other for testing...')
    for condition in tqdm(range(0, 2)):
        # Load data.
        # Training set.
        dir_imu_train = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                     'data_segmented', 'condition_'+str(condition)+'.csv')
        df_imu_train = pd.read_csv(dir_imu_train)
        # Gyroscope.
        data_gyro_x_train = df_imu_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_gyro_y_train = df_imu_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_gyro_z_train = df_imu_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        # Accelerometer.
        data_acc_x_train = df_imu_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_acc_y_train = df_imu_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_acc_z_train = df_imu_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_imu_train = np.concatenate((data_gyro_x_train[:, np.newaxis, :], 
                                         data_gyro_y_train[:, np.newaxis, :], 
                                         data_gyro_z_train[:, np.newaxis, :], 
                                         data_acc_x_train[:, np.newaxis, :], 
                                         data_acc_y_train[:, np.newaxis, :], 
                                         data_acc_z_train[:, np.newaxis, :]), axis=1)
        # Test set.
        dir_imu_test = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                    'data_segmented', 'condition_'+str(list(set([0, 1]).difference([condition]))[0])+'.csv')
        df_imu_test = pd.read_csv(dir_imu_test)
        # Gyroscope.
        data_gyro_x_test = df_imu_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_gyro_y_test = df_imu_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_gyro_z_test = df_imu_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        # Accelerometer.
        data_acc_x_test = df_imu_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_acc_y_test = df_imu_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_acc_z_test = df_imu_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
        data_imu_test = np.concatenate((data_gyro_x_test[:, np.newaxis, :], 
                                        data_gyro_y_test[:, np.newaxis, :], 
                                        data_gyro_z_test[:, np.newaxis, :], 
                                        data_acc_x_test[:, np.newaxis, :], 
                                        data_acc_y_test[:, np.newaxis, :], 
                                        data_acc_z_test[:, np.newaxis, :]), axis=1)
        # Initialize the Hydra transform correspondance.
        transform = Hydra(Params.len_segment)
        # Transform into tensors.
        data_imu_train = torch.tensor(data_imu_train, dtype=torch.float32).unsqueeze(2)
        data_train = torch.Tensor([])
        data_imu_test = torch.tensor(data_imu_test, dtype=torch.float32).unsqueeze(2)
        data_test = torch.Tensor([])
        # Transform the original data into Hydra features.
        print('Hydra transforming process...')
        for i_channel in tqdm(range(data_imu_train.shape[1])):
            scaler = SparseScaler()
            data_train = torch.cat((data_train, scaler.fit_transform(transform(data_imu_train[:, i_channel, :]))), 1)
            data_test = torch.cat((data_test, scaler.transform(transform(data_imu_test[:, i_channel, :]))), 1)
        # Training set groundtruth.
        Y_train_gt = df_imu_train.loc[:, ['ID']].values.ravel()
        Y_train_gt = torch.tensor(Y_train_gt, dtype=torch.float32)
        # Test set groundtruth.
        Y_test_gt = df_imu_test.loc[:, ['ID']].values.ravel()
        Y_test_gt = torch.tensor(Y_test_gt, dtype=torch.float32)
        # Ridge classifier.
        model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))        
        model.fit(data_train.numpy(), Y_train_gt.numpy())
        # Training set prediction.
        Y_train_pred = model.predict(data_train.numpy())
        # Test set prediction.
        Y_test_pred = model.predict(data_test.numpy())
        # Record results.
        df_result = pd.concat((df_result, 
                               pd.DataFrame({'condition': condition, 'label_test_gt': Y_test_gt, 'label_test_pred': Y_test_pred})), axis=0)
        # Save results.
        dir_result = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, 'hydra_result.csv')
        df_result.to_csv(dir_result, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZLeft', 'manual_IMUZCenter', 'manual_IMUZRight']  # ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        # Print current data information.
        print('Data Type: '+type_imu+'.')
        main_HumanAuthentication_hydra(Params=Params)