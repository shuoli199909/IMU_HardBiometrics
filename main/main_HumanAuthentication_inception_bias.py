"""
Generate the recognition result for evaluating the model predictive bias. (InceptionTime)
"""

# Author: Shuo Li
# Date: 2024/01/09

import os
import sys
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from tsai.models import InceptionTime
warnings.filterwarnings('ignore')  # Ignore warnings.
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data


def main_HumanAuthentication_inception_bias(Params):
    """
    Parameters 
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # Two conditions. One for training, the other for testing.
    df_result = pd.DataFrame(columns=['batch', 'condition', 'label_test_gt', 'label_test_pred'])
    # Repeat experiments.
    for batch in tqdm(range(0, 100)):
        for condition in range(0, 2):
            # Load data.
            # Training set.
            dir_imu_train = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                         'data_segmented', 'condition_'+str(condition)+'.csv')
            df_imu_train = pd.read_csv(dir_imu_train)
            # Data sampling for balanced evaluation.
            if condition == 0:
                list_sample_idx = []
                list_age_range = [5, 15, 25, 35, 45, 55]
                # Loop over all age ranges.
                for gender in range(0, 2):
                    for i_age in range(0, len(list_age_range)-1):
                        age_left = list_age_range[i_age]
                        age_right = list_age_range[i_age+1]
                        list_sample_idx_tmp = ((df_imu_train['age'].values>=age_left) & 
                                               (df_imu_train['age'].values<age_right) & 
                                               (df_imu_train['gender'].values==gender) & 
                                               (df_imu_train['num_seq'].values==0))
                        list_sample_idx_tmp = np.random.choice(np.where(list_sample_idx_tmp)[0], 8)
                        for idx_tmp in list_sample_idx_tmp:
                            list_sample_idx = list_sample_idx + np.linspace(start=idx_tmp, stop=idx_tmp+99, num=100).tolist()
            else:
                # The same index with condition 0.
                pass
            df_imu_train = df_imu_train.loc[list_sample_idx, :]
            # Gyroscope.
            data_gyro_x_train = df_imu_train.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_gyro_y_train = df_imu_train.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_gyro_z_train = df_imu_train.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_acc_x_train = df_imu_train.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_acc_y_train = df_imu_train.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_acc_z_train = df_imu_train.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # IMU.
            data_imu_train = np.concatenate((data_gyro_x_train[:, np.newaxis, :], 
                                             data_gyro_y_train[:, np.newaxis, :], 
                                             data_gyro_z_train[:, np.newaxis, :], 
                                             data_acc_x_train[:, np.newaxis, :], 
                                             data_acc_y_train[:, np.newaxis, :], 
                                             data_acc_z_train[:, np.newaxis, :]), axis=1)
            # Groundtruth ID labels.
            list_ID_train = np.unique(df_imu_train.loc[:, ['ID']].values)
            for i_tmp in range(0, len(list_ID_train)):
                ID_tmp = list_ID_train[i_tmp]
                df_imu_train.loc[df_imu_train['ID'].values==ID_tmp, 'ID_label'] = i_tmp
            Y_train_gt = df_imu_train.loc[:, ['ID_label']].values
            # Test set.
            dir_imu_test = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                        'data_segmented', 'condition_'+str(list(set([0, 1]).difference([condition]))[0])+'.csv')
            df_imu_test = pd.read_csv(dir_imu_test)
            df_imu_test = df_imu_test.loc[list_sample_idx, :]
            # Gyroscope.
            data_gyro_x_test = df_imu_test.loc[:, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_gyro_y_test = df_imu_test.loc[:, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_gyro_z_test = df_imu_test.loc[:, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # Accelerometer.
            data_acc_x_test = df_imu_test.loc[:, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_acc_y_test = df_imu_test.loc[:, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            data_acc_z_test = df_imu_test.loc[:, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]].values
            # IMU.
            data_imu_test = np.concatenate((data_gyro_x_test[:, np.newaxis, :], 
                                            data_gyro_y_test[:, np.newaxis, :], 
                                            data_gyro_z_test[:, np.newaxis, :], 
                                            data_acc_x_test[:, np.newaxis, :], 
                                            data_acc_y_test[:, np.newaxis, :], 
                                            data_acc_z_test[:, np.newaxis, :]), axis=1)
            # Groundtruth ID labels.
            list_ID_test = np.unique(df_imu_test.loc[:, ['ID']].values)
            for i_tmp in range(0, len(list_ID_test)):
                ID_tmp = list_ID_test[i_tmp]
                df_imu_test.loc[df_imu_test['ID'].values==ID_tmp, 'ID_label'] = i_tmp
            Y_test_gt = df_imu_test.loc[:, ['ID_label']].values
            # Initialize InceptionTime model.
            model = InceptionTime.InceptionTime(c_in=6, c_out=len(np.unique(df_imu_train['ID'].values)), seq_len=300, 
                                                nf=32, nb_filters=None, ks=40, bottleneck=True)
            model.train()
            model = model.to(Params.device)
            # Initialize optimizer.
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            # Criterion.
            criterion = torch.nn.CrossEntropyLoss()
            # Start training.
            epochs = 50
            for epoch in tqdm(range(0, epochs)):
                for num_seq in range(0, 100):
                    # Zero gradient.
                    optimizer.zero_grad()
                    # Select a subset from the training & testing data.
                    idx_tmp = df_imu_train['num_seq'].values == num_seq
                    # IMU.
                    data_imu_train_tmp = data_imu_train[idx_tmp, :, :]
                    data_imu_train_tmp = torch.tensor(data_imu_train_tmp).to(Params.device)  # Move the training data to target device.
                    data_imu_test_tmp = data_imu_test[idx_tmp, :, :]
                    data_imu_test_tmp = torch.tensor(data_imu_test_tmp).to(Params.device)  # Move the testing data to target device.
                    # Groundtruth.
                    Y_train_gt_tmp = Y_train_gt[idx_tmp]
                    Y_train_gt_tmp = torch.tensor(Y_train_gt_tmp.ravel()).to(Params.device)
                    Y_test_gt_tmp = Y_test_gt[idx_tmp]
                    Y_test_gt_tmp = torch.tensor(Y_test_gt_tmp.ravel()).to(Params.device)
                    # Predicted subject ID on the training set.
                    Y_train_pred_tmp = model(data_imu_train_tmp.type(torch.float32))
                    # Compute training loss.
                    loss_train = criterion(Y_train_pred_tmp, Y_train_gt_tmp.long())
                    # Backpropagation.
                    loss_train.backward()
                    optimizer.step()
                    # Compute training accuracy.
                    Y_train_pred_tmp = torch.argmax(input=Y_train_pred_tmp, dim=1)
                    acc_train = torch.mean((Y_train_pred_tmp == Y_train_gt_tmp).float())
                    # Compute testing loss.
                    Y_test_pred_tmp = model(data_imu_test_tmp.type(torch.float32))
                    loss_test = criterion(Y_test_pred_tmp, Y_test_gt_tmp.long())
                    # Compute testing accuracy.
                    Y_test_pred_tmp = torch.argmax(input=Y_test_pred_tmp, dim=1)
                    acc_test = torch.mean((Y_test_pred_tmp == Y_test_gt_tmp).float())
            # Record prediction results.
            Y_test_gt = []
            Y_test_pred = []
            model.eval()
            for i_1 in range(0, len(np.unique(df_imu_test['ID'].values))):
                # Testing data of one subject.
                idx_tmp = (df_imu_test['ID_label'].values == i_1)
                # Ground truth label.
                Y_test_gt_tmp = df_imu_test.loc[idx_tmp, 'ID'].values.tolist()
                Y_test_gt = Y_test_gt + Y_test_gt_tmp
                # Model inference.
                data_imu_test_tmp = data_imu_test[idx_tmp, :, :]
                data_imu_test_tmp = torch.tensor(data_imu_test_tmp).to(Params.device)  # Move the testing data to target device.
                Y_test_pred_tmp = model(data_imu_test_tmp.type(torch.float32))
                Y_test_pred_tmp = torch.argmax(input=Y_test_pred_tmp, dim=1).cpu().numpy().tolist()
                # Manual label -> True label.
                for i_2 in range(0, len(Y_test_pred_tmp)):
                    ID_label_tmp = Y_test_pred_tmp[i_2]
                    ID_tmp = np.unique(df_imu_train.loc[df_imu_train['ID_label'].values==ID_label_tmp, 'ID'])[0]
                    Y_test_pred_tmp[i_2] = ID_tmp
                # Predicted label.
                Y_test_pred = Y_test_pred + Y_test_pred_tmp
            df_result = pd.concat((df_result, 
                                   pd.DataFrame({'batch': batch, 'condition': condition, 
                                                 'label_test_gt': Y_test_gt, 'label_test_pred': Y_test_pred})), axis=0)
        # Save results.
        dir_result = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, 'inception_result_bias.csv')
        df_result.to_csv(dir_result, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']  # ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        # Print current data information.
        print('Data Type: '+type_imu+'.')
        main_HumanAuthentication_inception_bias(Params=Params)