"""
Data segmentation for the human authentication pipeline.
"""

# Author: Shuo Li
# Date: 2024/01/09

import os
import sys
import tqdm
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings.
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data


def main_DataSegmentation(Params):
    """
    Parameters 
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # load data.
    GT = util_data.GroundTruth(Params=Params)
    # Load subject demographic information.
    df_info = pd.read_csv(Params.dir_info)
    # Random seed.
    df_info = df_info.sample(frac=1, random_state=Params.seed)
    # Data selection. Loop over all potential candidates.
    list_ID = []
    list_gender = []
    list_age = []
    print('Selecting data...')
    for i_info in tqdm.tqdm(range(0, len(df_info))):
        # Subject ID.
        ID = df_info.loc[i_info, 'ID']
        # Collect IMU data and groundtruth.
        gyro_x_0, gyro_y_0, gyro_z_0, acc_x_0, acc_y_0, acc_z_0, age, gender = GT.get_GT(specification=[Params.type_imu, 0, ID])
        gyro_x_1, gyro_y_1, gyro_z_1, acc_x_1, acc_y_1, acc_z_1, age, gender = GT.get_GT(specification=[Params.type_imu, 1, ID])
        if (np.isnan(gyro_x_0).any()) or (np.isnan(gyro_x_1).any()) == True:
            continue
        elif (len(gyro_x_0) < Params.len_segment+99) or (len(gyro_x_1) < Params.len_segment+99):
            continue
        else:
            list_ID.append(ID)
            list_gender.append(gender)
            list_age.append(age)
    # Data balancing across different age and gender.
    list_idx_female = (np.array(list_gender) == 0)
    list_idx_male = (np.array(list_gender) == 1)
    list_ID_filtered = []
    for age_tmp in tqdm.tqdm(np.sort(np.unique(list_age))):
        list_idx_age_tmp = (list_age == age_tmp)
        list_idx_female_tmp = (list_idx_female & list_idx_age_tmp)
        list_idx_male_tmp = (list_idx_male & list_idx_age_tmp)
        # All female subjects. 
        list_ID_female = np.array(list_ID)[list_idx_female_tmp]
        # All male subjects.
        list_ID_male = np.array(list_ID)[list_idx_male_tmp]
        # Selected subjects.
        if len(list_ID_female) >= len(list_ID_male):
            list_ID_filtered = list_ID_filtered + (list_ID_male.tolist()+list_ID_female[:len(list_ID_male)].tolist())
        else:
            list_ID_filtered = list_ID_filtered + (list_ID_female.tolist()+list_ID_male[:len(list_ID_female)].tolist())
    # Original signal segments.
    df_imu = pd.DataFrame(columns=['ID', 'condition', 'num_seq', 'age', 'gender'], index=[0])
    df_imu = pd.concat((
        df_imu, 
        pd.DataFrame(columns=['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Gyroscope x-axis.
        pd.DataFrame(columns=['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Gyroscope y-axis.
        pd.DataFrame(columns=['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer z-axis.
        pd.DataFrame(columns=['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer x-axis.
        pd.DataFrame(columns=['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0]),  # Accelerometer y-axis.
        pd.DataFrame(columns=['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)], index=[0])  # Accelerometer z-axis.
        ), ignore_index=True)
    df_imu.drop(df_imu.index, inplace=True)
    # Manual features.
    df_feature_manual = pd.DataFrame([])
    # Auto correlation features.
    df_feature_autocorr = pd.DataFrame([])
    # Loop over all subjects for training.
    print('Data transformation...')
    for idx in tqdm.tqdm(range(0, len(list_ID_filtered))):
        ID_tmp = list_ID_filtered[idx]
        # Collect IMU data and groundtruth.
        gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, age_tmp, gender_tmp = GT.get_GT(specification=[Params.type_imu, Params.condition, ID_tmp])
        for i_start in range(0, 100):
            # Current data collection.
            gyro_x_tmp = gyro_x[i_start:i_start+Params.len_segment]
            gyro_y_tmp = gyro_y[i_start:i_start+Params.len_segment]
            gyro_z_tmp = gyro_z[i_start:i_start+Params.len_segment]
            acc_x_tmp = acc_x[i_start:i_start+Params.len_segment]
            acc_y_tmp = acc_y[i_start:i_start+Params.len_segment]
            acc_z_tmp = acc_z[i_start:i_start+Params.len_segment]
            data_imu = np.row_stack((gyro_x_tmp, gyro_y_tmp, gyro_z_tmp, acc_x_tmp, acc_y_tmp, acc_z_tmp))
            # Original IMU data segment.
            df_imu_tmp = pd.DataFrame(columns=df_imu.columns.values.tolist())
            df_imu_tmp.loc[0, ['gyro_x_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_x_tmp
            df_imu_tmp.loc[0, ['gyro_y_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_y_tmp
            df_imu_tmp.loc[0, ['gyro_z_'+str(i_f) for i_f in range(0, Params.len_segment)]] = gyro_z_tmp
            df_imu_tmp.loc[0, ['acc_x_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_x_tmp
            df_imu_tmp.loc[0, ['acc_y_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_y_tmp
            df_imu_tmp.loc[0, ['acc_z_'+str(i_f) for i_f in range(0, Params.len_segment)]] = acc_z_tmp
            df_imu_tmp.loc[0, ['ID', 'condition', 'num_seq', 'age', 'gender']] = [ID_tmp, Params.condition, i_start, age_tmp, gender_tmp]
            df_imu = pd.concat((df_imu, df_imu_tmp), ignore_index=True)
            # Manual feature.
            feature_manual = util_data.IMU2ManualFeature(data_imu)
            feature_manual = pd.concat((pd.Series({'ID': ID_tmp, 'condition': Params.condition, 
                                                   'num_seq': i_start, 'age': age_tmp, 'gender': gender_tmp}), feature_manual))
            df_feature_manual = pd.concat((df_feature_manual, feature_manual.to_frame().T), ignore_index=True)
            # Auto correlation feature.
            feature_autocorr = util_data.IMU2AutoCorrFeature(data_imu)
            feature_autocorr = pd.concat((pd.Series({'ID': ID_tmp, 'condition': Params.condition, 
                                                     'num_seq': i_start, 'age': age_tmp, 'gender': gender_tmp}), feature_autocorr))
            df_feature_autocorr = pd.concat((df_feature_autocorr, feature_autocorr.to_frame().T), ignore_index=True)
        # Save dataframes.
        df_imu.to_csv(os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                   'data_segmented', 'conditionn_'+str(Params.condition)+'.csv'), index=None)
        df_feature_manual.to_csv(os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                   'feature_manual', 'conditionn_'+str(Params.condition)+'.csv'), index=None)
        df_feature_autocorr.to_csv(os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                   'feature_autocorr', 'conditionn_'+str(Params.condition)+'.csv'), index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']  # ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_condition = [0, 1]
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        for condition in list_condition:
            Params.condition = condition
            # Print current data information.
            print('Data Type: '+type_imu+'. Walk Condition: '+str(condition)+'.')
            main_DataSegmentation(Params=Params)
