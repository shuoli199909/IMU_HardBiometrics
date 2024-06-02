# Predictive parity analysis.

# Author: Shuo Li
# Date: 2024/05/23

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')  # Ignore warnings.
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data
from tqdm import tqdm


def main_PredictiveParity(Params):
    """Main function.
    Parameters
    ----------
    Params: The pre-defined class of parameter settings.
    
    Returns
    -------

    """

    # Subject info.
    df_info = pd.read_csv(Params.dir_info)
    # Human recognition results.
    dir_result = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_result_bias.csv')
    df_result = pd.read_csv(dir_result)
    # Fairness evaluation metrics.
    df_metric_pp = pd.DataFrame(columns=['batch', 'gender', 'age_left', 'age_right', 'FMR', 'FNMR'])
    # Age range.
    list_age_range = [5, 15, 25, 35, 45, 55]
    list_gender = [0, 1]
    # Loop over all batches
    for batch in tqdm(np.sort(np.unique(df_result['batch'].values))):
        df_result_batch = df_result.loc[df_result['batch'].values==batch, :]
        df_result_batch.reset_index(inplace=True)
        df_result_batch = df_result_batch.drop(columns=['index', 'batch', 'condition'])
        # Loop over female and male subjects.
        for gender in list_gender:
            # Loop over all age ranges.
            for i_age in range(0, len(list_age_range)-1):
                # Current age range.
                age_left = list_age_range[i_age]
                age_right = list_age_range[i_age+1]
                list_ID_tmp = df_info.loc[(df_info['Gender(0:Female;1:Male)'].values==gender) & 
                                          (df_info['Age'].values>=age_left) & 
                                          (df_info['Age'].values<age_right), 'ID'].values
                df_result_tmp = df_result_batch.loc[np.in1d(df_result_batch['label_test_gt'].values, list_ID_tmp), :]
                list_FMR = []
                list_FNMR = []
                for label_test_gt_tmp in np.unique(df_result_tmp['label_test_gt'].values):
                    # False Match Rate (FMR).
                    FMR_tmp = np.sum((df_result_batch['label_test_gt'].values!=label_test_gt_tmp) & 
                                     (df_result_batch['label_test_pred'].values==label_test_gt_tmp))/\
                              np.sum(df_result_batch['label_test_gt'].values!=label_test_gt_tmp)
                    list_FMR.append(FMR_tmp)
                    # False Non-Match Rate (FNMR).
                    FNMR_tmp = np.sum((df_result_batch['label_test_gt'].values==label_test_gt_tmp) & 
                                     (df_result_batch['label_test_pred'].values!=label_test_gt_tmp))/\
                              np.sum(df_result_batch['label_test_gt'].values==label_test_gt_tmp)
                    list_FNMR.append(FNMR_tmp)
                    
                # Record results.
                df_metric_pp.loc[len(df_metric_pp)] = [batch, gender, age_left, age_right, 
                                                       np.mean(list_FMR), np.mean(list_FNMR)]
    # Summary statistics.
    # Loop over female and male subjects.
    for gender in list_gender:
        # Loop over all age ranges.
        for i_age in range(0, len(list_age_range)-1):
            # Current age range.
            age_left = list_age_range[i_age]
            age_right = list_age_range[i_age+1]
            # Record results.
            df_metric_pp.loc[len(df_metric_pp)] = ['Average', gender, age_left, age_right, 
                                                   np.mean(df_metric_pp.loc[(df_metric_pp['gender'].values==gender) & 
                                                                            (df_metric_pp['age_left'].values==age_left) & 
                                                                            (df_metric_pp['age_right'].values==age_right), 'FMR']), 
                                                   np.mean(df_metric_pp.loc[(df_metric_pp['gender'].values==gender) & 
                                                                            (df_metric_pp['age_left'].values==age_left) & 
                                                                            (df_metric_pp['age_right'].values==age_right), 'FNMR'])]
    dir_metric_pp = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_bias.csv')
    df_metric_pp.to_csv(dir_metric_pp, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZRight', 'manual_IMUZLeft', 'manual_IMUZCenter']  #, 'manual_IMUZRight'] #, 'manual_IMUZCenter']  #, ]  # ['auto_IMUZCenter', 'manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_method = ['combined', 'hydra', 'inception']
    # Loop over all IMU locations.
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        # Loop over all baseline algorithms.
        for method in list_method:
            Params.method = method
            # Print current data information.
            print('Data Type: '+type_imu+'. '+'Algorithm: '+method+'.')
            main_PredictiveParity(Params=Params)
    