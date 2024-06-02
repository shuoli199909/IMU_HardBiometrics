# Correlation analysis of age.

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')  # Ignore warnings.
from tqdm import tqdm
from scipy import stats
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data

def main_AnalysisAge(Params):
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
    dir_data = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, Params.method+'_result_bias.csv')
    df_data = pd.read_csv(dir_data)
    idx_misidentification = np.where(df_data['label_test_gt'].values != df_data['label_test_pred'].values)[0]
    idx_misidentification = np.random.choice(idx_misidentification, int(0.5*len(idx_misidentification)), replace=False)
    list_age_gt = []
    list_age_pred = []
    data_dist =np.zeros(shape=[5, 5])
    for i1 in tqdm(range(0, len(idx_misidentification))):
        ID_gt = df_data.loc[idx_misidentification[i1], 'label_test_gt']
        ID_pred = df_data.loc[idx_misidentification[i1], 'label_test_pred']
        age_gt = df_info.loc[df_info['ID'].values==ID_gt, 'Age'].values[0]
        age_pred = df_info.loc[df_info['ID'].values==ID_pred, 'Age'].values[0]
        list_age_gt.append(age_gt)
        list_age_pred.append(age_pred)
        data_dist[int((age_gt-5)/10), int((age_pred-5)/10)] = data_dist[int((age_gt-5)/10), int((age_pred-5)/10)] + 1
    print(stats.pearsonr(list_age_gt, list_age_pred))
        


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZRight', 'manual_IMUZLeft', 'manual_IMUZCenter']  # ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    list_method = ['combined', 'hydra', 'inception']
    # Loop over all IMU locations.
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        # Loop over all baseline algorithms.
        for method in list_method:
            Params.method = method
            # Print current data information.
            print('Data Type: '+type_imu+'. '+'Algorithm: '+method+'.')
            main_AnalysisAge(Params=Params)