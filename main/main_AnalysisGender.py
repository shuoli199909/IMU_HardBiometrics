# Statistical analysis of gender.

# Author: Shuo Li
# Date: 2024/05/24

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')  # Ignore warnings.
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data


def main_AnalysisGender(Params):
    """Main function.
    Parameters
    ----------
    
    Returns
    -------
    Params: The pre-defined class of parameter settings.

    """

    # Subject info.
    df_info = pd.read_csv(Params.dir_info)
    # Human recognition results.
    dir_data = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, 'Manual'+'_result_bias.csv')
    df_data = pd.read_csv(dir_data)
    idx_misidentification = np.where(df_data['label_test_gt'].values != df_data['label_test_pred'].values)[0]
    idx_misidentification = np.random.choice(idx_misidentification, len(idx_misidentification), replace=False)
    num_gender_same = 0
    num_gender_different = 0
    for i1 in tqdm(range(0, len(idx_misidentification))):
        ID_gt = df_data.loc[idx_misidentification[i1], 'label_test_gt']
        ID_pred = df_data.loc[idx_misidentification[i1], 'label_test_pred']
        gender_gt = df_info.loc[df_info['ID'].values==ID_gt, 'Gender(0:Female;1:Male)'].values[0]
        gender_pred = df_info.loc[df_info['ID'].values==ID_pred, 'Gender(0:Female;1:Male)'].values[0]
        if gender_gt == gender_pred:
            num_gender_same = num_gender_same + 1
        else:
            num_gender_different = num_gender_different + 1
    print('Number of misidentifications of same gender: ', num_gender_same, 
          'Number of misidentifications of different gender: ', num_gender_different)
        


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
            main_AnalysisGender(Params=Params)