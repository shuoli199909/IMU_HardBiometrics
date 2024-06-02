"""
Main function of the human authentication pipeline. (Combined Feature + KNN)
"""

# Author: Shuo Li
# Date: 2024/01/09

import os
import sys
import matlab
import matlab.engine
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings.
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_data


def main_HumanAuthentication_combined(Params):
    """
    Parameters 
    ----------
    Params: The pre-defined class of parameter settings.

    Returns
    -------

    """

    # Two conditions. One for training, the other for testing.
    df_result = pd.DataFrame(columns=['condition', 'label_test_gt', 'label_test_pred'])
    for condition in range(0, 2):
        # Load data.
        # Training set.
        # Manual feature.
        dir_manual_train = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                        'feature_manual', 'condition_'+str(condition)+'.csv')
        df_manual_train = pd.read_csv(dir_manual_train)
        # Auto-correlation feature.
        dir_autocorr_train = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                          'feature_autocorr', 'condition_'+str(condition)+'.csv')
        df_autocorr_train = pd.read_csv(dir_autocorr_train)
        # Combined feature.
        df_combine_train = pd.concat([df_manual_train.drop(['condition', 'num_seq', 'age', 'gender'], axis=1), 
                                      df_autocorr_train.drop(['ID', 'condition', 'num_seq', 'age', 'gender'], axis=1)], axis=1)
        # Test set.
        # Manual feature.
        dir_manual_test = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                       'feature_manual', 'condition_'+str(list(set([0, 1]).difference([condition]))[0])+'.csv')
        df_manual_test = pd.read_csv(dir_manual_test)
        # Auto-correlation feature.
        dir_autocorr_test = os.path.join(dir_crt, 'data', Params.name_dataset, Params.type_imu, 
                                         'feature_autocorr', 'condition_'+str(list(set([0, 1]).difference([condition]))[0])+'.csv')
        df_autocorr_test = pd.read_csv(dir_autocorr_test)
        # Combined feature.
        df_combine_test = pd.concat([df_manual_test.drop(['condition', 'num_seq', 'age', 'gender'], axis=1), 
                                     df_autocorr_test.drop(['ID', 'condition', 'num_seq', 'age', 'gender'], axis=1)], axis=1)
        
        # Start training.
        # X & Y assignment for training set.
        X_train = matlab.double(df_combine_train.drop('ID', axis=1).values.tolist())
        Y_train_gt = matlab.double(df_combine_train['ID'].values.tolist())
        # Fit model.
        eng = matlab.engine.start_matlab()
        model = eng.fitcknn(X_train, Y_train_gt, 'Standardize', True)
        # X & Y assignment for test set.
        X_test = matlab.double(df_combine_test.drop('ID', axis=1).values.tolist())
        Y_test_gt = matlab.double(df_combine_test['ID'].values.tolist())
        # Model evaluation.
        Y_train_pred = np.array(eng.predict(model, X_train)).ravel()
        Y_test_pred = np.array(eng.predict(model, X_test)).ravel()
        Y_train_gt = np.array(Y_train_gt).ravel()
        Y_test_gt = np.array(Y_test_gt).ravel()
        # Record results.
        df_result = pd.concat((df_result, 
                               pd.DataFrame({'condition': condition, 'label_test_gt': Y_test_gt, 'label_test_pred': Y_test_pred})), axis=0)
    # Save results.
    dir_result = os.path.join(dir_crt, 'result', Params.name_dataset, Params.type_imu, 'combined_result.csv')
    df_result.to_csv(dir_result, index=None)


if __name__ == "__main__":
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    name_dataset = 'OU_ISIR_Inertial_Sensor'
    Params = util_data.Params(dir_option, name_dataset)
    list_type_imu = ['manual_IMUZRight', 'manual_IMUZLeft', 'manual_IMUZCenter']  # ['manual_IMUZCenter', 'manual_IMUZLeft', 'manual_IMUZRight']
    for type_imu in list_type_imu:
        Params.type_imu = type_imu
        # Print current data information.
        print('Data Type: '+type_imu+'.')
        main_HumanAuthentication_combined(Params=Params)