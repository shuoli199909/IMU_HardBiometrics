"""
Utils for data collection and pre-processing.
"""

# Author: Shuo Li
# Date: 2024/01/10

import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import robust
from statsmodels.tsa.stattools import acf


class Params():
    """Load the pre-defined parameters for preliminary analysis from a YAML file. 
       Create a class.
    """

    def __init__(self, dir_option, name_dataset) -> None:
        """Parameter calss initialization.

        Parameters
        ----------
        dir_option: Directory of the YAML file.
        name_dataset: Name of datasets. ['OU_ISIR_Inertial_Sensor', 'OU_ISIR_Similar_Activity']

        Returns
        -------

        """

        # Options.
        self.options = yaml.safe_load(open(dir_option))
        # Dataset name.
        self.name_dataset = self.options[name_dataset]['name_dataset']
        # Dataset directory.
        self.dir_dataset = self.options[name_dataset]['dir_dataset']
        # Directory of the ID-Age-Gender list.
        self.dir_info = self.options[name_dataset]['dir_info']
        # Random seed for reproducible results.
        self.seed = self.options[name_dataset]['seed']
        # IMU data type.
        self.type_imu = self.options[name_dataset]['type_imu']
        # Walking condition.
        self.condition = self.options[name_dataset]['condition']
        # Length of data segments for pre-processing.
        self.len_segment = self.options[name_dataset]['len_segment']
        # Length of moving the sliding window.
        self.len_slide = self.options[name_dataset]['len_slide']
        # Maximum time delay for auto-correlation features (number of frames).
        self.delay_max_AutoCorrFeature = self.options[name_dataset]['delay_max_AutoCorrFeature']
        # Device for DL models.
        self.device = self.options[name_dataset]['device']
        # Selected algorithm for training.
        self.method = self.options[name_dataset]['method']

class GroundTruth(): 
    """Load the groundtruth data. (gyroscope, acceleration, age, gender). 
       Create a class.
    """

    def __init__(self, Params) -> None:
        """groundtruth class initialization.

        Parameters
        ----------
        Params: The pre-defined class of parameter settings.

        Returns
        -------

        """

        # Parse the pre-defined parameters.
        self.Params = Params
    
    
    def get_GT(self, specification):
        """Get a single ground truth data.

        Parameters
        ----------
        specification: Specificy the dataset.
                       OU_ISIR_Inertial_Sensor: [type_imu, condition, num_attendant].
                                  type_imu: ['auto_IMUZCenter'-AutomaticExtractionData(IMUZCenter), 
                                             'manual_IMUZCenter'-ManualExtractionData(IMUZCenter), 
                                             'manual_IMUZLeft'-ManualExtractionData(IMUZLeft), 
                                             'manual_IMUZRight'-ManualExtractionData(IMUZRight)]
                                  condition: [0-'walk-1', 1-'walk-2'].
                                  num_attendant: [104~471437].

        Returns
        -------
        gyro_x: x axis of gyroscope data. size = [num_frames].
        gyro_y: y axis of gyroscope data. size = [num_frames].
        gyro_z: z axis of gyroscope data. size = [num_frames].
        acc_x: x axis of acceleration data. size = [num_frames].
        acc_y: y axis of acceleration data. size = [num_frames].
        acc_z: z axis of acceleration data. size = [num_frames].
        age: Age of the subject. size = [1].
        gender: Gender of the subject. 1-female, 0-male. size = [1].
        """

        if self.Params.name_dataset == 'OU_ISIR_Inertial_Sensor':  # OU_ISIR_Inertial_Sensor dataset.

            # IMU data type.
            type_imu = specification[0]
            # Condition of the walking sequence.
            condition = specification[1]
            # Serial number of the subject.
            num_attendant = specification[2]
            # Load the ID info csv.
            info_raw = pd.read_csv(self.Params.dir_info)
            # Age info.
            age = info_raw.loc[info_raw['ID']==num_attendant, 'Age'].values[0]
            # Gender info.
            gender = info_raw.loc[info_raw['ID']==num_attendant, 'Gender(0:Female;1:Male)'].values[0]
            # Load the target csv.
            if type_imu == 'auto_IMUZCenter':
                dir_data = os.path.join(self.Params.dir_dataset, 
                                        'AutomaticExtractionData_IMUZCenter', 
                                        'T0_ID'+str(num_attendant).zfill(6)+'_Center_seq'+str(condition)+'.csv')
            elif type_imu == 'manual_IMUZCenter':
                dir_data = os.path.join(self.Params.dir_dataset, 
                                        'ManualExtractionData', 'IMUZCenter', 
                                        'T0_ID'+str(num_attendant).zfill(6)+'_Walk'+str(condition+1)+'.csv')
            elif type_imu == 'manual_IMUZLeft':
                dir_data = os.path.join(self.Params.dir_dataset, 
                                        'ManualExtractionData', 'IMUZLeft', 
                                        'T0_ID'+str(num_attendant).zfill(6)+'_Walk'+str(condition+1)+'.csv')
            elif type_imu == 'manual_IMUZRight':
                dir_data = os.path.join(self.Params.dir_dataset, 
                                        'ManualExtractionData', 'IMUZRight', 
                                        'T0_ID'+str(num_attendant).zfill(6)+'_Walk'+str(condition+1)+'.csv')
            else:
                print('Wrong type of IMU data.')
                return False
            
            if os.path.exists(dir_data):
                data_raw = np.genfromtxt(fname=dir_data, delimiter=',', skip_header=2)
                # Gyroscope data.
                gyro_x = data_raw[:, 0]
                gyro_y = data_raw[:, 1]
                gyro_z = data_raw[:, 2]
                # Acceleration data.
                acc_x = data_raw[:, 3]
                acc_y = data_raw[:, 4]
                acc_z = data_raw[:, 5]
            
            else:
                gyro_x = np.nan
                gyro_y = np.nan
                gyro_z = np.nan
                acc_x = np.nan
                acc_y = np.nan
                acc_z = np.nan
            
            return gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, age, gender


def IMU2ManualFeature(data_imu):
    """Extract the corresponding features for an IMU data sequence.

    References
    ----------
    [1] Khabir, K. M., Siraj, M. S., Ahmed, M., & Ahmed, M. U. (2019, May). Prediction of gender and age from inertial sensor-based gait dataset. In 2019 Joint 8th International Conference on Informatics, Electronics & Vision (ICIEV) and 2019 3rd International Conference on Imaging, Vision & Pattern Recognition (icIVPR) (pp. 371-376). IEEE.
    [2] Pathan, R. K., Uddin, M. A., Nahar, N., Ara, F., Hossain, M. S., & Andersson, K. (2020, December). Gender classification from inertial sensor-based gait dataset. In International Conference on Intelligent Computing & Optimization (pp. 583-596). Cham: Springer International Publishing.

    Parameters
    ----------
    data_imu: IMU sequence data. Numpy array. size = [6(gx, gy, gz, ax, ay, az), length].

    Returns
    -------
    series_feature_manual: The corresponding features extracted from the IMU sequence. Data type = Pandas.Series.
    """

    # Condition of the walking sequence.
    series_feature_manual = pd.Series([])
    # Mean.
    series_feature_manual['mean_gx'] = np.mean(data_imu[0, :])  # gx.
    series_feature_manual['mean_gy'] = np.mean(data_imu[1, :])  # gy.
    series_feature_manual['mean_gz'] = np.mean(data_imu[2, :])  # gz.
    series_feature_manual['mean_ax'] = np.mean(data_imu[3, :])  # ax.
    series_feature_manual['mean_ay'] = np.mean(data_imu[4, :])  # ay.
    series_feature_manual['mean_az'] = np.mean(data_imu[5, :])  # az.
    # Medium.
    series_feature_manual['median_gx'] = np.median(data_imu[0, :])  # gx.
    series_feature_manual['median_gy'] = np.median(data_imu[1, :])  # gy.
    series_feature_manual['median_gz'] = np.median(data_imu[2, :])  # gz.
    series_feature_manual['median_ax'] = np.median(data_imu[3, :])  # ax.
    series_feature_manual['median_ay'] = np.median(data_imu[4, :])  # ay.
    series_feature_manual['median_az'] = np.median(data_imu[5, :])  # az.
    # Maximum.
    series_feature_manual['max_gx'] = np.max(data_imu[0, :])  # gx.
    series_feature_manual['max_gy'] = np.max(data_imu[1, :])  # gy.
    series_feature_manual['max_gz'] = np.max(data_imu[2, :])  # gz.
    series_feature_manual['max_ax'] = np.max(data_imu[3, :])  # ax.
    series_feature_manual['max_ay'] = np.max(data_imu[4, :])  # ay.
    series_feature_manual['max_az'] = np.max(data_imu[5, :])  # az.
    # Minimum.
    series_feature_manual['min_gx'] = np.min(data_imu[0, :])  # gx.
    series_feature_manual['min_gy'] = np.min(data_imu[1, :])  # gy.
    series_feature_manual['min_gz'] = np.min(data_imu[2, :])  # gz.
    series_feature_manual['min_ax'] = np.min(data_imu[3, :])  # ax.
    series_feature_manual['min_ay'] = np.min(data_imu[4, :])  # ay.
    series_feature_manual['min_az'] = np.min(data_imu[5, :])  # az.
    # Mean.
    series_feature_manual['mean_gx'] = np.mean(data_imu[0, :])  # gx.
    series_feature_manual['mean_gy'] = np.mean(data_imu[1, :])  # gy.
    series_feature_manual['mean_gz'] = np.mean(data_imu[2, :])  # gz.
    series_feature_manual['mean_ax'] = np.mean(data_imu[3, :])  # ax.
    series_feature_manual['mean_ay'] = np.mean(data_imu[4, :])  # ay.
    series_feature_manual['mean_az'] = np.mean(data_imu[5, :])  # az.
    # Median absolute deviation.
    series_feature_manual['mad_gx'] = robust.mad(data_imu[0, :])  # gx.
    series_feature_manual['mad_gy'] = robust.mad(data_imu[1, :])  # gy.
    series_feature_manual['mad_gz'] = robust.mad(data_imu[2, :])  # gz.
    series_feature_manual['mad_ax'] = robust.mad(data_imu[3, :])  # ax.
    series_feature_manual['mad_ay'] = robust.mad(data_imu[4, :])  # ay.
    series_feature_manual['mad_az'] = robust.mad(data_imu[5, :])  # az.
    # Standard error of the mean.
    series_feature_manual['sem_gx'] = stats.sem(data_imu[0, :])  # gx.
    series_feature_manual['sem_gy'] = stats.sem(data_imu[1, :])  # gy.
    series_feature_manual['sem_gz'] = stats.sem(data_imu[2, :])  # gz.
    series_feature_manual['sem_ax'] = stats.sem(data_imu[3, :])  # ax.
    series_feature_manual['sem_ay'] = stats.sem(data_imu[4, :])  # ay.
    series_feature_manual['sem_az'] = stats.sem(data_imu[5, :])  # az.
    # Skewness.
    series_feature_manual['skew_gx'] = stats.skew(data_imu[0, :])  # gx.
    series_feature_manual['skew_gy'] = stats.skew(data_imu[1, :])  # gy.
    series_feature_manual['skew_gz'] = stats.skew(data_imu[2, :])  # gz.
    series_feature_manual['skew_ax'] = stats.skew(data_imu[3, :])  # ax.
    series_feature_manual['skew_ay'] = stats.skew(data_imu[4, :])  # ay.
    series_feature_manual['skew_az'] = stats.skew(data_imu[5, :])  # az.
    # Kurtosis.
    series_feature_manual['kurt_gx'] = stats.kurtosis(data_imu[0, :])  # gx.
    series_feature_manual['kurt_gy'] = stats.kurtosis(data_imu[1, :])  # gy.
    series_feature_manual['kurt_gz'] = stats.kurtosis(data_imu[2, :])  # gz.
    series_feature_manual['kurt_ax'] = stats.kurtosis(data_imu[3, :])  # ax.
    series_feature_manual['kurt_ay'] = stats.kurtosis(data_imu[4, :])  # ay.
    series_feature_manual['kurt_az'] = stats.kurtosis(data_imu[5, :])  # az.
    # Standard deviation.
    series_feature_manual['std_gx'] = np.std(data_imu[0, :])  # gx.
    series_feature_manual['std_gy'] = np.std(data_imu[1, :])  # gy.
    series_feature_manual['std_gz'] = np.std(data_imu[2, :])  # gz.
    series_feature_manual['std_ax'] = np.std(data_imu[3, :])  # ax.
    series_feature_manual['std_ay'] = np.std(data_imu[4, :])  # ay.
    series_feature_manual['std_az'] = np.std(data_imu[5, :])  # az.
    # Variance.
    series_feature_manual['var_gx'] = np.var(data_imu[0, :])  # gx.
    series_feature_manual['var_gy'] = np.var(data_imu[1, :])  # gy.
    series_feature_manual['var_gz'] = np.var(data_imu[2, :])  # gz.
    series_feature_manual['var_ax'] = np.var(data_imu[3, :])  # ax.
    series_feature_manual['var_ay'] = np.var(data_imu[4, :])  # ay.
    series_feature_manual['var_az'] = np.var(data_imu[5, :])  # az.
    # Root mean square.
    series_feature_manual['rms_gx'] = np.sqrt(np.mean(np.square(data_imu[0, :])))  # gx.
    series_feature_manual['rms_gy'] = np.sqrt(np.mean(np.square(data_imu[1, :])))  # gy.
    series_feature_manual['rms_gz'] = np.sqrt(np.mean(np.square(data_imu[2, :])))  # gz.
    series_feature_manual['rms_ax'] = np.sqrt(np.mean(np.square(data_imu[3, :])))  # ax.
    series_feature_manual['rms_ay'] = np.sqrt(np.mean(np.square(data_imu[4, :])))  # ay.
    series_feature_manual['rms_az'] = np.sqrt(np.mean(np.square(data_imu[5, :])))  # az.
    # Vector sum.
    vs_gxzy = np.sum(a=data_imu[0:3, :], axis=1)  # gxyz.
    series_feature_manual['vs_gx'] = vs_gxzy[0]  # gx.
    series_feature_manual['vs_gy'] = vs_gxzy[1]  # gy.
    series_feature_manual['vs_gz'] = vs_gxzy[2]  # gz.
    vs_axzy = np.sum(a=data_imu[3:6, :], axis=1)  # axyz.
    series_feature_manual['vs_ax'] = vs_axzy[0]  # ax.
    series_feature_manual['vs_ay'] = vs_axzy[1]  # ay.
    series_feature_manual['vs_az'] = vs_axzy[2]  # az.

    return series_feature_manual


def IMU2AutoCorrFeature(data_imu, delay_max=10):
    """Extract the autocorrelation features from the IMU data sequence.

    References
    ----------
    [1] Mostafa, A., Elsagheer, S. A., & Gomaa, W. (2021). BioDeep: A Deep Learning System for IMU-based Human Biometrics Recognition. In ICINCO (pp. 620-629).

    Parameters
    ----------
    data_imu: IMU sequence data. Numpy array. size = [6(gx, gy, gz, ax, ay, az), length].
    delay_max: Maximum time delay (number of frames). size = [1].

    Returns
    -------
    series_feature_autocorr: The corresponding autocorrelation feature extracted from the IMU sequence. Data type = Pandas.Series.
    """
    
    # Initialization of autocorrelation feature.
    feature = []
    # Loop over all IMU dimensions.
    for i_dim in range(0, data_imu.shape[0]):
        # Compute auto correlation features.
        data_imu_tmp = data_imu[i_dim]
        feature.append(acf(data_imu_tmp, nlags=delay_max, fft=True)[1:])

    # Reshape into 1-dimensional feature.
    feature = np.ravel(feature)
    # Save as Pandas Series.
    series_feature_autocorr = pd.Series(data=feature, index=['f_'+str(i_f) for i_f in range(0, len(feature))])

    return series_feature_autocorr

    """Split the dataset in a balanced way. Make sure the age and gender are balanced in each fold.

    Parameters
    ----------
    list_idx: A list or 1D Numpy array of the subject index. size = [num_subject].
    list_age: A list or 1D Numpy array of the subject age. size = [num_subject].
    list_gender: A list or 1D Numpy array of the subject gender. size = [num_subject].
    range_age: A range list or 2D Numpy array for the age classification task. size = [num_class_age].
    num_fold: N-fold (num_fold) cross validation. size = [1].

    Returns
    -------
    list_idx_balance: A list of balanced subject index. size = [num_fold].
    """
    
    # List -> Numpy array.
    list_idx = np.array(list_idx)  # Subject index.
    list_age = np.array(list_age)  # Age.
    list_gender = np.array(list_gender)  # Gender.
    range_age = np.array(range_age)  # Age range.
    num_fold = np.array(num_fold)  # Number of folds.
    # Create index-age-gender dataframe.
    df_info = pd.DataFrame({'index': list_idx, 'age': list_age, 'gender': list_gender})
    # Initialize the balanced subject index list.
    list_idx_balance = [[] for _ in range(0, num_fold)]
    # Fill the list.
    state_empty = False
    while state_empty == False:
        # Loop over all folds.
        for fold in range(0, num_fold):
            # Loop over all age ranges.
            for i_age in range(0, len(range_age)-1):
                # Loop over all genders.
                for gender in range(0, 2):
                    idx_crt = df_info.loc[((df_info['age'].between(left=range_age[i_age], right=range_age[i_age+1], inclusive='left')) & 
                                          (df_info['gender'] == gender)), 'index'].values
                    if len(idx_crt) == 0:  # Empty data. Break out of the while loop.
                        state_empty = True
                        break
                    else:
                        idx_crt = idx_crt[0]
                        df_info = df_info.drop(index=np.ravel(np.argwhere((df_info['index']==idx_crt).values)), axis=0)
                        df_info.reset_index(inplace=True, drop=True)
                        list_idx_balance[fold].append(idx_crt)
                if state_empty == True:  # Empty data. Break out of the while loop.
                    break
                else:
                    continue
            if state_empty == True:  # Empty data. Break out of the while loop.
                break
            else:
                continue
    
    return list_idx_balance