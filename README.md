# IMU_HardBiometrics

## Introduction
This project provides the code for the experiments of hard biometrics based on IMU gait sequences. It serves as an important part of the author's master thesis (30ECTS) in ETH Zurich.
## Summary
This project mainly comprises five parts:
1. Segmenting the original IMU gait signals into IMU sequences of equal length. (csv format)
2. Transforming the IMU sequences into hand-crafted features. (csv format)
3. Transforming the IMU sequences into autocorrelation features. (csv format)
5. Evaluating the performance of all included algorithms for hard biometrics.
6. Evaluating the underlying bias of hard biometrics models.
## Package Requirements
- matlabengine==23.2
- numpy==1.26.4
- pandas==2.2.2
- PyYAML==6.0.1
- scikit_learn==1.4.1.post1
- scipy==1.13.1
- statsmodels==0.14.1
- torch==2.2.1+cu118
- tqdm==4.66.2
- tsai==0.3.9
## Configuration
The experiments were runned on the author's personal laptop. The configurations are provided as the reference:
- CPU: AMD Ryzen 9 5900HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU
- CUDA Version: 11.7
- Operating System: Microsoft Windows 11 (version-10.0.22631)
## Code Structure
```bash
IMU_HardBiometrics
├─config
├─data
│  └─OU_ISIR_Inertial_Sensor
│      ├─manual_IMUZCenter
│      │  ├─data_segmented
│      │  ├─feature_autocorr
│      │  └─feature_manual
│      ├─manual_IMUZLeft
│      │  ├─data_segmented
│      │  ├─feature_autocorr
│      │  └─feature_manual
│      └─manual_IMUZRight
│          ├─data_segmented
│          ├─feature_autocorr
│          └─feature_manual
├─main
├─result
│  └─OU_ISIR_Inertial_Sensor
│      ├─manual_IMUZCenter
│      ├─manual_IMUZLeft
│      └─manual_IMUZRight
└─util
```
## Datasets
1. OU-ISIR Inertial Sensor Dataset: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html
2. OU-ISIR Similar Action Inertial Dataset: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/SimilarActionsInertialDB.html
## Usage
First, activate the local environment and then set the folder containing this README file as the current folder.  
For Windows, execute: **python (...).py**  
For Linux, execute: **python3 (...).py**  
1. Segment the original IMU gait signals into IMU sequences of equal length: **python "./main/main_DataSegmentation.py"**
2. Transform segmented IMU signals into hand-crafted features: **python "./main/main_IMU2ManualFeature.py"**
3. Transform segmented IMU signals into autocorrelation features: **python "./main/main_IMU2AutoCorrFeature.py"**
4. Evaluate the performance of Combined Feature + KNN for hard biometrics: **python "./main/main_HumanAuthentication_combined.py"**
5. Evaluate the performance of HYDRA for hard biometrics: **python "./main/main_HumanAuthentication_hydra.py"**
6. Evaluate the performance of InceptionTime for hard biometrics: **python "./main/main_HumanAuthentication_inception.py"**
7. Generate the recognition result of Combined Feature + KNN for bias evaluation: **python "./main/main_HumanAuthentication_combined_bias.py"**
8. Generate the recognition result of HYDRA for bias evaluation: **python "./main/main_HumanAuthentication_hydra_bias.py"**
9. Generate the recognition result of InceptionTime for bias evaluation: **python "./main/main_HumanAuthentication_inception_bias.py"**
10. Quantitatively evaluate the False Match Rate (FMR) and False Non-Match Rate (FNMR): **python "./main/main_PredictiveParity.py"**
11. Correlation analysis of age for misidentification pairs: **python "./main/main_AnalysisAge.py"**
12. Quantify misidentification pairs of same/different gender: **python "./main/main_AnalysisGender.py"**
## Example Results
### User Recognition (Training: walk-1. Testing: walk2.)
| **Baseline Method**              | **IMUZ\-Left** | **IMUZ\-Center** | **IMUZ\-Right** |
|:--------------------------------:|:--------------:|:----------------:|:---------------:|
| Combined Feature \+ KNN          | 80\.66\%      | 79\.72\%        | 75\.74\%       |
| Original Signal \+ InceptionTime | 81\.02\%      | 84\.86\%        | 78\.92\%       |
| HYDRA                            | **95\.26\%**  | **94\.85\%**    | **92\.79\%**   |
### Contact
If you have any questions, please feel free to contact me through email (shuoli199909@outlook.com)!
## Authors and acknowledgment
This master thesis was supervised by Dr. Mohamed Elgendi (ETH Zurich), Prof. Dr. Carlo Menon (ETH Zurich), and Prof. Dr. Rosa Chan (City University of Hong Kong). The code was developed by Shuo Li. Also thank to all my colleagues and providers of datasets for the continuous help!
## License - MIT License.
