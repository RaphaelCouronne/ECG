import os
from core.utils import download_kaggle

if not os.path.exists('../ECG_Data/ECG_Heartbeat_Categorization_Dataset'):
    print("Download dataset shayanfazeli/heartbeat")
    download_kaggle("shayanfazeli/heartbeat", '../ECG_Data/ECG_Heartbeat_Categorization_Dataset')

if not os.path.exists('../ECG_Data/Cuff-Less_Blood_Pressure_Estimation'):
    print("Download dataset mkachuee/BloodPressureDataset")
    download_kaggle("mkachuee/BloodPressureDataset", '../ECG_Data/Cuff-Less_Blood_Pressure_Estimation')