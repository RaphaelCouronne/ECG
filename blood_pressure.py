#%%

# importing libraries
import numpy as np # For numerical computation
import pandas as pd # Data manipulation
import seaborn as sns # plotting
import scipy.io # reading matlab files in python
from scipy import signal #signal processing
from scipy.fftpack import fft, dct #signal processing

from sklearn.linear_model import LinearRegression #linear regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split # cross validation split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt # For plotting graphs(Visualization)

import os # system-wide functions


# Data
data_path = "/Users/raphael.couronne/Programming/Perso/Data/ECG"
#data_path = "../Data"

# dataset

# ECG Heartbeat Categorization Dataset
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat
dataset = "Cuff-Less_Blood_Pressure_Estimation"

print(os.listdir(os.path.join(data_path, dataset)))

#%%

# defining our evaluation error function
def rmse(y_true, y_pred):
    """Computes the Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

sample_file = scipy.io.loadmat(os.path.join(data_path, dataset,"part_1.mat"))

print(f'sample_file Data type: {type(sample_file)}')
print(f'sample_file keys:\n{sample_file.keys()}')


#%%
test_sample = scipy.io.loadmat(os.path.join(data_path, dataset,"part_1.mat"))["p"]
print(f'sample_file Data type: {type(test_sample)}')
print(f'sample_file keys:\n{test_sample.keys()}')



#%%
# Loading a sample .mat file to understand the data dimensions
test_sample = scipy.io.loadmat(f'../input/BloodPressureDataset/part_{1}.mat')['p']
print(f'test_sample Data type: {type(test_sample)}')
print(f'test_sample shape/dimensions: {test_sample.shape}')



#%%

print(f"Total Samples: {len(test_sample[0])}")
print(f"Number of readings in each sample(column): {len(test_sample[0][0])}")
print(f"Number of samples in each reading(ECG): {len(test_sample[0][0][2])}")

temp_mat = test_sample[0, 999]
temp_length = temp_mat.shape[1]
sample_size = 125


print(temp_length)
print((int)(temp_length/sample_size))
