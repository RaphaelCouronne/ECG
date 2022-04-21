import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')


"""
-How many rows per patients ?
"""


"""
-Adam ?
-Initialize a Network ?
-Deep in General
"""


#%% Import data

# Data
data_path = "../ECG_Data"

# dataset

# ECG Heartbeat Categorization Dataset
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat
dataset = "ECG_Heartbeat_Categorization_Dataset"
path_df_test = os.path.join(data_path, dataset, "mitbih_test.csv")
path_df_train = os.path.join(data_path, dataset, "mitbih_train.csv")
#ptbdb_abnormal.csv ptbdb_normal.csv

# From physionet
# https://www.kaggle.com/competitions/1056lab-cardiac-arrhythmia-detection/data


# Also (GitHub)
# https://github.com/awni/ecg

#%% Load data

df_train = pd.read_csv(path_df_train, header=None)
df_test = pd.read_csv(path_df_test, header=None)

#%% Plot

# Class proportion
plt.bar(df_train[187].value_counts().index, df_train[187].value_counts())
plt.show()


#%% Input

# Show examples
num_plot = 20
num_hist = 500
num_classes = 5
classes_lim = {
    0: 75,
    1: 75,
    2: 100,
    3: 75,
    4: 110,
}
fig, axes = plt.subplots(num_classes, 3, figsize=(12,12))

for i, lim in classes_lim.items():
    axes[i,0].set_title(f"Class {i}")
    df_chosenclass = df_train.loc[df_train[187] == i]
    axes[i,0].plot(df_chosenclass.iloc[:num_plot,:].T)
    axes[i,1].hist2d(np.tile(np.arange(0,188), num_hist),
                   df_chosenclass.iloc[:num_hist,:].values.reshape(-1),
                   bins=(60,10), cmap=plt.cm.jet)
    axes[i,2].hist2d(np.tile(np.arange(0,lim), num_hist),
                   df_chosenclass.iloc[:num_hist,:lim].values.reshape(-1),
                   bins=(lim,30), cmap=plt.cm.jet)
    axes[i,0].set_ylim(0,1)
plt.show()

