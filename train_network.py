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
from sklearn.utils import resample


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

data_path = "/Users/raphael.couronne/Programming/Perso/Data/ECG"
#data_path = "../Data"


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


#%% Tensorflow


"""
TODO : 

INPUT DATA PROCESSING
-take subset | balance classes | weight classes
-data augmentation ?

NETWORK MODIFICATIONS ?
-
-

"""

n_samples_train = 20000
n_samples_test = 3000

# Set to 2000 per class, except class 3 where we upsample from 600 to 2000

from sklearn.utils import resample
df_1=df_train[df_train[187]==1]
df_2=df_train[df_train[187]==2]
df_3=df_train[df_train[187]==3]
df_4=df_train[df_train[187]==4]
df_0=(df_train[df_train[187]==0]).sample(n=n_samples_train,random_state=42)
df_1_upsample=resample(df_1,replace=True,n_samples=n_samples_train,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=n_samples_train,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=n_samples_train,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=n_samples_train,random_state=126)

df_train_resample=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])


def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.1,186)
    return (signal+noise)

target_train=df_train_resample[187]
target_test=df_test[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)
X_train=df_train_resample.iloc[:,:186].values
X_test=df_test.iloc[:,:186].values

# Data Augmentation
for i in range(len(X_train)):
    X_train[i,:186]= add_gaussian_noise(X_train[i,:186])
X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)


X_test = X_test[:n_samples_test,:]
y_test = y_test[:n_samples_test]

#%% Train Network

from networks import network, train_model, evaluate_model, plot_confusion_matrix

model = network(X_train.shape[1])
model, history = train_model(model, X_train, y_train, X_test, y_test, n_epochs=10)
evaluate_model(history, X_test, y_test, model)
y_pred=model.predict(X_test)


#%%

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()