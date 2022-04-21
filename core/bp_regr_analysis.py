"""
https://pubmed.ncbi.nlm.nih.gov/31156106/

Ressources : https://www.youtube.com/watch?v=tZLotOFiyZ4
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6696196/ --> Personalization (overfitting of individual)

    Preprocessing : do we have to split data with peak detection algorithms ?

    What do you predict ? BP ? or SBP and DBP ? If so, in what frames ?
    Do we use spectograms (eg 5s) in DL

    Models :
        -ARMA
        -Standard ML
        -Deep L

    Library for time series : Skits or other ?
    https://www.ethanrosenthal.com/2018/03/22/time-series-for-scikit-learn-people-part2/

    Deep Learning

    lag between ECG et PPG --> informative

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
from core.utils import load_bp_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
import os

#%% parameters
num_files_to_load = 1

# %% Data - ECG Heartbeat Categorization Dataset - https://www.kaggle.com/datasets/shayanfazeli/heartbeat
data_path = "/Users/raphael.couronne/Programming/Perso/Data/ECG"
dataset = "Cuff-Less_Blood_Pressure_Estimation"
bp_path = os.path.join(data_path, dataset)
data = load_bp_data(bp_path, num_files_to_load)

#%% Data visualization

begin = 60900
end = 61300

names = ["BP", "PPG", "ECG"]


fig, axes = plt.subplots(3, 1)
for j, name in zip(range(3), names):
    axes[j].plot(data[begin:end, j])
    axes[j].set_title(name)
    peaks = scipy.signal.find_peaks(-data[begin:end, j])[0]

    # Find peaks
    axes[j].vlines(peaks,
                   ymin=min(data[begin:end, j]),
                   ymax=max(data[begin:end, j]),
                   color="red")
plt.show()

#%% Cross correlation


begin, end = 11200, 12000
correlation = scipy.signal.correlate(data[begin:end, 0],
                                     data[begin:end, 1], mode="full")
lags = scipy.signal.correlation_lags(data[begin:end, 0].size,
                                     data[begin:end, 1].size, mode="full")
lag_bp_ppg = lags[np.argmax(correlation)]

# Cross correlations
fig, axes = plt.subplots(2, 1, sharex=True)
correlation_bp_ppg = scipy.signal.correlate(data[begin:end,0],
                                            data[begin:end,1], mode="full")
correlation_ecg_ppg = scipy.signal.correlate(data[begin:end,2],
                                             data[begin:end,1], mode="full")
lag_ecg_ppg = lags[np.argmax(correlation_ecg_ppg)]

axes[0].plot(correlation_bp_ppg)
axes[0].set_title(lag_bp_ppg)
axes[1].plot(correlation_ecg_ppg)
axes[1].set_title(lag_ecg_ppg)
plt.show()

# Corrected plots
_, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(data[begin:end, 0])
axes[1].set_title("lag {}".format(lag_bp_ppg))
axes[1].plot(np.arange(0, end-begin)-lag_bp_ppg, data[begin:end, 1])
axes[2].set_title("lag {}".format(lag_ecg_ppg))
axes[2].plot(np.arange(0, end-begin)+lag_ecg_ppg, data[begin:end, 2])
plt.show()


#%% ML

"""
Task : 
    -ECG, PPG (125Hz) -> BP (12hz)

Conclusion : 
    -Not sufficient, have to take into account more info (a window ?)
"""


# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0],
                                                    test_size=0.50, shuffle=False)

#%% Model with LR

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# See on test data
begin, end = 1100, 1500
_, ax = plt.subplots(1,1)
ax.plot(model.predict(X=X_test[begin:end]), c="red")
ax.plot(y_test[begin:end])
plt.show()