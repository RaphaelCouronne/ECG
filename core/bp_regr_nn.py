import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
from core.networks import network_regression, train_model
from core.utils import load_bp_data


# %% Deep Learning

"""
Ressources :
    -Estimating Systolic Blood Pressure Using Convolutional Neural Networks  https://pubmed.ncbi.nlm.nih.gov/31156106/
    -Systolic Blood Pressure Estimation from Electrocardiogram and Photo Plethysmogram Signals Using Convolutional Neural Networks

Task : 
    -ECG, PPG (125Hz) -> SBP (1hz)

Research :
    -Better windows split ? (according to peaks ?)
"""


def launch_bp_regression(data_path="../ECG_Data", output_path= "../trained_models", model_name="sbp_regression", num_files_to_load=1, freq=125, max_samples_train=1000, max_samples_test=200,n_epochs=1):

    #%% Data - ECG Heartbeat Categorization Dataset - https://www.kaggle.com/datasets/shayanfazeli/heartbeat
    dataset = "Cuff-Less_Blood_Pressure_Estimation"
    bp_path = os.path.join(data_path, dataset)

    print("Getting data at : ", os.path.abspath(bp_path))

    data = load_bp_data(bp_path, num_files_to_load)


    # %% Dataset Preparation for Deep Learning Task
    num_windows = int(data.shape[0] / freq)

    X = np.zeros(shape=(num_windows, freq, 2))
    y = np.zeros(shape=(num_windows))

    for i in range(num_windows):
        y[i] = min(data[freq * i:freq * (i + 1), 0])
        X[i] = data[freq * i:freq * (i + 1), 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    X_train = X_train[:max_samples_train]
    y_train = y_train[:max_samples_train]
    X_test = X_test[:max_samples_test]
    y_test = y_test[:max_samples_test]


    #%% Launch Network training
    model = network_regression(X_train.shape[1], X_train.shape[2])
    model, history = train_model(model, X_train, y_train, X_test, y_test,
                                 model_name=model_name, n_epochs=n_epochs, output_path=output_path)

    #%% Look at the results
    fig, axes = plt.subplots(2,1)
    y_pred = model.predict(X_train[:300])
    axes[0].set_title("Train set")
    axes[0].plot(y_pred)
    axes[0].plot(y_train[:300])
    y_pred = model.predict(X_test[:300])
    axes[1].plot(y_pred)
    axes[1].plot(y_test[:300])
    axes[1].set_title("Test set")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,"{}_pred.jpg".format(model_name)))
    plt.show()



