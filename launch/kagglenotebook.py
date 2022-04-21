import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
from core.networks import network_regression, train_model
from core.utils import load_bp_data

# Parameters
data_path = "../../ECG_Data"

this_dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.abspath(os.path.join(this_dir_path, "../trained_models"))

print(f"Output Path : {output_path}")

num_files_to_load = 1
max_samples_train = 1000
max_samples_test = 200
freq = 125
n_epochs = 1

# %% Data - ECG Heartbeat Categorization Dataset - https://www.kaggle.com/datasets/shayanfazeli/heartbeat
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


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers


def Model(input_dim, activation, num_class):
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim))
    model.add(Activation(activation))
    # model.add(Dropout(0.1))

    model.add(Dense(512))
    model.add(Activation(activation))
    # model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation(activation))
    # model.add(Dropout(0.25))

    model.add(Dense(num_class))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=0.002),
                  metrics=['MeanAbsoluteError']
                  )
    return model

input_dim = X_train.shape[1]
activation = 'relu'
classes = 1
model = Model(input_dim=input_dim, activation=activation, num_class=classes)
model.summary()

# Training the model
history = model.fit(X_train[:1000000, :, 0], # using the first 1million rows for speed.
                    y_train[:1000000].squeeze(),
                    epochs=20,
                    batch_size=32,
                    verbose = 1
                   )