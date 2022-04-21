import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from core.networks import network_classification, train_model, evaluate_model, plot_confusion_matrix, evaluate_confusion_matrix

"""
Q abut the data ?
-How many rows per patients ?

INPUT DATA PROCESSING
-take subset | balance classes | weight classes
-data augmentation change ?

NETWORK MODIFICATIONS ?
"""

def launch_ecg_classif(data_path = "../ECG_Data",  output_path="../trained_models", model_name= "ecg-classif", max_samples_train=1000, max_samples_test=200, n_epochs=1):


    #%% Import data
    # ECG Heartbeat Categorization Dataset
    # https://www.kaggle.com/datasets/shayanfazeli/heartbeat

    dataset = "ECG_Heartbeat_Categorization_Dataset"
    print(f"Getting data at : {os.path.abspath(os.path.join(data_path, dataset))}")
    path_df_test = os.path.join(data_path, dataset, "mitbih_test.csv")
    path_df_train = os.path.join(data_path, dataset, "mitbih_train.csv")
    df_train = pd.read_csv(path_df_train, header=None)
    df_test = pd.read_csv(path_df_test, header=None)
    
    #%% Prepare Data for Deep Learning
    max_samples_train_fifth = int(max_samples_train/5)

    # Set to 2000 per class, except class 3 where we upsample from 600 to 2000
    from sklearn.utils import resample
    df_1=df_train[df_train[187] == 1]
    df_2=df_train[df_train[187] == 2]
    df_3=df_train[df_train[187] == 3]
    df_4=df_train[df_train[187] == 4]
    df_0=(df_train[df_train[187] == 0]).sample(n=max_samples_train_fifth, random_state=42)
    df_1_upsample = resample(df_1, replace=True, n_samples=max_samples_train_fifth, random_state=123)
    df_2_upsample = resample(df_2, replace=True, n_samples=max_samples_train_fifth, random_state=124)
    df_3_upsample = resample(df_3, replace=True, n_samples=max_samples_train_fifth, random_state=125)
    df_4_upsample = resample(df_4, replace=True, n_samples=max_samples_train_fifth, random_state=126)
    
    df_train_resample = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])
    
    def add_gaussian_noise(signal):
        noise = np.random.normal(0, 0.1, 186)
        return signal+noise
    
    target_train = df_train_resample[187]
    target_test = df_test[187]
    y_train = to_categorical(target_train)
    y_test = to_categorical(target_test)
    X_train = df_train_resample.iloc[:, :186].values
    X_test = df_test.iloc[:, :186].values
    
    # Data Augmentation
    for i in range(len(X_train)):
        X_train[i, :186] = add_gaussian_noise(X_train[i,:186])
    X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
    X_test = X_test[:max_samples_test, :]
    y_test = y_test[:max_samples_test]
    
    
    #%% Train Network
    model = network_classification(X_train.shape[1], 1, dim_out=5)
    model, history = train_model(model, X_train, y_train, X_test, y_test, output_path=output_path, model_name=model_name, n_epochs=n_epochs)

    """
    #evaluate_model(history, X_test, y_test, model)
    #evaluate_confusion_matrix(X_test, y_test, model)
    y_pred=model.predict(X_test)
    
    #%% Check results
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'], normalize=True,
                          title='Confusion matrix, with normalization', output_path=output_path)
    """