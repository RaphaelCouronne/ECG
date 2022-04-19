import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
num_plot = 10
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


# Set to 2000 per class, except class 3 where we upsample from 600 to 2000
from sklearn.utils import resample
df_1=df_train[df_train[187]==1].sample(n=2000,random_state=42)
df_2=df_train[df_train[187]==2].sample(n=2000,random_state=42)
df_3=df_train[df_train[187]==3]
df_4=df_train[df_train[187]==4].sample(n=2000,random_state=42)
df_0=(df_train[df_train[187]==0]).sample(n=2000,random_state=42)


df_3_upsample=resample(df_3,replace=True,n_samples=2000,random_state=42)
df_train_resample = pd.concat([df_0,df_1,df_2,df_3_upsample,df_4])


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



def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.5,186)
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


def network(input_size):
    im_shape = (input_size, 1)
    inputs_cnn = Input(shape=(im_shape), name='inputs_cnn')
    conv1_1 = Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1 = Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    pool2 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
    conv3_1 = Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    pool3 = MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
    flatten = Flatten()(pool3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)

    model = Model(inputs=inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model,X_train, y_train, X_test, y_test, n_epochs=1):
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, y_train, epochs=n_epochs, callbacks=callbacks, batch_size=32,
                        validation_data=(X_test, y_test))

    model.load_weights('best_model.h5')

    return (model, history)


def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    target_names = ['0', '1', '2', '3', '4']

    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)


from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = network(X_train.shape[1])
model,history = train_model(model, X_train, y_train, X_test, y_test, n_epochs=1)
evaluate_model(history, X_test, y_test, model)
y_pred=model.predict(X_test)


#%%

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()