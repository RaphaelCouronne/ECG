import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import datasets, layers, models



def network_classif(input_dim, num_class, activation="relu",learning_rate=0.001):
    model = Sequential()

    #model.add(tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim)))

    model.add(Dense(1024, input_dim=(input_dim)))
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

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy']
                  )
    return model



def network_regr(im_shape, activation="relu", learning_rate=0.001):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 1), activation='relu', input_shape=im_shape))
    model.add(layers.MaxPooling2D((3, 1)))
    model.add(layers.Conv2D(32, (3, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    #model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['mean_squared_error']
                  )
    return model


def train_model(model, X_train, y_train, X_test, y_test, model_name, output_path,
                n_epochs=1, batch_size=16):

    filepath = os.path.join(output_path, 'best_model.{}.h5'.format(model_name))
    print(f"Save filepath in {os.path.abspath(filepath)}")

    #if not os.path.exists(output_path):
    #    print("Warning folder does not exists")

    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, y_train, epochs=n_epochs, callbacks=callbacks,
                        batch_size=batch_size, validation_data=(X_test, y_test),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_steps=X_test.shape[0] // batch_size,
                        use_multiprocessing=False,
                        verbose=1
    )

    model.load_weights(filepath)

    return (model, history)


def evaluate_model(history, X_test, y_test, model, metric="accuracy", path_out="../"):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_{}".format(metric)])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(os.path.join(path_out, "metric.jpg"))
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig(os.path.join(path_out, "loss.jpg"))
    plt.show()


def evaluate_confusion_matrix(X_test, y_test, model):
    target_names = ['0', '1', '2', '3', '4']
    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          output_path="../trained_models"):
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
    plt.savefig(os.path.join(output_path, "ecg-classif.jpg"))
