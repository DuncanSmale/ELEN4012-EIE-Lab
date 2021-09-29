import os
from tensorflow import keras
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import LeakyReLU
from keras.constraints import maxnorm
from sklearn.metrics import accuracy_score
import numpy as numpy
from numpy import mean
from numpy import std
import tensorflow as tf
from tqdm.keras import TqdmCallback
from matplotlib import pyplot

# tf.config.set_visible_devices([], 'GPU')

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_dataset():
    X = numpy.loadtxt('../dataNaive.txt', delimiter=",")
    y = numpy.loadtxt('../messagesNaive.txt', delimiter=",")
    print(X.shape, y.shape)
#     print(X[:, 0:100] == y[:, :])
    return X, y


def get_model(n_inputs, n_ouputs):
    initializer = tf.keras.initializers.GlorotNormal()
    num_hidden = 2
    drop = 0.3
    # dense = int(n_inputs*(1+2/3))
    dense = int(n_inputs*2)
    opt = tf.keras.optimizers.SGD(
        learning_rate=0.25, momentum=0.9)
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))
    for i in range(0, num_hidden):
        model.add(Dense(dense,
                        kernel_initializer="he_uniform", activation="relu", kernel_constraint=maxnorm(4)))
        model.add(Dropout(drop))
    model.add(Dense(n_ouputs, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=["accuracy", tf.keras.metrics.BinaryAccuracy()])
    return model


# def evaluate_model(X, y):
#     results = list()
#     n_inputs, n_outputs = X.shape[1], y.shape[1]
#     cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
#     for train_ix, test_ix in cv.split(X):
#         X_train, X_test = X[train_ix], X[test_ix]
#         y_train, y_test = y[train_ix], y[test_ix]
#         model = get_model(n_inputs, n_outputs)
#         model.fit(X_train, y_train, verbose=0, epochs=45)
#         yhat = model.predict(X_test)
#         yhat = yhat.round()
#         acc = accuracy_score(y_test, yhat)
#         print('>%.3f' % acc)
#         results.append(acc)
#     return results

def evaluate_model(X, y, n_train, n_batch):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    X_train, X_test = X[:n_train, :], X[n_train:, :]
    y_train, y_test = y[:n_train], y[n_train:]
    history, model = fit_model(
        n_inputs, n_outputs,
        X_train, X_test,
        y_train, y_test,
        n_batch)
    _, _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    pyplot.plot(history.history['binary_accuracy'], label='train')
    pyplot.plot(history.history['val_binary_accuracy'], label='test')
    pyplot.title('batch='+str(n_batch), pad=-40)
    pyplot.legend()
    pyplot.show()


def fit_model(n_inputs, n_outputs, X_train, X_test, y_train, y_test, n_batch):
    model = get_model(n_inputs, n_outputs)
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), verbose=0, epochs=300, batch_size=n_batch)
    return history, model


def train_model(X, y, n_train, n_batch):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    X_train, X_test = X[:n_train, :], X[n_train:, :]
    y_train, y_test = y[:n_train], y[n_train:]
    history, model = fit_model(
        n_inputs, n_outputs,
        X_train, X_test,
        y_train, y_test,
        n_batch)
    model.save('test_model.h5')


X, y = get_dataset()
n_train = int(X.shape[0] * 0.1)
batch_size = 64
evaluate_model(X, y, n_train, batch_size)
# print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
train_model(X, y, n_train, batch_size)
