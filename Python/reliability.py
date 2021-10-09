import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RepeatedKFold, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import LeakyReLU
from keras.initializers import GlorotNormal, he_uniform
from keras.constraints import max_norm
from sklearn.metrics import accuracy_score
import numpy as numpy
from numpy import mean
from numpy import std
from tqdm.keras import TqdmCallback
from matplotlib import pyplot
import time

SEED = 3
# DATATYPE = 'LLR'
DATATYPE = 'Naive'
SIZE = '10K'
SNR = '_0_3SNR'
NOISE_PERCENT = '100'
PATH = '../' + DATATYPE + '/' + SIZE + SNR + NOISE_PERCENT
X_PATH_TRAIN = PATH + 'dataTRAIN' + DATATYPE + '.txt'
Y_PATH_TRAIN = PATH + 'messagesTRAIN' + DATATYPE + '.txt'
X_PATH_TEST = PATH + 'dataTEST' + DATATYPE + '.txt'
Y_PATH_TEST = PATH + 'messagesTEST' + DATATYPE + '.txt'
ACTIVATION = 'tanh'
NUM_HIDDEN = 2
# MODEL_NAME = 'models/' + DATATYPE + SIZE + \
#     SNR + NOISE_PERCENT + f'H{NUM_HIDDEN}' + ACTIVATION + '.h5'
MODEL_NAME = 'models/' + DATATYPE + '.h5'
EPOCHS = 10
N = 200
models = ["Naive", "LLR", "Vote", "NaiveMultVote",
          "LLRMultVote", "LLRMultVoteMultNaive", "LLRVoteRange"]


def get_dataset():
    # X = numpy.loadtxt('../LLR/2M75dataLARGELLR.txt', delimiter=",")
    X = numpy.loadtxt(X_PATH_TRAIN, delimiter=",")
    X_test = numpy.loadtxt(X_PATH_TEST, delimiter=",")
    # y = numpy.loadtxt('../LLR/2M75messagesLARGELLR.txt', delimiter=",")
    y = numpy.loadtxt(Y_PATH_TRAIN, delimiter=",")
    y_test = numpy.loadtxt(Y_PATH_TEST, delimiter=",")
    print(X.shape, y.shape)
#     print(X[:, 0:100] == y[:, :])
    return X, y, X_test, y_test


def get_model(n_inputs, n_ouputs):
    num_hidden = NUM_HIDDEN
    drop = 0.2
    dense = int(30*N)
    # opt = tf.keras.optimizers.SGD(
    #     learning_rate=0.02, momentum=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))
    for i in range(0, num_hidden):
        model.add(Dense(dense,
                        kernel_initializer=GlorotNormal(),
                        kernel_constraint=max_norm(6),
                        activation=ACTIVATION))
        model.add(Dropout(drop))
    model.add(Dense(n_ouputs, activation="sigmoid"))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=opt,
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def evaluate_model(X, y, X_test, y_test, n_batch):
    n_inputs, n_outputs = X.shape[1], y.shape[1]

    history, model = fit_model(
        n_inputs, n_outputs,
        X, X_test,
        y, y_test,
        n_batch)
    _, train_acc = model.evaluate(X, y, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.plot(history.history['binary_accuracy'], label='train')
    ax1.plot(history.history['val_binary_accuracy'], label='test')
    ax1.legend()
    ax1.set_title('Accuracy')
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.legend()
    ax2.set_title('Loss')


def fit_model(n_inputs, n_outputs, X_train, X_test, y_train, y_test, n_batch):
    model = get_model(n_inputs, n_outputs)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test), verbose=1,
                        epochs=EPOCHS, batch_size=n_batch, shuffle=True)
    model.save(MODEL_NAME)
    model.summary()
    return history, model


def train_model(X, y, n_test, n_batch):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=SEED)
    history, model = fit_model(
        n_inputs, n_outputs,
        X_train, X_test,
        y_train, y_test,
        n_batch)
    model.summary()
    model.save(MODEL_NAME)


start_time = time.time()
X, y, X_test, y_test = get_dataset()
batch_size = 16
evaluate_model(X, y, X_test, y_test, batch_size)
# print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
# train_model(X, y, n_test, batch_size)
print(f"training time: {(time.time() - start_time)/60}")
pyplot.show()
