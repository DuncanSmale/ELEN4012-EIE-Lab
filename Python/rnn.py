import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RepeatedKFold, train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, Embedding, LSTM, GRU
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


n_inputs = 201
n_ouputs = 100
DATATYPE = 'LLR'
SIZE = '100K'
SNR = '_4_10SNR'
PATH = '../' + DATATYPE + '/' + SIZE + SNR
X_PATH_TRAIN = PATH + 'dataTRAIN' + DATATYPE + '.txt'
Y_PATH_TRAIN = PATH + 'messagesTRAIN' + DATATYPE + '.txt'
X_PATH_TEST = PATH + 'dataTEST' + DATATYPE + '.txt'
Y_PATH_TEST = PATH + 'messagesTEST' + DATATYPE + '.txt'
EPOCHS = 15
dense = 10*(n_inputs-1)
models = ["Naive", "LLR", "NaiveMultVote",
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


n_features = n_inputs
n_steps = 1


def get_model():

    drop = 0.2
    model = Sequential()
    model.add(GRU(dense, input_shape=(
        n_steps, n_features), return_sequences=True, dropout=drop))
    model.add(GRU(dense, dropout=drop))
    model.add(Dense(n_ouputs, activation='sigmoid'))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


n_batch = 128

for mod in models:
    print(mod)
    model = get_model()
    DATATYPE = mod
    PATH = '../' + DATATYPE + '/' + SIZE + SNR
    X_PATH_TRAIN = PATH + 'dataTRAIN' + DATATYPE + '.txt'
    Y_PATH_TRAIN = PATH + 'messagesTRAIN' + DATATYPE + '.txt'
    X_PATH_TEST = PATH + 'dataTEST' + DATATYPE + '.txt'
    Y_PATH_TEST = PATH + 'messagesTEST' + DATATYPE + '.txt'
    MODEL_NAME = 'models/' + DATATYPE + 'RNN1.h5'
    start_time = time.time()
    X, y, X_test, y_test = get_dataset()

    X = X.reshape((X.shape[0], 1, n_features))
    X_test = X_test.reshape((X_test.shape[0], 1, n_features))
    history = model.fit(X, y,
                        validation_data=(X_test, y_test), verbose=1,
                        epochs=EPOCHS, batch_size=n_batch, shuffle=True)
    _, train_acc = model.evaluate(X, y, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    print(f'Saving: {MODEL_NAME}')
    model.save(MODEL_NAME)

    # fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # ax1.plot(history.history['binary_accuracy'], label='train')
    # ax1.plot(history.history['val_binary_accuracy'], label='test')
    # ax1.legend()
    # ax1.set_title('Accuracy')
    # ax2.plot(history.history['loss'], label='train')
    # ax2.plot(history.history['val_loss'], label='validation')
    # ax2.legend()
    # ax2.set_title('Loss')
    # pyplot.show()
