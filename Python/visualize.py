import visualkeras
import tensorflow as tf
from tensorflow import keras
import numpy
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# keras.utils.plot_model(model, 'MLP.pdf', show_shapes=True)
# model = tf.keras.models.load_model("models/TEST6LLRVoteRangeNEW.h5")
# keras.utils.plot_model(model, 'MLPInputRef.pdf', show_shapes=True)

models = ["NaiveMultVote", "LLRVoteRange",
          "NaiveMultVoteNEW", "LLRVoteRangeNEW"]


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


labels = ["".join("c" + str(i)) for i in range(0, 100)]


for mod in models:
    no_new = mod.replace('NEW', '')
    DATATYPE = no_new
    SIZE = '10K'
    SNR = '_6_6SNR'
    PATH = '../' + DATATYPE + '/' + SIZE + SNR
    X_PATH_TRAIN = PATH + 'dataTRAIN' + DATATYPE + '.txt'
    Y_PATH_TRAIN = PATH + 'messagesTRAIN' + DATATYPE + '.txt'
    X_PATH_TEST = PATH + 'dataTEST' + DATATYPE + '.txt'
    Y_PATH_TEST = PATH + 'messagesTEST' + DATATYPE + '.txt'
    model = tf.keras.models.load_model("models/TEST6" + mod + ".h5")
    X, y, X_test, y_test = get_dataset()
    y_pred = y
    for i in range(10000):
        inp = X[i, :]
        inp = numpy.reshape(inp, (1, 201))
        # print(inp)
        y_pred[i, :] = model.predict(inp)
    y_pred = numpy.round(y_pred)
    y = numpy.round(y)
    # multilabel_confusion_matrix(y, y_pred)
    f, axes = plt.subplots(20, 5, figsize=(30, 50))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.rcParams.update({'font.size': 10})
    axes = axes.ravel()
    for i in range(100):
        # print(y[i, :])
        # print(y_pred[i, :])
        # print(numpy.shape(y[:, i]))
        # print(numpy.shape(y_pred[:, i]))
        disp = ConfusionMatrixDisplay(confusion_matrix(y[:, i], y_pred[:, i]),
                                      display_labels=[0, 1])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'c {i}')
        disp.ax_.set_ylabel('')
        disp.ax_.set_xlabel('')
        # if i == 92 or i == 97:
        #     disp.ax_.set_xlabel('')
        # if i % 10 != 0:
        #     disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    # for a in axes:
    #     a.set_aspect('equal')
    plt.subplots_adjust(right=0.4, wspace=0.3, hspace=1.2)
    plt.tight_layout()
    f.colorbar(disp.im_, ax=axes)
    plt.show()
    f.savefig("confusion/" + mod + "2.pdf", bbox_inches='tight')
