from transformer_classifier import TransformerClassifier
from utils import *
import tensorflow as tf
import logging

logging.basicConfig(filename='./results/test_accuracy_base.log',format='%(asctime)s : %(message)s')
def classify(feature_engineering_techniques):
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    except Exception as e:
        pass

    num_folds = 10
    fs, initial_attention = 50, 50
    folds_X, folds_y, dict1 = load_clf_data_kfold(sample_len=initial_attention, fs=fs, feature_extraction=feature_engineering_techniques)
    feed_shape = folds_X[0][0].shape
    # print(feed_shape)
    # feed_shape = feed_shape[1]
    # Creating the model
    # model = TransformerClassifier(feed_shape, vocabulary_size=12050)

    accuracies = []
    for k in range(num_folds):

        x_test, y_test = folds_X[k], folds_y[k]

        x_train = folds_X[0]
        y_train = folds_y[0]
        for i in range(num_folds):
            print(i)
            if i is 0 and k is 0:
                i += 1
                x_train = folds_X[1]
                y_train = folds_y[1]
            elif i is not k:
                x_train = np.concatenate((x_train, folds_X[i]), axis=0)
                y_train = np.concatenate((y_train, folds_y[i]), axis=0)

        # x_train = np.concatenate((folds_X[0], folds_X[1], folds_X[2], folds_X[3], folds_X[4], folds_X[5]), axis=0)
        # y_train = np.concatenate((folds_y[0], folds_y[1], folds_y[2], folds_y[3], folds_y[4], folds_y[5]), axis=0)

        model = TransformerClassifier(feed_shape, vocabulary_size=12050)
        model.fit(x_train, x_test, y_train, y_test)
        accuracies.append(model.model.evaluate(x_test, y_test))

    logging.warning(accuracies)

# Choosing whether training or testing the model
# model.fit(x_train, x_test, y_train, y_test)
# model.model.evaluate(x_test, y_test)
# predictions = model.model.predict(x_test)





