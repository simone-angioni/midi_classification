from transformer_classifier import TransformerClassifier
from utils import *
import tensorflow as tf
import logging
import json
import pandas as pd

def classify(feature_engineering_techniques, time_frame, sample_len, log_name, shuffle):
    logging.basicConfig(filename=f'{base_dir}/results/{log_name}.log',format='%(asctime)s : %(message)s')
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    except Exception as e:
        pass

    num_folds = 10
    fs, initial_attention = time_frame, sample_len
    folds_X, folds_y, dict1 = load_clf_data_kfold(feature_size=initial_attention,
                                                  time_frame=fs,
                                                  feature_extraction=feature_engineering_techniques,
                                                  shuffle=shuffle)
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
            if i == 0 and k == 0:
                i += 1
                x_train = folds_X[1]
                y_train = folds_y[1]
            elif i != k:
                x_train = np.concatenate((x_train, folds_X[i]), axis=0)
                y_train = np.concatenate((y_train, folds_y[i]), axis=0)

        # x_train = np.concatenate((folds_X[0], folds_X[1], folds_X[2], folds_X[3], folds_X[4], folds_X[5]), axis=0)
        # y_train = np.concatenate((folds_y[0], folds_y[1], folds_y[2], folds_y[3], folds_y[4], folds_y[5]), axis=0)
        if time_frame % sample_len == 0:
            #max_size = 12050
            max_size = max(pd.DataFrame(x_train).max().max(), pd.DataFrame(x_test).max().max())+1
        else:
            max_size = max(pd.DataFrame(x_train).max().max(), pd.DataFrame(x_test).max().max())+1
        model = TransformerClassifier(feed_shape, vocabulary_size=max_size, maxlen=max_size)
        model.fit(x_train, x_test, y_train, y_test)
        metrics = model.model.evaluate(x_test, y_test)
        accuracies.append(metrics)
        logging.warning(f"> Accuracy on k: {k} are: {metrics}")

    avg_metrics = {}
    avg_metrics['used_strategy'] = feature_engineering_techniques
    avg_metrics['time_frame'] = time_frame
    avg_metrics['feature_size'] = sample_len
    avg_metrics['avg_loss'] = sum([l[0] for l in accuracies])/len(accuracies) if not len(accuracies) == 0 else 0
    avg_metrics['avg_accuracy'] = sum([l[1] for l in accuracies])/len(accuracies) if not len(accuracies) == 0 else 0
    with open(f'{base_dir}/results/{log_name}.json', "w") as f:
        json.dump(avg_metrics, f)
    logging.warning(accuracies)

# Choosing whether training or testing the model
# model.fit(x_train, x_test, y_train, y_test)
# model.model.evaluate(x_test, y_test)
# predictions = model.model.predict(x_test)





