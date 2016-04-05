
import os
import sys
import time
import logging
import csv

import cPickle as pkl
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

loglevel = logging.DEBUG
logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)

train_file = 'data/train.csv'
test_file = 'data/test.csv'

output_conf_score = 'conf_score.pkl'
output_predict = 'predict.pkl'
output_model = 'model.pkl'

Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 300]


def my_read_file(fname, is_train=True):
    with open(fname, 'r') as fr:
        n_rows = len(fr.readlines())

    with open(fname, 'r') as fr:
        for row in fr:
            n_cols = len(row.split(','))
            break

    n_examples = n_rows - 1
    dim = n_cols - 1 if is_train else n_cols
    logging.debug('n_cols=%d' % n_cols)


    X_ret = np.zeros((n_examples, dim))
    y_ret = np.zeros(n_examples)

    logging.info('X_shape = %s' % str(X_ret.shape))

    with open(train_file, 'r') as fr:
        csvreader = csv.reader(fr, delimiter=',')
        row_idx = 0
        for row in csvreader:
            if row_idx == n_examples:
                break
            if row_idx > 0:
                X = [float(ele) for ele in row[0:dim]]
                X_ret[row_idx-1] = np.array(X)
                if is_train:
                    y = float(row[dim])
                    y_ret[row_idx-1] = y
                    
            row_idx += 1

    logging.info('number of positives = %d' % (sum(y_ret)))
    return X_ret, y_ret


if __name__ == '__main__':

    # read data
    X_train, y_train = my_read_file(train_file, is_train=True)
    X_test, y_test = my_read_file(test_file, is_train=False)

    # normalization
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5-fold cross-validation on C
    kfolder = KFold(n=X_train.shape[0], n_folds=5, shuffle=True)
    best_c = 0
    best_score = 0

    for c in Cs:
        sum_score = 0.0
        for (i, (train_index, dev_index)) in enumerate(kfolder):
            logging.info("cross-validation fold %d: train=%d, test=%d" % (i, len(train_index), len(dev_index)))
            X_train_sub, X_dev_sub, y_train_sub, y_dev_sub = X_train[train_index], X_train[dev_index], y_train[train_index], y_train[dev_index]
            classifier = svm.LinearSVC(C=c)
            classifier.fit(X_train_sub, y_train_sub)
            score = classifier.score(X_dev_sub, y_dev_sub)
            
            logging.debug('score = %.5f' % (score))
            sum_score += score

        mean_score = sum_score/len(kfolder)
        logging.info('C = %f, mean_score = %f' % (c, mean_score))
        if mean_score > best_score:
            logging.info('BEST SCORE!!!')
            best_c = c
            best_score = mean_score


    # train with the best C and with all examples
    logging.info('best_c = %f' % best_c)
    classifier = svm.LinearSVC(C=best_c)
    classifier.fit(X_train, y_train)
    conf_score = classifier.decision_function(X_test)
    y_predict = classifier.predict(X_test)

    # save results
    pkl.dump(conf_score, open(output_conf_score, 'wb'))
    pkl.dump(y_predict, open(output_predict, 'wb'))
    pkl.dump(classifier, open(output_model, 'wb'))

