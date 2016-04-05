
import os
import sys
import time
import logging
import csv
import argparse

import cPickle as pkl
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold


def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(float(part))
    return sorted(result)

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SVM training for Santander')
    parser.add_argument('input_folder', metavar='INPUT_FOLDER', 
                        help='data folder')
    parser.add_argument('-k', '--kfold', metavar='NFOLD', type=int, default=5, 
                        help='k for kfold cross-validtion. If the value less than 2, we skip the cross-validation and choose the first parameter of -c (DEFAULT: 5)')
    parser.add_argument('-m', '--output_model', metavar='OUTPUT_MODEL', default='model.pkl', 
                        help='output, model file name (DEFAULT: model.pkl)')
    parser.add_argument('-p', '--output_predict', metavar='OUTPUT_PREDICT', default='predict.pkl', 
                        help='output, predict file name (DEFAULT: predict.pkl)')
    parser.add_argument('-f', '--output_confidence', metavar='OUTPUT_CONFIDENCE', default='confidence.pkl', 
                        help='output, confidence-level file name (DEFAULT: confidence.pkl)')
    parser.add_argument('-c', '--Cs', metavar='C', type=parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def my_read_file(fname, logger, is_train=True):
    with open(fname, 'r') as fr:
        n_rows = len(fr.readlines())

    with open(fname, 'r') as fr:
        for row in fr:
            n_cols = len(row.split(','))
            break

    n_examples = n_rows - 1
    dim = n_cols - 1 if is_train else n_cols
    logger.debug('n_cols=%d' % n_cols)


    X_ret = np.zeros((n_examples, dim))
    y_ret = np.zeros(n_examples)

    logger.info('X_shape = %s' % str(X_ret.shape))

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

    logger.info('number of positives = %d' % (sum(y_ret)))
    return X_ret, y_ret


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    train_file = os.path.join(args.input_folder, 'train.csv')
    test_file = os.path.join(args.input_folder, 'test.csv')


    # read data
    X_train, y_train = my_read_file(train_file, logger, is_train=True)
    X_test, y_test = my_read_file(test_file, logger, is_train=False)

    # normalization
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5-fold cross-validation on C
    kfolder = KFold(n=X_train.shape[0], n_folds=5, shuffle=True)
    best_c = 0
    best_score = 0
    logger.info('cross validation with C = %s' %(str(args.Cs)))
    for c in args.Cs:
        sum_score = 0.0
        for (i, (train_index, dev_index)) in enumerate(kfolder):
            start_t = time.time()
            logger.info("cross-validation fold %d: train=%d, test=%d" % (i, len(train_index), len(dev_index)))
            X_train_sub, X_dev_sub, y_train_sub, y_dev_sub = X_train[train_index], X_train[dev_index], y_train[train_index], y_train[dev_index]
            classifier = svm.LinearSVC(C=c)
            classifier.fit(X_train_sub, y_train_sub)
            score = classifier.score(X_dev_sub, y_dev_sub)
            logger.info('training time: %.3f seconds' % (time.time()-start_t))

            logger.debug('score = %.5f' % (score))
            sum_score += score

        mean_score = sum_score/len(kfolder)
        logger.info('C = %f, mean_score = %f' % (c, mean_score))
        if mean_score > best_score:
            logger.info('BEST SCORE!!!')
            best_c = c
            best_score = mean_score


    # train with the best C and with all examples
    logger.info('best_c = %f' % best_c)
    classifier = svm.LinearSVC(C=best_c)
    classifier.fit(X_train, y_train)
    conf_score = classifier.decision_function(X_test)
    y_predict = classifier.predict(X_test)

    # save results
    logger.info('dumping %s' % args.output_confidence)
    pkl.dump(conf_score, open(args.output_confidence, 'wb'))
    logger.info('dumping %s' % args.output_predict)
    pkl.dump(y_predict, open(args.output_predict, 'wb'))
    logger.info('dumping %s' % args.output_model)
    pkl.dump(classifier, open(args.output_model, 'wb'))

