import sys

import numpy as np


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def fold(dataset):
    y = dataset[:, 0]
    X = np.delete(dataset, 0, 1)
    x0 = np.ones(len(X), dtype=int)
    X = np.insert(X, 0, x0, axis=1)
    return X, y


def train(theta, X, y, num_epoch, learning_rate):
    for i in range(num_epoch):
        for j in range(len(X)):
            gradient = (sigmoid(np.dot(theta, X[j])) - y[j]) * X[j]
            theta = theta - learning_rate * gradient
    return theta


def predict(theta, X):
    predictions = []
    for i in range(len(X)):
        pred = sigmoid(np.dot(theta, X[i]))
        if pred >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def compute_error(y_pred, y):
    return np.sum(y_pred != y) / len(y)

def write_labels(path, labels):
    '''
    write the label files
    :param data: np array of data
    :param path: output path
    :param majority_label: most common label
    :return: None
    '''
    with open(path, mode='w') as file:
        for label in labels:
            file.write(str(label) + '\n')
    file.close()

def write_metrics(error_train, error_test, metrics):
    '''
    write the metrics file
    :param error_train: error rate of training set
    :param error_test: error rate of testing set
    :param metrics: path to the metrics file
    :return: None
    '''
    with open(metrics, mode='w') as file:
        file.write("error(train): " + str(error_train) + '\n')
        file.write("error(test): " + str(error_test))


if __name__ == "__main__":
    train_file = sys.argv[1]
    validation_file = sys.argv[2]
    test_file = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    num_epoch = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    dataset_train = np.loadtxt(train_file, delimiter='\t', comments=None, encoding='utf-8', dtype=np.float16)
    X, y= fold(dataset_train)
    theta = np.zeros(len(X[0]), dtype=np.float16)
    theta = train(theta, X, y, num_epoch, learning_rate)
    y_pred_train = np.array(predict(theta, X))
    train_error = format(compute_error(y_pred_train, y), '.6f')
    write_labels(train_out, y_pred_train)

    dataset_test = np.loadtxt(test_file, delimiter='\t', comments=None, encoding='utf-8', dtype=np.float16)
    X_test,y_test = fold(dataset_test)
    y_test = dataset_test[:,0]
    y_pred_test = np.array(predict(theta, X_test))
    test_error = format(compute_error(y_pred_test, y_test), '.6f')
    write_labels(test_out,y_pred_test)

    write_metrics(train_error,test_error,metrics_out)