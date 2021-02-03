# def predict(row, coefficients):
#     yhat = coefficients[0]
#     for i in range(len(row)-1):
#         yhat += coefficients[i+1] * row[i]
#     return yhat
#
# def coefficients_sgd(train, l_rate, n_epoch):
#     coef = [0.0 for i in range(len(train[0]))]
#     count = 0 # early loop break for gradient descent
#     sum_error = 0
#     for epoch in range(n_epoch):
#         prior_error = sum_error
#         sum_error = 0
#         for row in train:
#             yhat = predict(row, coef)
#             error = yhat - row[-1]
#             sum_error += error**2
#             coef[0] = coef[0] - l_rate * error
#             for i in range(len(row)-1):
#                 coef[i+1] = coef[i+1] - l_rate * error * row[i]
#         print('>epoch=%d, 1rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
#         if round(sum_error, 3) == round(prior_error, 3):
#             count += 1
#             print(count)
#         if count == 15:
#             break
#     return coef
#
# dataset = [[1,1],[2,3],[4,3],[3,2],[5,5]]
# # coef = [0.4,0.8]
# # for row in dataset:
# #     yhat = predict(row, coef)
# #     print('Expected=%.3f, Predicted=%.3f' % (row[-1], yhat))
# l_rate = 0.001
# n_epoch = 100
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)

from random import seed
from random import randrange
from csv import reader
from math import sqrt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())

def dataset_minax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset,n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores

def predict(row,coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1]*row[i]
    return yhat

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for i in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] - l_rate * error * row[i]
    return coef

def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return(predictions)

seed(1)
filename = 'wineQualityWhite.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset,i)
minmax = dataset_minax(dataset)
normalize_dataset(dataset,minmax)
n_folds = 5
l_rate = 0.01
n_epoch = 100
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
