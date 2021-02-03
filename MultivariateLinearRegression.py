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

def load_csv_file(filename):
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



