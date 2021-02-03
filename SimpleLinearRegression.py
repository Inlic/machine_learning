# simple linear regression in python

from math import sqrt

def mean(values):
    return sum(values) / float(len(values))

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    b1 = covariance(x,mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    return [b0, b1]


def simple_linear_regression(train,test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
    predicted = algorithm(dataset,test_set)
    print(predicted)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse


dataset = [[1,1], [2,3], [4,3], [3,2], [5,5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

mean_x, mean_y = mean(x), mean(y)
# var_x, var_y = variance(x, mean_x), variance(y, mean_y)
# print('x stats: mean=%.3f variance=%.3f' % (mean_x,var_x))
# print('y stats: mean=%.3f variance=%.3f' % (mean_y,var_y))
covar = covariance(x, mean_x, y, mean_y)
print('Covariance: %.3f' % (covar))
b0, b1 = coefficients(dataset)
print('Coefficients: B=%.3f, B=%f' % (b0,b1))
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))