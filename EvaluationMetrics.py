from math import sqrt

def accuracy_metric(actual,predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100

actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual,predicted)
print(accuracy)
print("#######################")

def confusion_matrix(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return unique, matrix

def print_confusion_matrix(unique, matrix):
    print('(A)'+' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print_confusion_matrix(unique,matrix)
print("#######################")

# Interpretation of confusion matrix
# Prediction errors on the left diagonal - One 1 counted as 0, Two 0 counted as 1

# Precision and recall

def prec_recall(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return (matrix[0][0]/(matrix[0][0]+matrix[0][1])), (matrix[0][0]/(matrix[0][0]+matrix[1][0]))

precision, recall = prec_recall(actual,predicted)
print("precision")
print(precision)
print("recall")
print(recall)


def f_measure(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return matrix[0][0] / (matrix[0][0]+0.5*(matrix[0][1]+matrix[1][0]))

print("F Measure Ranges from 0 to 1")

fmeasure = f_measure(actual,predicted)
print(fmeasure)

print("######################")

# Matthews Correlation Coefficient
# -1 bad, 0 random, 1 great

def matthews_corr(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / sqrt(float(matrix[0][0]+matrix[0][1])*float(matrix[0][0]+matrix[1][0])*float(matrix[1][1]+matrix[0][1])*float(matrix[1][1]+matrix[1][0]))

corr = matthews_corr(actual,predicted)

print("This is the MCC Ranges from -1 to 1")

print(corr)

print("######################")

print("Youdens J Ranges from 0 to 1")

def yodens_jay(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return (matrix[0][0] / (matrix[0][0]+matrix[1][0])) + (matrix[1][1] / (matrix[1][1]+matrix[0][1])) - 1

jay = yodens_jay(actual,predicted)

print(jay)

print("######################")

def cohens_kappa(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    tot = a+b+c+d
    p0 = (a+d) / tot
    pyes = ((a+b) / tot) * ((a+c) / tot)
    pno = ((c+d) / tot) * ((b+d) / tot)
    pe = pyes + pno
    return (p0 - pe) / (1 - pe)


kappa = cohens_kappa(actual,predicted)
print("Cohens Kappa, Range between 0 and 1 usually possible to be negative")
print(kappa)

print("#######################")


def mae_metric(actual,predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i]-actual[i])
    return sum_error / float(len(actual))

# Test out MAE
actual = [0.1,0.2,0.3,0.4,0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)
print("#######################")

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Test out RMSE
rmse = rmse_metric(actual, predicted)
print(rmse)