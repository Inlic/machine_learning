from random import seed
from random import randrange

# def train_test_split(dataset,split=0.60):
#     train = list()
#     train_size = split * len(dataset)
#     dataset_copy = list(dataset)
#     while len(train) < train_size:
#         index = randrange(len(dataset_copy))
#         train.append(dataset_copy.pop(index))
#     return train, dataset_copy

def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

seed(1)
dataset = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# train test datasets
#train, test = train_test_split(dataset)
# k fold validation
folds = cross_validation_split(dataset,4)

# print(train)
# print(test)
print(folds)


