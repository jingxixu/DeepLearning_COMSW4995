# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:19:10 2017

@author: kaelc
"""
import datetime
def split(X, y, test_size):
    indices = np.random.permutation(X.shape[1])
    test_num = int(test_size * X.shape[1])
    return X[:, indices[test_num:]], X[:, indices[:test_num]], y[indices[test_num:]], y[indices[:test_num]]

print(datetime.datetime.now())

X_trn, X_val, y_trn, y_val = split(X_train, y_train, test_size=0.1)

NN = NeuralNetwork(layer_dimensions)
NN.load_validation_set(X_val, y_val)

# trn_acc = 0.643, val_acc = 0.5066, 5 min
layer_dimensions = [X_train.shape[0], 500, 100, 25, 10]
NN.train(X_trn, y_trn, iters=5000, alpha=1, batch_size=100, print_every=100)

## trn_acc = 0.652, val_acc = 0.5068, 5 min
#layer_dimensions = [X_train.shape[0], 1000, 200, 50, 10]
#NN.train(X_trn, y_trn, iters=5000, alpha=1, batch_size=100, print_every=100)

print('Train Score: {}, Test Score: {}'.format(
        NN.score(NN.predict(X_trn), y_trn),
        NN.score(NN.predict(X_val), y_val)))
print(datetime.datetime.now())