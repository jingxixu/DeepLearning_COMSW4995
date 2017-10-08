# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:08:24 2017

@author: kaelc
"""

# Load the data
data_root_path = 'cifar10-hw1/'
X_train, y_train = get_train_data(data_root_path) # this may take a few minutes
X_test = get_images(data_root_path + 'test')
print('Data loading done')

print(X_train.shape)
print(y_train.shape)
print(y_train[:5])