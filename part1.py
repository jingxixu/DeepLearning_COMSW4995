# -*- coding: utf-8 -*-
layer_dimensions = [X_train.shape[0], 500, 100, 25, 10]  # including the input and output layers
NN = NeuralNetwork(layer_dimensions)
NN.train(X_train, y_train, iters=5000, alpha=2, batch_size=100, print_every=10)
#NN.train(X_train, y_train, iters=10, alpha=0.00001, batch_size=1000, print_every=10)

y_predicted = NN.predict(X_test)
save_predictions('ans1-uni', y_predicted)

# test if your numpy file has been saved correctly
loaded_y = np.load('ans1-uni.npy')
print(loaded_y.shape)
loaded_y[:10]