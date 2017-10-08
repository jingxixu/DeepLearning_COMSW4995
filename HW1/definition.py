class NeuralNetwork(object):
    """
    Abstraction of neural network.
    Stores parameters, activations, cached values. 
    Provides necessary functions for training and prediction. 
    """
    def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0):
        """
        Initializes the weights and biases for each layer
        :param layer_dimensions: (list) number of nodes in each layer
        :param drop_prob: drop probability for dropout layers. Only required in part 2 of the assignment
        :param reg_lambda: regularization parameter. Only required in part 2 of the assignment
        """
        np.random.seed(1)
        
        self.parameters = {}
        self.num_layers = len(layer_dimensions)
        self.drop_prob = drop_prob
        self.reg_lambda = reg_lambda
        self.X_val = None
        self.y_val = None
        
        # init parameters
        for i in range(1, self.num_layers):
            self.parameters[('W', i)] = np.random.normal(0, 1, (layer_dimensions[i], layer_dimensions[i - 1]))
            self.parameters[('W', i)] /= np.sqrt(layer_dimensions[i - 1])
            self.parameters[('b', i)] = np.zeros([layer_dimensions[i], 1])

    def affineForward(self, A, W, b):
        """
        Forward pass for the affine layer.
        :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
        the number of samples
        :returns: the affine product WA + b, along with the cache required for the backward pass
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b, Z)
        return Z, cache

    def activationForward(self, A, activation="relu"):
        """
        Common interface to access all activation functions.
        :param A: input to the activation function
        :param prob: activation funciton to apply to A. Just "relu" for this assignment.
        :returns: activation(A)
        """ 
        return self.relu(A)

    def relu(self, X):
        A = np.maximum(0, X)
        assert (X.shape == A.shape)
        return A
            
    def dropout(self, A, prob):
        """
        :param A: 
        :param prob: drop prob
        :returns: tuple (A, M) 
            WHERE
            A is matrix after applying dropout
            M is dropout mask, used in the backward pass
        """
        # todo
        return A, M

    def forwardPropagation(self, X):
        """
        Runs an input X through the neural network to compute activations
        for all layers. Returns the output computed at the last layer along
        with the cache required for backpropagation.
        :returns: (tuple) AL, cache
            WHERE 
            AL is activation of last layer
            cache is cached values for each layer that
                     are needed in further steps
        """
        cache = {}
        A = X
        for l in range(1, self.num_layers):
            Z, cache_l = self.affineForward(A,
                                            self.parameters[('W', l)], 
                                            self.parameters[('b', l)])
            cache[l] = cache_l
            A = self.activationForward(Z)
        # return AL, cache
        return A, cache
    
    def costFunction(self, AL, y):
        """
        :param AL: Activation of last layer, shape (num_classes, S)
        :param y: labels, shape (S)
        :param alpha: regularization parameter
        :returns cost, dAL: A scalar denoting cost and the gradient of cost
        """
        # compute loss
        S = AL.shape[1]
        probs = np.exp(AL - np.max(AL, axis=0, keepdims=True))
        probs /= np.sum(probs, axis=0, keepdims=True)
        cost = -np.sum(np.log(probs[y, np.arange(S)])) / S
        
#         if self.reg_lambda > 0:
#             # add regularization
       
        dAL = probs.copy()
        dAL[y, np.arange(S)] -= 1
        dAL /= S
        return cost, dAL

    def affineBackward(self, dA_prev, cache):
        """
        Backward pass for the affine layer.
        :param dA_prev: gradient from the next layer.
        :param cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """
#        A = cache[0]
#        W = cache[1]
#        b = cache[2]
        A, W, b, Z = cache
        S = A.shape[1]
        
        dZ = self.activationBackward(dA_prev, cache)
        dA = np.dot(W.T, dZ)
        dW = np.dot(dZ, A.T) / S
        db = np.sum(dZ, axis = 1, keepdims=True) / S # or np.mean() ?
        # dA = W.T.dot(dA_prev) 
        # dW = dA_prev.dot(A.T) / S
        # db = np.sum(dA_pre, axis = 1) / S
        return dA, dW, db
    
    def activationBackward(self, dA, cache, activation="relu"):
        """
        Interface to call backward on activation functions.
        In this case, it's just relu. 
        """
        return self.relu_derivative(dA, cache[3]) # cache[3] == Z[l]
        # return self.relu_derivative(dA, cache[0]) 
        
    def relu_derivative(self, dx, cached_x):
        out = np.maximum(0, cached_x)
        out[out > 0] = 1
        dx = out * dx
        return dx

    def dropout_backward(self, dA, cache):
        # todo
        return dA

    def backPropagation(self, dAL, Y, cache):
        """
        Run backpropagation to compute gradients on all paramters in the model
        :param dAL: gradient on the last layer of the network. Returned by the cost function.
        :param Y: labels
        :param cache: cached values during forwardprop
        :returns gradients: dW and db for each weight/bias
        """
        gradients = {}
        dA = dAL
        for l in range(self.num_layers-1, 0, -1):
            if self.drop_prob > 0:
                dA = self.dropout_backward(dA, cache[l])
            dA, dW, db = self.affineBackward(dA, cache[l])
            # assert (dW.shape == self.parameters[('W', l)].shape), '{} - {}'.format(dW.shape, self.parameters[('W', l)].shape)
            # assert (db.shape == self.parameters[('b', l)].shape), '{} - {}'.format(db.shape, self.parameters[('b', l)].shape)
            gradients[('W', l)] = dW
            gradients[('b', l)] = db
            if self.reg_lambda > 0:
                # add gradients from L2 regularization to each dW
                gradients[('W', l)] += reg_lambda * self.parameters[('W', l)]
                gradients[('b', l)] += reg_lambda * self.parameters[('b', l)]
        return gradients

    def updateParameters(self, gradients, alpha):
        """
        :param gradients: gradients for each weight/bias
        :param alpha: step size for gradient descent 
        """
        for key in gradients.keys():
            self.parameters[key] -= alpha * gradients[key]
            
    def train(self, X, y, iters=1000, alpha=0.0001, batch_size=100, print_every=100):
        """
        :param X: input samples, each column is a sample
        :param y: labels for input samples, y.shape[0] must equal X.shape[1]
        :param iters: number of training iterations
        :param alpha: step size for gradient descent
        :param batch_size: number of samples in a minibatch
        :param print_every: no. of iterations to print debug info after
        """
        assert (alpha * self.reg_lambda < 1)
        self.parameters['mean'] = np.mean(X, axis = 1, keepdims = True)
        self.parameters['var'] = np.var(X, axis = 1, keepdims = True)
        X = (X - self.parameters['mean']) / np.sqrt(self.parameters['var'])
        for i in range(0, iters):
            # get minibatch
            X_batch, y_batch = self.get_batch(X, y, batch_size)
            # forward prop
            AL, cache = self.forwardPropagation(X_batch)
            # compute loss
            cost, dAL = self.costFunction(AL, y_batch)
            # compute gradients
            gradients = self.backPropagation(dAL, y_batch, cache)
            # update weights and biases based on gradient
            self.updateParameters(gradients, alpha)
            if i % print_every == 0:
                # print cost, train and validation set accuracies
                trn_acc = self.score(self.predict(X), y)
                if self.X_val is not None:
                    val_acc = self.score(self.predict(self.X_val), self.y_val)
                else:
                    val_acc = np.nan
                print('iter={:5}, cost={:.4f}, trn_acc={:.4f}, val_acc={:.4f}'.format(i, cost, trn_acc, val_acc))
                
    def predict(self, X):
        """
        Make predictions for each sample
        """
        X = (X - self.parameters['mean']) / np.sqrt(self.parameters['var'])
        AL, _ = self.forwardPropagation(X)
        y_pred = np.argmax(AL, axis = 0)
        return y_pred

    def get_batch(self, X, y, batch_size):
        """
        Return minibatch of samples and labels
        
        :param X, y: samples and corresponding labels
        :parma batch_size: minibatch size
        :returns: (tuple) X_batch, y_batch
        """
        batch_idx = np.random.randint(X.shape[1], size = batch_size)
        X_batch = X[:, batch_idx]
        y_batch = y[batch_idx]
#         print y_batch.shape
        return X_batch, y_batch
    
    def score(self, y_pred, y_gold):
        return np.mean(y_pred == y_gold)
    
    def load_validation_set(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val