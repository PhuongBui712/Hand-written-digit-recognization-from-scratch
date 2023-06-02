import numpy as np
import os
import pickle
import gzip
import matplotlib.pyplot as plt

def read_mnist(mnist_file):
    if os.path.isfile(mnist_file) == False:
        mnist_file = os.path.join(os.path.expanduser('~'), 'data', 'mnist.pkl.gz')
    
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    
    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data    
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def add_ones(X):
    return np.hstack((np.ones((len(X), 1)), X))

def compute_nnet_output(Ws, X, return_what='class'):
    # YOUR CODE HERE
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(Z):
        e_Z = np.exp(Z)
        A = e_Z / e_Z.sum()
        return A

    res = []
    neural = X
    res.append(neural)
    for w in Ws[:-1]:
        neural = np.hstack((np.ones((neural.shape[0], 1)), sigmoid(neural @ w)))
        res.append(neural)
    
    res.append(np.apply_along_axis(softmax, axis=1, arr=(neural @ Ws[-1])))

    if return_what == 'all':
        return res
    elif return_what == 'prob':
        return res[-1]    
    else:
        return np.argmax(res[-1], 1)

def train_nnet(X, y, hid_layer_sizes, initial_Ws, mb_size, lr, max_epoch):
    # Initialize weights
    n_classes = len(np.unique(y)) 
    if initial_Ws is None:
        layer_sizes = [X.shape[1] - 1] + hid_layer_sizes + [n_classes]
        Ws = [np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) 
              / np.sqrt(layer_sizes[i] + 1) 
              for i in range(len(layer_sizes) - 1)] 
    else:
        Ws = initial_Ws
    
    # Build model
    prob_y = np.apply_along_axis(lambda x : np.eye(10)[x, :], axis=0, arr=y)

    n_mb = int(X.shape[0] / mb_size)
    index_arr = np.arange(X.shape[0])
    cross_entropy = []
    for epoch in range(max_epoch):
        np.random.shuffle(index_arr)
        
        for b in range(n_mb):
            batch_X = X[index_arr[b * mb_size : (b + 1) * mb_size], :]
            batch_prob_y = prob_y[index_arr[b * mb_size : (b + 1) * mb_size], :]
            
            all_layer_output = compute_nnet_output(Ws, batch_X, 'all')
            delta = all_layer_output[-1] - batch_prob_y
            Ws[-1] -= lr * (1/mb_size) * (all_layer_output[-2].T @ delta)

            for l in range(len(hid_layer_sizes)-1, -1, -1):
                delta = delta.dot(Ws[l + 1].T[:, 1:]) * all_layer_output[l + 1][:, 1:] * (1 - all_layer_output[l + 1][:, 1:])
                Ws[l] -= lr * (1/mb_size) * (all_layer_output[l].T @ delta)
        
        cross_entropy.append((1 / X.shape[0]) * ((prob_y * np.log(compute_nnet_output(Ws, X, 'prob'))).sum(axis=1) * -1).sum())

    return Ws, cross_entropy

def training():
    train_X, train_y, val_X, val_y, test_X, test_y = read_mnist('mnist.pkl.gz')
    train_Z = add_ones(train_X)
    val_Z = add_ones(val_X)
    test_Z = add_ones(test_X)

    Z = np.vstack((train_Z, val_Z, test_Z))
    y = np.hstack((train_y, val_y, test_y))

    Ws, ces = train_nnet(Z, y, hid_layer_sizes=[50], initial_Ws=None, mb_size=32, lr=0.3, max_epoch=100)
    
    if not os.path.exists('./model/'): os.makedirs('./model/')
    for i in range(len(Ws)):
        np.savetxt(f'model/layer{i}.txt', Ws[i])

    return Ws