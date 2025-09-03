import numpy as np
import pandas as pd

# preparing data for the network - here we have mnist dataset
data = pd.read_csv("path_to_your_mnist_dataset")
print(data.head())

# changing it into np array
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# splitting into train and test set 
data_test = data[0: 6000].T
y_test = data_test[0]
x_test = data_test[1:n]
x_test = x_test/255

data_train = data[6000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train/255

# MAKING NEURAL NETWORK 
# creating initializing parameters - weights and biases
def init_params():
    w1 = np.random.rand(10, 784) - 0.5 
    # here we are making our parameters for first layer of neural network 
    # as a result we are getting a 10 x m size matrix 
    b1 = np.random.rand(10, 1) - 0.5
    # same things here - just different size
    w2 = np.random.rand(10, 10) - 0.5 
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

# activation function - ReLU
def ReLU(Z): 
    return np.maximum(Z, 0)

def deriv_ReLU(Z):
    return Z > 0 # deriv of Z is 1 so it only depends on value of Z (bigger than 0 or no)

# creating softmax function - to examine probabilities of our prediction being good one
def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

# forward propagation
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# one-hot function - it provides us correct way to put labels into neural network 
# (ex. (2, 1, 4) -> 1st column has 1 on 3rd row tohers are 0, 2nd column has 1 on 2nd row other are 0, 3rd column has 1 on 5th row other are 0)
def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# back propagation
def back_prop(z1, a1, z2, a2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y) 
    dz2 = a2 - one_hot_Y # essentially our prediction (based on softmax) - truth values 
    dw2 = 1/m * dz2.dot(a1.T) # result is a 10x10 matrix 
    db2 = 1/m * np.sum(dz2) # 10x1 matrix 
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1) 
    dw1 = 1/m * dz1.dot(X.T)
    db1 = 1/m * np.sum(dz1)
    return dw1, db1, dw2, db2

# updating parameters based on back propagation
# alpha here is our learning rate 
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    return w1, w2, b1, b2

# getting our predictions 
def get_predictions(a2): 
    return np.argmax(a2, 0)

# calculating accuracy - times when predictions were equal to Y 
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
    
# gradient descent 
def grad_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations): 
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, X, Y)
        w1, w2, b1, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0: 
            print("iteration : ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), Y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = grad_descent(x_train, y_train, 500, 0.1)

