import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_derivative(h):
    """
    Calculate sigmoid's derivative
    """
    return sigmoid(h) * (1 - sigmoid(h))

learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5) # target y

# Initial weights
w = np.array([0.5, -0.5])

nn_output = sigmoid(np.dot(x,w)) # output y^

error = y - nn_output

del_w = [ learnrate * error * sigmoid_derivative(np.dot(x,w)) * x[0],
          learnrate * error * sigmoid_derivative(np.dot(x,w)) * x[1] ]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
