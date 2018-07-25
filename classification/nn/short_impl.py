import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * sigmoid(1-x)
    else:
        return 1 / (1 + np.exp(-x))

def one_hidden_layer():
    W1 = np.random.uniform(0, 0.01, (input_size, hidden_size))
    # W2 = np.random.random((hidden_size, hidden_size))
    O = np.random.uniform(0, 0.01, (hidden_size, output_size))

    for i in range(it):
        l0 = X
        l1 = sigmoid(np.dot(X, W1))
        # l2 = sigmoid(np.dot(l1, W2))
        o = sigmoid( np.dot(l1, O))

        loss_value = np.sum(np.square(y - o))
        print("Iteration: {}, Loss: {}".format(i, loss_value))
        if loss_value<1e-6:
            print("Loss is small enough")
            break

        o_error = o - y
        # l2_error = sigmoid(l2, derivative=True) * np.dot(o_error, O.T)
        l1_error = sigmoid(l1, derivative=True) * np.dot(o_error, O.T)

        o_pd =  l1[:, :, np.newaxis] * o_error [:, np.newaxis, :]
        # l2_pd = l1[:, :, np.newaxis] * l2_error[:, np.newaxis, :]
        l1_pd = X [:, :, np.newaxis] * l1_error[:, np.newaxis, :]

        total_o_gradient = np.average(o_pd, axis=0)
        # total_l2_gradient = np.average(l2_pd, axis=0)
        total_l1_gradient = np.average(l1_pd, axis=0)

        O =  O - learning_rate *  total_o_gradient
        # W2 = W2 - learning_rate * total_l2_gradient
        W1 = W1 - learning_rate * total_l1_gradient

    print("Predictions:")
    print(np.around(o))
    print("Accuracy:", len(o[np.around(o) == y]) / len(y))


def two_hidden_layers():
    W1 = np.random.uniform(0, 0.1, (input_size, hidden_size))
    W2 = np.random.uniform(0, 0.1, (hidden_size, hidden_size))
    O = np.random.uniform(0, 0.1, (hidden_size, output_size))

    for i in range(it):
        l0 = X
        l1 = sigmoid(np.dot(X, W1))
        l2 = sigmoid(np.dot(l1, W2))
        o = sigmoid(np.dot(l2, O))

        print("Iteration: {}, Loss: {}".format(i, np.sum(np.square(y-o))))
        o_error = o - y

        l2_error = sigmoid(l2, derivative=True) * np.dot(o_error, O.T)
        l1_error = sigmoid(l1, derivative=True) * np.dot(l2_error, W2.T)

        o_pd = l2[:, :, np.newaxis] * o_error[:, np.newaxis, :]
        l2_pd = l1[:, :, np.newaxis] * l2_error[:, np.newaxis, :]
        l1_pd = X[:, :, np.newaxis] * l1_error[:, np.newaxis, :]

        total_o_gradient = np.average(o_pd, axis=0)
        total_l2_gradient = np.average(l2_pd, axis=0)
        total_l1_gradient = np.average(l1_pd, axis=0)

        O = O - learning_rate * total_o_gradient
        W2 = W2 - learning_rate * total_l2_gradient
        W1 = W1 - learning_rate * total_l1_gradient

    print("Predictions:")
    print(np.around(o))
    print("Accuracy:", len(o[np.around(o) == y]) / len(y))

X = np.asarray([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

y = np.asarray([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
])


print(X, y)
it = 1000
input_size = 3
output_size = 1
hidden_size = 24

learning_rate = 0.1
print("=================")
print("One Hidden Layer")
print("=================")
one_hidden_layer()

print("=================")
print("Two Hidden Layers")
print("=================")
two_hidden_layers()

