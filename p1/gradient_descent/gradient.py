import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs    = 1000
learnrate = 0.25

for e in range(epochs):
    del_w = np.zeros(weights.shape)

    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        output = sigmoid(np.dot(x, weights)) # + b ???

        error = y - output

        del_w +=  learnrate * error * output * (1 - output) * x

    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out  = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out     = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy    = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

# --------- Output ---------

# Train loss:  0.264007377577
# Train loss:  0.24897418938
# Train loss:  0.238140550234
# Train loss:  0.229995093545
# Train loss:  0.223719039368
# Train loss:  0.218822281186
# Train loss:  0.214972682744
# Train loss:  0.211926845358
# Train loss:  0.209500624401
# Train loss:  0.207553625323
# Prediction accuracy: 0.800
