import numpy as np


def train_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    n = len(y)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    weights = np.zeros(X.shape[1])

    losses = []  # 👈 store loss

    for i in range(epochs):
        y_pred = np.dot(X, weights)
        error = y_pred - y

        gradient = (1 / n) * np.dot(X.T, error)
        weights = weights - learning_rate * gradient

        loss = (1 / n) * np.sum(error ** 2)
        losses.append(loss)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return weights, losses  # 👈 return losses too


def predict(X, weights):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    return np.dot(X, weights)