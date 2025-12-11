import numpy as np
from scipy.special import expit  # numerically stable sigmoid

class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, penalty=None, lambda_=0.1,
                 verbose=False, cost_interval=100):
        """
        penalty: None, 'l1', or 'l2'
        lambda_: regularization strength
        verbose: print cost while training
        cost_interval: iterations between cost recordings/prints
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty
        self.lambda_ = lambda_
        self.verbose = verbose
        self.cost_interval = cost_interval

        self.weights = None
        self.bias = None
        self.costs = []

    def sigmoid(self, z):
        # Use scipy's expit which is stable and vectorized
        return expit(z)

    def loss_function(self, y, y_pred):
        m = len(y)
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        cost = -(1.0 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        if self.penalty == 'l2':
            cost += (self.lambda_ / (2.0 * m)) * np.sum(self.weights ** 2)
        elif self.penalty == 'l1':
            cost += (self.lambda_ / m) * np.sum(np.abs(self.weights))

        return cost

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        # initialize
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.costs = []

        for i in range(self.max_iter):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1.0 / n_samples) * X.T.dot(y_pred - y)
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            # regularization gradients (weights only)
            if self.penalty == 'l2':
                dw += (self.lambda_ / n_samples) * self.weights
            elif self.penalty == 'l1':
                dw += (self.lambda_ / n_samples) * np.sign(self.weights)

            # parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (i % self.cost_interval) == 0:
                cost = self.loss_function(y, y_pred)
                self.costs.append((i, cost))
                if self.verbose:
                    print(f"Iter {i}: cost {cost:.6f}")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        linear_model = X.dot(self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.mean(y_true == y_pred)
