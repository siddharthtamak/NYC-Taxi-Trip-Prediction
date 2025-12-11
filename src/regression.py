# Contains: MyLinearRegression, CreatePolynomialFeatures, MyRidgeRegression, MyLassoRegression classes

import numpy as np
import itertools

#Implementation of the class MyLinearRegression
class MyLinearRegression:
    def __init__(self):
        self.theta = None

    def _add_bias_term(self, X):
        """Adds a column of 1s to the start of X for the intercept term."""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        """
        Fits the linear regression model.
        Args:
            X : Feature matrix (n_samples, n_features)
            y : Target vector (n_samples,)
        """
        X_b = self._add_bias_term(X)
        try:
            part_1 = np.linalg.pinv(X_b.T @ X_b)
            part_2 = X_b.T @ y
            self.theta = part_1 @ part_2
            
        except np.linalg.LinAlgError:
            print("Error: The matrix (X.T @ X) is singular and cannot be inverted.")
            print("This can happen with perfectly collinear features.")
            self.theta = None

    def predict(self, X):
        """
        Makes predictions using the fitted model.
        Args:
            X : Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
            
        # 1. Add the bias term to X
        X_b = self._add_bias_term(X)
        
        # 2. Calculate predictions: y_pred = X_b @ theta
        y_pred = X_b @ self.theta
        return y_pred


# function to create polynomial features for polynomial regression
def create_polynomial_features(X, degree):
    """
    Generates a new feature matrix with polynomial features.
    
    Args:
        X : The original feature matrix (n_samples, n_features).
        degree : The degree of the polynomial.
        
    Returns:
        The new feature matrix with polynomial features.
    """
    n_samples, n_features = X.shape
    X_poly = [X]
    for d in range(2, degree + 1):
        indices = itertools.combinations_with_replacement(range(n_features), d)
        
        for index_comb in indices:
            new_feature = np.prod(X[:, index_comb], axis=1)
            X_poly.append(new_feature[:, np.newaxis])
    return np.hstack(X_poly)


# Implementation of the class MyRidgeRegression
class MyRidgeRegression:
    """
    This model adds L2 regularization to the Normal Equation to prevent the overfitting
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # The regularization strength
        self.theta = None

    def _add_bias_term(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        """
        Fits the Ridge regression model.
        Args: X : Feature matrix , y : Target vector
        """
        X_b = self._add_bias_term(X)
        
        # Create the Identity matrix for regularization, It's X_b.shape[1] because we need to match the dimensions of (X_b.T @ X_b)
        identity_matrix = np.eye(X_b.shape[1])
        
        identity_matrix[0, 0] = 0
        try:
            A = X_b.T @ X_b + self.alpha * identity_matrix
            b = X_b.T @ y
            self.theta = np.linalg.pinv(A) @ b
            
        except np.linalg.LinAlgError:
            print("Error: Matrix inversion still failed.")
            self.theta = None

    def predict(self, X):
        """Makes predictions using the fitted model."""
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
            
        X_b = self._add_bias_term(X)
        y_pred = X_b @ self.theta
        return y_pred


class MyLassoRegression:
    
    # ... (init and _soft_threshold methods are the same) ...
    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.tol = tol
        self.theta = None
        self.intercept_ = None
        
    def _soft_threshold(self, rho, lamda):
        if rho > lamda:
            return rho - lamda
        elif rho < -lamda:
            return rho + lamda
        else:
            return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.intercept_ = np.mean(y) 
        
        l1_penalty = self.alpha

        for i in range(self.n_iterations):
            theta_old = np.copy(self.theta)
            
            for j in range(n_features):
                y_pred_no_j = X.dot(self.theta) - X[:, j] * self.theta[j]
                residual = y - self.intercept_ - y_pred_no_j
                rho = np.dot(X[:, j], residual)
                
                denominator = np.dot(X[:, j], X[:, j])
                
                if denominator == 0:
                    # This feature is constant (all zeros), its weight must be 0
                    self.theta[j] = 0
                else:
                    # Original logic
                    self.theta[j] = self. _soft_threshold(rho, l1_penalty) / denominator

            if np.sum(np.abs(self.theta - theta_old)) < self.tol:
                break
                
    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        return self.intercept_ + X.dot(self.theta)
    
