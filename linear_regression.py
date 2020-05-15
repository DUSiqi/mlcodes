"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    [n,d] = X.shape
    y_ = np.dot(X,w)
    err = np.float64( (np.sum(np.abs(y_-y)))/n) #mae
    #####################################################
    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
  #####################################################
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    [n,d] = X.shape
    A = np.dot(X.T,X)
    eigen_values,eigen_vectors = np.linalg.eig(A)
    min_eigen_value = np.min(np.abs(eigen_values))
    lam = 0
    while min_eigen_value < 1e-5:
        lam += 1e-1
        I = np.eye(d)
        A = A + lam*I
        eigen_values, eigen_vectors = np.linalg.eig(A)
        # print(eigen_values)
        # print(eigen_vectors)
        min_eigen_value = np.min(np.abs(eigen_values))

    w = np.dot(np.dot(np.linalg.inv(A), X.T), y)
    #####################################################
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
    [n,d] = X.shape
    if lambd is None:
        lambd = 0
    A = np.dot(X.T,X)
    I = np.eye(d)
    A = A + lambd * I
    # print(A.shape)
    w = np.dot(np.dot(np.linalg.inv(A), X.T), y)
  #####################################################
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    mae_val_list = []
    for i in range(-19,20):
        lambd = 10 ** i
        w = regularized_linear_regression(Xtrain,ytrain,lambd)
        mae_val = mean_absolute_error(w,Xval,yval)
        mae_val_list.append(mae_val)
    bestlambda = 10 ** int(np.argmin(mae_val_list) - 19)
    #####################################################
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    [n,d] = X.shape
    XX = X
    for i in range(2,power+1):
        XX = np.hstack((XX,np.power(X,i)))
    #####################################################
    X = XX
    return X


