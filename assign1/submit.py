import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def my_fit(X_train, y0_train, y1_train):
    model0 = LogisticRegression(max_iter=1000)
    model0.fit(X_train, y0_train)
    w0 = model0.coef_[0]
    b0 = model0.intercept_[0]
    
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_train, y1_train)
    w1 = model1.coef_[0]
    b1 = model1.intercept_[0]
    
    return w0, b0, w1, b1

def my_map(X):
    return np.cumprod(np.flip(2 * X - 1, axis=1), axis=1)


if __name__ == "__main__":
    train_data = np.loadtxt("C:\\Users\\rupes\\OneDrive\\Documents\\CS771\\assn1\\public_trn.txt", delimiter=' ')
    X_train = train_data[:, :-2]
    y0_train = train_data[:, -2]
    y1_train = train_data[:, -1]

    test_data = np.loadtxt("C:\\Users\\rupes\\OneDrive\\Documents\\CS771\\assn1\\public_tst.txt", delimiter=' ')
    X_test = test_data[:, :-2]
    y0_test = test_data[:, -2]
    y1_test = test_data[:, -1]

    w0, b0, w1, b1 = my_fit(X_train, y0_train, y1_train)

    print("Weights for Response0:", w0)
    print("Bias for Response0:", b0)
    print("Weights for Response1:", w1)
    print("Bias for Response1:", b1)

    X_test_mapped = my_map(X_test)

    pred0 = (np.dot(X_test_mapped, w0) + b0 > 0).astype(int)
    pred1 = (np.dot(X_test_mapped, w1) + b1 > 0).astype(int)

    accuracy0 = np.mean(pred0 == y0_test)
    accuracy1 = np.mean(pred1 == y1_test)

    print("Test Accuracy for Response0:", accuracy0)
    print("Test Accuracy for Response1:", accuracy1)
