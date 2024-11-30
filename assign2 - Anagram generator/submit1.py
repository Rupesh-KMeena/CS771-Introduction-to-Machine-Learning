import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.linalg import khatri_rao

# Function to create the feature mapping
def my_map(X):
    n_samples, n_bits = X.shape
    # Initialize the feature matrix with the appropriate dimensions
    feat = np.zeros((n_samples, 64))
    
    for i in range(n_samples):
        x = X[i]
        # Constructing the feature vector as described in the derivation
        for j in range(32):
            feat[i, j] = np.prod([1 - 2 * x[k] for k in range(j, 32)])
        for j in range(32, 64):
            feat[i, j] = np.prod([1 - 2 * x[k - 32] for k in range(j - 32, 32)])
    
    return feat

# Function to train the linear models
def my_fit(X_train, y0_train, y1_train):
    # Map the challenges to the feature space
    X_mapped = my_map(X_train)

    # Train the model for Response0
    model0 = LogisticRegression(fit_intercept=True, solver='liblinear')
    model0.fit(X_mapped, y0_train)
    w0 = model0.coef_[0]
    b0 = model0.intercept_[0]
    
    # Train the model for Response1
    model1 = LogisticRegression(fit_intercept=True, solver='liblinear')
    model1.fit(X_mapped, y1_train)
    w1 = model1.coef_[0]
    b1 = model1.intercept_[0]
    
    return w0, b0, w1, b1

def read_data(train_path, test_path):
    # Load training data
    train_data = np.loadtxt(train_path, delimiter=' ')
    X_train = train_data[:, :32].astype(int)
    y0_train = train_data[:, 32].astype(int)
    y1_train = train_data[:, 33].astype(int)

    # Load testing data
    test_data = np.loadtxt(test_path, delimiter=' ')
    X_test = test_data[:, :32].astype(int)

    return X_train, y0_train, y1_train, X_test

if __name__ == "__main__":
    train_path = "C:\\Users\\rupes\\OneDrive\\Documents\\CS771\\assn1\\public_trn.txt"
    test_path = "C:\\Users\\rupes\\OneDrive\\Documents\\CS771\\assn1\\public_tst.txt"

    # Read the training and testing data
    X_train, y0_train, y1_train, X_test = read_data(train_path, test_path)
    
    # Train the models
    w0, b0, w1, b1 = my_fit(X_train, y0_train, y1_train)
    
    # Map the test data to the feature space
    feat_test = my_map(X_test)
    print(feat_test)
