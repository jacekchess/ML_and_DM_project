# Importing libraries and data
# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind

X_test = pd.read_csv('../data/classification/X_test.csv').to_numpy()
X_train = pd.read_csv('../data/classification/X_train.csv').to_numpy()
y_test = pd.read_csv('../data/classification/y_test.csv').to_numpy()
y_train = pd.read_csv('../data/classification/y_train.csv').to_numpy()

# Learning logistic regression for different regularization parameters using cross validation
# %%
CV = KFold(n_splits=10, shuffle=True) # 10-fold cross validation
alphas = [1, 3, 5, 7, 9] # regularization values for testing - the higher the stronger regularization
errors = np.zeros((len(alphas), 10)) # empty error matrix

k = 0
for train_index, test_index in CV.split(X_train):
    # Create datasets for every cross validation fold
    X_train_CV, X_valid_CV = X_train[train_index], X_train[test_index]
    y_train_CV, y_valid_CV = y_train[train_index], y_train[test_index]

    for i, alpha in enumerate(alphas):
        # Test every parameter for every cross validation fold
        log_regression = LogisticRegression(C = alphas[i]).fit(X_train_CV, y_train_CV.ravel())
        errors[i, k] = accuracy_score(y_valid_CV, log_regression.predict(X_valid_CV))

    k += 1

# Calculate means from cross validation results
errors = np.mean(errors, axis = 1)

# %%
# Plot results
plt.plot(alphas, errors)

# Test the best logistic regression and explain
# %%
log_regression = LogisticRegression(C = 3.0).fit(X_train, y_train.ravel())
observation_number = 2 # because it works xd

print('Prediction: ', log_regression.predict(X_test)[observation_number])
print('Acctual value: ', y_test[observation_number][0])
print('Coefficients for this class: ', log_regression.coef_[1]) # For predicting the probability of certain class we have a set of coef for every class
print('Features for this observation: ', X_test[observation_number]) # Theoreticaly multiplying coef and this vector should result in probablity
print('Probability of beloging to this class: ', log_regression.predict_proba(X_test)[observation_number][1]) # Probability for this class