# Load libraries

import matplotlib.pyplot as plt
# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# Load data
# %%
data = pd.read_csv('../data/abalone.csv')
data.head()

# Detect outliers
# %%
for iter, column in enumerate(data.columns[1:9]):
    plt.subplot(3, 3, iter+1)
    plt.boxplot(data[column])
    
# %%
plt.boxplot(data.Height)

# Removing outliers
# %%
data = data[data.Height < 0.4]



### Regression

# Train test split
# %%
X = data.drop('Rings', axis=1).to_numpy()
y = data['Rings'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# One of K encoding
# %% 
ct = ColumnTransformer([("Sex", OneHotEncoder(), [0])], remainder = 'passthrough')
ct = ct.fit(X_train)

X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

# Standardization
# %%
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X = scaler_X.fit(X_train)
scaler_y = scaler_y.fit(y_train.reshape(-1, 1))

X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Save datasets
# %%
columns = ['Is_F', 'Is_I', 'Is_M', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']

pd.DataFrame(X_train, columns = columns).to_csv('../data/regression/X_train.csv', header=True, index=False)
pd.DataFrame(X_test, columns = columns).to_csv('../data/regression/X_test.csv', header=True, index=False)

pd.DataFrame(y_train, columns = ['Rings']).to_csv('../data/regression/y_train.csv', header=True, index=False)
pd.DataFrame(y_test, columns = ['Rings']).to_csv('../data/regression/y_test.csv', header=True, index=False)



### Classification

# Train test split
# %%
X = data.drop('Sex', axis=1).to_numpy()
y = data['Sex'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#For the sex column, M = 0, F = 1, and I = -1
sex = data.iloc[:,0].to_numpy()
m, f, i = sex == 'M', sex == 'F', sex == 'I'
sex[m], sex[f], sex[i] = 0., 1., -1.
sexVector = sex.reshape(len(sex), 1)


# Standardization
# %%
scaler_X = StandardScaler()
scaler_X = scaler_X.fit(X_train)

X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

# Save datasets
# %%
columns = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Rings']

pd.DataFrame(X_train, columns = columns).to_csv('../data/classification/X_train.csv', header=True, index=False)
pd.DataFrame(X_test, columns = columns).to_csv('../data/classification/X_test.csv', header=True, index=False)

pd.DataFrame(y_train, columns = ['Sex']).to_csv('../data/classification/y_train.csv', header=True, index=False)
pd.DataFrame(y_test, columns = ['Sex']).to_csv('../data/classification/y_test.csv', header=True, index=False)


#Logistic Regression

#Defines the logistic function for use in logistic regression (logistic sigmoid)
def logistic(self, z):
   return 1 / (1 + np.exp(-z))

#Computes the classification error for a set of results and given class labels
def classificationError(self, results, classLabels):
   n = results.size
   numErrors = 0.
   
   for i in range(n):
            if (results[i] >= 0.5 and classLabels[i]==0) or (results[i] < 0.5 and classLabels[i]==1):
                numErrors += 1

            return numErrors / n
        
#Classify Sex of Adult Abalone with Logistic Regression with removing Infants       
def doLogistic(self, doPCA=True):
    
        print('--Starting Logistic Regression for sex prediction--')
        
        #Reading in the data to training and testing subsets
        dat = self.readData(ordinate='sex')
        xTrain, xTest = dat['X_train'], dat['X_test']
        yTrain, yTest = dat['y_train'], dat['y_test']

        print("Removing infants from the data.")
        
        #Remove infants from training subsets
        keep = np.where(yTest != -1)[0]
        yTrain = yTrain[keep].astype(int)
        xTrain = xTrain[keep,:]
        
        #Remove infants from testing subsets
        keep = np.where(yTrain != -1)[0]
        yTest = yTest[keep].astype(int)
        xTest = xTest[keep,:]
        
        print('Training the logistic regression model.')
        #Train the logistic regression model
        logReg = linear_model.LogisticRegression()
        logReg.fit(xTrain, yTrain)
        
        print('Performing predictions based on the trained model.')
        #Do classification using the trained model
        results = logReg.predict(xTest)

        error = self.classificationError(results, yTest)
        print('Classification Error: {:4.2f}%'.format(error*100))
        
        print('--Logistic Regression for sex prediction completed.--')
        
#Other trials for logistic regression        
#clf = LogisticRegression(tol=0.1)

#clf.fit(X_train,y_train)

#y_pred=clf.predict([X_test])

#y_pred

#Decision tree classifier



#Baseline
