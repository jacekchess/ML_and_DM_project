# Load libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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
