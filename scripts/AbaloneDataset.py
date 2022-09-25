import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd

# Load dataset and convert the dataframe to numpy array
filename = 'Data/abalone.csv'
df = pd.read_csv(filename)
raw_data = df.values  
#print(raw_data)
#print('data shape', raw_data.shape)

# Data matrix X 
cols = range(1, 9) 
X = raw_data[:, cols].astype(np.float64)
#print('X', X)

# Extract the attribute names from the header of the csv
attributeNames = np.asarray(df.columns[cols])
print(attributeNames)

# Extract the strings for each sample from the first column
classLabels = raw_data[:, 0] 
#print('class labels', classLabels)

# Save the 3 class labels 
classNames = np.unique(classLabels)
#print('class names', classNames)

# Extract class names and encode with integers (dict)
classDict = dict(zip(classNames,range(len(classNames))))
#print('class dictionary', classDict)

y = np.array([classDict[v] for v in classLabels])
#print('y', y)

N, M = X.shape
print('N data objects = ', N, 'M attributes = ', M)

# Number of classes
C = len(classNames)

# Simple Visualization of the first two attributes
"""
fig = plt.figure()
plt.title('Abalone data')

# Data attributes to be plotted are the first 2 f.ex.
i = 0
j = 1

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.show()

"""
### CENTERED DATA ###
# Subtract mean value from data
CenteredData = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing Singular Value Decomposition of y
U, S, V = svd(CenteredData, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('Abalone_PCA.png')


# Attributes have different scales. Standard deviation of Rings is too high.
plt.figure()
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Abalone: attribute standard deviations')


### STANDARDIZED DATA ###
StandardizedData = X - np.ones((N, 1))*X.mean(0)
StandardizedData = StandardizedData*(1/np.std(StandardizedData,0))

Data = [CenteredData, StandardizedData]
titles = ['Centered Data', 'Standardized Data']
threshold = 0.9
# Choose two PCs to plot the projection
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.5)
plt.title('Abalone: Effect of standardization')
nrows=3
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD
    U,S,Vh = svd(Data[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()
        