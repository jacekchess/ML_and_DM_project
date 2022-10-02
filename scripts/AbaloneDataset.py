import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.linalg import svd
from scipy.stats import zscore

# Load dataset and convert to numpy array
filename = 'Data/abalone.csv'
df = pd.read_csv(filename)
raw_data = df.values  
cols = range(1, 9) 
X = raw_data[:, cols].astype(np.float64)

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, 0] 
#print('class labels', classLabels)
classNames = np.unique(classLabels)

# Extract class names and encode with integers (dict)
classDict = dict(zip(classNames,range(len(classNames))))
#print('class dictionary', classDict)

y = np.array([classDict[v] for v in classLabels])
#print('y', y)

N, M = X.shape
print('N data objects = ', N, 'M attributes = ', M)

# Number of classes
C = len(classNames)

###########################################################################
# Plotting the 8 attributes and their standard deviation
fig, ax = plt.subplots(figsize =(16, 9))
r = np.arange(1,X.shape[1]+1)
ax.barh(r, np.std(X,0))
plt.yticks(r, attributeNames, fontsize=12)

for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
plt.xlabel('Standard deviation')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
ax.invert_yaxis()
 
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')

ax.set_title('Attributes and their standard deviation',
             loc ='center')

# Box plot of each attribute
plt.figure()
plt.title('Abalone: Boxplot')
plt.boxplot(X)
plt.xticks(range(1,M+1), attributeNames, fontsize=6, rotation=30)

#########################################################################

### STANDARDIZED DATA ###
# CenteredData = X - np.ones((N,1))*X.mean(axis=0)
Y = X - np.ones((N, 1))*X.mean(axis=0)
Y = Y*(1/np.std(Y,0))
print('Standardized data', Y)

### FINDING THE OUTLIERS ###
# Box plot of each attribute after standardization
plt.figure()
plt.title('Abalone: Boxplot Standardized Data')
plt.boxplot(Y)
plt.xticks(range(1,M+1), attributeNames, fontsize=6, rotation=30)

# Plot histograms of all attributes.
plt.figure()
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(int(M)/u))

for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(Y[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Abalone: Histogram after standardization')

# Remove outliers from standardized data
print('Height column: ', Y[:,2])
outlier_mask = (Y[:,2] > 8)
print('Outlier mask', outlier_mask)
valid_mask = np.logical_not(outlier_mask)

Y = Y[valid_mask,:]
print('Y', Y)
y = y[valid_mask]
print('y', y)
N = len(y)
print('N', N)

plt.figure(figsize=(14,9))
u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(Y[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    if i==0: plt.title('Abalone Histogram after outlier detection')

########################### PCA ##########################################
# PCA by computing Singular Value Decomposition of y
# S matrix with the singular values (square root of the eigenvalues)
U, S, Vh = svd(Y, full_matrices=False)
print('S', S)
V = Vh.T
print('V matrix containing the principal component directions as columns: ', V)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
print('rho', rho)
threshold = 0.9

# Sanity check
pca1 = ((167.4450905)**2) / ((167.4450905)**2 + (53.90338916)**2 + (32.85600272)**2 + (26.33134205)**2 + (18.83705776)**2 + (16.28268244)**2 + (7.28172523)**2 + (5.1697003)**2)
pca2 = ((167.4450905)**2 + (53.90338916)**2) / ((167.4450905)**2 + (53.90338916)**2 + (32.85600272)**2 + (26.33134205)**2 + (18.83705776)**2 + (16.28268244)**2 + (7.28172523)**2 + (5.1697003)**2)
print('Variance explained by the first principal component', pca1*100)
print('Variance explained by the first two principal components', pca2*100)

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


# Simple Visualization of the first two attributes
fig = plt.figure()
plt.title('Abalone data')

# Data attributes to be plotted are the first 2 f.ex.
i = 0
j = 1

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Y[class_mask,i], Y[class_mask,j], 'o',alpha=.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.show()

# Compute the projection onto the principal components
Z = U*S;  
print('Z', Z)

# Choose two PCs to plot the projection
i = 0
j = 1  
# Plot projection
plt.figure()
C = len(classNames)
for c in range(C):
    plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.title('Projection: PCA' )
plt.legend(classNames)
plt.axis('equal')



plt.figure()
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, r)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Abalone: PCA Component Coefficients')

print('PC1:')
print(V[:,0].T)
print('PC2:')
print(V[:,1].T)
print('PC3:')
print(V[:,2].T)

plt.show()
        