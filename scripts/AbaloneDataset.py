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
#print(attributeNames)

# Extract the strings for each sample from the first column
classLabels = raw_data[:, 0] 
print('class labels', classLabels)

# Save the 3 class labels 
classNames = np.unique(classLabels)
#print('class names', classNames)

# Extract class names and encode with integers (dict)
classDict = dict(zip(classNames,range(len(classNames))))
#print('class dictionary', classDict)

y = np.array([classDict[v] for v in classLabels])
print('y', y)

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
#########################################################################


### STANDARDIZED DATA ###
# CenteredData = X - np.ones((N,1))*X.mean(axis=0)
Y = X - np.ones((N, 1))*X.mean(axis=0)
Y = Y*(1/np.std(Y,0))

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


'''
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

'''
plt.show()
        