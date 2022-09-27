from AbaloneDataset import *

# Simple Visualization of the first two attributes

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



# Compute the projection onto the principal components
# Z = U*S;  // same Z result
Z = Y @ V
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
plt.show()