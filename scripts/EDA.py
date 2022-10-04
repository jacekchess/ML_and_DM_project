# Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn import preprocessing

# Load data

# %%
abalone = pd.read_csv('../data/abalone.csv', header=0)
abalone.head()
# abalone.describe()

# %%
abalone_std = preprocessing.scale(abalone.iloc[:,1:])
abalone_std = pd.DataFrame(abalone_std, columns=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Rings'])

abalone_std.drop(abalone_std[abalone_std.Height > 8].index, inplace=True)
# Plots

# %%
# pairplot = sns.pairplot(abalone, hue='Sex')
# pairplot
for i in range(1, 9):
    mu, std = norm.fit(np.array(abalone.iloc[:, [i]])) 
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.subplot(3,3,i)
    plt.hist(np.array(abalone.iloc[:, [i]]), bins=30, density=True)
    plt.plot(x, p, 'k', linewidth=2, c='red')

plt.show()

# %%
mu, std = norm.fit(np.array(abalone.iloc[:, [1]])) 
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.hist(np.array(abalone.iloc[:, [1]]), bins=30)
plt.plot(x, p, 'k', linewidth=2)

# abalone.hist(bins=30, figsize=(15, 10))
# plt.savefig('../plots/pairplot.png', dpi=300, bbox_inches='tight', transparent = False)

# %%
corr_plot = sns.heatmap(abalone_std.corr(), vmax=1, vmin=0, cmap='Blues', annot=True)
corr_plot.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.savefig('../plots/heatmap.png', dpi=300, bbox_inches='tight', transparent = False)

# %%
abalone.corr()

# %%
np.corrcoef(abalone[:1])

# %%
cols = range(1, 9) 
pd.data_frame(np.corrcoef(abalone.values[:, cols].astype(np.float64)))



