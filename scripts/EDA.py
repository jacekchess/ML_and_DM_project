# Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data

# %%
abalone = pd.read_csv('../data/abalone.csv', header=0)
abalone.head()
# abalone.describe()

# Plots

# %%
pairplot = sns.pairplot(abalone, hue='Sex')
plt.savefig('../plots/pairplot.png', dpi=300, bbox_inches='tight', transparent = False)

# %%
corr_plot = sns.heatmap(abalone.corr(), vmax=1, vmin=0, cmap='Blues')
corr_plot.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.savefig('../plots/heatmap.png', dpi=300, bbox_inches='tight', transparent = False)

