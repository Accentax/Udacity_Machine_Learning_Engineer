from tsne import bh_sne

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('white')
sns.set_context('notebook', font_scale=2)
from bh_sne import BH_SNE
bh_sne(data, pca_d=None, d=2, perplexity=30., theta=0.5,
           random_state=None, copy_data=False)
           """
    Run Barnes-Hut T-SNE on _data_.
    @param data         The data.
    @param pca_d        The dimensionality of data is reduced via PCA
                        to this dimensionality.
    @param d            The embedding dimensionality. Must be fixed to
                        2.
    @param perplexity   The perplexity controls the effective number of
                        neighbors.
    @param theta        If set to 0, exact t-SNE is run, which takes
                        very long for dataset > 5000 samples.
    @param random_state A numpy RandomState object; if None, use
                        the numpy.random singleton. Init the RandomState
                        with a fixed seed to obtain consistent results
                        from run to run."""
                        
                        
                        
                        
from tsne import bh_sne
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X_2d = bh_sne(X)
len(X)
X_2d

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model



# Set seed for reproducibility
np.random.seed(0)

print("Loading data...")
# Load the data from the CSV files
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)


#scatter
pd.scatter_matrix(training_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
#correlation
fig, ax = plt.subplots(figsize=(50, 50))
sns.heatmap(training_data.corr(), square=True)

#violin
fig, ax = plt.subplots(figsize=(24, 12))

sns.violinplot(data=df_plot, x='feature', y='value', split=True, hue='target', scale='area', palette='Set3', cut=0, lw=1, inner='quart')
sns.despine(left=True, bottom=True)

ax.set_xticklabels(feature_cols, rotation=90);

# Transform the loaded CSV data into numpy arrays
Y = training_data['target']
X = training_data.drop('target', axis=1)
t_id = prediction_data['t_id']
x_prediction = prediction_data.drop('t_id', axis=1)
len(X[:1000])
X_2d_1k = bh_sne(X[:10000]) #takes around 1-5 min

dictdf={'adim':X_2d_1k[:,0],'bdim':X_2d_1k[:,1], 'index':t_id.values[:10000]}
id=t_id.values[:10000]
len(X_2d_1k[:,0])
#t_id.values[:10000]
X_2d_1k_df = pd.DataFrame(data=dictdf)
X_2d_1k_df.to_csv("X_2d_1k_df.csv", index=True)
tsne_data=X_2d_1k_df

Y
fig, ax = plt.subplots(figsize=(25, 25))
plt.scatter(tsne_data['adim'].values, tsne_data['bdim'].values, c=Y.values[:10000], cmap='Set3', alpha=0.8, s=4, lw=0)
#http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
#standard scaler
#remove mean divide std
#http://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features

#create pipe
#scale
#log remove mean std division
#polynomial features and scaling
#combination of above

#PCA on data above
#t-SNE on on raw data with different parameters
#check if clusters can be found. If not try other data from pipe.
#do t-sne on test data.
#do silouette scoring

#setup the different models and run hyperopt.


#http://hyperopt.github.io/hyperopt-sklearn/


