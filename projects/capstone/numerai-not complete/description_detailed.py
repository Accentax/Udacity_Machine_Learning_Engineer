from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score

#####################################

#iterativ metodikk. i hver pass sjekk numerai ranking
#1 pass med baseline, adveserial validation om det er bedre.
##LR submitted til numerai, LR submitted men sammenligner numerai score med adveserial validation

#2 pass med baseline med feature engineering hver for seg
##PCA, FastICA, Feature selection, clustering,t-sne
##Sammenlign om concatenering av features er bedre en dem alene

#3 pass med baseline og features men med flere modeller gjør grid search
##på concatenated features kjør grid search for modellene

#4 pass med pipe fra forrige pass men ensamble av modeller
##lag pipes av modellene og sjekk ensamble score

#laste datasets

print("Loading data...")
# Load the data from the CSV files
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
Y = training_data['target']
X = training_data.drop('target', axis=1)
t_id = prediction_data['t_id']
x_prediction = prediction_data.drop('t_id', axis=1)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.1, random_state=0)


###########################

#statistikk
#korrelasjoner
#distribusjon av data
#visualisering

#kjøre benchmark

#t-sne og PCA med varians
# kjør clustering på t-sne
#sjekk feature importance opp mot de 50 andre


#feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris as load_data
import matplotlib.pyplot as plt
from scikitplot import classifier_factory

X1, y1 = load_data(return_X_y=True)
X1.shape
y1.shape
X.shape
y.shape
rf = classifier_factory(RandomForestClassifier(random_state=1))
rf.fit(X, Y)
rf.plot_feature_importances(feature_names=["feature"+str(i)for i in range(50)])
plt.show()

# Using the more flexible functions API
from scikitplot import plotters as skplt
rf = RandomForestClassifier()
rf = rf.fit(X, y)
skplt.plot_feature_importances(rf, feature_names=['petal length', 'petal width',
                                                  'sepal length', 'sepal width'])
plt.show()
#kjøre PCA med 50 variable og se hvor mye dimensjonene forklarer varians

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits as load_data
import scikitplot.plotters as skplt
import matplotlib.pyplot as plt

pca = decomposition.PCA()
pca.fit(X)


skplt.plot_pca_component_variance(pca)
plt.show()

# Plot the PCA spectrum
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

###############################################################################
# Prediction



#Parameters of pipelines can be set using ‘__’ separated parameter names:
#print fitting model
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.feature_selection import SelectKBest, chi2

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LogisticRegression())
])

N_FEATURES_OPTIONS = [2,10,20,40]
C_OPTIONS = [1,2,3,4]
param_grid = [
    {
        'reduce_dim': [PCA(), FastICA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'FastICA', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=3, n_jobs=2, param_grid=param_grid, scoring="log_loss")
grid.fit(X[:3000],Y[:3000])
grid.best_estimator_
grid.grid_scores_
print("fitted")
mean_scores
mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = -1*mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0.68, 0.71))
plt.legend(loc='upper left')
plt.show()



##### XGBoost

import numpy
import xgboost
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
y_proba=model.predict_proba(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
log_loss=metrics.log_loss(y_test,y_proba)
print("Log loss: %.5f"% (log_loss))

#GBM

from sklearn import ensemble

clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba=clf.predict_proba(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
log_loss=metrics.log_loss(y_test,y_proba)
print("Log loss: %.5f"% (log_loss))


####################################




#print best 
#clustering på t-sne og PCA

#lage pipes og gjøre grid search på antall clusters på 2d tsne og 20 dim PCA

Ensamble 1 rådata + t-sne + poly + cluster
pipe 1: scale/dim red/lr
pipe 2: scale /dim red /xgboost
pipe 3: scale /dim red /xgboost
pipe 4: scale /dim red /GBM

Ensamble 2 på rådata
scale lr
scale xgboost
scale gbm

Ensamble 3 på pca + t-sne + cluster
scale lr
scale xgboost
scale gbm

#velge de beste pipene basert på grid search. En xgboost, en GBM en LogisticRegression. 
#for hver ensamble kjør på tsne data og rå data 

#sjekke score

#gjøre adveserial validation
#sjekke score



