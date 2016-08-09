
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project 2: Building a Student Intervention System

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
#
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
#
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ### Question 1 - Classification vs. Regression
# *Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

# **Answer: **
# The task at hand is to identify students who need help in order to prevent failiure to graduate. The task is then to identify students that need help or does not need help. We can divide the students into two groups, or classes. Therefore the problem is to classify help needed or not help needed.

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[1]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print ("Student data read successfully!")


# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
#

# In[2]:

# TODO: Calculate number of students
n_students = len(student_data)

# TODO: Calculate number of features
n_features = int(student_data.shape[1])-1

# TODO: Calculate passing students
n_passed = len(student_data[student_data['passed']=='yes'])

# TODO: Calculate failing students
n_failed = len(student_data[student_data['passed']=='no'])

# TODO: Calculate graduation rate
grad_rate = float(n_passed)/n_students

# Print the results
print ("Total number of students: {}".format(n_students))
print ("Number of features: {}".format(n_features))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))
print ("Graduation rate of the class: {:.2f}%".format(grad_rate))


# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
#
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
#
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[3]:

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print ("\nFeature values:")
print (X_all.head())


# ### Preprocess Feature Columns
#
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
#
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
#
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[4]:

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)

        # Collect the revised columns
        output = output.join(col_data)

    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[5]:

# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, test_size=num_test/float(X_all.shape[0]), random_state=42)


# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ## Training and Evaluating Models
# In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.

# ### Question 2 - Model Application
# *List three supervised learning models that are appropriate for this problem. What are the general applications of each model? What are their strengths and weaknesses? Given what you know about the data, why did you choose these models to be applied?*

# **Answer: **
# I think the following three models would be appropriate.
#
# Random forrest
#
# k-nearest neighbors algorithm - This is a relatively simple model that can be used for both regression and classification. For classification an object is classified based on majority vote of the k nearest objects based on their class, where k is an integer. Also it is possible to assign weight based on how close the object is. E.g 1/d wher d is the distance. The model is easy to understand and it is straight forward to evaluate the classifications of the model. However the model has short comings if there are local structures in the data that cannot be captured.
#
#
# Decision Trees - This classifier uses a decision tree to map observations about objects to draw conclusion. It consists of leafs and branches. Where each leaf represents a decision on a feature variable and branches out to new leafs, ending up in the end on a decision predicting the target variable. The model is particularly good for mining and for visualisation, as the tree is more easily understood compared to a black box algorithm. However a decision tree is prone to overfitting as the tree can grow too large if not pruned to avoid overfitting. Also at each decision is prone to fall into local minima as the decisions are made locally for each leaf. There are also consepts that decision trees have problems with representing without becoming overly complex, or not at all, such as the XOR function.
#
#

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[6]:

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[54]:

# TODO: Import the three supervised learning models from sklearn
# from sklearn import model_A
# from sklearn import model_B
# from skearln import model_C
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# TODO: Initialize the three models
clf_A = RandomForestClassifier(n_estimators=30,max_features="auto")
clf_B = KNeighborsClassifier(n_neighbors=5)
clf_C = DecisionTreeClassifier(random_state=0,max_features="auto")

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
for i in [clf_A,clf_B,clf_C]:
    print("-"*20)
    train_predict(i,X_train_100,y_train_100,X_test,y_test)
    print("\n")
    train_predict(i,X_train_200,y_train_200,X_test,y_test)
    print("\n")
    train_predict(i,X_train_300,y_train_300,X_test,y_test)
    print("-"*20)


# Results from run selected for evaluation.
# --------------------
#
# **Training a RandomForestClassifier using a training set size of 100. . .**
#
# Trained model in 0.0460 seconds
# Made predictions in 0.0010 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0020 seconds.
# F1 score for test set: 0.7660.
#
#
# **Training a RandomForestClassifier using a training set size of 200. . .**
#
# Trained model in 0.0440 seconds
# Made predictions in 0.0020 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0020 seconds.
# F1 score for test set: 0.7941.
#
#
# **Training a RandomForestClassifier using a training set size of 300. . .**
#
# Trained model in 0.0440 seconds
# Made predictions in 0.0030 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0010 seconds.
# F1 score for test set: 0.7914.
#
#
# **Training a KNeighborsClassifier using a training set size of 100. . .**
#
# Trained model in 0.0000 seconds
# Made predictions in 0.0010 seconds.
# F1 score for training set: 0.8060.
# Made predictions in 0.0010 seconds.
# F1 score for test set: 0.7246.
#
#
# **Training a KNeighborsClassifier using a training set size of 200. . .**
#
# Trained model in 0.0010 seconds
# Made predictions in 0.0020 seconds.
# F1 score for training set: 0.8800.
# Made predictions in 0.0010 seconds.
# F1 score for test set: 0.7692.
#
#
# **Training a KNeighborsClassifier using a training set size of 300. . .**
# Trained model in 0.0010 seconds
# Made predictions in 0.0050 seconds.
# F1 score for training set: 0.8809.
# Made predictions in 0.0020 seconds.
# F1 score for test set: 0.7801.
#
#
# **Training a DecisionTreeClassifier using a training set size of 100. . .**
#
# Trained model in 0.0000 seconds
# Made predictions in 0.0000 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0000 seconds.
# F1 score for test set: 0.6504.
#
#
# **Training a DecisionTreeClassifier using a training set size of 200. . .**
#
# Trained model in 0.0010 seconds
# Made predictions in 0.0000 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0000 seconds.
# F1 score for test set: 0.7519.
#
#
# **Training a DecisionTreeClassifier using a training set size of 300. . .**
#
# Trained model in 0.0010 seconds
# Made predictions in 0.0000 seconds.
# F1 score for training set: 1.0000.
# Made predictions in 0.0000 seconds.
# F1 score for test set: 0.7727.
#

# ### Tabular Results
# Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

# ** Classifer 1 - RandomForestClassifier **
#
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |          0.0460        |        0.0010          | 1.0000          |   0.7660              |
# | 200               |         0.0440          |        0.0020          | 1.0000          |    0.7941             |
# | 300               |            0.0440       |         0.0030         | 1.0000           |    0.7914      |
#
# ** Classifer 2 - KNeighborsClassifier **
#
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |     0.0000              |  0.0010                | 0.8060                 |  0.7246               |
# | 200               |     0.0010              |     0.0020             |  0.8800                |   0.7692              |
# | 300               |     0.0010              |   0.0050               |  0.8809                |     0.7801     |
#
# ** Classifer 3 - DecisionTreeClassifier **
#
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |    0.0000              |  0.0000                 | 1.0000           |0.6504
# | 200               |     0.0010             |   0.0000               |  1.0000         |    0.7519             |
# | 300               |    0.0010              |   0.0010                | 1.0000          |   0.7727             |

# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score.

# ### Question 3 - Chosing the Best Model
# *Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

# **Answer: **
# We ran the classifiers on a home computer.
#
# We can see that both the KNeigboorsClassifier and DecisionTreeClassifier gets an F1 score of 1.000 but a lower test score. Which is due to overfitting the training data. The RandomForrestClassifier (RF) gets the highest F1 Score on the test set of 0.7941 for 200 samples in the training set. The prediction time of RF classifier increases linearly with more samples. It takes 0.0030 seconds for 300 students. That is 0.00001 seconds per students. Meaning that making predictions for a million students should take 10 seconds. Which means that even though it is slowe than the DecisionTreeClassifier it is still more than fast enough with the current hardware to be used in production.
#
# My recommendation to the board is therefore that we continue with the Random Forrest Classifier.
#
#
#

# ### Question 4 - Model in Layman's Terms
# *In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. For example if you've chosen to use a decision tree or a support vector machine, how does the model go about making a prediction?*

# **Answer: **
# To explain how a random forrest works I will use an example where I want to figure out what movie I want to see, because I am so indecisive.
#
# So I ask my friend Steve if I will like the movie. In order for Steve to know this I need to tell him some of the movies I like and dont like. Then when I ask him if I will like a given movie, he will do 20 questions based on the knowledge he gained from what I told him. E.g "Is X a romantic movie?", "Does Johnny Depp star in the movie?" an so on. He asks informative questions first the gives me a yes or no question at the end. By doing this Steve has created a decision tree for my movie preferences. But Steve is only human and does not generalize my taste and preferences well. (He overfits). In order to get more accurate recomendations I ask a group of my friends if I will like the movie in the same way, i.e making them vote. Now my friends are decision trees will make up the forrest. But I dont want all my friends to think the same way about my preferences.
#
# Since I dont want the same answer from all my friends I give them slightly different data. In addition I will make my friends have to choose a random attribute to split on, i.e not all my friends can ask if Leonardo Di Caprio is in the movie, just because I told them I liked Titanic and Inception. Now my friends are asking different questions at different times. My friends now form a random forrest.
#

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
# - Initialize the classifier you've chosen and store it in `clf`.
# - Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
#  - Set the `pos_label` parameter to the correct value!
# - Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.

# In[ ]:

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.grid_search import GridSearchCV
# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators':[10,30,60,100,300,1000,1500,3000],'max_features':['auto','sqrt','log2']}#,'min_samples_split':[2,5,10],'max_depth':[None,5,10,100]   }

# TODO: Initialize the classifier
clf = RandomForestClassifier()

# TODO: Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Question 5 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: **
#
# The tunes model has an F1 score of 0.7917 which is higher than the F1 score for the untuned model, however the untuned model has a higher score at 200 samples. However since the forrest is random this is not always the case if we retrain. One could easily find a case where the F1 score is higher for more training samples.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
