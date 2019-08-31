# # Entomics Project
# 
# ## The Soldier Fly Hunger Game  
# ### _Predicting the hunger of insects using historical data

# ## Code only, neither results nor conclusions are shown in the following code

# ## Import the necessary `libraries` and upload the `data`
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential              # class of neural networks with one layer after the other
from keras.layers.core import Dense, Activation  # type of layers
from keras.optimizers import SGD                 # Optimisers, here the stochastic gradient descent 
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Data have been imported as data_set
data_set = pd.read_csv("Data/FeedRecords.csv")
# First check at the data
data_set.head()
# check the Null value
data_set[data_set['Previous Feed 3'].isnull()].head()
# check the features provided
data_set.describe()
data_set.info()
# How many different 'Feed' values for each condition:
data_set.groupby(['Age', 'Temperature', 'Moisture', 'Food Left',
       'Black Larvae', 'Dead Larvae', 'Overpopulated'])['Feed'].nunique().sort_values(ascending=False).head(20)
# Some funny data, like tray 382
data_set[data_set['Tray']==382][['Date', 'Tray', 'Food Left', 'Feed']]
# I start to create sub-set split by type of data. First is numerical:
data_set_num = data_set[['Age', 'Temperature', 'Moisture']]
# Boolean dataset:
data_set_bool = data_set[['Black Larvae', 'Dead Larvae', 'Overpopulated']]
# I convert the True/False values in 1/0
for col in data_set_bool.columns:
    data_set_bool[col] = data_set_bool[col].map({True: 1, False: 0})
# Category columns (so far only 'Food Left')
data_set_cat = data_set[['Food Left']]
# Quick check on number of categoris...
np.unique(data_set['Food Left'], return_counts=True)
# ...and the feature I want to predict
np.array(data_set['Feed'])
# Previous Feed data (numerical)
data_set_previous = data_set[['Previous Feed 3', 'Previous Feed 2', 'Previous Feed 1']]

# ### Following, some graphs to grasp a geneal comprehension of the data
# Distribution of the main features
col_check = ['Feed', 'Temperature', 'Moisture']
for col in col_check:
    sns.countplot(data_set[col])
    plt.xticks(rotation=90)
    plt.title(col + ' ({} possible outcomes)'.format(data_set[col].nunique()))
    plt.show()
# According to the correlation matrix, there is a positive correlation between 'Feed' and (previous feeds) and Temperature and an inverse correlation with 'Black Larvae'. Strange enough, there is no correlation between 'Feed' and 'Overpopulated'.
corrmat = data_set.corr()
plt.figure(figsize=(9,7))
plt.title('Correlation Matrix')
sns.heatmap(corrmat, 
            linewidths=0.5, 
            cmap="RdBu", 
            vmin=-1, 
            vmax=1, 
            annot=True)
plt.xticks(rotation=270);

# ## Model 1: logistic regression with all the features provided

data_set_num_bool = pd.concat([data_set_num , data_set_bool], axis=1)
enc = OneHotEncoder()
enc.fit(data_set_cat)
# I need only 2 columns, the third one is not necessary
new_cat_col = enc.transform(data_set_cat).toarray()[:,0:2]
# Converting a dataframe in array
X_data_set_previous = np.array(data_set_previous.fillna(0))
X_not_scaled = np.array(pd.concat([data_set_num , data_set_bool, data_set_previous.fillna(0)], axis=1))
X = np.hstack((X_not_scaled, new_cat_col))
y = np.array(data_set['Feed'])
# Usually test_size is about 0.2-0.3, but I prefer to split my data between train and test with an high 
# percentage (35%) on the testing side, to allow the model to be sufficiently general.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
penalty    = ['l1', 'l2']
C_range    = 2. ** np.arange(-10, 1, step=1)
# build a dictionary of parameters
parameters = [{'C': C_range, 'penalty': penalty}]
# pass the dictionary to GridSearchCV specifying that it's the LogisticRegression
# and indicating the number of cross validation folds
grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
grid.fit(X_train, y_train)
# display the results
bestC = grid.best_params_['C']
bestP = grid.best_params_['penalty']
print ("The best parameters are: cost=", bestC , " and penalty=", bestP, "\n")
# use the best parameters and check the accuracy
print("Accuracy: {0:.4f}".format(accuracy_score(y_test, grid.predict(X_test))))
lr =  LogisticRegression(penalty=bestP, C=bestC , multi_class='ovr')
lr.fit(X_train, y_train)
print('lr score: {}'.format(lr.score(X_test, y_test)))
# Accuracy
np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
# I order the data_set by date and tray
data_set = data_set.sort_values(by=['Tray', 'Date']).reset_index()
# I create a list of dictionaries with the previous values that will be added to the original dataset:
previous_black_dict = []
previous_dead_dict = []
previous_overpopulated_dict = []
previous_foodleft_dict = []
previous_temperature_dict = []
previous_moisture_dict = []
for ind in data_set.index:
    if ind == 0:
        previous_black_dict.append(False)
        previous_dead_dict.append(False)
        previous_overpopulated_dict.append(False)
        previous_foodleft_dict.append('None')
        previous_temperature_dict.append(0)
        previous_moisture_dict.append(0)
    elif data_set['Tray'][ind] == data_set['Tray'][ind-1]:
            previous_black_dict.append(data_set['Black Larvae'][ind-1])
            previous_dead_dict.append(data_set['Dead Larvae'][ind-1])
            previous_overpopulated_dict.append(data_set['Overpopulated'][ind-1])
            previous_foodleft_dict.append(data_set['Food Left'][ind-1])
            previous_temperature_dict.append(data_set['Temperature'][ind-1])
            previous_moisture_dict.append(data_set['Moisture'][ind-1])
    else:
        previous_black_dict.append(False)
        previous_dead_dict.append(False)
        previous_overpopulated_dict.append(False)
        previous_foodleft_dict.append('None')
        previous_temperature_dict.append(0)
        previous_moisture_dict.append(0)
# I create a new dataframe with only the new previous values:
prev_values = [previous_black_dict, previous_dead_dict, previous_overpopulated_dict, previous_foodleft_dict, previous_temperature_dict, previous_moisture_dict]
col_names = ['Prev Black Larvae', 'Prev Dead Larvae', 'Prev Overpopulated', 'Prev Food Left', 'Prev Temperature', 'Prev Moisture']
zipped = list(zip(col_names, prev_values))
data = dict(zipped)
previous_df = pd.DataFrame(data)
# I call the new dataset, data_set_large, which is the original dataset merged with the one I just created with the new previous values.
data_set_large = pd.merge(data_set, previous_df, left_index=True, right_index=True)
# Ordering by column index I get the original order.
data_set_large = data_set_large.sort_values(by=['index'])
# If I check a random tray, previous values are correct:
data_set_large[data_set_large['Tray']==338]
# Option with previous values
data_set_num = data_set_large[['Age', 'Temperature', 'Moisture', 'Prev Temperature', 'Prev Moisture']]
# Option with previous values
data_set_bool = data_set_large[['Black Larvae', 'Dead Larvae', 'Overpopulated', 'Prev Black Larvae', 'Prev Dead Larvae', 'Prev Overpopulated']]
for col in data_set_bool.columns:
    data_set_bool[col] = data_set_bool[col].map({True: 1, False: 0})
# Option with previous values
data_set_cat = data_set_large[['Food Left', 'Prev Food Left']]
data_set_previous_feed = data_set_large[['Previous Feed 3', 'Previous Feed 2', 'Previous Feed 1']]
data_set_num_bool = pd.concat([data_set_num , data_set_bool], axis=1)
# I introduce OneHotEncoder
enc = OneHotEncoder()
enc.fit(data_set_cat)
# Option with previous values
new_cat_col = enc.transform(data_set_cat).toarray()[:,[0, 1, 3, 4]]
X_previous_feed = np.array(data_set_previous_feed.fillna(0))
X_data_set_num_bool = np.array(data_set_num_bool)
X = np.hstack((X_data_set_num_bool, new_cat_col, X_previous_feed))
y = np.array(data_set_large['Feed'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)

penalty    = ['l1', 'l2']
C_range    = 2. ** np.arange(-10, 1, step=1)
# build a dictionary of parameters
parameters = [{'C': C_range, 'penalty': penalty}]
# pass the dictionary to GridSearchCV specifying that it's the LogisticRegression
# and indicating the number of cross validation folds
grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
grid.fit(X_train, y_train)
# display the results
bestC = grid.best_params_['C']
bestP = grid.best_params_['penalty']
print ("The best parameters are: cost=", bestC , " and penalty=", bestP, "\n")
# use the best parameters and check the accuracy
print("Accuracy: {0:.4f}".format(accuracy_score(y_test, grid.predict(X_test))))
lr =  LogisticRegression(penalty=bestP, C=bestC , multi_class='ovr')
lr.fit(X_train, y_train)
print('lr score: {}'.format(lr.score(X_test, y_test)))
# Accuracy
np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(lr.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]

# ## Model 3: Decision tree classifier
# Restoring the original dataset
data_set_num = data_set_large[['Age', 'Temperature', 'Moisture']]
data_set_bool = data_set_large[['Black Larvae', 'Dead Larvae', 'Overpopulated']]
data_set_cat = data_set_large[['Food Left']]
enc = OneHotEncoder()
enc.fit(data_set_cat)
new_cat_col = enc.transform(data_set_cat).toarray()[:,[0, 1]]
data_set_num_bool = pd.concat([data_set_num , data_set_bool], axis=1)
X_data_set_num_bool = np.array(data_set_num_bool)
X_previous_feed = np.array(data_set_previous_feed.fillna(0))
X = np.hstack((X_data_set_num_bool, new_cat_col, X_previous_feed))
y = np.array(data_set_large['Feed'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
# Create the dictionary of given parameters
depth_n = np.arange(2, 40, 1)
parameters = [{'max_depth': depth_n}] 
#Pass the dicitionary and other parameters to GridSearchCV to create a GridSearchCV object
gridCV = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, return_train_score=True)
gridCV.fit(X_train, y_train) 
bestdepth = gridCV.best_params_['max_depth']
print("Best parameters: max_depth=", bestdepth)
print("Accuracy: {0:.4f}".format(accuracy_score(y_test, gridCV.predict(X_test))))
grid_df = pd.DataFrame(gridCV.cv_results_)
plt.plot(depth_n, gridCV.cv_results_['mean_test_score'])
dtc = DecisionTreeClassifier(max_depth=bestdepth)
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))
# Accuracy
np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
# visualise the tree
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/marco/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
# I create a new decision tree with a max_depth of only 3 to make a graph intelligeble.
# Even if the accuracy is only 0.44, I can see that 'Previous Feed 1', 'Age' and 'Moisture' are the most important features
dtc_graph = DecisionTreeClassifier(max_depth=3)
dtc_graph.fit(X_train, y_train)
print(dtc_graph.score(X_test, y_test))
columns_name = list(data_set_num_bool.columns) + list(enc.categories_[0][0:2]) + list(data_set_previous_feed.columns)
dot_data = export_graphviz(dtc_graph, out_file=None, 
    feature_names=columns_name,  
    filled=True, rounded=True,  
    special_characters=True, rotate=True)
graph = graphviz.Source(dot_data)
# I create a new decision tree with a max_depth of only 3 to make a more intelligible graph. Even if the accuracy is only 0.44 in this case, I can see that 'Previous Feed 1', 'Age' and 'Moisture' are the most important features
graph

# ## Model 4: k-nearest neighbors classifier
# Create the dictionary of given parameters
n_neighb = np.arange(2, 20, 1)
parameters = [{'n_neighbors': n_neighb}] 
#Pass the dicitionary and other parameters to GridSearchCV to create a GridSearchCV object
gridCV = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, return_train_score=True)
gridCV.fit(X_train, y_train) 
bestNeighb = gridCV.best_params_['n_neighbors']
print("Best parameters: n_neighbours=", bestNeighb)
print("Accuracy: {0:.4f}".format(accuracy_score(y_test, gridCV.predict(X_test))))
grid_df = pd.DataFrame(gridCV.cv_results_)
plt.plot(n_neighb, gridCV.cv_results_['mean_test_score'])
knn = KNeighborsClassifier(n_neighbors=bestNeighb)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
# Accuracy
np.unique(np.abs(knn.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(knn.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(knn.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]

# ## Model 5: Neural Network
nb_classes = 8
labels_train = np_utils.to_categorical(y, nb_classes)
# First, declare a model with a sequential architecture
model = Sequential()
# Then add a first layer with 500 nodes and 11 inputs (the columns of the dataset without feed)
model.add(Dense(500,input_shape=(11,)))
# Define the activation function to use on the nodes of that first layer
model.add(Activation('relu'))
# Second hidden layer with 300 nodes
model.add(Dense(300))
model.add(Activation('relu'))
# Third hidden layer with 300 nodes
model.add(Dense(300))
model.add(Activation('relu'))
# Output layer with 8 categories, which are the food from 0 to 7 (using softmax)
model.add(Dense(8))
model.add(Activation('softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=["accuracy"])
X_train, X_test, y_train, y_test = train_test_split(X, labels_train, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
model.fit(X_train, y_train,
          batch_size=100,
          epochs=30,
          verbose=2,
          validation_data = (X_test, y_test))
y_predict = np.argmax(model.predict(X_test), axis=1)
y_test_NN = np.argmax(y_test, axis=1)
# Accuracy
np.unique(np.abs(y_predict - y_test_NN), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(y_predict - y_test_NN), return_counts=True)[1][0]
error_one = np.unique(np.abs(y_predict - y_test_NN), return_counts=True)[1][1]
(error_zero+error_one)/y_test_NN.shape[0]

# ## Model 6: Random Forest
# Restoring the original dataset
data_set_num = data_set_large[['Age', 'Temperature', 'Moisture']]
data_set_bool = data_set_large[['Black Larvae', 'Dead Larvae', 'Overpopulated']]
data_set_cat = data_set_large[['Food Left']]
enc = OneHotEncoder()
enc.fit(data_set_cat)
new_cat_col = enc.transform(data_set_cat).toarray()[:,[0, 1]]
data_set_num_bool = pd.concat([data_set_num , data_set_bool], axis=1)
X_data_set_num_bool = np.array(data_set_num_bool)
X_previous_feed = np.array(data_set_previous_feed.fillna(0))
X = np.hstack((X_data_set_num_bool, new_cat_col, X_previous_feed))
y = np.array(data_set_large['Feed'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# 250 estimatators provided a good result after a few attempts
param_grid = {"max_depth": np.arange(10, 40, 2),
              "max_features": [3, 7, 10, 11],
              "min_samples_split": [3, 5, 7]}
# run grid search
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=250), param_grid=param_grid, n_jobs=-1)
start = time()
grid_search.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
clf = RandomForestClassifier(n_estimators=250, random_state=0, max_depth=20, max_features=7, min_samples_split=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# Accuracy
np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
columns_name = list(data_set_num_bool.columns) + list(enc.categories_[0][0:2]) + list(data_set_previous_feed.columns)
df_fig = pd.DataFrame(clf.feature_importances_, index=columns_name, columns=["importance"]).sort_values("importance", ascending=False)
# Main features
df_fig.plot(kind="bar")
plt.title('Features importance NOT SCALED')
plt.xlabel('Features')
plt.ylabel('Parameter value')
# ### I try to standardise the data to see if 'Age' is still the most important one
scaler = StandardScaler()
scaler.fit(X)
data_set_scaled = scaler.transform(X)
X_scaled = data_set_scaled
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
# 250 estimatators
param_grid = {"max_depth": np.arange(10, 40, 2),
              "max_features": [3, 7, 10, 11],
              "min_samples_split": [3, 5, 7]}
# run grid search
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=250), param_grid=param_grid, n_jobs=-1)
start = time()
grid_search.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
clf = RandomForestClassifier(n_estimators=250, random_state=0, max_depth=32, max_features=7, min_samples_split=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
columns_name = list(data_set_num_bool.columns) + list(enc.categories_[0][0:2]) + list(data_set_previous_feed.columns)
df_fig = pd.DataFrame(clf.feature_importances_, index=columns_name, columns=["importance"]).sort_values("importance", ascending=False)
df_fig.plot(kind="bar")
plt.title('Features importance SCALED')
plt.xlabel('Features')
plt.ylabel('Parameter value')
# ## Model 7: Random Forest, with additional tray data
# Uploading the data per tray
data_set_tray = pd.read_csv("Data/TrayInfo.csv")
# Eliminating tray Null
data_set_tray = data_set_tray[~data_set_tray['Tray'].isnull()]
data_set_tray_spec = data_set_tray[['Tray', 'Egg Mass (g)', 'Diet (most common)']]
data_set_large_tray = pd.merge(data_set_large, data_set_tray_spec, how='left', left_on='Tray', right_on='Tray')
# No Egg Mass input nor Diet are null:
print('Number of Nan lines in Egg Mass: {}'.format(sum(data_set_large_tray['Egg Mass (g)'].isnull())))
print('Number of Nan lines in Diet: {}'.format(sum(data_set_large_tray['Diet (most common)'].isnull())))
# I'm going to add only the new columns to the original dataset 'X' since it wasn't scaled and the
# order of the lines is the same.
#Eggs is numerical
X_eggs = np.array(data_set_large_tray['Egg Mass (g)']).reshape(len(data_set_large_tray['Egg Mass (g)']),1)
X_tray = np.hstack((X, X_eggs))
# Destination is categorical
df_diet = data_set_large_tray[['Diet (most common)']]
enc_2 = OneHotEncoder()
enc_2.fit(df_diet)
X_diet = enc_2.transform(df_diet).toarray()[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
X = np.hstack((X_tray, X_diet))
y = np.array(data_set_large_tray['Feed'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.35, 
                                                    random_state=1337,
                                                    stratify=y)
# Create the dictionary of given parameters
depth_n = np.arange(10, 40, 1)
parameters = [{'max_depth': depth_n}] 
#Pass the dicitionary and other parameters to GridSearchCV to create a GridSearchCV object
gridCV = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, return_train_score=True)
gridCV.fit(X_train, y_train) 
bestdepth = gridCV.best_params_['max_depth']
print("Best parameters: max_depth=", bestdepth)
print("Accuracy: {0:.4f}".format(accuracy_score(y_test, gridCV.predict(X_test))))
grid_df = pd.DataFrame(gridCV.cv_results_)
plt.plot(depth_n, gridCV.cv_results_['mean_test_score'])
dtc = DecisionTreeClassifier(max_depth=bestdepth)
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))
# Accuracy
np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(dtc.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
# 250 estimatators
# use a full grid over all parameters
param_grid = {"max_depth": np.arange(10, 40, 2),
              "max_features": [3, 7, 10, 11],
              "min_samples_split": [3, 5, 7]}
# run grid search
grid_search = GridSearchCV(RandomForestClassifier(n_estimators=250), param_grid=param_grid, n_jobs=-1)
start = time()
grid_search.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
clf = RandomForestClassifier(n_estimators=250, random_state=0, max_depth=18, max_features=7, min_samples_split=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# Accuracy
np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(clf.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
columns_name = list(data_set_num_bool.columns) + list(enc.categories_[0][0:2]) + list(data_set_previous_feed.columns) + ['Egg Mass (g)'] + list(enc_2.categories_[0][0:11])
df_fig = pd.DataFrame(clf.feature_importances_, 
             index=columns_name, 
             columns=["importance"]).sort_values("importance", ascending=False)
df_fig.plot(kind="bar")
plt.title('Features importance')
plt.xlabel('Features')
plt.ylabel('Parameter value')

# ## Model 8: LightGBM
# I need to add the columns name to X_train in order to show at the end a graph with the most important features
columns_name = list(data_set_num_bool.columns) + list(enc.categories_[0][0:2]) + list(data_set_previous_feed.columns) + ['Egg Mass (g)'] + list(enc_2.categories_[0][0:11])
df_X_train = pd.DataFrame(X_train, 
             columns=columns_name).astype('float')
df_y_train = pd.DataFrame(y_train, 
             columns=['Feed']).astype('float')
get_ipython().run_cell_magic('time', '', "\ncv_params = {'max_depth': [3, 5, 7, 10, 11], 'min_child_weight': [1, 3, 5, 7], 'n_estimators': [50, 100, 150]}\nind_params = {'learning_rate': .1, 'random_state':0, 'subsample': 0.8, 'colsample_bytree': 0.8}\n\noptimized_LGB = GridSearchCV(lgb.LGBMClassifier(**ind_params), cv_params, scoring='accuracy', cv = 5, n_jobs = -1) \noptimized_LGB.fit(df_X_train, df_y_train)")
# Accuracy
np.unique(np.abs(optimized_LGB.predict(X_test) - y_test), return_counts=True)[1][0]/y_test.shape[0]
# Accuracy with one litre error allowed
error_zero = np.unique(np.abs(optimized_LGB.predict(X_test) - y_test), return_counts=True)[1][0]
error_one = np.unique(np.abs(optimized_LGB.predict(X_test) - y_test), return_counts=True)[1][1]
(error_zero+error_one)/y_test.shape[0]
plt.figure(figsize=(10,10))
lgb.plot_importance(optimized_LGB.best_estimator_, ax=plt.gca())

# ## Extra: Clustering the data to find some patters
from sklearn.cluster import KMeans
# Apply k-means with 2 clusters using a subset of features 
# (mean_spent and max_spent)
kmeans = KMeans(n_clusters = 2)
data_to_cust  = data_set_large_tray[['Age', 'Temperature', 'Moisture', 
       'Black Larvae', 'Dead Larvae', 'Overpopulated', 'Previous Feed 3',
       'Previous Feed 2', 'Previous Feed 1', 'Feed', 'Egg Mass (g)']].fillna(0)
kmeans.fit(data_to_cust)
cluster_assignment = kmeans.predict(data_to_cust)
np.unique(cluster_assignment)
# This function generates a pairplot enhanced with the result of k-means
def pairplot_cluster(df, cols, cluster_assignment):
    """
    Input
        df, dataframe that contains the data to plot
        cols, columns to consider for the plot
        cluster_assignments, cluster asignment returned 
        by the clustering algorithm
    """
    # seaborn will color the samples according to the column cluster
    df['cluster'] = cluster_assignment 
    sns.pairplot(df, vars=cols, hue='cluster')
    df.drop('cluster', axis=1, inplace=True)
# Visualise the clusters using pairplot_cluster()
pairplot_cluster(data_to_cust, ['Egg Mass (g)', 'Age'], cluster_assignment)
from sklearn.metrics import silhouette_score
# Computing the silhouette score
print('silhouette_score {0:.2f}'.format(silhouette_score(data_to_cust, cluster_assignment)))
from sklearn.cluster import DBSCAN
# Apply DBSCAN setting eps to 1.0 and min samples to 8​
db = DBSCAN(eps=2.0, min_samples=8)
cluster_assignment = db.fit_predict(data_to_cust)
# Display how many clusters were found
clusters_found = np.unique(cluster_assignment)
print ('Clusters found', len(clusters_found))
# Visualise the clusters using pairplot_cluster()
pairplot_cluster(data_to_cust, ['Egg Mass (g)', 'Age'], cluster_assignment)
# Compute the silhouette score of DBSCAN
from sklearn.metrics import silhouette_score
print(silhouette_score(data_to_cust, cluster_assignment))

# ## Extra: Trying to find different operators
# This code line converts the date into date values
data_set_large_tray['Date'] = data_set_large_tray['Date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
data_set_operator = data_set_large_tray[['Date', 'Age', 'Egg Mass (g)', 'Temperature', 'Moisture', 'Feed']]
data_set_operator.groupby(['Date', 'Age', 'Temperature', 'Moisture'])['Feed'].mean().reset_index().head()
data_set_operator_2 = data_set_operator.groupby(['Date', 'Age', 'Temperature', 'Moisture'])['Feed'].mean().reset_index()
data_set_operator_3 = data_set_operator_2[(data_set_operator_2['Age']==10) & (data_set_operator_2['Temperature']==0) & (data_set_operator_2['Moisture']==0)]
sns.set(style="darkgrid")
​sns.lineplot(x="Date", y="Feed", data=data_set_operator_3)
for age in list(np.unique(data_set_operator_2['Age'])):
    data_set_operator_3 = data_set_operator_2[(data_set_operator_2['Age']==age) & (data_set_operator_2['Temperature']==0) & (data_set_operator_2['Moisture']==0)]
    num_obs = data_set_operator_3.shape[0]
    print('Age: {} Number of observations: {}'.format(age, num_obs))
    sns.lineplot(x="Date", y="Feed", data=data_set_operator_3)
    plt.show()