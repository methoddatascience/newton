#####################################################
#
#   Author: @Franco Matias Ferrari (LinkedIn: https://www.linkedin.com/in/franco-matias-ferrari/ )
#           @Carlos Flury (LinkedIn: https://www.linkedin.com/in/carflury/ )
#   
#   Description: This is a script that have different functions to make easy preprocesing, modeling, tunning, evaluate, save and use diferent models (classification and regression)
#                       - Random and Search grid of parameters
#                       - Train with cross validation
#                       - Dummies staff
#                       - Save, open and use models
#                       - Get metrics, evaluate models and save results                   
#
#   Date: 2/12/2018 
#
#   Obs: Contact us if you have any question or suggestion about it :)
#
#####################################################

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#from Perceptron import BinaryMLP

from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import ParameterSampler
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold

import statsmodels.api as sm
import json
from sklearn.feature_selection import RFE

from sklearn import metrics



############### RANDOM SELECT HYPERPARAMETERS ###############
######## MAKE THE PARAM_GRID ########
def createRandomGrid(model):
    # RANDOM FOREST
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    learners = [
######### CLASSIFIER GRID ########
        {
        "learner": GradientBoostingClassifier,
        "params": {
            "learning_rate": [0.1], 
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25],
            "subsample": [0.5, 0.75, 1], #Denotes the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
            "n_estimators": 1000
            },
        },
        {
        "learner": XGBClassifier,
        "params": {
            "learning_rate": [0.1], #o ETA, Analogous to learning rate in GBM
            "colsample_bytree": [0.15, 0.25, 0.5, 0.75, 1], #Denotes the fraction of columns to be randomly samples for each tree
            "colsample_bylevel": [1],#Denotes the subsample ratio of columns for each split, in each level.
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25],#The maximum depth of a tree, same as GBM. Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
            "gamma": [0, 0.01, 0.05, 0.1, 0.25, 0.5], #A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
            "subsample": [0.5, 0.75, 1], #Denotes the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
            "min_child_weight": [1, 5, 10, 15, 25, 50, 100],#Defines the minimum sum of weights of all observations required in a child. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
            "base_score": [0.52],
            "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            },
        },
        {
        "learner": RandomForestClassifier,
        "params": {
            "max_depth":max_depth, # Maximum number of levels in tree
            "max_features": ["auto", 0.1, 0.25, 0.5, 0.75, 1], # Number of features to consider at every split
            "min_weight_fraction_leaf": [0, 0.01, 0.001], 
            "bootstrap": [True, False], # Method of selecting samples for training each tree
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)], # Number of trees in random forest
            "n_jobs": -1
            },
######### REGRESSION GRID ########
        },  
        {
        "learner": XGBRegressor,
        "params": {
            "learning_rate": [0.1], #o ETA, Analogous to learning rate in GBM
            "colsample_bytree": [0.15, 0.25, 0.5, 0.75, 1], #Denotes the fraction of columns to be randomly samples for each tree
            "colsample_bylevel": [1],#Denotes the subsample ratio of columns for each split, in each level.
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25],#The maximum depth of a tree, same as GBM. Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
            "gamma": [0, 0.01, 0.05, 0.1, 0.25, 0.5], #A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
            "subsample": [0.5, 0.75, 1], #Denotes the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
            "min_child_weight": [1, 5, 10, 15, 25, 50, 100],#Defines the minimum sum of weights of all observations required in a child. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
            "base_score": [0.52],
            "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            },
        },      
    ]
    
    #candidate = [i['learner'] for i in learners if i["learner"].__name__ == model][0]
    #params = candidate["params"]

    for i in learners:
        if i["learner"].__name__ == model:
            params = i["params"]
    return params


######## TRAIN RANDOM SELECTOR ########
def trainRandomGrid(candidate, random_grid, train_features, train_labels):
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model = candidate
    # Random search of parameters, using 3 fold cross validations
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)

    # We can view the best parameters from fitting the random search
    # rf_random.best_params_

    return rf_random

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #accuracy = metrics.accuracy_score(test_labels, predictions)
    mae = metrics.mean_absolute_error(test_features, test_labels)
    accuracy = 1 - mae
    return accuracy

############### GRID SELECT HYPERPARAMETERS ###############
######## MAKE THE PARAM_GRID ########
# Create the parameter grid based on the results of random search 

def trainGrid(candidate, param_grid, train_features, train_labels):
    # Create a based model
    model = candidate
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

    ######## TRAIN GRID SELECTOR #######
    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    #grid_search.best_params_
    return grid_search

def crossValidationSelector(X, y, folds):  
    skf = StratifiedKFold(n_splits=folds)
    return skf.split(X, y)

def trainWithCrossValidation(kfold, base, learner):    
    # kfolds are the folds that return crossValidationSelector
    # base is the dataset
    # learner is the model with the best parameters
    proba = pd.DataFrame()
    preds = pd.DataFrame()
    X, y = base.drop(base.columns[-1], axis=1), base.iloc[:,-1] 
    
    for train_index, test_index in kfold:
#    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.loc.__getitem__(train_index), X.loc.__getitem__(test_index)
        y_train, y_test = y.loc.__getitem__(train_index), y.loc.__getitem__(test_index)    
        auxTest = X_test.copy()
        X_train = X_train.values.astype(np.float32, order="C")
        X_test = X_test.values.astype(np.float32, order="C")
        scaler = StandardScaler()
        scaler.fit(np.r_[X_train, X_test])
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    
        probabilidades, predicciones = train(learner, X_train, y_train, X_test, y_test)     
        auxTest['prob'] = probabilidades
        proba = pd.concat([proba, auxTest[['prob']]])
        
        auxTest['pred'] = predicciones
        preds = pd.concat([preds, auxTest[['pred']]])

    return proba, preds, learner

def train(learner, X_train, y_train, X_test, y_test, esr=20):
    mask = y_test.isin(y_train.unique()).values
    if type(learner) in [XGBClassifier]:
        learner.fit(X_train, y_train,
                    eval_set=[(X_test, y_test),
                              (X_test[mask], y_test[mask])
                              ],
                    early_stopping_rounds=esr,
                    verbose=True)
        n_estimators = learner.best_iteration
        probs = learner.predict_proba(X_test, ntree_limit=n_estimators)[:, -1]
        pred = learner.predict(X_test, ntree_limit=n_estimators)

    elif type(learner) in [RandomForestClassifier, ExtraTreesClassifier,
                       AdaBoostClassifier, BaggingClassifier]:
        learner.set_params(warm_start = True)
        learner.set_params(n_estimators = 50)
        learner.fit(X_train, y_train)
        probs = learner.predict_proba(X_test)[:, -1]
        pred = learner.predict(X_test)
    return probs, pred
    
def saveModel(learner, modelName):
    filename = './{0}.sav'.format(modelName)
    pickle.dump(learner, open(filename, 'wb'))   
    
def openModel(modelName):
    filename = './{0}.sav'.format(modelName)   
    return pickle.load(open(filename, 'rb'))

def predictProbaSavedModel(modelName, base):
    X, y = base.drop(base.columns[-1], axis=1), base.iloc[:,-1] 
    learner = openModel(modelName)
    probs = learner.predict_proba(X)[:, -1]   
    base['probs'] = probs
    return base

def convertCatVariablesIntoDummies(base, cat_vars):
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(base[var], prefix=var)
        data1=base.join(cat_list)
        base=data1
        
    return base

def featureSelecciontLogisticRegression(X, y):
    logreg = LogisticRegression()
    rfe = RFE(logreg, 18)
    rfe = rfe.fit(X, y)
    columns = [x[1] for x in zip(rfe.support_, X.columns) if x[0] == True]
    
    return X[columns]


def saveResults(modelName, learnerName, *metricas, **kargs):
    directory = './results'
    if not os.path.exists(directory):
        os.makedirs(directory)
     
    if os.path.exists(directory+'/'+modelName+'.xlsx'):
        results = pd.read_excel(directory+'/'+modelName+'.xlsx', index_col = 0)
    else:
        results = pd.DataFrame()

    cols = []
    vals = []
    for metric in metricas:
        cols.append(list(metric.keys())[0])
        vals.append(list(metric.values())[0])
        
    cols.append('params')
    vals.append(json.dumps(kargs))
    results = pd.concat([results,pd.DataFrame([vals], columns=cols,
                                          index=[learnerName])])
    results.to_excel(directory+'/'+modelName+'.xlsx')

def getMetrics(y, yPred, multiple_labels, target_names = None):
    if multiple_labels == False:
        accuracy = metrics.accuracy_score(y, yPred)
        mean_absolute_error = metrics.mean_absolute_error(y, yPred, multioutput='raw_values')
        recall = metrics.recall_score(y, yPred)
        f1Score = metrics.f1_score(y, yPred)
        auc = metrics.roc_auc_score(y, yPred)
        metricas = [
                {'accuracy': accuracy},
                {'mean_absolute_error': mean_absolute_error},
                {'recall': recall},
                {'f1Score': f1Score},
                {'auc': auc}        
                ]
    else:
        metricas = metrics.classification_report(y, yPred, target_names=target_names)
    
    return metricas