import optuna

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
from itertools import chain, combinations

import time

import warnings
warnings.filterwarnings("ignore")

from inspect import signature

from raw_data import*

### Start of defining all parameters we wish to search over
### in a single model "combination":
# X and y feature scalings:    
standard_scaler = StandardScaler
min_max_scaler = MinMaxScaler
robust_scaler = RobustScaler

# linear models
linear_regression = LinearRegression
lasso = Lasso
ridge = Ridge
elastic_net = ElasticNet

# nonlinear models
random_forest = RandomForestRegressor
svr = SVR
mlp = MLPRegressor
decision_tree = DecisionTreeRegressor
ada_boost = AdaBoostRegressor
knn = KNeighborsRegressor
nu_svr = NuSVR
gradient_boosting = GradientBoostingRegressor

### linear model parameter grids: 
# Ordinary Linear Regression
linear_regression_param_grid = {
# no hyperparmeters for ordinary linear regression
}

# Lasso Linear Regression
lasso_param_grid = {
'alpha': {'type': 'float', 'range': [0.00001,  1E5]}
}

# Ridge Linear Regression
ridge_param_grid = {
    'alpha': {'type': 'float', 'range': [0.00001,  1E5]}
}

# Elastic Net
elastic_net_param_grid = {
    'alpha': {'type': 'float', 'range': [0.0001,  1E4]},
    'l1_ratio': {'type': 'float', 'range': [0.0,  1.0]} # this is the balance between L1 and L2 regularization
}


### nonlinear model parameter grids:
# RandomForest
rf_param_grid = {
    'n_estimators': {'type': 'int', 'range': [1,  500]},
    'max_depth': {'type': 'int', 'range': [1,  100]},
    'min_samples_split': {'type': 'int', 'range': [2,  20]},
    'min_samples_leaf': {'type': 'int', 'range': [1,  20]},
}

# SVR
svr_param_grid = {
    'C': {'type': 'float', 'range': [1e-4,  1e4]},
    'kernel': {'type': 'categorical', 'values': ['linear', 'poly', 'rbf', 'sigmoid']},
    'degree': {'type': 'int', 'range': [1,  5]},
    'gamma': {'type': 'categorical', 'values': ['scale', 'auto']},
    'coef0': {'type': 'float', 'range': [-1.0,  1.0]},
    'epsilon': {'type': 'float', 'range': [0.0,  1.0]}
}

# Multilayer Perceptron
mlp_param_grid = {
    'hidden_layer_sizes': {'type': 'categorical', 'values': [(50,), (100,), (150,)]},
    'activation': {'type': 'categorical', 'values': ['identity', 'logistic', 'tanh', 'relu']},
    'solver': {'type': 'categorical', 'values': ['lbfgs', 'adam']},
    'alpha': {'type': 'float', 'range': [0.0001,  0.1]},
    'batch_size': {'type': 'categorical', 'values': ['auto',  10,  50,  100,  200]}
}

# Decision Trees
dt_param_grid = {
'max_depth': {'type': 'int', 'range': [1,  100]},
'min_samples_split': {'type': 'int', 'range': [2,  20]},
'min_samples_leaf': {'type': 'int', 'range': [1,  20]}
}

# AdaBoostRegressor
ada_boost_param_grid = {
    'n_estimators': {'type': 'int', 'range': [1,  500]},
    'learning_rate': {'type': 'float', 'range': [0.01,  1.0]}
}

# KNeighborsRegressor
knn_param_grid = {
    'n_neighbors': {'type': 'int', 'range': [3,  10]},
    'weights': {'type': 'categorical', 'values': ['uniform', 'distance']},
    'algorithm': {'type': 'categorical', 'values': ['auto', 'ball_tree', 'kd_tree', 'brute']}
}

# NuSVR
nu_svr_param_grid = {
    'nu': {'type': 'float', 'range': [0.01,  1.0]},
    'kernel': {'type': 'categorical', 'values': ['linear', 'poly', 'rbf', 'sigmoid']},
    'degree': {'type': 'int', 'range': [1,  5]},
    'gamma': {'type': 'categorical', 'values': ['scale', 'auto']},
    'coef0': {'type': 'float', 'range': [-1.0,  1.0]}
}

# GradientBoostingRegressor
gradient_boosting_param_grid = {
    'n_estimators': {'type': 'int', 'range': [10,  1000]},
    'learning_rate': {'type': 'float', 'range': [0.01,  1.0]},
    'max_depth': {'type': 'int', 'range': [3,  10]},
    'subsample': {'type': 'float', 'range': [0.5,  1.0]},
    'loss': {'type': 'categorical', 'values': ['absolute_error', 'squared_error', 'huber', 'quantile']}
}


### Foramtting of scaler and model classes for loading 
### the best studies and datasets:
# Define the scaler classes
scaler_classes = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler
}

# Define the model classes
model_classes = {
    'SVR': SVR,
    'MLPRegressor': MLPRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'AdaBoostRegressor': AdaBoostRegressor,
    'KNeighborsRegressor': KNeighborsRegressor,
    'LinearRegression': LinearRegression,
    'ElasticNet': ElasticNet,
    'Lasso': Lasso,
    'Ridge': Ridge,
    'NuSVR': NuSVR,
    'GradientBoostingRegressor': GradientBoostingRegressor
}