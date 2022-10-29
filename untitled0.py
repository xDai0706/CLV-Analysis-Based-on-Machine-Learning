#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 21:39:15 2022

@author: demix9
"""

"""
Created on Wed Dec 29 08:26:55 2021
# 使用sklearn做测试各种回归

# 基本回归：线性、决策树、SVM、KNN

# 集成方法：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees
@author: lenovo
"""  
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
        from sklearn.neural_network import MLPRegressor

        from xgboost import XGBRegressor

        from sklearn.model_selection import GridSearchCV

        import shap
        class machinelearning():
            def __init__(self,model_name, Train_X,Train_Y,Test_X, Test_Y):
                self.model_name = model_name
                self.Train_X,self.Train_Y,self.Test_X, self.Test_Y = Train_X,Train_Y,Test_X, Test_Y


            def model_predict(self):
                names = ['Decision Tree', 'Linear Regression', 'Ridge','Lasso','SVR', 'KNN', 'RFR', 'Ada Boost', 
                    'Gradient Boost', 'Bagging', 'Extra Tree','XGBoost','BPNN']
                regressors = [
                    DecisionTreeRegressor(),
                    LinearRegression(),
                    Ridge(),
                    Lasso(),
                    SVR(gamma='scale'),
                    KNeighborsRegressor(),
                    RandomForestRegressor(),
                    AdaBoostRegressor(),
                    GradientBoostingRegressor(),
                    BaggingRegressor(),
                    ExtraTreeRegressor(),
                    XGBRegressor(),
                    MLPRegressor()
                ]
                param_grid =[{'criterion':['mse','friedman_mse','mae'],'splitter':['best'],'min_samples_leaf':[3,5],'max_features':[6,9,15],'max_depth':[8,10,15]},
                             {'fit_intercept':[True,False]},
                             {'fit_intercept':[True,False],'alpha':[0.00001,0.0001,0.001,0.01, 0.1, 1.0, 10.0,100.0]},
                             {'fit_intercept':[True,False],'alpha':[0.00001,0.0001,0.001,0.01, 0.1, 1.0, 10.0,100.0]},
                             {'kernel':['linear','poly','rbf','sigmoid'],'C':[0.00001,0.0001,0.001,0.01, 0.1, 1.0, 10.0,100.0]},
                             {'weights':['uniform','distance'],'n_neighbors':range(2,15)},
                             {'n_estimators':range(10,300,50),'min_samples_leaf':[3,5],'max_features':[6,9],'max_depth':[8,10,15]},
                             {'max_depth':[8,10,15], 'min_samples_split':[5,10,15,20], 'min_samples_leaf':[3,5],'n_estimators':range(10,300,30), 'learning_rate':[0.2,0.5,0.8]},
                             {'max_depth':[8,10,15], 'min_samples_split':[5,10,15,20], 'min_samples_leaf':[3,5],'n_estimators':range(10,300,30), 'learning_rate':[0.2,0.5,0.8]},
                             {'max_depth':[8,10,15], 'min_samples_split':[5,10,15,20], 'min_samples_leaf':[3,5],'n_estimators':range(10,300,30), 'learning_rate':[0.2,0.5,0.8]},
                             {'min_samples_leaf':[3,5],'max_features':[6,9,15],'max_depth':[8,10,15]},
                             {'n_estimators': range(10,300,50),'max_depth': [8,10,15], 'min_child_weight': [1,3,5],'learning_rate':[0.2,0.5,0.8]},
                             {'hidden_layer_sizes':[(6,2),(6,6),(10,5)], 'activation':['relu'], 'solver':['adam'], 'alpha':[0.00001,0.0001,0.001,0.01, 0.1, 1.0, 10.0,100.0],'learning_rate':['constant']}]  #XGBoost
                                
                # print(param_grid.index[self.model_name])
                grid_search = GridSearchCV(regressors[names.index(self.model_name)],param_grid[names.index(self.model_name)],cv=5, scoring='neg_mean_squared_error',n_jobs=-1)
                grid_search.fit( self.Train_X, self.Train_Y)
                predict_y=grid_search.predict(self.Test_X)
                resu = pd.DataFrame({'test':self.Test_Y,'predict':predict_y})
                
                mse = mean_squared_error(self.Test_Y,predict_y)
                r2 = r2_score(self.Test_Y,predict_y)
                mae = mean_absolute_error(self.Test_Y,predict_y)
                print(grid_search.best_params_)
                return [mse,r2,mae]
                  
                # grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
                # pd.set_option('display.max_columns', None) # 显示所有列
                # pd.set_option('max_colwidth',100) # 设置value的显示长度为100，默认为50
                # print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
            
    