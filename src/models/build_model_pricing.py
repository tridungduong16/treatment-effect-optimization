#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:34:58 2021

@author: trduong
"""

import numpy as np 
import pandas as pd
import yaml 
import pickle 

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    """Read configuration"""
    with open('config.yaml') as file:
        conf = yaml.safe_load(file)
    
    """Load data"""
    df = pd.read_csv(conf['data_pricing'])
    
    """Preprocess data"""
    df['treatment'] = np.where(df['price'] == 1, 1, 0)
    df['price'] = np.where(df['price'] == 1, 1, 0.8)
    df['revenue'] = df['demand']*df['price']

    
    """Set up features"""
    treatment = 'treatment'
    features = ['account_age', 
             'age', 
             'avg_hours', 
             'days_visited', 
             'friends_count', 
             'has_membership', 
             'is_US', 
             'songs_purchased', 
             'income', 'treatment']
    covariates = ['account_age', 
             'age', 
             'avg_hours', 
             'days_visited', 
             'friends_count', 
             'has_membership', 
             'is_US', 
             'songs_purchased', 
             'income']
    outcome = 'demand'
    
    """Build treament and outcome model"""

    classifier = GradientBoostingClassifier(random_state=0)
    classifier.fit(df[covariates].values, df[treatment].values.reshape(-1,1))
    
    with open(conf['model_directory'].format(conf['treatment_model_pricing']), 'wb') as f:
        pickle.dump(classifier, f)
    
    
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(df[features].values, df[outcome].values.reshape(-1,1))
    
    with open(conf['model_directory'].format(conf['outcome_model_pricing']), 'wb') as f:
        pickle.dump(reg, f)
        
        
    """Check performance"""
    df['ps_score'] = classifier.predict_proba(df[covariates].values)[:, 1].reshape(-1)
    df['y_estimator'] = reg.predict(df[features].values).reshape(-1)
    prediction = classifier.predict(df[covariates].values).reshape(-1)

    
    acc = accuracy_score(df['treatment'].values, prediction)
    mse = mean_squared_error(df[outcome], df['y_estimator'])

    print("Accuracy {:.4f}".format(acc))
    print("Mean squared error {:.4f}".format(mse))
    
    """Save processed data"""
    df.to_csv(conf['processed_data_directory'].format('pricing.csv'), index = False)
    
