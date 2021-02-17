#!/usr/bin/env python
# coding: utf-8

# ## Import library

# In[1]:


import pandas as pd
import numpy as np
import pickle 
import yaml 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor


"""Read configuration"""
with open('config.yaml') as file:
  conf = yaml.safe_load(file)
  print(conf)

DATA_PATH = "/data/trduong/treatment-effect-optimization/data/raw/{}"
MODEL_PATH = "/data/trduong/treatment-effect-optimization/models/{}"


df = pd.read_csv(DATA_PATH.format('stream.csv'))



treatment = 'logTN'
outcome = 'Taxonrich'
features = ['logTN', 'ELEV', 'longitude', 'logprecip', 'logAREA',
       'logCL', 'logHCO3', 'logSO4', 'SED', 'STRMTEMP', 'Percent.AGT',
       'Percent.URB', 'Percent.Canopy', 'Riparian.Disturb.']
covariates = ['ELEV', 'longitude', 'logprecip', 'logAREA',
       'logCL', 'logHCO3', 'logSO4', 'SED', 'STRMTEMP', 'Percent.AGT',
       'Percent.URB', 'Percent.Canopy', 'Riparian.Disturb.']


reg = LinearRegression()
reg = ElasticNet(random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(df[covariates].values, df[treatment].values.reshape(-1,1))

with open(MODEL_PATH.format('treatment_model.pkl'), 'wb') as f:
    pickle.dump(reg, f)



df['ps_score'] = reg.predict(df[covariates].values).reshape(-1)


reg = LinearRegression()
reg = ElasticNet(random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(df[features].values, df[outcome].values.reshape(-1,1))

with open(MODEL_PATH.format('outcome_model.pkl'), 'wb') as f:
    pickle.dump(reg, f)


df['y_estimator'] = reg.predict(df[features].values).reshape(-1)

df.to_csv("/data/trduong/treatment-effect-optimization/data/processed/stream_ps.csv", index = False)


mse1 = mean_squared_error(df[treatment], df['ps_score'])
mse2 = mean_squared_error(df[outcome], df['y_estimator'])



print("Mean squared error in treatment model: {:.4f}".format(mse1))
print("Mean squared error in outcome model: {:.4f}".format(mse2))

