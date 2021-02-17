#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:48:36 2021

@author: trduong
"""

import pandas as pd
import numpy as np
import pickle 
import yaml
import sys

from off_pol_eval_functions import gaussian_kernel

"""Import genetic library"""
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize 
from pymoo.model.problem import Problem
import autograd.numpy as anp


class PolicyOptimize(Problem):
    def __init__(self,
                 df,
                 conf,
                 bandwidth,
                 covariates,
                 n_var=670, 
                 **kwargs):
        super().__init__(n_var=n_var, 
                         n_obj=1, 
                         n_constr=0, 
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         xl=np.array([0]*670), 
                         xu=np.array([100]*670),
                         **kwargs)
        self.n_var = 670
        self.n_obj = 1
        self.conf = conf 
        self.treatment_model = self.load_model(conf['model_directory'].format(conf['treatmet_model']))
        self.outcome_model = self.load_model(conf['model_directory'].format(conf['outcome_model']))
        self.df = df
        self.bandwidth = bandwidth
        self.covariates = covariates
        
        # print(self.df)


    def _evaluate(self, delta, out, *args, **kwargs):
        # print("Propensity score ", self.df['ps_score'].values)
        ps = self.df['ps_score'].values
        # print(ps.shape)
        # print("Delta ", delta)
        policy_value = (delta * ps) / (delta * ps + 1 - ps)
        # policy_value = policy_value.reshape(-1,1)
        cov_value = self.df[self.covariates].values
        # print(policy_value.shape, cov_value.shape)
        feature_value = np.concatenate([policy_value.reshape(-1,1),cov_value], axis = 1)
        
        
        # print(policy_value)
        v_DR = self.estimator_DR(self.df[treatment].values, 
                        self.df[outcome].values,
                        self.df['ps_score'], 
                        policy_value,
                        feature_value,
                        self.df['y_estimator'],
                        self.bandwidth)
        # print(v_DR)
        out["F"] = v_DR
    
    def load_model(self, PATH):
        """
        
    
        Parameters
        ----------
        PATH : str
            Directory for treatment and outcome model.
    
        Returns
        -------
        model : TYPE
            DESCRIPTION.
    
        """
        model = pickle.load(open(PATH, 'rb'))
        return model
    
    def estimator_DR(self, treatment, outcome, ps_score, policy_value, feature_value, y_estimator, bandwith):
        """
        
    
        Parameters
        ----------
        treatment : TYPE
            treatment model.
        outcome : TYPE
            outcome model.
        ps_score : TYPE
            propensity score.
        policy : TYPE
            proposed policy.
        y_estimator : TYPE
            DESCRIPTION.
        bandwith : TYPE
            DESCRIPTION.
    
        Returns
        -------
        v_DR : TYPE
            DESCRIPTION.
    
        """
        
        h = 0.1 ##bandwith
        a = outcome - y_estimator
        b = 1/(h*ps_score)*gaussian_kernel((policy_value - treatment)/bandwith)
        response = self.outcome_model.predict(feature_value)
        v_DR = np.mean(response + b*a)
        
        return -v_DR

        

if __name__ == "__main__":
    """Read configuration"""
    with open('config.yaml') as file:
      conf = yaml.safe_load(file)
      print(conf)
      
    """Load data"""
    df = pd.read_csv(conf['processed_data_directory'].format(conf['data_name']))
    
    """Load model"""
    # treatment_model = load_model(conf['model_directory'].format(conf['treatmet_model']))
    # outcome_model = load_model(conf['model_directory'].format(conf['outcome_model']))
    
    """Define features"""
    treatment = 'logTN'
    outcome = 'Taxonrich'
    
    
    features = ['logTN', 'ELEV', 'longitude', 'logprecip', 'logAREA',
           'logCL', 'logHCO3', 'logSO4', 'SED', 'STRMTEMP', 'Percent.AGT',
           'Percent.URB', 'Percent.Canopy', 'Riparian.Disturb.']
    covariates = ['ELEV', 'longitude', 'logprecip', 'logAREA',
           'logCL', 'logHCO3', 'logSO4', 'SED', 'STRMTEMP', 'Percent.AGT',
           'Percent.URB', 'Percent.Canopy', 'Riparian.Disturb.']
    
    """Off-Policy Continuous Estimator"""
    bandwidth = 0.1
    
    problem = PolicyOptimize(df, conf, bandwidth, covariates)
    
    # bandwith = 0.1 ##bandwith    
    # cov_value = df[covariates].values
    # policy_value =  df['ps_score'].values.reshape(-1,1)
    # policy_value = np.concatenate([policy_value,cov_value], axis = 1)
    # v_DR = estimator_DR(df[treatment].values, 
    #                     df[outcome].values,
    #                     df['ps_score'], 
    #                     policy_value,
    #                     df['y_estimator'],
    #                     bandwith)
    delta = [0.95]*670
    # delta[0] = 5
    ps = df['ps_score'].values
    incre_ps = (delta * ps) / (delta * ps + 1 - ps)
    policy_value = incre_ps.reshape(-1,1)
    cov_value = df[covariates].values
    # print(policy_value.shape, cov_value.shape)
    feature_value = np.concatenate([policy_value,cov_value], axis = 1)
    
    
    # print(policy_value)
    v_DR = problem.estimator_DR(df[treatment].values, 
                    df[outcome].values,
                    df['ps_score'], 
                    incre_ps,
                    feature_value,
                    df['y_estimator'],
                    bandwidth)
    print(v_DR)
        
    """Optimization"""
    algorithm = GA(pop_size=100, eliminate_duplicates=True)
    res = minimize(problem, algorithm, seed=1, verbose=False)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    
    
    

  