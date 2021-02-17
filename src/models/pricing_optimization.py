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
                 covariates,
                 n_var=1, 
                 **kwargs):
        super().__init__(n_var=n_var, 
                         n_obj=1, 
                         n_constr=0, 
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         xl=np.array([0]*1), 
                         xu=np.array([100]*1),
                         **kwargs)
        self.n_var = n_var
        self.n_obj = 1
        self.conf = conf 
        self.treatment_model = self.load_model(conf['model_directory'].format(conf['treatment_model_pricing']))
        self.outcome_model = self.load_model(conf['model_directory'].format(conf['outcome_model_pricing']))
        self.df = df
        self.covariates = covariates
        
        # print(self.df)


    def _evaluate(self, delta, out, *args, **kwargs):
        # print("Propensity score ", self.df['ps_score'].values)
        ps = self.df['ps_score'].values
        # print(ps.shape)
        # print("Delta ", delta)
        propensity_score = (delta * ps) / (delta * ps + 1 - ps)        
        policy_value = np.where(propensity_score >= 0.5, 1, 0)
    
        cov_value = self.df[self.covariates].values
        feature_value = np.concatenate([cov_value, policy_value.reshape(-1,1)], axis = 1)
        
        
        # print(policy_value)
        v_DR = self.estimator_DR(self.df[treatment].values, 
                        self.df[outcome].values,
                        self.df['ps_score'], 
                        policy_value,
                        feature_value,
                        self.df['y_estimator'])
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
    
    def estimator_DR(self, treatment, outcome, ps_score, policy_value, feature_value, y_estimator):
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
        
        # h = 0.1 ##bandwith
        # a = outcome - y_estimator
        # b = 1/(h*ps_score)*gaussian_kernel((policy_value - treatment)/bandwith)
        policy_value = np.where(policy_value == 1, 1, 0.8)
        demand = self.outcome_model.predict(feature_value)
        response = demand*policy_value
        # print("Feature ", feature_value)
        # print("Policy value ", policy_value)
        # print("Demand ", demand)
        # print("Response ", response)

        v_DR = np.mean(response)
        
        return -v_DR

        

if __name__ == "__main__":
    """Read configuration"""
    with open('config.yaml') as file:
      conf = yaml.safe_load(file)
      # print(conf)
      
    """Load data"""
    df = pd.read_csv(conf['processed_data_directory'].format('pricing.csv'))
        
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
             'income',
             'treatment']
    covariates = ['account_age', 
             'age', 
             'avg_hours', 
             'days_visited', 
             'friends_count', 
             'has_membership', 
             'is_US', 
             'songs_purchased', 
             'income']
    outcome = 'revenue'
    
    """Off-Policy Pricing Estimator"""
    problem = PolicyOptimize(df, conf, covariates)
    
    delta = 1
    ps = df['ps_score'].values
    incre_ps = (delta * ps) / (delta * ps + 1 - ps)
    policy_value = incre_ps.reshape(-1,1)
    policy_value = np.where(policy_value >= 0.5,1,0)
    cov_value = df[covariates].values
    feature_value = np.concatenate([cov_value, policy_value], axis = 1)
    
    v_DR = problem.estimator_DR(df[treatment], 
                                df[outcome], 
                                df['ps_score'], 
                                policy_value, 
                                feature_value, 
                                df['y_estimator'])
    print(v_DR)
        
    """Optimization"""
    algorithm = GA(pop_size=100, eliminate_duplicates=True)
    res = minimize(problem, algorithm, seed=1, verbose=False)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    
    
    

  