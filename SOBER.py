# -*- coding: utf-8 -*-
"""
# This project is licensed under the MIT License - see the LICENSE file for details

Created on Wed May 28 05:29:57 2025

SOBER - Github package

@author: Mahesh Kumbhar
"""

import numpy as np

from collections import Counter
from sklearn.neighbors import NearestNeighbors
import random
from sklearn import preprocessing
import pandas as pd
from itertools import combinations 
import re
from scipy.optimize import minimize
import ast

class SOBER:
    def __init__(self, k_obj=5, con_para=0.95):

        assert k_obj in [5, 7, 9, 11]
        assert con_para in [0.90, 0.95]
        

        self.k_obj = k_obj
        self.con_para = con_para
        
        self.maj_index = []
        self.min_index = []

  

    def ratio_objective(self, X, df_combine, k_input):
            
        distances = np.linalg.norm(df_combine[:, :-1] - X, axis=1)
        
        # Get k nearest neighbors
        k = k_input + 1
        nearest_neighbor_ids = np.argpartition(distances, k)[:k]  # Faster than argsort for partial sorting
        
        # Remove first element (self-neighbor)
        nearest_neighbor_ids = nearest_neighbor_ids[1:]
        
        min_index = np.flatnonzero(df_combine[:,-1] == 1).tolist()
        maj_index = np.flatnonzero(df_combine[:,-1] == 0).tolist()
        
        target_class_filter = df_combine[min_index]
        non_target_class_filter = df_combine[maj_index]
    
        
        target_samples_index = np.intersect1d(nearest_neighbor_ids, target_class_filter, assume_unique=True)
        non_target_samples_index = np.intersect1d(nearest_neighbor_ids, non_target_class_filter, assume_unique=True)
        
        
        initial_term = len(non_target_samples_index) / (len(target_samples_index) + 1.0)
    
        second_term = np.random.normal(0, )    
        final_term =  initial_term + second_term
        return(final_term)
    

    def normalize(self, a):
        min_max_scaler = preprocessing.MinMaxScaler()
        a = min_max_scaler.fit_transform(a)
        return a

    def cal_bounds(self, X_inp):
        ######### for capping bounds #####
        lower_bound = list(X_inp.min(axis=0))        
        upper_bound = list(X_inp.max(axis=0))
        return lower_bound, upper_bound

    def get_keys(self, X_inp):
        features_selection = int(np.floor(np.sqrt(X_inp.shape[1])))
        if(features_selection>=3):
            features_selection = 3        
        else:            
            features_selection = 2         
        
        pair_keys = [] 
    
        for item in combinations(range(X_inp.shape[1]), features_selection):     
            pair_keys.append(item)
        
        return pair_keys

    

    def roulette_wheel_selection(self, pair_keys):
        pair_keys.sort_values(by="columns_selected_index", ascending=True, inplace=True)
        sum_prob_pair = pair_keys["prob_pair"].sum()
        probabilities = np.array(pair_keys["prob_pair"]) / sum_prob_pair 
        cumulativeprobability = np.cumsum(probabilities)
            
        randomNumber = round(random.uniform(0, cumulativeprobability[-1]),3)
        
        randomNumber = random.uniform(0, 1)
        selected_index = np.searchsorted(cumulativeprobability, randomNumber)
        return(selected_index)
    
    
    def bayesian_Reinforcement(self, smooth_obj, historical_virtual_obs_loop, concentration_parameter):
        
        # --------------------------Bayesian Reinforcement----------------------------------
        
        
        smooth_obj.sort_values(by="columns_selected_index", ascending = True, inplace=True)
        smooth_obj.reset_index(drop=True, inplace = True)
        df_smooth = smooth_obj.copy(deep = True)
        
        del smooth_obj
        
        pair_count = df_smooth.shape[0]
            
        max_obj_value = df_smooth.fun_value.max()
        df_smooth["fun_scaled"] = np.abs(df_smooth["fun_value"] - max_obj_value) 
        
        df_smooth_adjusted = df_smooth.copy(deep=True)
        
        df_smooth_adjusted.sort_values(by="columns_selected_index", ascending = True, inplace=True)
        df_smooth_adjusted["fun_normalized"] = df_smooth_adjusted["fun_scaled"]/ sum(df_smooth_adjusted["fun_scaled"])
        df_smooth_adjusted["virtual_obs"] = df_smooth_adjusted["fun_normalized"] * concentration_parameter
        
        historical_virtual_obs_loop.sort_values(by="columns_selected_index", ascending = True, inplace=True)
        historical_virtual_obs_loop.reset_index(drop=True, inplace = True)
            
        df_smooth_adjusted = pd.merge(left=df_smooth_adjusted, right=historical_virtual_obs_loop,
                 on = "columns_selected_index", how = "inner")
        df_smooth_adjusted["agg_virtual_obs"] = df_smooth_adjusted["virtual_obs"] + df_smooth_adjusted["agg_virtual_obs"]
        
        historical_virtual_obs_loop["agg_virtual_obs_new"] = df_smooth_adjusted["agg_virtual_obs"] 
        historical_virtual_obs_loop.drop(["agg_virtual_obs"], inplace = True, axis = 1)
        historical_virtual_obs_loop.columns = ["columns_selected_index","agg_virtual_obs"]
        
        aggregate_virtual_obs = np.sum(df_smooth_adjusted["agg_virtual_obs"])
        
        df_smooth_adjusted["prob_pair"] = (df_smooth_adjusted["agg_virtual_obs"]+ 1 )/(aggregate_virtual_obs + pair_count)
        
        df_smooth_adjusted = df_smooth_adjusted[["columns_selected_index", "agg_virtual_obs", "prob_pair"]]
        return df_smooth_adjusted, historical_virtual_obs_loop


    
    def fit_sample(self, X, y):
        
        column_names = X.columns
        X = np.array(X)
        label = np.array(y)
        columns = [k for k in range(X.shape[1])] 
        
        if(Counter(label)[1]>Counter(label)[0]):
            Min_num = Counter(label)[0]
            Maj_num = Counter(label)[1]
        else:
            Maj_num = Counter(label)[0]
            Min_num = Counter(label)[1]
        
        replication_needed = (Maj_num - Min_num)
        
                
        min_index = np.flatnonzero(y == 1).tolist()
        maj_index = np.flatnonzero(y == 0).tolist()
        
        x_scaled = self.normalize(X)
        
        target_scaled = x_scaled[min_index]
        non_target_scaled = x_scaled[maj_index]
            
        lower_bound, upper_bound = self.cal_bounds(target_scaled)
        pair_keys = self.get_keys(target_scaled)
        pair_keys_counter = [0]*len(pair_keys)
    
        
        br_itr_df_aggregate = pd.DataFrame()        
        br_df_agg = pd.DataFrame()
        generated_points_algo = pd.DataFrame()    
        
        historical_virtual_obs_agg_ref = pd.DataFrame()
        counter_creation_virtual_obs = 0
                
        concentration_parameter = ((self.con_para * len(pair_keys)) -1.0)/((1.0-self.con_para)*replication_needed)
                
        excluded_pairs_prior = pd.DataFrame()
        pair_ele_prev = 0
        track_child = 0
    
        
        while(track_child < replication_needed):
                
    
            br_df = pd.DataFrame()
    
                        
            if(track_child < 2):
                scoping_column_index = random.sample(range(0, len(pair_keys)), 1)[0]
                
            else:
                
                br_itr_df_aggregate_selection = br_itr_df_aggregate.groupby("columns_selected_index", as_index=False)["fun_value"].mean()
                
                excluded_items = set(pair_keys) - set(br_itr_df_aggregate_selection["columns_selected_index"]) 
                if excluded_pairs_prior.empty & len(excluded_items) > 0:
                    
                    excluded_pairs = pd.DataFrame({"columns_selected_index": list(excluded_items)}) 
                    excluded_pairs["fun_value"] = br_itr_df_aggregate_selection["fun_value"].max()                     
                    excluded_pairs_prior = pd.concat([excluded_pairs_prior, excluded_pairs], ignore_index= True)
                    br_itr_df_aggregate_selection = pd.concat([br_itr_df_aggregate_selection, excluded_pairs], ignore_index=True)
                                    
                elif not excluded_pairs_prior.empty:
                    
                    excluded_pairs_prior = excluded_pairs_prior[excluded_pairs_prior["columns_selected_index"]!=pair_ele_prev]
                
                    br_itr_df_aggregate_selection = pd.concat([br_itr_df_aggregate_selection, excluded_pairs_prior], ignore_index=True)
                    br_itr_df_aggregate_selection
                else:
                    
                    br_itr_df_aggregate_selection
                    
                if(counter_creation_virtual_obs == 0):
                    
                    historical_virtual_obs = br_itr_df_aggregate_selection.copy(deep=True)
                    historical_virtual_obs.drop("fun_value", axis=1, inplace = True)
                    historical_virtual_obs["agg_virtual_obs"] = 0
                    br_df, historical_virtual_obs_agg = self.bayesian_Reinforcement(br_itr_df_aggregate_selection, historical_virtual_obs, concentration_parameter)            
                    scoping_column_index = self.roulette_wheel_selection(br_df)
           
                    historical_virtual_obs_agg_ref = historical_virtual_obs_agg
                    counter_creation_virtual_obs +=1
                
                else:
                    
                    br_df, historical_virtual_obs_agg = self.bayesian_Reinforcement(br_itr_df_aggregate_selection, historical_virtual_obs_agg_ref, concentration_parameter)            
                    historical_virtual_obs_agg_ref = historical_virtual_obs_agg
                    scoping_column_index = self.roulette_wheel_selection(br_df)              
                
                    counter_creation_virtual_obs +=1
    
            
            pair_ele = pair_keys[scoping_column_index]
            pair_ele_export = tuple(pair_ele)
            
            dim_1_samples = random.sample(range(0, len(min_index)), len(list(pair_ele))+1)
            
            pair_ele_prev = pair_ele
            
            pair_keys_counter[scoping_column_index] += 1
             
            lower_bound_scope = [lower_bound[x] for x in list(pair_ele)]
            upper_bound_scope = [upper_bound[x] for x in list(pair_ele)]
            lower_bound_scope = np.array([float(x) if x is not None else -np.inf for x in lower_bound_scope])
            upper_bound_scope = np.array([float(x) if x is not None else np.inf for x in upper_bound_scope])
            
            selected_random_sample = target_scaled[dim_1_samples,:]
            selected_random_sample_scoping = selected_random_sample[:, list(pair_ele)] ##### initial simplex
            columns_not_in_scope = [k for k in columns if k not in pair_ele]
        
            df_column_not_in_scope = selected_random_sample[0, columns_not_in_scope]
            
            scaled_df_loop_scoping = np.concatenate((x_scaled[:, list(pair_ele)], label.reshape(-1, 1)), axis = 1)
        
            
            # --------------------------Oversampling----------------------------------

            function_call_output = minimize(self.ratio_objective, x0= selected_random_sample_scoping[0,:], 
                    args=(scaled_df_loop_scoping, self.k_obj) ,method='nelder-mead', 
                                            options={'adaptive': True, 'initial_simplex': selected_random_sample_scoping})
            
            
            br_itr_df = pd.DataFrame({"columns_selected_index": [pair_ele]}, index=[0])            
    
            br_itr_df["Iterations"] = function_call_output.nit    
            br_itr_df["fun_value"] = np.round(function_call_output.fun,5)
    
            br_df["columns_selected"] = [pair_ele_export]*len(br_df)            
            br_df["Iterations"] = function_call_output.nit        
            br_df["concentration_parameter"] = concentration_parameter
            
            br_df_agg = pd.concat([br_df_agg, br_df], axis = 0, ignore_index=True)
            
            br_itr_df_aggregate = pd.concat([br_itr_df_aggregate, br_itr_df], axis=0, ignore_index= True)
            
            function_call_output = function_call_output.final_simplex[0][0]
            
    
            function_call_output = function_call_output.tolist()
            
            # --------------------------Oversampling----------------------------------
                    
            function_call_output = np.clip(function_call_output, lower_bound_scope, upper_bound_scope)
    
            function_call_output = function_call_output.tolist()
            ##### Columns re-ordering
    
            
            df_column_not_in_scope_first = list(df_column_not_in_scope)
            function_call_output.extend(df_column_not_in_scope_first)
            
            
            column_name_list_scope_after_loop = list(pair_ele)
            column_name_list_scope_after_loop.extend(list(columns_not_in_scope))
            
            sorted_idx = sorted(range(len(column_name_list_scope_after_loop)), key=lambda i: column_name_list_scope_after_loop[i])

            function_call_output_sorted = [function_call_output[i] for i in sorted_idx]

            final_simplex_row_output = pd.DataFrame([function_call_output_sorted], columns=column_names)
            final_simplex_row_output = final_simplex_row_output[column_names]
            generated_points_algo = pd.concat([generated_points_algo,final_simplex_row_output], axis=0, ignore_index= True)
            track_child = track_child + 1
    
        target_complete = np.concatenate((target_scaled, generated_points_algo), axis=0)
        
        new_target_class = pd.DataFrame(target_complete, columns=column_names)
        
        new_target_class["class"] = 1
                
        old_target_scaled = pd.DataFrame(non_target_scaled, columns=column_names)
            
        old_target_scaled["class"] = 0
                        
        oversample_result = pd.concat([old_target_scaled, new_target_class], ignore_index = True)

        return oversample_result, br_itr_df_aggregate, historical_virtual_obs_agg_ref, br_df_agg
    
    

