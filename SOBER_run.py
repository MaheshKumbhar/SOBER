# -*- coding: utf-8 -*-
"""
# This project is licensed under the MIT License - see the LICENSE file for details

Created on Wed May 28 08:01:21 2025

@author: Mahesh Kumbhar

"""


from SOBER_package import SOBER  
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

if __name__ == '__main__':
    
    file_name = 'dataSet_name'

    try:    
        df = pd.read_excel(fr"{file_name}.xlsx", engine='openpyxl')
        df.sort_values(by= df.columns[-1], ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)
        ori_column_names = [s.strip() for s in df.columns]
        df.columns = ori_column_names
        df_original = df.copy(deep=True)
    except Exception as e:
        try:
            df = pd.read_csv(fr"{file_name}.csv")
            df.sort_values(by= df.columns[-1], ascending=False, inplace=True)
            df.reset_index(inplace=True, drop=True)
            ori_column_names = [s.strip() for s in df.columns]
            df.columns = ori_column_names
            df_original = df.copy(deep=True)
        except Exception as e:
            None
        
    ######################################
    df.reset_index(inplace=True, drop=True)
    df_original = df.copy(deep=True)
    

    
    train_X, valid_X, train_y, valid_y = train_test_split(df_original.iloc[:,:-1], df_original.iloc[:,-1], 
                    test_size=0.2, stratify=df_original.iloc[:,-1], shuffle=True)
    

    oversample_result, br_itr_df_aggregate, historical_virtual_obs_agg_ref, br_df_agg = SOBER().fit_sample(train_X, train_y)#, 5, 0.95)
    
    # oversample_result - oversampled dataset
    # br_itr_df_aggregate - used to plot feature importance
    # historical_virtual_obs_agg_ref - virtual observations feature pair wise
    # br_df_agg - concentration parameter with feature pair selection
    

