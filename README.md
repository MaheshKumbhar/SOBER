# SOBER
SOBER for imbalanced learning

## Introduction
The implementation of Imbalanced Data Oversampling through Subspace Optimization with Bayesian Reinforcement (SOBER), which is under review in Artificial Intelligence Review. It generates synthetic samples using optimization, along with feature importance using Bayesian reinforcement. 

## Usage
There is one method `SOBER().fit_sample(train_X, train_y)`, which takes the following two inputs:

- `train_X` : Training dataset features 
- `train_y` : Training dataset class label

Output:
 - `oversample_result`: Oversampled dataset
- `br_itr_df_aggregate`: Feature subspaces importance
- `historical_virtual_obs_agg_ref`: Virtual observations generated during Bayesian reinforcement 
- `br_df_agg`: Feature subspaces selection with concentration parameter (alpha)
