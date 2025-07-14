# SOBER
SOBER for imbalanced learning

The implementation of Imbalanced Data Oversampling through Subspace Optimization with Bayesian Reinforcement (SOBER), which is under review in Artificial Intelligence Review. It generates synthetic samples using optimization, along with feature importance using Bayesian reinforcement. 
The following are the objects generated after oversampling

1. Oversampled dataset: oversample_result
2. Feature subspaces importance: br_itr_df_aggregate
3. Virtual observations generated during Bayesian reinforcement: historical_virtual_obs_agg_ref
4. Feature subspaces selection with concentration parameter (alpha): br_df_agg
