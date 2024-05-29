# Robustifying-Conditional-Portfolio-Decisions-via-Optimal-Transport

The data, code and results for mean-CVaR models (folder "/mean-CVaR") and mean-variance models (folder "/mean-variance") are described in "Robustifying Conditional Portfolio Decisions via Optimal Transport", where

"./data” includes the US stock return data and side information downloaded from Wharton Research Data Services and yfinance, respectively.

All the core computational code is in the folder "./modeling", in which

    "./modeling/data" is for data preprocessing
    "./modeling/factors" is for side information preprocessing
    "./modeling/simulations" are for the parameter tuning and out-of-sample performance computation framework
    "./modeling/solvers" is the code for solving different types of portfolio optimization problems. 

"./py_scripts" is used to record output, e.g., the hyperparameters of each portfolio allocation method.

Run the file "run_seed.py" to generate the optimal hyperparameters for each eta (corresponding
to different possible trade-offs between the portfolio mean return and portfolio CVaR or variance for the portfolio
manager) and the different types of portfolio optimization problems. With the optimal hyperparameters, run the file "run_optimal_models.py" to get all the tables described in the paper.

"results_eat_#.out" is the results involved in the tables for different eta values.



