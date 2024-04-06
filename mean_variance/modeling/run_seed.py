import pandas as pd
import numpy as np
import sys
import glob
import itertools
#sys.path.append('E:\projects\DR-portfolio-optimization')
sys.path.append('./')
import modeling.data.modelstate
import modeling.simulations.US_sim
from modeling.simulations import SimulationAnalyzer
start_date = '20190101'
end_date = '20210101'

seed_id_list = list(range(256))
cmd = 'python ../py_scripts/US_sim.py'
exp_params = {
    'equal_weight': {'reg_params': [9]},
	'mean_variance': {'reg_params': [9]},
    'dr_mean_variance': {'reg_params': [9], 'rho': [0.05,0.1,0.25]},
    'cond_mean_variance': {'reg_params': [9], 'neighbor_quantile': [0.05,0.1,0.25]},
    'dr_winf_cond_mean_variance': {'reg_params': [9], 'gamma_quantile': [0.05,0.1,0.25], 'rho_quantile': [0.05,0.1,0.25]},
    'dr_w2_cond_mean_variance': {'reg_params': [9], 'epsilon': [0.05, 0.1,0.15], 'rho_div_rho_min':  [1.05,1.1,1.15]},
}
import itertools
def iterdict(input_dict):
    for vals in itertools.product(*input_dict.values()):
        yield dict(zip(input_dict.keys(), vals))

for model_name in exp_params:
    for solver_kwargs in iterdict(exp_params[model_name]):
        #if 'variance' not in model_name:
        #    continue
        sim = modeling.simulations.US_sim.USSimulation(
            portfolio_solver_type = model_name,
            portfolio_solver_kwargs = solver_kwargs,
        )
        sim.batch_run_from_seed_id_list(seed_id_list=list(range(1)), stock_num=20, start_date=start_date, end_date=end_date, save_result=True)

import itertools
def iterdict(input_dict):
    for vals in itertools.product(*input_dict.values()):
        yield dict(zip(input_dict.keys(), vals))

def read_pickle_regex(path_regex):
    import re
    import glob
    prog = re.compile(path_regex)
    glob_pattern = re.sub('\(.*?\)', '*', path_regex)
    if path_regex == glob_pattern:
        return pd.read_pickle(path_regex)
    summary_table_dict = {}
    for path in glob.glob(glob_pattern, recursive=True):
        path=path.replace('\\','/')
        match = prog.match(path)
        if match:
            res_table = pd.read_pickle(path)
            group = tuple(float(key) for key in match.groups())
            summary_table_dict[group] = pd.read_pickle(path)
    summary_table = pd.concat(summary_table_dict, names=prog.groupindex)
    return summary_table
def read_model_sim_ana_summary(model_name, opt_params_dict = None, start_date = 20190101, end_date = 20210101, summarizer = lambda x: x.mean()):
    if opt_params_dict is None:
        opt_params_dict = {
            'equal_weight': ['reg_params'],
            'mean_variance': ['reg_params'],
            'dr_mean_variance': ['reg_params', 'rho'],
            'cond_mean_variance': ['reg_params','neighbor_quantile'],
            'dr_winf_cond_mean_variance': ['reg_params', 'gamma_quantile', 'rho_quantile'],
            'dr_w2_cond_mean_variance': ['reg_params','epsilon', 'rho_div_rho_min'],
        }
    pname2regex = lambda param: f"{param}_(?P<{param}>.*?)"
    head = './sim_results/US'
    tail = f'return_ana/seed_0_{start_date}_{end_date}.pkl'
    path_regex = '/'.join([head,model_name, '/'.join(map(pname2regex, sorted(opt_params_dict[model_name]))),tail])
    if len(opt_params_dict[model_name]) > 0:
        res = summarizer(read_pickle_regex(path_regex).groupby(level=list(range(len(opt_params_dict[model_name])))))
    else:
        res = summarizer(read_pickle_regex(path_regex)).to_frame().transpose()
    return res
def show_latex(df):
    head='''\\begin{table}[ht]
\centering
\small'''
    print(head)
    col = ['pnl', 'stdDev', 'gSharpe', 'tcBySd', 'Sharpe', 'maxDraw', 'totalMv', 'tradedMv','CVaRUtil@0.1']
    print(df[col].round(3).to_latex())
    print('\end{table}')
    return df
opt_params_dict = {
            'equal_weight': ['reg_params'],
            'mean_variance': ['reg_params'],
            'dr_mean_variance': ['reg_params','rho'],
            'cond_mean_variance': ['reg_params','neighbor_quantile'],
            'dr_winf_cond_mean_variance': ['reg_params', 'gamma_quantile', 'rho_quantile'],
            'dr_w2_cond_mean_variance': ['reg_params','epsilon', 'rho_div_rho_min'],
        }
model_list = ['equal_weight', 'mean_variance', 'dr_mean_variance','cond_mean_variance','dr_winf_cond_mean_variance','dr_w2_cond_mean_variance']

with open('./py_scripts/post_cv_commands.txt','w') as f:
    for model_name in model_list:
        cv_summary_df = read_model_sim_ana_summary(model_name = model_name,start_date='20190101',end_date='20210101')
        if model_name == 'equal_weight1':
            args = (f'--start_date {start_date} --end_date {end_date}'
                    f' --portfolio_solver_type {model_name}\n')
        else:
            if len(cv_summary_df.index.names) > 1:
                portfolio_solver_kwargs = ','.join([f'{name}:{val}' for name, val in 
                                                    zip(cv_summary_df.index.names, cv_summary_df['obj_series'].idxmin())])
            else:
                portfolio_solver_kwargs = '{name}:{val}'.format(name=cv_summary_df.index.names[0], val=cv_summary_df['obj_series'].idxmin())
            args = (f'--start_date {start_date} --end_date {end_date}'
                    f' --portfolio_solver_type {model_name}'
                    f' --portfolio_solver_kwargs {portfolio_solver_kwargs}'
                    '\n')        
        full_cmd = (cmd+' '+args)
        f.write(full_cmd)
print(pd.concat({
    model: read_model_sim_ana_summary(model_name = model, opt_params_dict = opt_params_dict, 
                                      start_date='20190101', end_date='20210101').loc[lambda df: [df['obj_series'].idxmin()]].reset_index()
    for model in model_list
}).droplevel(-1).eval('grossPnl = std_series * sharp_series')[
    ['pnl_series','obj_series', 'cvar_series', 'std_series','sharp_series', 'drawdown_series']
].loc[['equal_weight', 'mean_variance', 'dr_mean_variance','cond_mean_variance','dr_winf_cond_mean_variance','dr_w2_cond_mean_variance']].round(4).to_latex())