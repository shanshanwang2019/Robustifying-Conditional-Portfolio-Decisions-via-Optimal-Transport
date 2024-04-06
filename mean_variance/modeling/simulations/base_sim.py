import pandas as pd
import numpy as np
import pathlib
from os import path
from tqdm.auto import tqdm
from abc import abstractmethod, ABC
from typing import Callable, Optional, Dict, Any, List
from ..solvers import portfolio_solver_wrapper
from ..data import (
    SIM_RESULT_PATH_FMT,
)
from .data_type import (
    SimDataWrapper,
    SolverInputDataWrapper,
)
from time import process_time 
class BaseSimulation(ABC):
    def __init__(
        self,
        portfolio_solver: Callable = portfolio_solver_wrapper,
        portfolio_solver_type: str = 'equal_weight',
        portfolio_solver_kwargs: Optional[Dict[str, Any]] = {},
    ):
        self.portfolio_solver = portfolio_solver
        self.portfolio_solver_kwargs = portfolio_solver_kwargs
        self.solver_kwargs_str = self._kwargs2pathstr(portfolio_solver_kwargs)
        self.portfolio_solver_type = portfolio_solver_type
        self.region = None
        self.sim_data = None
        self.goal_position_df = None
        self.return_df = None
        self.return_ana_df = None
        self.traded_mv_df = None
        self.SIM_RESULT_PATH_FMT = SIM_RESULT_PATH_FMT
        self.sample_stock_series_func = None

    def batch_run_from_seed_id_list(
        self,
        seed_id_list: List[int],
        stock_num: int,
        start_date: str,
        end_date: str,
        window_size: str = '730d',
        show_single_seed_progress: bool = False,
        save_result: bool = True,
        save_path_params: Dict = {},
    ):
        for seed_id in tqdm(seed_id_list, desc="Seed progress", position=0):
            self.run_from_seed_id(
                seed_id=seed_id,
                stock_num=stock_num,
                start_date=start_date,
                end_date=end_date,
                window_size=window_size,
                show_progress=show_single_seed_progress,
                save_result=save_result,
                save_path_params=save_path_params,
            )

    def run_from_seed_id(
        self,
        seed_id: int,
        stock_num: int,
        start_date: str,
        end_date: str,
        window_size: str = '730d',
        show_progress: bool = True,
        save_result: bool = True,
        save_path_params: Dict = {},
    ):
        stock_code_series = self.sample_stock_series_func(stock_num, np.random.RandomState(seed_id))
        save_path_params['seed_id'] = seed_id
        return self.run(
            stock_code_series=stock_code_series,
            seed_id=seed_id,
            stock_num=stock_num,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            show_progress=show_progress,
            save_result=save_result,
            save_path_params=save_path_params
        )

    def run(
        self,
        stock_code_series: pd.Series,
        seed_id: int,
        stock_num: int,
        start_date: str,
        end_date: str,
        window_size: str = '730d',
        show_progress: bool = True,
        save_result: bool = True,
        save_path_params: Dict = {},

    ):
        if save_result:
            save_path_params['result_type'] = 'goal_position'
            save_path_params['start_date'] = start_date
            save_path_params['end_date'] = end_date
            save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            save_path_params['result_type'] = 'traded_mv'
            mv_save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            save_path_params['result_type'] = 'return'
            rt_save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            save_path_params['result_type'] = 'return_ana'
            rt_ana_save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            if path.exists(save_path) and path.exists(mv_save_path) and path.exists(rt_save_path) and path.exists(rt_ana_save_path):
                self.goal_position_df = pd.read_pickle(save_path)
                self.traded_mv_df = pd.read_pickle(mv_save_path)
                self.return_df = pd.read_pickle(rt_save_path)
                self.return_ana_df = pd.read_pickle(rt_ana_save_path)
                print(f'save path exist: \n{save_path}\nskip.')
                return self.goal_position_df
        self.sim_data = self.read_data_for_simulation(stock_code_series)        
        trading_dates = pd.Series(self.sim_data.pct_chg_df.index)
        #print(trading_dates)
        trading_dates_filter = (trading_dates >= start_date) & (trading_dates <= end_date)
        trading_dates = trading_dates.loc[trading_dates_filter]
        goal_money_position_dict = {}
        traded_mv_dict = {}
        return_dict = {}
        # set the initial position
        #goal_money_position_dict[pd.Timestamp(start_date) - pd.Timedelta('1d')] = current_money_position
        if show_progress:
            trading_dates_iterator = tqdm(trading_dates, desc='Trading Dates', leave=False, position=1)
        else:
            trading_dates_iterator = trading_dates
        total_trade_date=60
        t1_start = process_time() 
        for trading_date in trading_dates_iterator:
            stock_code_series = self.sample_stock_series_func(stock_num, np.random.RandomState(seed_id))
            seed_id=seed_id+1
            self.sim_data = self.read_data_for_simulation(stock_code_series)               
            solver_input_data = self.wrap_input_data_for_solver(self.sim_data, trading_date.strftime('%Y%m%d'), window_size)
            current_money_position = pd.Series(1/len(stock_code_series), index=stock_code_series)
            goal_money_position = self.portfolio_solver(solver_input_data,
                                                        current_money_position,
                                                        self.portfolio_solver_type,
                                                        **self.portfolio_solver_kwargs)
            # assert np.isclose(goal_money_position.sum(), current_money_position.sum(), rtol = 1e-4, atol= 1e-4)
            #print(self.portfolio_solver_kwargs['reg_params'])
            traded_mv_dict[trading_date] = goal_money_position - current_money_position
            goal_money_position_dict[trading_date] = goal_money_position
            trading_dates_ = pd.Series(self.sim_data.pct_chg_df.index)
            trading_date_end=(pd.Timestamp(trading_date) + pd.Timedelta('200d')).strftime('%Y%m%d')
            trading_dates_filter_ = (trading_dates_ >= trading_date) & (trading_dates_ <= trading_date_end)
            trading_dates_ = trading_dates_.loc[trading_dates_filter_]
            goal_money_position_return=[]
            num_trade=0
            for trading_date_iter in trading_dates_:
                num_trade=num_trade+1
                goal_money_position_return_each=goal_money_position * self.sim_data.pct_chg_df.loc[trading_date_iter]
                goal_money_position_return.append( goal_money_position_return_each.sum()) 
                if num_trade>=total_trade_date:
                    break
            #print(goal_money_position_return)          
            return_dict[trading_date] = goal_money_position_return
            #print(len(goal_money_position_return))
        t1_end = process_time()  
        print("Elapsed time during the whole program in seconds:", 
                                         t1_end-t1_start)
        self.goal_position_df = pd.DataFrame(goal_money_position_dict).transpose()
        self.traded_mv_df = pd.DataFrame(traded_mv_dict).transpose()
        self.return_df = pd.DataFrame(return_dict).transpose()
        def _drawdown_tranform(df: pd.DataFrame) -> pd.Series:
            row, col=df.shape
            drawdown=[]
            for i in range(row):
                s_drawdown=df.iloc[i, :].cummax()-df.iloc[i, :]
                drawdown.append(s_drawdown.max())
            return pd.Series(drawdown,index=df.index)
        def _compute_CVaR_(df: pd.DataFrame, q:float = 0.05) -> pd.Series:
            row, col=df.shape
            quan_df=df.quantile(q=0.05,axis=1)
            cvar_val=[]
            for i in range(row):
                s_quantile_diff=df.iloc[i, :]-quan_df[i]
                cvar_val.append(-s_quantile_diff[s_quantile_diff < 0].mean())
            return pd.Series(cvar_val,index=df.index)
        self.return_ana_df = pd.DataFrame({
            'pnl_series': self.return_df.mean(axis=1),
			'std_series': self.return_df.std(axis=1),
			'cvar_series': _compute_CVaR_(self.return_df,q=0.05),
            'obj_series': self.return_df.std(axis=1)*self.return_df.std(axis=1)-self.portfolio_solver_kwargs['reg_params']*self.return_df.mean(axis=1),
            'sharp_series': self.return_df.mean(axis=1)/self.return_df.std(axis=1)*np.sqrt(252),
            'drawdown_series': _drawdown_tranform(self.return_df),#compute based on goal position
        })
        if save_result:
            save_path_params['result_type'] = 'goal_position'
            save_path_params['start_date'] = start_date
            save_path_params['end_date'] = end_date
            save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.goal_position_df.to_pickle(save_path)
            save_path_params['result_type'] = 'traded_mv'
            save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.traded_mv_df.to_pickle(save_path)
            save_path_params['result_type'] = 'return'
            save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.return_df.to_pickle(save_path)
            save_path_params['result_type'] = 'return_ana'
            save_path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **save_path_params)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.return_ana_df.to_pickle(save_path)
        return self.goal_position_df

    def load_sim_result(
        self,
        start_date: str,
        end_date: str,
        seed_id: Optional[int] = None,
        path_params: Dict = {},
        load_traded_mv: bool = True,
    ):
        if seed_id is not None:
            path_params['seed_id'] = seed_id
        path_params['start_date'] = start_date
        path_params['end_date'] = end_date
        path_params['result_type'] = 'goal_position'
        path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **path_params)
        self.goal_position_df = pd.read_pickle(path)
        if load_traded_mv:
            path_params['result_type'] = 'traded_mv'
            path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **path_params)
            self.traded_mv_df = pd.read_pickle(path)

    def get_sim_ana_result_path(
        self,
        seed_id_list: List[int],
        start_date: str,
        end_date: str,
        path_params: Dict = {},
    ):
        path_params['start_date'] = start_date
        path_params['end_date'] = end_date
        path_params['result_type'] = 'sim_ana_table'
        path_params['seed_id'] = f'{np.array(seed_id_list).min()}_to_{np.array(seed_id_list).max()}'
        path = self.SIM_RESULT_PATH_FMT.format(**self.__dict__, **path_params)
        pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
        return path

    def batch_load_total_position_result(
        self,
        seed_id_list: List[int],
        start_date: str,
        end_date: str,
        path_params: Dict = {},
    ):
        total_position_dict = {}
        for seed_id in tqdm(seed_id_list, desc="Seed progress"):
            self.load_sim_result(
                start_date=start_date,
                end_date=end_date,
                seed_id=seed_id,
                path_params=path_params,
                load_traded_mv=False
            )
            total_position_dict[f'seed_{seed_id}'] = self.goal_position_df.sum(axis = 1)
        return pd.DataFrame(total_position_dict)

    def _kwargs2pathstr(self, kwargs:Dict) -> str:
        '''
        a helper function to transform kwargs to str.

        Parameters
        ----------
        kwargs: solver kwargs

        Returns
        -------
        a string as partial path to save the simulation result
        '''
        return '/'.join(f'{key}_{kwargs[key]:.4f}' for key in sorted(kwargs))

    @abstractmethod
    def read_data_for_simulation(
        self,
        stock_code_series: pd.Series
    ) -> SimDataWrapper:
        pass

    @abstractmethod
    def wrap_input_data_for_solver(
        self,
        sim_data: SimDataWrapper,
        trading_date: str,
        window_size: str = '730d',
    ) -> SolverInputDataWrapper:
        """
        Wraps the input data for solver. Assume that trade happens right before the market close
        Parameters
        ----------
        self
        sim_data: DataWrapper that contains return and factor datas
        trading_date: Time for solver to determine position change, in 'YYYYMMDD' format
        window_size: Length of historical data to lookback, like '730d'
        """
        pass