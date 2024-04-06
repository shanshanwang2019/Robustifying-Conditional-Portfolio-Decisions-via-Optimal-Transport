import pandas as pd
from typing import Callable, Optional, Dict, Any
from ..data.modelstate import read_china_modelstate_from_code_series, sample_china_stock_series
from ..factors import read_china_fama_french_factors_df
from ..solvers import portfolio_solver_wrapper
from .data_type import (
    SimDataWrapper,
    SolverInputDataWrapper,
)
from .base_sim import BaseSimulation

class ChinaSimulation(BaseSimulation):

    def __init__(
        self,
        portfolio_solver: Callable = portfolio_solver_wrapper,
        portfolio_solver_type: str = 'equal_weight',
        portfolio_solver_kwargs: Optional[Dict[str, Any]] = {},
    ):
        super(ChinaSimulation, self).__init__(
            portfolio_solver=portfolio_solver,
            portfolio_solver_type=portfolio_solver_type,
            portfolio_solver_kwargs=portfolio_solver_kwargs,
        )
        self.region = 'China'
        self.sample_stock_series_func = sample_china_stock_series

    def read_data_for_simulation(self, stock_code_series: pd.Series) -> SimDataWrapper:
        all_stock_data_df = read_china_modelstate_from_code_series(stock_code_series)
        pct_chg_df = all_stock_data_df.pct_chg.to_frame().reset_index().pivot(
            index='time', columns='stock_code'
        ).pct_chg.loc[:, stock_code_series]
        fama_french_factors_df = read_china_fama_french_factors_df()
        is_tradable_df = all_stock_data_df.is_tradable.to_frame().reset_index().pivot(
            index='time', columns='stock_code'
        ).is_tradable.loc[:, stock_code_series]
        return_data = SimDataWrapper(
            pct_chg_df=pct_chg_df,
            fama_french_factors_df=fama_french_factors_df,
            is_tradable_df=is_tradable_df
        )
        return return_data

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
        window_end = trading_date
        window_start = (pd.Timestamp(trading_date) - pd.Timedelta(window_size)).strftime('%Y%m%d')
        historical_returns_df = sim_data.pct_chg_df.loc[window_start:window_end]
        historical_factors_df = sim_data.fama_french_factors_df.shift(1).fillna(0) \
                                    .loc[window_start:window_end]
        current_factor_series = sim_data.fama_french_factors_df.loc[trading_date]
        is_tradable_series = sim_data.is_tradable_df.loc[window_end]
        return_data = SolverInputDataWrapper(
            historical_returns_df=historical_returns_df,
            historical_factors_df=historical_factors_df,
            current_factor_series=current_factor_series,
            is_tradable_series=is_tradable_series,
        )
        return return_data
