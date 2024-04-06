import pandas as pd
from typing import NamedTuple

__all__ = ['SimDataWrapper', 'SolverInputDataWrapper']

class SimDataWrapper(NamedTuple):
    """
        Wraps the input data for simulation.
        ----------
        pct_chg_df: DataFrame in wide-form that contains the the stock return datas
        fama_french_factors_df: DataFrame that stores the fama_french factors
        is_tradable_df: DataFrame in wide-form that contains if a stock is tradable at a trading_date
    """
    pct_chg_df: pd.DataFrame
    fama_french_factors_df: pd.DataFrame
    is_tradable_df: pd.DataFrame

class SolverInputDataWrapper(NamedTuple):
    """
        Wraps data as input for solver.
        ----------
        historical_returns_df: DataFrame in wide-form that contains historical stock return
        historical_factors_df: DataFrame that stores the prev historical fama_french factors
        current_factor_series: pd.Series that stores current fama_french factor vector
        is_tradable_series: pd.Series[bool] that indicates if a stock is tradable
    """
    historical_returns_df: pd.DataFrame
    historical_factors_df: pd.DataFrame
    current_factor_series: pd.Series
    is_tradable_series: pd.Series