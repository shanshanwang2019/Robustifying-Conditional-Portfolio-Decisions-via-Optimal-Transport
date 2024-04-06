import numpy as np
import pandas as pd
from typing import Optional
from ..data.constants import (
    CHINA_PATH_TUPLE,
    US_PATH_TUPLE,
)
from ..data.modelstate import (
    get_china_stock_codes,
    get_US_stock_codes,
    read_china_modelstate_from_code_series,
    read_US_modelstate_from_code_series,
)

def market_pct_chg_kernel(df: pd.DataFrame) -> float:
    return (df.pct_chg*df.total_mv).sum() / df.total_mv.sum()

def smb_pct_chg_kernel(df: pd.DataFrame) -> float:
    smb_filters = df.total_mv.lt(df.total_mv.median())
    small_cap_portfolio_pct_ret = df.loc[smb_filters].pct_chg.mean()
    big_cap_portfolio_pct_ret = df.loc[~smb_filters].pct_chg.mean()
    SMB_pct_ret = small_cap_portfolio_pct_ret - big_cap_portfolio_pct_ret
    return SMB_pct_ret

def hml_pct_chg_kernel(df: pd.DataFrame) -> float:
    high_value_filter = df.pe.le(df.pe.quantile(q = 0.3))
    low_value_filter = df.pe.ge(df.pe.quantile(q = 0.7))
    # mid_value_filter = ~(low_value_filter|high_value_filter)
    # compute portfolio return
    high_value_filter_portfolio_pct_ret = df.loc[high_value_filter].pct_chg.mean()
    low_value_filter_portfolio_pct_ret = df.loc[low_value_filter].pct_chg.mean()
    # mid_value_filter_portfolio_pct_ret = df.loc[mid_value_filter].pct_chg.mean()
    HML_pct_ret = high_value_filter_portfolio_pct_ret - low_value_filter_portfolio_pct_ret
    return HML_pct_ret

def gen_china_fama_french_factors_df(save_path: Optional[str] = None):
    print('Reading all china stock data...')
    all_china_stock_data_df = read_china_modelstate_from_code_series(get_china_stock_codes())
    print('Generating Fama-French Factors...')
    fama_french_df = pd.DataFrame({
        'market_pct_chg': all_china_stock_data_df.groupby('time').apply(market_pct_chg_kernel),
        'SMB_pct_chg': all_china_stock_data_df.groupby('time').apply(smb_pct_chg_kernel),
        'HML_pct_chg': all_china_stock_data_df.groupby('time').apply(hml_pct_chg_kernel),
    })
    if save_path is None:
        save_path = CHINA_PATH_TUPLE.FAMA_FRENCH_FACTORS
    fama_french_df.to_pickle(save_path)
    print('Fama-French factors saved to path:\n', save_path)

def gen_US_fama_french_factors_df(save_path: Optional[str] = None):
    print('Reading all US stock data...')
    all_US_stock_data_df = read_US_modelstate_from_code_series(get_US_stock_codes())
    print('Generating Fama-French Factors...')
    fama_french_df = pd.DataFrame({
        'market_pct_chg': all_US_stock_data_df.groupby('time').apply(market_pct_chg_kernel),
        'SMB_pct_chg': all_US_stock_data_df.groupby('time').apply(smb_pct_chg_kernel),
        'HML_pct_chg': all_US_stock_data_df.groupby('time').apply(hml_pct_chg_kernel),
    })
    if save_path is None:
        save_path = US_PATH_TUPLE.FAMA_FRENCH_FACTORS
    fama_french_df.to_pickle(save_path)
    print('Fama-French factors saved to path:\n', save_path)

def read_china_fama_french_factors_df(path: Optional[str] = None):
    if path is None:
        path = CHINA_PATH_TUPLE.FAMA_FRENCH_FACTORS
    return pd.read_pickle(path)

def read_US_fama_french_factors_df(path: Optional[str] = None):
    if path is None:
        path = US_PATH_TUPLE.FAMA_FRENCH_FACTORS
    return pd.read_pickle(path)