import numpy as np
import pandas as pd
import functools
from .constants import CHINA_PATH_TUPLE, US_PATH_TUPLE

def fill_china_missing_data_for_all_trading_date(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    trading_date = pd.read_pickle(CHINA_PATH_TUPLE.CHINA_TRADING_DATE)
    filled_df = raw_data_df.reindex(trading_date)
    filled_df = filled_df.assign(is_tradable = ~filled_df['close'].isna())
    prefill_columns = ['ts_code', 'pre_close', 'vol', 'amount', 'turnover_rate',
                       'volume_ratio','pe','pb', 'dv_ratio', 'total_mv']
    filled_df[prefill_columns] = filled_df[prefill_columns].fillna(method = 'backfill')
    for column in ['open', 'high', 'low', 'close']:
        filled_df.loc[filled_df[column].isna(), column] = filled_df.loc[filled_df[column].isna(), 'pre_close']
    for column in ['change', 'pct_chg']:
        filled_df.loc[filled_df[column].isna(), column] = 0
    return filled_df

def fill_US_missing_data_for_all_trading_date(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    trading_date = pd.read_pickle(US_PATH_TUPLE.US_TRADING_DATE)
    filled_df = raw_data_df.reindex(trading_date)
    filled_df = filled_df.assign(is_tradable = ~filled_df['close'].isna(),
                                 pre_close = filled_df['close'].shift(1))
    prefill_columns = ['tic', 'pre_close', 'total_mv', 'pe']
    filled_df[prefill_columns] = filled_df[prefill_columns].fillna(method = 'backfill')
    for column in ['close']:
        filled_df.loc[filled_df[column].isna(), column] = filled_df.loc[filled_df[column].isna(), 'pre_close']
    for column in ['pct_chg']:
        filled_df.loc[filled_df[column].isna(), column] = 0
    return filled_df
    
def missing_data_filled_for_all_trading_date(region):
    '''preprocessing decorator for data reading'''
    def _decorator(func):
        @functools.wraps(func)
        def wrapper_filled_missing_data(*args, **kwargs):
            if region == 'china':
                return fill_china_missing_data_for_all_trading_date(func(*args, **kwargs))
            elif region == 'US':
                return fill_US_missing_data_for_all_trading_date(func(*args, **kwargs))
            else:
                raise NotImplementedError(f'Unknown region={region} for missing_data_filled_for_all_trading_date')
        return wrapper_filled_missing_data
    return _decorator

@missing_data_filled_for_all_trading_date(region='china')
def read_china_modelstate_from_code(stock_code: str) -> pd.DataFrame:
    raw_df = pd.read_pickle(CHINA_PATH_TUPLE.CHINA_MODELSTATE_FMT.format(code=stock_code))
    return raw_df

@missing_data_filled_for_all_trading_date(region='US')
def read_US_modelstate_from_code(stock_code: str) -> pd.DataFrame:
    #print(stock_code)
    raw_df = pd.read_pickle(US_PATH_TUPLE.US_MODELSTATE_FMT.format(code=stock_code))
    return raw_df

def read_china_modelstate_from_code_series(stock_code_series: pd.Series) -> pd.DataFrame:
    data_series = stock_code_series.apply(read_china_modelstate_from_code)
    concat_df = pd.concat({stock_code_series.loc[idx]: data_series.loc[idx] 
                           for idx in stock_code_series.index})
    concat_df.index.set_names('stock_code', level = 0, inplace = True)
    return concat_df

def read_US_modelstate_from_code_series(stock_code_series: pd.Series) -> pd.DataFrame:
    data_series = stock_code_series.apply(read_US_modelstate_from_code)
    concat_df = pd.concat({stock_code_series.loc[idx]: data_series.loc[idx]
                           for idx in stock_code_series.index})
    concat_df.index.set_names('stock_code', level = 0, inplace = True)
    return concat_df

def sample_china_stock_series(
    sample_stock_num: int,
    random_generator: np.random.RandomState = np.random.RandomState(42)
) -> pd.Series:
    total_code_list = get_china_stock_codes()
    sampled_code_series = pd.Series(random_generator.choice(total_code_list,
                                                            size=sample_stock_num,
                                                            replace=False))
    return sampled_code_series

def sample_US_stock_series(
    sample_stock_num: int,
    random_generator: np.random.RandomState = np.random.RandomState(42)
) -> pd.Series:
    total_code_list = get_US_stock_codes()
    sampled_code_series = pd.Series(random_generator.choice(total_code_list,
                                                            size=sample_stock_num,
                                                            replace=False))
    return sampled_code_series

def get_china_stock_codes() -> pd.Series:
    '''return all China stock codes'''
    SZ_code_list = pd.read_pickle(CHINA_PATH_TUPLE.SZ_CODE)
    SH_code_list = pd.read_pickle(CHINA_PATH_TUPLE.SH_CODE)
    total_code_list = pd.concat([SZ_code_list, SH_code_list])
    return pd.Series(total_code_list)

def get_US_stock_codes() -> pd.Series:
    '''return all US stock codes'''
    code_list = pd.read_pickle(US_PATH_TUPLE.CODE)
    return pd.Series(code_list)