from collections import namedtuple

CHINA_DATA_PATH = './data/TUShareData'
US_DATA_PATH = './data/WRDSData'
SIM_RESULT_PATH_FMT = './sim_results/{region}/' \
                      '{portfolio_solver_type}/{solver_kwargs_str}/{result_type}/' \
                      'seed_{seed_id}_{start_date}_{end_date}.pkl'

USPathTuple = namedtuple(
    'USPathTuple',
    [
        'ROOT', 'CODE',
        'US_TRADING_DATE',
        'US_MODELSTATE_FMT',
        'FAMA_FRENCH_FACTORS'
    ]
)

US_DATA_PATH_DICT = {
    'ROOT': US_DATA_PATH,
    'CODE': US_DATA_PATH + '/meta/code_list.pkl',
    'US_TRADING_DATE': US_DATA_PATH + '/meta/trading_dates.pkl',
    'US_MODELSTATE_FMT': US_DATA_PATH + '/modelstate/join_table_{code}.pkl',
    'FAMA_FRENCH_FACTORS': US_DATA_PATH + '/factors/fama_french.pkl',
}

ChinaPathTuple = namedtuple(
    'ChinaPathTuple', 
    [
        'ROOT', 'SZ_CODE', 'SH_CODE', 
        'CHINA_TRADING_DATE',
        'CHINA_MODELSTATE_FMT',
        'FAMA_FRENCH_FACTORS'
    ]
)

CHINA_DATA_PATH_DICT = {
    'ROOT': CHINA_DATA_PATH,
    'SZ_CODE': CHINA_DATA_PATH + '/meta/SZ_code_list.pkl',
    'SH_CODE': CHINA_DATA_PATH + '/meta/SH_code_list.pkl',
    'CHINA_TRADING_DATE': CHINA_DATA_PATH + '/meta/trading_dates.pkl',
    'CHINA_MODELSTATE_FMT': CHINA_DATA_PATH + '/modelstate/join_table_{code}.pkl',
    'FAMA_FRENCH_FACTORS': CHINA_DATA_PATH + '/factors/fama_french.pkl',
}

CHINA_PATH_TUPLE = ChinaPathTuple(
    **CHINA_DATA_PATH_DICT
)
US_PATH_TUPLE = USPathTuple(
    **US_DATA_PATH_DICT
)