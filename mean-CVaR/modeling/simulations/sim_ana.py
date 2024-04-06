import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Dict
from .base_sim import BaseSimulation

class SimulationAnalyzer(object):

    def __init__(self, analysis_input_df: pd.DataFrame):
        self.analysis_input_df = analysis_input_df

    @staticmethod
    def from_simulation(
        sim_model: BaseSimulation,
        ana_start_date: Optional[str] = None,
        ana_end_date: Optional[str] = None,
    ):
        def _drawdown_tranform(df: pd.DataFrame) -> pd.Series:
            row, col=df.shape
            drawdown=[]
            for i in range(row):
                s_drawdown=df.iloc[i, :].cummax()-df.iloc[i, :]
                drawdown.append(s_drawdown.max())
            return pd.Series(drawdown)
        def _compute_CVaR_(df: pd.DataFrame, q:float = 0.05) -> pd.Series:
            row, col=df.shape
            quan_df=df.quantile(q=0.05,axis=1)
            cvar_val=[]
            for i in range(row):
                s_quantile_diff=df.iloc[i, :]-quan_df[i]
                cvar_val.append(s_quantile_diff[s_quantile_diff < 0].mean())
            return pd.Series(cvar_val)
        analysis_input_df = pd.DataFrame({
            'pnl_series': sim_model.goal_position_df.mean(axis=1),
			'std_series': sim_model.goal_position_df.std(axis=1),
			'cvar_series': _compute_CVaR_(sim_model.goal_position_df,q=0.05),
            'traded_mv_series': sim_model.traded_mv_df.abs().sum(axis=1),
            'drawdown_series': _drawdown_tranform(sim_model.goal_position_df),#compute based on goal position
        })
        if ana_start_date is not None:
            analysis_input_df = analysis_input_df.loc[ana_start_date:]
        if ana_end_date is not None:
            analysis_input_df = analysis_input_df.loc[:ana_end_date]
        simulation_analyzer = SimulationAnalyzer(analysis_input_df)
        return simulation_analyzer

    @staticmethod
    def batch_load_pnl_series(
        sim_model: BaseSimulation,
        seed_id_list: List[int],
        start_date: str,
        end_date: str,
        ana_start_date: Optional[str] = None,
        ana_end_date: Optional[str] = None,
        path_params: Dict = {},
    ):
        result_dict = {}
        for seed_id in tqdm(seed_id_list, desc="Seed progress"):
            sim_model.load_sim_result(
                start_date=start_date,
                end_date=end_date,
                seed_id=seed_id,
                path_params=path_params,
                load_traded_mv=True
            )
            result_dict[seed_id] = \
                SimulationAnalyzer.from_simulation(
                    sim_model,
                    ana_start_date=ana_start_date,
                    ana_end_date=ana_end_date,
                ).analysis_input_df.pnl_series
        return_df = pd.DataFrame(result_dict)
        return return_df

    @staticmethod
    def sim_model_batch_annual_analysis_from_seed_id_list(
        sim_model: BaseSimulation,
        seed_id_list: List[int],
        start_date: str,
        end_date: str,
        ana_start_date: Optional[str] = None,
        ana_end_date: Optional[str] = None,
        path_params: Dict = {},
        save_sim_ana_result: bool = True,
        CVaR_q: float = 0.05,
        risk_aversion:float=0.5,
        util_inverse_ra: float = 0.1,
        tc_scale: float = 0.0001,
    ):
        result_dict = {}
        for seed_id in tqdm(seed_id_list, desc="Seed progress"):
            sim_model.load_sim_result(
                start_date=start_date,
                end_date=end_date,
                seed_id=seed_id,
                path_params=path_params,
                load_traded_mv=True
            )
            result_dict[seed_id] = \
                SimulationAnalyzer.from_simulation(
                    sim_model,
                    ana_start_date=ana_start_date,
                    ana_end_date=ana_end_date,
                ).analyze(
                    index_type='annual_only',
                    CVaR_q=CVaR_q,
                    risk_aversion=risk_aversion,
                    util_inverse_ra=util_inverse_ra,
                    tc_scale=tc_scale,
                ).loc['ANNUAL']
        return_df = pd.DataFrame(result_dict).transpose()
        return_df.index.name = 'seed'
        if save_sim_ana_result:
            start_date = ana_start_date if ana_start_date is not None else start_date
            end_date = ana_end_date if ana_end_date is not None else end_date
            path = sim_model.get_sim_ana_result_path(
                seed_id_list=seed_id_list,
                start_date=start_date,
                end_date=end_date,
                path_params=path_params,
            )
            return_df.to_pickle(path)
        return return_df

    def _compute_mean(self, series: pd.Series, normalize=None):
        if normalize is None:
            scale = series.shape[0]
        else:
            scale = normalize
        return series.mean() * scale# annualized mean= day's average return * 252
    def _compute_sharp(self, series: pd.Series, series_: pd.Series,  normalize=None):
        if normalize is None:
            scale = series.shape[0]
        else:
            scale = normalize
        sharp_=series/series_ * np.sqrt(scale)
        return sharp_.mean()# annualized mean= day's average return * 252
    def _compute_obj(self, series_mean: pd.Series, series_cvar: pd.Series,  normalize=None,risk_aversion:float=10):
        if normalize is None:
            scale = series_mean.shape[0]
        else:
            scale = normalize
        obj=series_mean*risk_aversion-series_cvar
        return obj.mean()*scale# annualized mean= day's average return * 252

    
    def _compute_stdDev(self, series: pd.Series, normalize=None):
        if normalize is None:
            scale = series.shape[0]
        else:
            scale = normalize
        return series.mean() * np.sqrt(scale)# annualized variance= variance for day * 252
    def _compute_CVaR(self, series: pd.Series, normalize=None):
        if normalize is None:
            scale = series.shape[0]
        else:
            scale = normalize
        return series.mean() * scale# annualized variance= variance for day * 252



    def _sim_ana_kernel(
        self,
        analysis_input_df:pd.DataFrame,
        normalize:Optional[float]=None,
        CVaR_q:float=0.05,
        risk_aversion:float=10,
        util_inverse_ra:float=0.1,
        tc_scale:float=0.0001,
    ) -> pd.Series:
        mean_pnl = self._compute_mean(analysis_input_df.pnl_series, normalize=normalize)
        stdDev = self._compute_stdDev(analysis_input_df.std_series, normalize=normalize)
        gSharpe = self._compute_sharp(analysis_input_df.pnl_series,analysis_input_df.std_series,normalize=normalize)
        obj=self._compute_obj(analysis_input_df.pnl_series,analysis_input_df.std_series,normalize=normalize,risk_aversion=risk_aversion)
        CVaR = self._compute_CVaR(analysis_input_df.cvar_series, q=CVaR_q, normalize=normalize)
        maxDrawDown = analysis_input_df.drawdown_series.max()
        traded_mv = analysis_input_df.traded_mv_series.mean()
        days = analysis_input_df.shape[0] if normalize is None else normalize
        res_wrap_series = pd.Series({
            'grossPnl':mean_pnl,
            'obj': obj,
            'stdDev': stdDev,
            'CVaR': CVaR,
            'gSharpe': gSharpe,
            'maxDraw': maxDrawDown,
            'tradedMv': traded_mv,
            'days': days,
        })
        return res_wrap_series

    def analyze(
        self,
        index_type:str = 'default',
        CVaR_q:float=0.05,
        risk_aversion:float=0.5,
        util_inverse_ra:float=0.1,
        tc_scale: float = 0.0001,
    ):
        index_type_list = ['default', 'annual_only']
        if index_type not in index_type_list:
            raise ValueError(f'Unknown index_type {index_type}, please set index type from '
                             f'{index_type_list}.')
        if index_type == 'default':
            result_df = pd.concat([
                self.analysis_input_df.groupby(self.analysis_input_df.index.year).apply(
                    self._sim_ana_kernel,
                    CVaR_q=CVaR_q,
                    risk_aversion=risk_aversion,
                    util_inverse_ra=util_inverse_ra,
                    tc_scale=tc_scale,
                ),
                self._sim_ana_kernel(
                    self.analysis_input_df,
                    normalize=252,
                    CVaR_q=CVaR_q,
                    risk_aversion=risk_aversion,
                    util_inverse_ra=util_inverse_ra,
                    tc_scale=tc_scale,
                ).to_frame(name='ANNUAL').transpose(),
            ])
        if index_type == 'annual_only':
            result_df = self._sim_ana_kernel(
                self.analysis_input_df,
                normalize=252,
                CVaR_q=CVaR_q,
                risk_aversion=risk_aversion,
                util_inverse_ra=util_inverse_ra,
                tc_scale=tc_scale,
            ).to_frame(name='ANNUAL').transpose()
        result_df.index.name = 'time'
        return result_df



