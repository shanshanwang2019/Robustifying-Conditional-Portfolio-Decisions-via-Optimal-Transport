import pandas as pd
import logging

logger = logging.getLogger(__name__)

from .equal_weight import equal_weight_kernel
from .mean_variance import (
    mean_variance_kernel,
    DR_mean_variance_kernel,
    cond_mean_variance_kernel,
    DR_W2_conditional_mean_variance_kernel,
    DR_Winf_conditional_mean_variance_kernel,
)

from .CVaR import (
    mean_CVaR_kernel,
    cond_mean_CVaR_kernel,
    DR_mean_CVaR_kernel,
    DR_W2_conditional_mean_CVaR_kernel,
    DR_Winf_conditional_mean_CVaR_kernel,
    DR_trim_conditional_mean_CVaR_kernel,
)

from ..simulations.data_type import SolverInputDataWrapper

def portfolio_solver_wrapper(
    solver_input_data: SolverInputDataWrapper,
    current_money_position_series: pd.Series,
    portfolio_solver_type: str,
    **portfolio_solver_kwargs
) -> pd.Series:
    X_mat = solver_input_data.historical_factors_df.values
    Y_mat = solver_input_data.historical_returns_df.values
    X0 = solver_input_data.current_factor_series.values
    is_tradable = solver_input_data.is_tradable_series.values
    current_money_position = current_money_position_series.values

    solver_kernel_dict = {
        'equal_weight': equal_weight_kernel,
        'mean_variance': mean_variance_kernel,
        'mean_cvar': mean_CVaR_kernel,
        'cond_mean_variance': cond_mean_variance_kernel,
        'cond_mean_cvar': cond_mean_CVaR_kernel,
        'dr_mean_cvar': DR_mean_CVaR_kernel,
        'dr_mean_variance': DR_mean_variance_kernel,
        'dr_w2_cond_mean_cvar': DR_W2_conditional_mean_CVaR_kernel,
        'dr_w2_cond_mean_variance': DR_W2_conditional_mean_variance_kernel,
        'dr_winf_cond_mean_cvar': DR_Winf_conditional_mean_CVaR_kernel,
        'dr_winf_cond_mean_variance': DR_Winf_conditional_mean_variance_kernel,
        'dr_trim_cond_mean_cvar': DR_trim_conditional_mean_CVaR_kernel,
    }

    if portfolio_solver_type not in solver_kernel_dict:
        raise NotImplementedError(f'Unknown Portfolio Solver Type: {portfolio_solver_type}')

    try:
        goal_money_position = solver_kernel_dict[portfolio_solver_type](
            X_mat, Y_mat, X0, is_tradable, current_money_position, **portfolio_solver_kwargs
        )

    except:
        logger.error(f'Solver {portfolio_solver_type} fails, try equal weight.')
        goal_money_position = equal_weight_kernel(
            X_mat, Y_mat, X0, is_tradable, current_money_position
        )

    goal_money_position_series = pd.Series(goal_money_position, index=current_money_position_series.index)
    logger.info(f'goal_money_position_series = {goal_money_position_series}')
    return goal_money_position_series

