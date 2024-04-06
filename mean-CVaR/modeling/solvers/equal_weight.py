import numpy as np
from typing import Optional, List, Dict

def equal_weight_kernel(
    X_mat:np.array, Y_mat:np.array, X0:np.array, is_tradable:np.array, current_money_position:np.array,reg_params: Optional[float]=None,
)->np.array:
    goal_money_position = current_money_position.copy()
    goal_money_position[is_tradable] = goal_money_position[is_tradable].mean()
    return goal_money_position
