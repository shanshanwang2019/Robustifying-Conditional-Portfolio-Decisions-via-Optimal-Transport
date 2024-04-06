import numpy as np
import cvxpy as cp

def mean_CVaR_kernel(
    X_mat:np.array,
    Y_mat:np.array,
    X0:np.array,
    is_tradable:np.array,
    current_money_position:np.array,
    reg_params:float,
    tau:float,
)->np.array:
    """
    CVXPY solver kernel for vanilla CVAR optimization
    """
    num_sample = Y_mat.shape[0]
    dim_beta = Y_mat.shape[1]
    alpha = cp.Variable(shape = (1,), name = 'alpha')
    beta = cp.Variable(shape = (dim_beta,), name = 'beta', nonneg=True)
    lambda_ = cp.Variable(shape = (num_sample,), name = 'lambda')
    constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
        lambda_ >= -reg_params*(Y_mat@beta) + alpha,
        lambda_ >= -(reg_params+1/tau)*(Y_mat@beta) + (1-1/tau)*alpha,
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(cp.sum(lambda_)), constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

def cond_mean_CVaR_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    tau: float,
    neighbor_quantile: float,
)->np.array:
    """
    CVXPY solver kernel for conditional CVAR optimization
    """
    X_dist = np.linalg.norm(X_mat-X0, axis = 1)
    idx = (X_dist <= np.quantile(X_dist, neighbor_quantile))
    dim_beta = Y_mat.shape[1]
    dim_data = np.sum(idx)
    alpha = cp.Variable(shape = (1,), name = 'alpha')
    beta = cp.Variable(shape = (dim_beta,), name = 'beta', nonneg=True)
    lambda_ = cp.Variable(shape = (dim_data,), name = 'lambda')
    constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
        lambda_ >= -reg_params*(Y_mat[idx,:]@beta) + alpha,
        lambda_ >= -(reg_params+1/tau)*(Y_mat[idx,:]@beta) + (1-1/tau)*alpha,
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(cp.sum(lambda_)), constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

def DR_mean_CVaR_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    tau: float,
    rho: float,
)->np.array:
    """
    CVXPY solver kernel for vanilla CVAR optimization
    """
    num_sample = Y_mat.shape[0]
    dim_beta = Y_mat.shape[1]
    alpha = cp.Variable(shape = (1,), name = 'alpha')
    beta = cp.Variable(shape = (dim_beta,), name = 'beta', nonneg=True)
    lambda_ = cp.Variable(shape = (1,), name = 'lambda', nonneg=True)
    inside_exp = cp.Variable(shape = (num_sample,), name = 'inside_exp')
    constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
        inside_exp >= -reg_params*(Y_mat@beta) + alpha + cp.quad_over_lin(reg_params*beta,4*lambda_),
        inside_exp >= (-(reg_params+1/tau)*(Y_mat@beta) + 
                       (1-1/tau)*alpha + cp.quad_over_lin((reg_params+1/tau)*beta,4*lambda_)),
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(lambda_*rho + cp.sum(inside_exp)/num_sample), constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value


def DR_W2_conditional_mean_CVaR_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    tau: float,
    epsilon: float,
    rho_div_rho_min: float,
) -> np.array:
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:
    """
    def compute_rho_min(X_mat, X0, epsilon):
        X_dist = np.linalg.norm(X_mat - X0, axis=1)
        X_dist[np.isnan(X_dist)] = 1e8
        X_cut = np.quantile(X_dist, q=epsilon, interpolation='higher')
        return (X_dist[X_dist <= X_cut]**2).mean() * epsilon

    rho = rho_div_rho_min * compute_rho_min(X_mat, X0, epsilon)
    X_dist = np.linalg.norm(X_mat - X0, axis=1)
    eta = reg_params
    epsilon_inv = 1 / epsilon
    tau_inv = 1 / tau

    N, stock_num = Y_mat.shape
    beta = cp.Variable(stock_num, nonneg=True)
    alpha = cp.Variable(1)
    lambda1 = cp.Variable(1, nonneg=True)
    lambda2 = cp.Variable(1)
    theta = cp.Variable(N, nonneg=True)
    z = cp.Variable(N, nonneg=True)
    z_tilde = cp.Variable(N, nonneg=True)

    obj = cp.Minimize(lambda1 * rho + lambda2 * epsilon + cp.sum(theta) / N)
    linear_constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
        z == theta + lambda1 * X_dist ** 2 + lambda2 + epsilon_inv * eta * (Y_mat @ beta - alpha),
        z_tilde == (theta + lambda1 * X_dist ** 2 + lambda2
                    + epsilon_inv * (eta + tau_inv) * (Y_mat @ beta)
                    - epsilon_inv * (1 - tau_inv) * alpha)
    ]
    if (~is_tradable).sum() > 0:
        linear_constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    quad_over_lin_constraints = [
        z >= cp.quad_over_lin(epsilon_inv * eta * beta, 4 * lambda1),
        z_tilde >= cp.quad_over_lin(epsilon_inv * (eta + tau_inv) * beta, 4 * lambda1),
    ]
    problem = cp.Problem(obj, linear_constraints + quad_over_lin_constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value


def DR_Winf_conditional_mean_CVaR_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    tau: float,
    gamma_quantile: float,
    rho_quantile: float,
) -> np.array:

    eta = reg_params
    tau_inv = 1 / tau
    X_dist = np.linalg.norm(X_mat - X0, axis=1)
    X_dist[np.isnan(X_dist)] = 1e8
    gamma = np.quantile(X_dist, gamma_quantile)
    rho = np.quantile(X_dist, rho_quantile)
    #import warnings
    #warnings.filterwarnings("error")
    try:
        idx_I = (X_dist <= gamma + rho)
        idx_I1 = (X_dist + rho <= gamma)
        idx_I2 = idx_I & (~idx_I1)
    except RuntimeWarning:
        print(X_dist)
        print(gamma)
        print(rho)
    norm_x_minus_xp_in_I = X_dist[idx_I] - gamma
    norm_x_minus_xp_in_I[norm_x_minus_xp_in_I < 0] = 0
    y_I = Y_mat[idx_I]
    # if len(y_I) == 0:
    #    dim_beta = Y_mat.shape[1]
    #    return np.ones(dim_beta) / dim_beta

    stock_num = Y_mat.shape[1]
    beta = cp.Variable(stock_num, nonneg=True)
    alpha = cp.Variable(1)
    lambda_ = cp.Variable(shape=(1,))
    u = cp.Variable(shape=(len(y_I),), name='u')
    v_term_1 = alpha - eta * (Y_mat[idx_I] @ beta) + eta * cp.norm(beta) * (rho - norm_x_minus_xp_in_I)
    v_term_2 = ((1 - tau_inv) * alpha
                - (eta + tau_inv) * (Y_mat[idx_I] @ beta)
                + (eta + tau_inv) * cp.norm(beta) * (rho - norm_x_minus_xp_in_I))
    constraints = [
        u[idx_I2[idx_I]] >= 0,
        cp.sum(u) <= 0,
        cp.sum(beta) == cp.sum(current_money_position),
        lambda_ + u >= v_term_1,
        lambda_ + u >= v_term_2
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(lambda_), constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value


def DR_trim_conditional_mean_CVaR_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    tau: float,
    rho_quantile: float,
    gamma: float,
) -> np.array:
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem with trimming:
    """
    X_dist = np.linalg.norm(X_mat - X0, axis=1)
    rho = np.quantile(X_dist, rho_quantile) ** 2
    gamma_inv = 1 / gamma
    tau_inv = 1 / tau

    N, stock_num = Y_mat.shape
    beta = cp.Variable(stock_num, nonneg=True)
    alpha = cp.Variable(1)
    lambda1 = cp.Variable(1, nonneg=True)
    lambda2 = cp.Variable(1)
    theta = cp.Variable(N, nonneg=True)

    obj = cp.Minimize(lambda1 * rho + lambda2 + gamma_inv * cp.sum(theta) / N)
    constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
        theta + lambda2 >= (-reg_params * (Y_mat @ beta) + alpha +
                            cp.quad_over_lin(reg_params * beta, 4 * lambda1) - lambda1 * (X_dist ** 2)),

        theta + lambda2 >= (-(reg_params + tau_inv) * (Y_mat @ beta) + (1 - tau_inv) * alpha +
                            cp.quad_over_lin((reg_params + tau_inv) * beta, 4 * lambda1) - lambda1 * (X_dist ** 2)),
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(obj, constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

