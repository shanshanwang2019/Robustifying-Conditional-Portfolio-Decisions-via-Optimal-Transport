import cvxpy as cp
import numpy as np
import mosek

def mean_variance_kernel(
    X_mat:np.array,
    Y_mat:np.array,
    X0:np.array,
    is_tradable:np.array,
    current_money_position:np.array,
    reg_params:float
)->np.array:
    '''mean variance optimization for long only portfolio
    
    beta: goal money position
    model:
    min beta^T Cov beta - reg * beta^T mean
    s.t. beta >= 0; sum(beta) = 1;
         beta[is_not_tradable] = current_money_position[is_not_tradable]
         reg_params is the inverse of risk aversion
    '''
    dim_beta = Y_mat.shape[1]
    beta = cp.Variable(shape=(dim_beta,), name='beta', nonneg=True)
    mean_arr = Y_mat.mean(axis = 0)
    cov_arr = np.cov(Y_mat, rowvar=False)
    obj = cp.Minimize(cp.quad_form(beta, cov_arr) - reg_params*mean_arr@beta)
    constraints = [
        cp.sum(beta) == cp.sum(current_money_position)
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(obj, constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

def DR_mean_variance_kernel(
        X_mat: np.array,
        Y_mat: np.array,
        X0: np.array,
        is_tradable: np.array,
        current_money_position: np.array,
        reg_params:float,
        rho:float,
        p:int = 2
):
    '''CVXPY solver kernel for distributionally robust optimization problem:

    min_{beta} sqrt{beta^T Cov_{mathbb{P}_0}(R) beta} 
    - lambda beta^toprm{E}_{mathbb{P}_0}(R)
    + sqrt{1+lambda^2}cdot sqrt{rho} |beta|_p
    '''
    def ReLU(np_vec):
        res = np.zeros_like(np_vec)
        res[np_vec > 0] = np_vec[np_vec > 0]
        return res

    mean_t = np.mean(Y_mat, axis=0)
    cov_t = np.cov(Y_mat, rowvar=False)
    sample_stock_num = mean_t.shape[0]
    eigval, eigvecs = np.linalg.eig(cov_t)
    F = np.diag(np.sqrt(ReLU(eigval))) @ eigvecs.T
    beta = cp.Variable(sample_stock_num, name = 'beta', nonneg=True)
    objective_expression = (cp.norm2(cp.matmul(F, beta))
                            - reg_params * cp.matmul(mean_t, beta)
                            + np.sqrt(rho * (1 + reg_params ** 2)) * cp.norm(beta, p=p))
    objective = cp.Minimize(objective_expression)
    constraints = [cp.sum(beta) == cp.sum(current_money_position)]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return beta.value

def cond_mean_variance_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    neighbor_quantile: float,
)->np.array:
    """
    CVXPY solver kernel for conditional mean-variance optimization
    """
    X_dist = np.linalg.norm(X_mat-X0, axis = 1)
    idx = (X_dist <= np.quantile(X_dist, neighbor_quantile))
    dim_beta = Y_mat.shape[1]
    beta = cp.Variable(shape = (dim_beta,), name = 'beta', nonneg=True)

    mean_arr = Y_mat[idx,:].mean(axis=0)
    cov_arr = np.cov(Y_mat[idx,:], rowvar=False)
    obj = cp.Minimize(cp.quad_form(beta, cov_arr) - reg_params * mean_arr @ beta)

    constraints = [
        cp.sum(beta) == cp.sum(current_money_position),
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(obj, constraints)
    problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    #print(beta.value)
    return beta.value


def DR_W2_conditional_mean_variance_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    epsilon: float,
    rho_div_rho_min: float,
):
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:

    See problem formulation in DR_Conditional_EstimationW2.ipynb
    """

    def compute_rho_min(X_mat, X0, epsilon):
        X_dist = np.linalg.norm(X_mat - X0, axis=1)
        X_dist[np.isnan(X_dist)] = 1e8
        X_cut = np.quantile(X_dist, q=epsilon, interpolation='higher')
        return X_dist[X_dist <= X_cut].mean() * epsilon

    rho = rho_div_rho_min * compute_rho_min(X_mat, X0, epsilon)
    X_dist = np.linalg.norm(X_mat - X0, axis=1)
    X_dist[np.isnan(X_dist)] = 1e8

    eta = reg_params
    epsilon_inv = 1 / epsilon;

    N, sample_stock_num = Y_mat.shape

    m = cp.Variable(N, nonneg=True)
    beta = cp.Variable(sample_stock_num, nonneg=True)
    alpha = cp.Variable(1)
    lambda1 = cp.Variable(1, nonneg=True)
    denom = cp.Variable(1, nonneg=True)
    lambda2 = cp.Variable(1)
    linear_expr = Y_mat @ beta - alpha - 0.5 * eta
    obj = lambda1 * rho + lambda2 * epsilon + cp.sum(cp.pos(epsilon_inv * m
                                                            - 0.25 * epsilon_inv * eta ** 2
                                                            - epsilon_inv * eta * alpha
                                                            - lambda1 * X_dist - lambda2)) / N
    constraints = [m >= cp.hstack([cp.quad_over_lin(linear_expr[i], denom) for i in range(N)]),
                   epsilon_inv * cp.quad_over_lin(beta, lambda1) + denom <= 1,
                   cp.sum(beta) == cp.sum(current_money_position),]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(mosek_params={"MSK_DPAR_OPTIMIZER_MAX_TIME":  100.0,"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4})
    #problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

def DR_Winf_conditional_mean_variance_kernel(
    X_mat: np.array,
    Y_mat: np.array,
    X0: np.array,
    is_tradable: np.array,
    current_money_position: np.array,
    reg_params: float,
    gamma_quantile: float,
    rho_quantile: float,
):
    """
    CVXPY solver kernel for conditional distributionally robust optimization problem:

    See problem formulation in DR_Conditional_EstimationWinfty.ipynb
    """
    X_dist = np.linalg.norm(X_mat - X0, axis=1)
    X_dist[np.isnan(X_dist)] = 1e8
    gamma = np.quantile(X_dist, gamma_quantile)
    rho = np.quantile(X_dist, rho_quantile)
    eta = reg_params
    # import warnings
    # warnings.filterwarnings("error")
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
    dim_beta = Y_mat.shape[1]
    if len(y_I) == 0:
        return np.ones(dim_beta) / dim_beta
    alpha = cp.Variable(shape=(1,), name='alpha')
    beta = cp.Variable(shape=(dim_beta,), name='beta', nonneg=True)
    lambda_ = cp.Variable(shape=(1,), name='lambda')
    u = cp.Variable(shape=(len(y_I),), name='u')
    v_exp_term_1 = cp.abs(cp.matmul(Y_mat[idx_I], beta) - alpha - 0.5 * eta)
    v_exp_term_2 = cp.norm(beta) * (rho - norm_x_minus_xp_in_I)
    constraints = [
        u[idx_I2[idx_I]] >= 0,
        cp.sum(u) <= 0,
        cp.sum(beta) == cp.sum(current_money_position),
        lambda_ + u + eta * alpha + 0.25 * eta ** 2 >= cp.square(v_exp_term_1 + v_exp_term_2)
    ]
    if (~is_tradable).sum() > 0:
        constraints.append(beta[~is_tradable] == current_money_position[~is_tradable])
    problem = cp.Problem(cp.Minimize(lambda_), constraints)
    problem.solve(mosek_params={"MSK_DPAR_OPTIMIZER_MAX_TIME":  100.0,"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4})
    #problem.solve()
    if problem.status != 'optimal':
        raise ValueError('problem is not optimal')
    return beta.value

