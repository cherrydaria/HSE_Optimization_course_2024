from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from datetime import datetime
import scipy.optimize as opt


def barrier_method_lasso(oracle, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    def backtracking(oracle, y_k, d_k, alpha, t, c1):
        gamma = 1 / 2
        while (oracle.func_directional(y_k, d_k, alpha, t) > oracle.func_directional(y_k, 0, 0, t) +
               c1 * alpha * oracle.grad_directional(y_k, d_k, 0, t)):
            alpha *= gamma
        return alpha

    def armijo_search(oracle, y_k, d_k, alpha, t, c1):
        if (oracle.func_directional(y_k, d_k, alpha, t) > oracle.func_directional(y_k, 0, 0, t) +
                c1 * alpha * oracle.grad_directional(y_k, d_k, 0, t)):
            alpha = backtracking(oracle, y_k, d_k, alpha, t, c1)
            return alpha
        else:
            return alpha

    history = defaultdict(list) if trace else None
    n = x_0.shape[0]
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    y_k = np.hstack((x_k, u_k))
    t_k = t_0

    start_time = datetime.now()
    for k in range(max_iter):
        grad0_norm = np.linalg.norm(oracle.grad(y_k, t_k)) ** 2
        for j in range(max_iter_inner):
            hess = oracle.hess(y_k, t_k)
            grad = oracle.grad(y_k, t_k)
            d_k = np.linalg.solve(hess, -grad)
            scalar_products = oracle.q_matrix.dot(d_k)
            mask = scalar_products > 0
            y_k = np.hstack((x_k, u_k))
            if mask.sum() != 0:
                alphas = -oracle.q_matrix.dot(y_k)[mask] / oracle.q_matrix.dot(d_k)[mask]
                alpha_max = np.min(alphas)
                alpha = armijo_search(oracle, y_k, d_k, min(1, 0.99 * alpha_max), t_k, c1)
            else:
                alpha = armijo_search(oracle, y_k, d_k, 1, t_k, c1)
            y_k = y_k + alpha * d_k
            x_k = y_k[:n]
            u_k = y_k[n:]
            grad_norm = np.linalg.norm(oracle.grad(y_k, t_k)) ** 2
            if grad_norm <= tolerance_inner * grad0_norm:
                break
        if grad_norm > tolerance_inner * grad0_norm:
            print(f'Newton did not converge on {k} iteration')
        gap = oracle.lasso_duality_gap(x_k)
        if trace:
            current_time = datetime.now()
            time_delta = (current_time - start_time).total_seconds()
            history['time'].append(time_delta)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(gap)
            if x_0.shape[0] <= 2:
                history['x'].append(x_k)
        if gap <= tolerance:
            break
        t_k = gamma * t_k
    if gap <= tolerance:
        return 'success', history
    else:
        return 'iterations_exceeded', history