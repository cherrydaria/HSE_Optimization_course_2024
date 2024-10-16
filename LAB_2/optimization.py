import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from datetime import datetime


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    def update_history(x_k, d_k_norm, dt, history):
        history['residual_norm'].append(d_k_norm)
        history['time'].append((datetime.now() - dt).total_seconds())
        if x_k.size <= 2:
            history['x'].append(x_k)
        return history

    history = defaultdict(list) if trace else None

    if max_iter is None:
        max_iter = x_0.size

    stop = tolerance * np.linalg.norm(b)

    x_k, a_k = np.copy(x_0), None
    g_k = matvec(x_k) - b
    d_k = -g_k

    dt = datetime.now()

    for step in range(max_iter + 1):
        g_k_norm = np.linalg.norm(g_k)

        if trace:
            history = update_history(x_k, g_k_norm, dt, history)

        if g_k_norm <= stop:
            return x_k, 'success', history

        if step == max_iter:
            break

        A_d_k = matvec(d_k)
        g_k_squared = g_k @ g_k
        a_k = g_k_squared / (d_k @ A_d_k)

        x_k += a_k * d_k
        g_k += a_k * A_d_k

        b_k = (g_k @ g_k) / g_k_squared
        d_k = -g_k + b_k * d_k

    return x_k, 'iterations exceeded', history


def update_history(oracle, x_k, d_k_norm, dt, history):
    history['func'].append(oracle.func(x_k))
    history['grad_norm'].append(d_k_norm)
    history['time'].append((datetime.now() - dt).total_seconds())
    if x_k.size <= 2:
        history['x'].append(x_k)
    return history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.

    def find_direction(grad, s_history, y_history):
        if len(s_history) == 0:
            return -grad

        coefficients = []
        q = -grad

        # Forward pass from q_k to q_k_m:
        for s, y in zip(reversed(s_history), reversed(y_history)):
            c = (s @ q) / (y @ s)
            coefficients.append(c)
            q -= c * y

        # From q_k_m to r_k_m
        r = q * (s_history[-1] @ y_history[-1]) / (y_history[-1] @ y_history[-1])

        # Backward pass from r_k_m to r_k
        for s, y, c in zip(s_history, y_history, reversed(coefficients)):
            b = (y @ r) / (s @ y)
            r += (c - b) * s

        return r

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    a_k = None
    s_history, y_history = deque(), deque()

    dt = datetime.now()

    for step in range(max_iter + 1):
        g_k_norm = np.linalg.norm(g_k)
        g_k_prev = np.copy(g_k)

        if trace:
            history = update_history(oracle, x_k, g_k_norm, dt, history)

        if step == 0:
            stop = tolerance * np.linalg.norm(g_k_norm) ** 2

        if g_k_norm ** 2 <= stop:
            return x_k, 'success', history

        if step == max_iter:
            break

        d_k = find_direction(g_k, s_history, y_history)
        a_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=2 * a_k if a_k else None)
        x_k += a_k * d_k
        g_k = oracle.grad(x_k)

        if memory_size > 0:
            if (len(y_history) == memory_size):
                s_history.popleft()
                y_history.popleft()

            s_history.append(a_k * d_k)
            y_history.append(g_k - g_k_prev)

    return x_k, 'iterations exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    dt = datetime.now()

    for step in range(max_iter + 1):
        g_k = oracle.grad(x_k)
        g_k_norm = np.linalg.norm(g_k)

        if trace:
            history = update_history(oracle, x_k, g_k_norm, dt, history)

        if step == 0:
            stop = tolerance * np.linalg.norm(g_k_norm) ** 2

        if g_k_norm ** 2 <= stop:
            return x_k, 'success', history

        if step == max_iter:
            break

        eta_k = min(0.5, np.sqrt(g_k_norm))
        conjugate_flaq = False

        while not conjugate_flaq:
            d_k, _, _ = conjugate_gradients(lambda d: oracle.hess_vec(x_k, d), -g_k, -g_k, eta_k)
            eta_k /= 10
            conjugate_flaq = (g_k @ d_k < 0)

        a_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        x_k += a_k * d_k

    return x_k, 'iterations exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k, alpha = np.copy(x_0), None
    dt = datetime.now()
    stop = tolerance * np.linalg.norm(oracle.grad(x_k)) ** 2

    for _ in range(max_iter):
        d_k = -oracle.grad(x_k)
        d_k_norm = np.linalg.norm(-d_k)
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=2 * alpha if alpha else None)
        history = update_history(oracle, x_k, d_k_norm, dt, history)

        if d_k_norm ** 2 <= stop:
            return x_k, 'success', history
        x_k = x_k + alpha * d_k

    d_k_norm = np.linalg.norm(oracle.grad(x_k))
    history = update_history(oracle, x_k, d_k_norm, dt, history)

    if d_k_norm ** 2 > stop:
        return x_k, 'iterations_exceeded', history

    return x_k, 'success', history
