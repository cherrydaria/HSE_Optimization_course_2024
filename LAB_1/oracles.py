import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        add = np.logaddexp(0, -self.b * self.matvec_Ax(x))
        return np.linalg.norm(add, 1) / self.b.size + (self.regcoef / 2) * np.linalg.norm(x) ** 2

    def grad(self, x):
        sigmoid = scipy.special.expit(-self.b * self.matvec_Ax(x))
        return self.regcoef * x - self.matvec_ATx(self.b * sigmoid) / self.b.size

    def hess(self, x):
        tmp = scipy.special.expit(self.b * self.matvec_Ax(x))
        return self.regcoef * np.eye(x.size) + self.matmat_ATsA(tmp * (1 - tmp)) / self.b.size


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x

    if scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T @ x
        matmat_ATsA = lambda x: matvec_ATx(matvec_ATx(scipy.sparse.diags(x)).T)
    else:
        matmat_ATsA = lambda x: (A.T @ np.diag(x)) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    n = x.size
    result, e = np.zeros(n), np.eye(n)
    f_x = func(x)

    for i in range(n):
        result[i] = (func(x + eps * e[i, :]) - f_x) / eps

    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    n = x.size
    result, e = np.zeros((n, n)), np.eye(n)
    f_x = func(x)
    e_2 = eps ** 2

    tmp = np.zeros(n)
    for i in range(n):
        tmp[i] = func(x + eps * e[i])

    for i in range(n):
        for j in range(n):
            result[i][j] = (func(x + eps * e[i] + eps * e[j])
                            - tmp[i]
                            - tmp[j]
                            + f_x) / e_2
    return result


def check_grad(m, n):
    def gen_dataset(m, n):
        return np.random.rand(m, n), np.random.choice([-1, 1], size=m)

    A, y = gen_dataset(m, n)
    oracle = create_log_reg_oracle(A, y, 0.1, oracle_type='usual')
    x = np.random.randn(n)
    log_grad = oracle.grad(x)
    finite_grad = grad_finite_diff(oracle.func, x)
    print(f'log_grad: {log_grad}\nfinite_grad: {finite_grad}\ndiff: {np.abs(log_grad - finite_grad)}')

def check_hess(m, n, sparse=False):
    def gen_dataset(m, n, sparse=False):
        if sparse:
            return scipy.sparse.random(m, n), np.random.choice([-1, 1], size=m)
        return np.random.rand(m, n), np.random.choice([-1, 1], size=m)

    A, y = gen_dataset(m, n, sparse)
    oracle = create_log_reg_oracle(A, y, 0.1, oracle_type='usual')
    x = np.random.randn(n)
    log_hess = oracle.hess(x)
    finite_hess = hess_finite_diff(oracle.func, x)
    print(f'log_hess: {log_hess}\nfinite_hess: {finite_hess}\ndiff: {np.abs(log_hess - finite_hess)}')
