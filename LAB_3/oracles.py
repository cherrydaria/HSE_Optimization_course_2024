import numpy as np
import scipy
from scipy.special import expit


class LogRegOracle:
    def __init__(self, A: np.ndarray, b: np.ndarray, regcoef: float):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self.Ax_b = lambda x: self.A.dot(x) - self.b
        self.ATAx_b = lambda x: self.A.T.dot(self.A.dot(x) - self.b)

        self.n = A.shape[1]
        self.q_matrix = np.zeros((2 * self.n, 2 * self.n))
        for i in range(self.n):
            self.q_matrix[2 * i, i] = 1
            self.q_matrix[2 * i + 1, i] = -1
            self.q_matrix[2 * i, i + self.n] = -1
            self.q_matrix[2 * i + 1, i + self.n] = -1

    def func(self, x: np.ndarray) -> float:
        """
        Compute the logistic regression function
        :param x: x vector
        :return:
        """
        return 1 / 2 * np.linalg.norm(self.Ax_b(x)) ** 2 + self.regcoef * np.linalg.norm(x, ord=1)

    def mod_func(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the modified logistic regression function in the form of:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
        where:
        f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        :param y: (x, u) concatenation
        :param t: t parameter
        :return:
        """
        x = y[:self.n]
        u = y[self.n:]
        f = 1 / 2 * np.linalg.norm(self.Ax_b(x)) ** 2 + self.regcoef * np.sum(u)
        phi = t * f - np.sum(np.log(u + x) + np.log(u - x))

        return phi

    def func_directional(self, y, d, alpha, t):
        """
        Computes phi(alpha) = f(y + alpha*d).
        """
        return np.squeeze(self.mod_func(y + alpha * d, t))

    def grad_directional(self, y, d, alpha, t):
        """
        Computes phi'(alpha) = (f(y + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(y + alpha * d, t).dot(d))

    def grad(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the gradient of the modified logistic regression model in the form of:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
        where:
        f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        :param y: (x, u) concatenation
        :param t: t parameter
        :return: gradient of the modified logistic regression model
        """
        x = y[:self.n]
        u = y[self.n:]
        grad_x = t * self.ATAx_b(x) - (1 / (u + x) - 1 / (u - x))
        grad_u = self.regcoef * t * np.ones(len(x)) - (1 / (u + x) + 1 / (u - x))
        total_grad = np.hstack((grad_x, grad_u))

        return total_grad

    def hess(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the hessian of the modified logistic regression model in the form of:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
        :param y: (x, u) concatenation
        :param t: t parameter
        :return: hessian of the modified logistic regression model
        """
        x = y[:self.n]
        u = y[self.n:]
        hess_xx = t * self.A.T @ self.A + np.diag(1 / (u + x) ** 2 + 1 / (u - x) ** 2)
        hess_uu = np.diag(1 / (u + x) ** 2 + 1 / (u - x) ** 2)
        hess_ux = np.diag(1 / (u + x) ** 2 - 1 / (u - x) ** 2)
        top_rows = np.hstack((hess_xx, hess_ux))
        bottom_rows = np.hstack((hess_ux, hess_uu))
        total_hess = np.vstack((top_rows, bottom_rows))

        return total_hess

    def lasso_duality_gap(self, x: np.ndarray) -> float:
        """
        Estimates f(x) - f* via duality gap for
            f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        """
        mu = min(1, self.regcoef / np.linalg.norm(self.ATAx_b(x), ord=np.inf)) * self.Ax_b(x)
        gap = (1 / 2 * np.linalg.norm(self.Ax_b(x)) ** 2 + self.regcoef * np.linalg.norm(x, ord=1)
               + 1 / 2 * np.linalg.norm(mu) ** 2 + self.b.dot(mu))

        return gap