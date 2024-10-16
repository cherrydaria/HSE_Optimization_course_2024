from oracles import *
import numpy as np
import scipy


def gen_dataset(min_val=0, max_val=1000, m=50, n=10, sparse=False):
    if sparse:
        return scipy.sparse.uniform(min_val, max_val, (m, n)), np.random.choice([-1, 1], size=m)
    return np.random.uniform(min_val, max_val, (m, n)), np.random.choice([-1, 1], size=m)

def check():
    A, b = gen_dataset(n=10)
    oracle = create_log_reg_oracle(A, b, 0.1, oracle_type='usual')
    x = np.random.uniform(-100, 100, 10)
    v = np.random.uniform(-100, 100, 10)
    hess_vec_oracle = oracle.hess_vec(x, v)
    hess_vec_finite = hess_vec_finite_diff(oracle.func, x, v)
    diff = np.abs(hess_vec_oracle - hess_vec_finite)
    #print(f'hess_vec_oracle: {hess_vec_oracle}\nfinite_hess: {hess_vec_finite}\ndiff: {np.abs(hess_vec_oracle - hess_vec_finite)}')
    return diff

for i in range(1000):
    if max(check()) > 0.8:
        print("Big difference, maybe bad implementation")


