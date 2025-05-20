import numpy as np
from scipy.optimize import minimize

INIT_BIG_VAL = 1e+10


class AnisotropyOptimizer:
    """Optimizer for anisotropic energy density minimization.
    Finds the equilibrium orientation of magnetization"""
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2
        self.f = self._f_without_K2 if K2 is None else self._f_with_K2

    def _f_without_K2(self, x):
        x1, x2, x3 = x
        cross_terms = x1 ** 2 * x2 ** 2 + x2 ** 2 * x3 ** 2 + x3 ** 2 * x1 ** 2
        return self.K1 * cross_terms

    def _f_with_K2(self, x):
        x1, x2, x3 = x
        cross_terms = x1 ** 2 * x2 ** 2 + x2 ** 2 * x3 ** 2 + x3 ** 2 * x1 ** 2
        cubic_term = x1 ** 2 * x2 ** 2 * x3 ** 2
        return self.K1 * cross_terms + self.K2 * cubic_term

    @staticmethod
    def constraint(x):
        """Constraint on x: the point lies on the surface of a sphere of radius 1,
            because the components of x are direction cosines"""
        return np.sum(x ** 2) - 1.0

    def optimize(self, tol=1e-6, max_iter=1000, num_trials=3):
        """Solves optimization problem with SLSQP method.
         Runs optimization [num_trials] times, then chooses the best min"""

        # Optimization parameters
        constraints = {
            'type': 'eq',
            'fun': self.constraint
        }

        options = {
            'maxiter': max_iter,
            'ftol': tol,
            'disp': False
        }

        result = {
            'solution': None,
            'function_value': None,
        }

        best_func_val = INIT_BIG_VAL  # Initialization of value that will be compared

        for trial in range(num_trials):

            # Random initial point (Function is not strictly convex, that leads to several local minimum values.
            # Choosing different initial points will lead to different results).
            x0 = np.random.sample(3)
            # Making sure that it fits constraint
            x0 = x0 / np.linalg.norm(x0)

            res = minimize(
                fun=self.f,
                x0=x0,
                method='SLSQP',
                # Added bounds, because without them SQP sometimes moves out of constraints
                bounds=[(-1.1, 1.1) for _ in range(3)],  # Bounds are a bit wider than expected just in case
                constraints=constraints,
                options=options
            )

            # Projection onto the surface of a sphere (in case method has deviated from the boundary)
            x_opt = res.x / np.linalg.norm(res.x) if self.constraint(res.x) > tol else res.x

            # Normalizing results to compare with expected ones
            x_opt_cube = x_opt / np.linalg.norm(x_opt, np.inf)

            func_val = self.f(x_opt)

            # Result will be the minimal value within trials
            if func_val < best_func_val:
                best_func_val = func_val

                result['solution'] = x_opt_cube
                result['function_value'] = self.f(x_opt)

        return result


param_combinations = [
# (K1,K2) * 1e+5 ergs/cc, (expected result)
    (4.2, None),         # (100)
    (-4.2, None),        # (111)
    (4.2, 1.5),          # (100)
    (1.0, -4.5),         # (100)
    (1.0, -10.0),        # (111)
    (-1.0, -10.0),       # (111)
    (-1.0, 4.5),         # (110)
    (-1.0, 10.0)         # (110)
]

for K1, K2 in param_combinations:
    print(f"\n>>> Optimization results for K1 = {K1}, K2 = {K2} <<<")
    optimizer = AnisotropyOptimizer(K1, K2)
    result = optimizer.optimize()

    print(f"f(x) = {result['function_value']:.6f} * 10^5 ergs/cc")
    print(f"Solution: {[f'{x:.3f}' for x in result['solution']]}")
