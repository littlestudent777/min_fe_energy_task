import anisotropy_optimize as my

param_combinations = [
    # (K1, K2)     (expected result)
    (4.2, 1.5),     # (100)
    (1.0, -4.5),    # (100)
    (1.0, -10.0),   # (111)
    (-1.0, -10.0),  # (111)
    (-1.0, 4.5),    # (110)
    (-1.0, 10.0)    # (110)
]

for K1, K2 in param_combinations:
    print(f"\n>>> Optimization results for K1 = {K1}, K2 = {K2} <<<")

    optimizer = my.AnisotropyOptimizer(K1, K2)
    result = optimizer.optimize()

    print(f"  f(x) = {result['function_value']:.6f}")
    print(f"  Solution: {[f'{x:.3f}' for x in result['solution']]}")
    print(f"  Iterations: {result['nit']}")
    print()
