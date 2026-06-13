import numpy as np
from scipy.integrate import simpson
from typing import Callable, Union, Tuple


class IntegralService:
    def __init__(self, num_points: int = 10000):
        self.num_points = num_points
        self._safe_namespace = {
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
        }

    def _parse_function(self, expr: str) -> Callable[[np.ndarray], np.ndarray]:
        def f(x: np.ndarray) -> np.ndarray:
            namespace = self._safe_namespace.copy()
            namespace['x'] = x
            return eval(expr, {'__builtins__': {}}, namespace)
        return f

    def integrate(
        self,
        func_expr: Union[str, Callable[[float], float]],
        a: float,
        b: float,
        num_points: int = None
    ) -> Tuple[float, dict]:
        if num_points is None:
            num_points = self.num_points

        if num_points % 2 == 0:
            num_points += 1

        x = np.linspace(a, b, num_points)

        if isinstance(func_expr, str):
            f = self._parse_function(func_expr)
            y = f(x)
        elif callable(func_expr):
            y = np.vectorize(func_expr)(x)
        else:
            raise TypeError("func_expr must be a string expression or a callable function")

        dx = x[1] - x[0]
        result = simpson(y=y, dx=dx)

        info = {
            'method': 'Simpson',
            'interval': [a, b],
            'num_points': num_points,
            'step_size': dx,
            'function': func_expr if isinstance(func_expr, str) else 'callable',
        }

        return float(result), info

    def integrate_adaptive(
        self,
        func_expr: Union[str, Callable[[float], float]],
        a: float,
        b: float,
        tol: float = 1e-8,
        max_iter: int = 20
    ) -> Tuple[float, dict]:
        old_result, _ = self.integrate(func_expr, a, b, num_points=3)
        num_points = 3

        for i in range(max_iter):
            num_points = num_points * 2 - 1
            new_result, info = self.integrate(func_expr, a, b, num_points=num_points)
            error = abs(new_result - old_result)

            if error < tol:
                info['tolerance'] = tol
                info['actual_error'] = error
                info['iterations'] = i + 1
                info['converged'] = True
                return float(new_result), info

            old_result = new_result

        info['tolerance'] = tol
        info['actual_error'] = abs(new_result - old_result)
        info['iterations'] = max_iter
        info['converged'] = False
        return float(new_result), info


def compute_integral(
    func_expr: str,
    a: float,
    b: float,
    adaptive: bool = False,
    **kwargs
) -> Tuple[float, dict]:
    service = IntegralService()
    if adaptive:
        return service.integrate_adaptive(func_expr, a, b, **kwargs)
    return service.integrate(func_expr, a, b, **kwargs)
