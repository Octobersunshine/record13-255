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

    def _simpson(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        fa: float = None,
        fb: float = None,
        fm: float = None
    ) -> Tuple[float, float, float, float]:
        if fa is None:
            fa = f(a)
        if fb is None:
            fb = f(b)
        m = (a + b) / 2.0
        if fm is None:
            fm = f(m)
        h = b - a
        S = (h / 6.0) * (fa + 4.0 * fm + fb)
        return S, fa, fb, fm

    def _adaptive_simpson(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        eps: float,
        S: float,
        fa: float,
        fb: float,
        fm: float,
        depth: int,
        min_depth: int,
        max_depth: int,
        counters: dict
    ) -> float:
        counters['recursions'] += 1
        m = (a + b) / 2.0
        h = b - a
        m_left = (a + m) / 2.0
        m_right = (m + b) / 2.0

        f_left = f(m_left)
        f_right = f(m_right)
        counters['fevals'] += 2

        S_left = (h / 12.0) * (fa + 4.0 * f_left + fm)
        S_right = (h / 12.0) * (fm + 4.0 * f_right + fb)
        S2 = S_left + S_right

        error = abs(S2 - S) / 15.0

        if depth >= max_depth:
            counters['intervals'] += 1
            counters['max_depth_hit'] = True
            return S2 + (S2 - S) / 15.0

        if depth >= min_depth and error <= eps:
            counters['intervals'] += 1
            return S2 + (S2 - S) / 15.0

        return (
            self._adaptive_simpson(f, a, m, eps / 2.0, S_left, fa, fm, f_left, depth + 1, min_depth, max_depth, counters)
            + self._adaptive_simpson(f, m, b, eps / 2.0, S_right, fm, fb, f_right, depth + 1, min_depth, max_depth, counters)
        )

    def integrate_adaptive(
        self,
        func_expr: Union[str, Callable[[float], float]],
        a: float,
        b: float,
        tol: float = 1e-8,
        min_depth: int = 6,
        max_depth: int = 50
    ) -> Tuple[float, dict]:
        if isinstance(func_expr, str):
            f = self._parse_function(func_expr)
        elif callable(func_expr):
            f = np.vectorize(func_expr)
        else:
            raise TypeError("func_expr must be a string expression or a callable function")

        counters = {'fevals': 3, 'recursions': 0, 'intervals': 0, 'max_depth_hit': False}

        S, fa, fb, fm = self._simpson(f, a, b)

        result = self._adaptive_simpson(f, a, b, tol, S, fa, fb, fm, 0, min_depth, max_depth, counters)

        num_points = counters['fevals']
        converged = not counters['max_depth_hit']

        info = {
            'method': 'Adaptive Simpson (recursive)',
            'interval': [a, b],
            'num_points': num_points,
            'tolerance': tol,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'recursions': counters['recursions'],
            'intervals': counters['intervals'],
            'converged': converged,
            'function': func_expr if isinstance(func_expr, str) else 'callable',
        }

        return float(result), info


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
