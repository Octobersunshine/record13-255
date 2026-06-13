import numpy as np
from scipy.integrate import simpson
from typing import Callable, Union, Tuple, Optional


_SYMPY_AVAILABLE = False
try:
    import sympy as _sp
    _SYMPY_AVAILABLE = True
except ImportError:
    pass


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
        self._sympy_namespace = None
        if _SYMPY_AVAILABLE:
            self._sympy_namespace = {
                'sin': _sp.sin,
                'cos': _sp.cos,
                'tan': _sp.tan,
                'exp': _sp.exp,
                'log': _sp.log,
                'sqrt': _sp.sqrt,
                'abs': _sp.Abs,
                'pi': _sp.pi,
                'e': _sp.E,
                'sinh': _sp.sinh,
                'cosh': _sp.cosh,
                'tanh': _sp.tanh,
                'arcsin': _sp.asin,
                'arccos': _sp.acos,
                'arctan': _sp.atan,
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

    def _numeric_diff_7point(
        self,
        f: Callable[[float], float],
        x0: float,
        h: float,
        order: int = 1
    ) -> float:
        if order == 1:
            return (f(x0 + 3 * h) - 9 * f(x0 + 2 * h) + 45 * f(x0 + h) - 45 * f(x0 - h) + 9 * f(x0 - 2 * h) - f(x0 - 3 * h)) / (60 * h)
        elif order == 2:
            return (2 * f(x0 + 3 * h) - 27 * f(x0 + 2 * h) + 270 * f(x0 + h) - 490 * f(x0) + 270 * f(x0 - h) - 27 * f(x0 - 2 * h) + 2 * f(x0 - 3 * h)) / (180 * h ** 2)
        elif order == 3:
            return (-f(x0 + 3 * h) + 8 * f(x0 + 2 * h) - 13 * f(x0 + h) + 13 * f(x0 - h) - 8 * f(x0 - 2 * h) + f(x0 - 3 * h)) / (8 * h ** 3)
        elif order == 4:
            return (-f(x0 + 3 * h) + 12 * f(x0 + 2 * h) - 39 * f(x0 + h) + 56 * f(x0) - 39 * f(x0 - h) + 12 * f(x0 - 2 * h) - f(x0 - 3 * h)) / (6 * h ** 4)
        else:
            raise ValueError("order must be 1, 2, 3, or 4")

    def _richardson_extrapolate(self, series, hs, order):
        if len(series) < 2:
            return series[-1]
        p = 6.0
        r = hs[-2] / hs[-1]
        extrap = (r ** p * series[-1] - series[-2]) / (r ** p - 1.0)
        return extrap

    def _adaptive_numeric_diff(
        self,
        f: Callable[[float], float],
        x0: float,
        order: int,
        tol: float,
        max_iter: int
    ) -> Tuple[float, float, int, bool]:
        eps = np.finfo(float).eps
        x_scale = max(1.0, abs(x0))
        h = 0.05 * x_scale

        results = []
        hs = []
        err_ests = []

        for i in range(max_iter):
            current = self._numeric_diff_7point(f, x0, h, order)
            results.append(current)
            hs.append(h)

            if len(results) >= 2:
                err_est = abs(results[-1] - results[-2])
                err_ests.append(err_est)

                if err_est < tol and len(results) >= 3:
                    prev_err = err_ests[-2]
                    if err_est < prev_err:
                        return current, err_est, i + 1, True

                if len(err_ests) >= 4:
                    if err_ests[-1] > err_ests[-2] > err_ests[-3] > err_ests[-4]:
                        break
            h = h / 2.0

        if len(results) >= 3:
            best_idx = int(np.argmin(err_ests))
            best_idx = min(best_idx + 1, len(results) - 1)

            if best_idx >= 2:
                r = hs[best_idx - 1] / hs[best_idx]
                p = 6.0
                extrap = (r ** p * results[best_idx] - results[best_idx - 1]) / (r ** p - 1.0)
                err = min(err_ests[best_idx], abs(extrap - results[best_idx]))
                return extrap, err, len(results), err < tol * 100

            best_result = results[best_idx]
            best_err = err_ests[best_idx]
            return best_result, best_err, len(results), best_err < tol * 100

        if len(results) >= 2:
            return results[-1], abs(results[-1] - results[-2]), len(results), False

        return results[-1], float('inf'), len(results), False

    def _symbolic_diff(
        self,
        expr: str,
        x0: float,
        order: int
    ) -> Optional[float]:
        if not _SYMPY_AVAILABLE or self._sympy_namespace is None:
            return None
        try:
            x = _sp.Symbol('x', real=True)
            namespace = self._sympy_namespace.copy()
            namespace['x'] = x
            sym_expr = eval(expr, {'__builtins__': {}}, namespace)
            deriv = sym_expr
            for _ in range(order):
                deriv = _sp.diff(deriv, x)
            deriv_func = _sp.lambdify(x, deriv, modules=['numpy'])
            return float(deriv_func(x0))
        except Exception:
            return None

    def differentiate(
        self,
        func_expr: Union[str, Callable[[float], float]],
        x0: float,
        order: int = 1,
        method: str = 'auto',
        tol: float = 1e-10,
        max_iter: int = 20
    ) -> Tuple[float, dict]:
        if not isinstance(order, int) or order < 1 or order > 4:
            raise ValueError("order must be an integer between 1 and 4")

        if method not in ('auto', 'symbolic', 'numeric'):
            raise ValueError("method must be 'auto', 'symbolic', or 'numeric'")

        use_symbolic = False
        sym_result = None

        if method in ('auto', 'symbolic') and isinstance(func_expr, str):
            sym_result = self._symbolic_diff(func_expr, x0, order)
            if sym_result is not None:
                use_symbolic = True
                if method == 'symbolic':
                    return sym_result, {
                        'method': 'Symbolic (sympy)',
                        'point': x0,
                        'order': order,
                        'exact': True,
                        'function': func_expr,
                    }

        if isinstance(func_expr, str):
            f = self._parse_function(func_expr)
        elif callable(func_expr):
            f = np.vectorize(func_expr)
        else:
            raise TypeError("func_expr must be a string expression or a callable function")

        num_result, err, iters, converged = self._adaptive_numeric_diff(f, x0, order, tol, max_iter)

        method_used = 'Numeric (7-point central, adaptive step, Richardson extrapolation)'
        if use_symbolic:
            method_used = 'Symbolic (sympy)'
            final_result = sym_result
        else:
            final_result = num_result

        info = {
            'method': method_used,
            'point': x0,
            'order': order,
            'exact': use_symbolic,
            'converged': converged if not use_symbolic else True,
            'error_estimate': err if not use_symbolic else 0.0,
            'iterations': iters if not use_symbolic else 0,
            'function': func_expr if isinstance(func_expr, str) else 'callable',
        }
        if use_symbolic and not method == 'symbolic':
            info['numeric_result'] = num_result

        return float(final_result), info


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


def compute_derivative(
    func_expr: Union[str, Callable[[float], float]],
    x0: float,
    order: int = 1,
    **kwargs
) -> Tuple[float, dict]:
    service = IntegralService()
    return service.differentiate(func_expr, x0, order=order, **kwargs)
