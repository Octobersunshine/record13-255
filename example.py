from integral_service import IntegralService, compute_integral


def main():
    service = IntegralService(num_points=10001)

    print("=" * 60)
    print("数值积分服务示例 - 辛普森法")
    print("=" * 60)

    examples = [
        ("x**2", 0, 1, "∫ x² dx, [0,1] = 1/3"),
        ("sin(x)", 0, "pi", "∫ sin(x) dx, [0,π] = 2"),
        ("exp(x)", 0, 1, "∫ eˣ dx, [0,1] = e - 1"),
        ("1/x", 1, 2, "∫ 1/x dx, [1,2] = ln(2)"),
        ("sqrt(1 - x**2)", 0, 1, "∫ √(1-x²) dx, [0,1] = π/4"),
    ]

    print("\n--- 固定点数辛普森法 (num_points=10001) ---")
    for expr, a, b, desc in examples:
        a_val = float(a)
        b_val = float(eval(str(b), {'pi': 3.141592653589793}))
        result, info = service.integrate(expr, a_val, b_val)
        print(f"\n{desc}")
        print(f"  表达式: {expr}")
        print(f"  区间: [{a}, {b}]")
        print(f"  结果: {result:.10f}")
        print(f"  步长: {info['step_size']:.6e}")

    print("\n" + "=" * 60)
    print("--- 自适应辛普森法 (tol=1e-10) ---")

    adaptive_examples = [
        ("x**2", 0, 1),
        ("sin(x)", 0, 3.141592653589793),
        ("exp(-x**2)", 0, 2),
    ]

    for expr, a, b in adaptive_examples:
        result, info = service.integrate_adaptive(expr, a, b, tol=1e-10)
        print(f"\n∫ {expr} dx, [{a}, {b}]")
        print(f"  结果: {result:.12f}")
        print(f"  收敛: {info['converged']}")
        print(f"  迭代次数: {info['iterations']}")
        print(f"  采样点数: {info['num_points']}")
        print(f"  实际误差: {info['actual_error']:.2e}")

    print("\n" + "=" * 60)
    print("--- 使用 compute_integral 快捷函数 ---")
    result, info = compute_integral("cos(x)", 0, 1.5707963267948966, adaptive=True, tol=1e-12)
    print(f"\n∫ cos(x) dx, [0, π/2] = {result:.12f}")
    print(f"  收敛: {info['converged']}, 迭代: {info['iterations']}")

    print("\n" + "=" * 60)
    print("--- 使用 Python 函数作为输入 ---")
    def f(x):
        return x**3 + 2*x**2 - 5*x + 1

    result, info = service.integrate(f, -2, 2)
    exact_value = 44/3
    print(f"\n∫ (x³ + 2x² - 5x + 1) dx, [-2, 2]")
    print(f"  数值结果: {result:.10f}")
    print(f"  精确值:   {exact_value:.10f}")
    print(f"  误差:     {abs(result - exact_value):.2e}")


if __name__ == "__main__":
    main()
