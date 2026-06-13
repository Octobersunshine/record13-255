from integral_service import IntegralService, compute_integral
import math


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
    print("--- 递归自适应辛普森法 (tol=1e-10) ---")

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
        print(f"  递归次数: {info['recursions']}")
        print(f"  采样点数: {info['num_points']}")
        print(f"  区间数: {info['intervals']}")

    print("\n" + "=" * 60)
    print("--- 震荡/尖峰函数对比：固定步长 vs 自适应 ---")

    oscillatory_examples = [
        {
            "expr": "exp(-x**2 / 0.001)",
            "a": -10,
            "b": 10,
            "desc": "尖峰函数 exp(-x²/0.001)，区间 [-10, 10]",
            "ref_pts": 50001,
        },
        {
            "expr": "x * sin(1/x)",
            "a": 0.01,
            "b": 1.0,
            "desc": "震荡函数 x·sin(1/x)，区间 [0.01, 1]",
            "ref_pts": 50001,
        },
    ]

    for ex in oscillatory_examples:
        print(f"\n{ex['desc']}")
        ref_result, _ = service.integrate(ex["expr"], ex["a"], ex["b"], num_points=ex["ref_pts"])
        print(f"  参考值 (n={ex['ref_pts']}): {ref_result:.10f}")

        print(f"  固定步长:")
        for n in [11, 101, 1001]:
            result, info = service.integrate(ex["expr"], ex["a"], ex["b"], num_points=n)
            err = abs(result - ref_result)
            print(f"    n={n:>5d}: 结果={result:.10f}, 误差={err:.2e}")

        result_adapt, info_adapt = service.integrate_adaptive(ex["expr"], ex["a"], ex["b"], tol=1e-8)
        err_adapt = abs(result_adapt - ref_result)
        print(f"  递归自适应:  结果={result_adapt:.10f}, 误差={err_adapt:.2e}, 采样点={info_adapt['num_points']}")

    print("\n" + "=" * 60)
    print("--- 使用 compute_integral 快捷函数 ---")
    result, info = compute_integral("cos(x)", 0, 1.5707963267948966, adaptive=True, tol=1e-12)
    print(f"\n∫ cos(x) dx, [0, π/2] = {result:.12f}")
    print(f"  收敛: {info['converged']}, 区间数: {info['intervals']}")

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
