import torch

# -----------------------------------------------------------------------------
# 2.1 Простые вычисления с градиентами
# -----------------------------------------------------------------------------

def simple_gradients():
    """Вычисляет градиенты функции f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z."""
    print("2.1 Простые вычисления с градиентами")

    # Создаём тензоры с requires_grad=True
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # Определяем функцию
    f = x**2 + y**2 + z**2 + 2*x*y*z

    # Вычисляем градиенты
    f.backward()

    print(f"x = {x.item()}, y = {y.item()}, z = {z.item()}")
    print(f"f = {f.item():.4f}")
    print(f"df/dx = {x.grad.item():.4f}")
    print(f"df/dy = {y.grad.item():.4f}")
    print(f"df/dz = {z.grad.item():.4f}")

    # Аналитическая проверка
    analytic_dx = 2*x + 2*y*z
    analytic_dy = 2*y + 2*x*z
    analytic_dz = 2*z + 2*x*y
    print(f"Аналитически: "
          f"df/dx = {analytic_dx.item():.4f}, "
          f"df/dy = {analytic_dy.item():.4f}, "
          f"df/dz = {analytic_dz.item():.4f}")


# -----------------------------------------------------------------------------
# 2.2 Градиент функции потерь
# -----------------------------------------------------------------------------

def mse_gradients():
    """Вычисляет градиенты MSE для линейной модели y_pred = w*x + b."""
    print("2.2 Градиент функции потерь MSE")

    torch.manual_seed(42)
    n = 10
    x = torch.randn(n, 1, requires_grad=False)  # вход
    y_true = 3 * x + 5 + 0.1 * torch.randn(n, 1)  # истинные значения с шумом

    # Параметры модели
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Предсказание
    y_pred = w * x + b

    # MSE
    mse = torch.mean((y_pred - y_true) ** 2)

    # Градиенты
    mse.backward()

    print(f"w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"MSE = {mse.item():.4f}")
    print(f"Градиент по w: {w.grad.item():.4f}")
    print(f"Градиент по b: {b.grad.item():.4f}")    


# -----------------------------------------------------------------------------
# 2.3 Цепное правило
# -----------------------------------------------------------------------------

def chain_rule():
    """Вычисляет градиент составной функции f(x) = sin(x^2 + 1) двумя способами."""
    print("2.3 Цепное правило")

    x1 = torch.tensor(2.0, requires_grad=True)
    f1 = torch.sin(x1**2 + 1)
    f1.backward()
    grad_backward = x1.grad.item()
    print(f"f(x) = sin(x^2 + 1) при x = {x1.item()}")
    print(f"Градиент через backward(): {grad_backward:.4f}")

    x2 = torch.tensor(2.0, requires_grad=True)
    f2 = torch.sin(x2**2 + 1)
    grad_autograd = torch.autograd.grad(f2, x2, retain_graph=False)[0].item()  # или create_graph=False
    print(f"Градиент через torch.autograd.grad: {grad_autograd:.4f}")

    # Аналитически: df/dx = cos(x^2+1) * 2x
    x3 = torch.tensor(2.0)
    analytic = torch.cos(x3**2 + 1) * 2 * x3
    print(f"Аналитическое значение: {analytic.item():.4f}")
    print()


simple_gradients()
mse_gradients()
chain_rule()