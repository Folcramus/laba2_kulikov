import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
# Определяем систему уравнений с ограничением значений
def f(x, y):
    # Ограничиваем значения x и y, чтобы избежать переполнения
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    dxdt =  2*x - 2  # F
    dydt = x - 2*y**2  # G
    return dxdt, dydt


# Якобиан системы
def jacobian(x, y):
    dFdx = 2
    dFdy = 0
    dGdx = 1
    dGdy = -4*y
    return np.array([[dFdx, dFdy], [dGdx, dGdy]])


# Метод для нахождения особых точек
def find_equilibrium_points():
    equilibrium_points = []
    equilibrium_points.append((1, (1/(np.sqrt(2)))))
    equilibrium_points.append((1, -(1 / (np.sqrt(2)))))
    return equilibrium_points


# Анализ устойчивости точки
def stability_analysis(J, eq):
    eigvals, eigvecs = np.linalg.eig(J)
    if np.all(eigvals.real < 0):
        return "Устойчивая", eigvals, eigvecs
    elif np.all(eigvals.real > 0):
        return "Неустойчивая", eigvals, eigvecs
    else:
        return "Седло", eigvals, eigvecs


# Решение с помощью метода Эйлера с выводом каждых 100 значений
def euler_method(f, x0, y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0], y[0] = x0, y0

    print(f"Траектория (x0={x0:.1f}, y0={y0:.1f}):")

    for i in range(1, len(t)):
        dxdt, dydt = f(x[i - 1], y[i - 1])
        x[i] = x[i - 1] + dxdt * dt
        y[i] = y[i - 1] + dydt * dt

        # Ограничиваем значения для предотвращения переполнения
        if abs(x[i]) > 1e6 or abs(y[i]) > 1e6:
            x[i:] = np.nan
            y[i:] = np.nan
            break

        # Выводим каждые 100 значений
        if i % 30 == 0:
            print(f"t={t[i]:.2f}, x={x[i]:.6f}, y={y[i]:.6f}")

    return x, y, t


# Для отображения векторного поля
def vector_field(f, x_range, y_range, density=20):
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], density),
                       np.linspace(y_range[0], y_range[1], density))
    U, V = np.zeros(X.shape), np.zeros(Y.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx, dy = f(X[i, j], Y[i, j])
            U[i, j] = dx
            V[i, j] = dy

    return X, Y, U, V


# Найдем особые точки
equilibrium_points = find_equilibrium_points()

# Параметры
t0, tf = 0, 10
dt = 0.01

# Начальные условия для траекторий на сетке
x_initial = np.linspace(-3, 3, 3)
y_initial = np.linspace(-3, 3, 3)
initial_conditions = [(x0, y0) for x0 in x_initial for y0 in y_initial]

# Рисуем фазовый портрет
plt.figure(figsize=(10, 8))

# Векторное поле
x_range = (-5, 5)
y_range = (-5, 5)
X, Y, U, V = vector_field(f, x_range, y_range, density=20)
plt.streamplot(X, Y, U, V, color=np.sqrt(U ** 2 + V ** 2), linewidth=1, cmap='Blues')

# Траектории
for x0, y0 in initial_conditions:
    x, y, t = euler_method(f, x0, y0, t0, tf, dt)
    plt.plot(x, y, label=f"Траектория (x0={x0:.1f}, y0={y0:.1f})", alpha=0.7)

# Особые точки и асимптоты
for eq in equilibrium_points:
    J = jacobian(eq[0], eq[1])
    stability, eigvals, eigvecs = stability_analysis(J, eq)
    print(f"Особая точка {eq}:")
    print(f"  Собственные значения: {eigvals}")
    print(f"  Собственные векторы:\n{eigvecs}")

    if stability == "Устойчивая":
        plt.plot(eq[0], eq[1], 'go', label=f"Устойчивая ({eq[0]}, {eq[1]})")
    elif stability == "Неустойчивая":
        plt.plot(eq[0], eq[1], 'ro', label=f"Неустойчивая ({eq[0]}, {eq[1]})")
    else:
        plt.plot(eq[0], eq[1], 'bo', label=f"Седло ({eq[0]}, {eq[1]})")

        # Рисуем асимптоты для седловой точки
        for eigvec in eigvecs.T:
            x_vals = np.linspace(eq[0] - 3, eq[0] + 3, 100)
            y_vals = eq[1] + eigvec[1] / eigvec[0]  * (x_vals - eq[0])
            plt.plot(x_vals, y_vals, 'purple', linestyle='--', linewidth=1.5)

# Настройка графика
plt.title("Фазовый портрет системы")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend(fontsize=8)  # Уменьшаем размер шрифта легенды
plt.grid(True)

plt.show()
