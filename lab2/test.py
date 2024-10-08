import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg


# Определение пользовательской задачи оптимизации
class MyOptimizationProblem:
    def __init__(self):
        self.dim = 2  # Количество переменных задачи

    def fitness(self, x):
        # Целевая функция (например, функция Розенброка)
        return [(1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2]

    def get_bounds(self):
        # Границы для переменных (нижние и верхние)
        return ([-2, -2], [2, 2])

    def get_name(self):
        return "My Custom Optimization Problem"


# Создание задачи
problem = pg.problem(MyOptimizationProblem())

# Создание популяции
population = pg.population(problem, size=20)

# Определение алгоритма оптимизации (например, дифференциальная эволюция)
# algorithm = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
algorithm = pg.algorithm(pg.cmaes())

# Для визуализации процесса эволюции
num_generations = 50  # Количество поколений
pop_history = [population.get_x()]

# Выполнение оптимизации и сохранение истории популяций
for _ in range(num_generations):
    population = algorithm.evolve(population)
    pop_history.append(population.get_x())

# Визуализация процесса эволюции
x_min, x_max = -2, 2
y_min, y_max = -1, 3
x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)
Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2  # Функция Розенброка

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")

for generation in range(num_generations):
    ax.clear()
    ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    pop = np.array(pop_history[generation])
    ax.scatter(pop[:, 0], pop[:, 1], color="red", marker="o")
    ax.set_title(f"Generation {generation + 1}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.pause(0.5)

plt.show()
