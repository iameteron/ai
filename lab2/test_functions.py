import numpy as np


def Rastrigin_function(
    x: np.array,
    A: float = 10,
) -> np.array:
    y = (
        A * 2
        + x[0] ** 2
        - A * np.cos(2 * np.pi * x[0])
        + x[1] ** 2
        - A * np.cos(2 * np.pi * x[1])
    )
    return y


def Rastrigin_gradient(
    x: np.array,
    A: float = 10,
) -> np.array:
    df_dx = 2 * x[0] + 2 * np.pi * A * np.sin(2 * np.pi * x[0])
    df_dy = 2 * x[1] + 2 * np.pi * A * np.sin(2 * np.pi * x[1])
    return np.array([df_dx, df_dy])


def Rosenbrock_function(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    y = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    return y


def Rosenbrock_gradient(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    df_dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    df_dy = 2 * b * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])


def normalized_Rosenbrock_function(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    y = ((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2) / 1000
    return y


def normalized_Rosenbrock_gradient(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    df_dx = (-2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)) / 1000
    df_dy = (2 * b * (x[1] - x[0] ** 2)) / 1000
    return np.array([df_dx, df_dy])


def sphere_function(
    x: np.array,
    A: float = 1,
    B: float = 1,
) -> np.array:
    y = A * x[0] ** 2 + B * x[1] ** 2
    return y


def sphere_gradient(
    x: np.array,
    A: float = 1,
    B: float = 1,
) -> np.array:
    df_dx = 2 * A * x[0]
    df_dy = 2 * B * x[1]
    return np.array([df_dx, df_dy])


def ellipse_function(
    x: np.array,
    A: float = 1,
    B: float = 5,
) -> np.array:
    y = A * x[0] ** 2 + B * x[1] ** 2
    return y


def ellipse_gradient(
    x: np.array,
    A: float = 1,
    B: float = 5,
) -> np.array:
    df_dx = 2 * A * x[0]
    df_dy = 2 * B * x[1]
    return np.array([df_dx, df_dy])


def Booths_function(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    y = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    return y


def Booths_gradient(
    x: np.array,
    a: float = 1,
    b: float = 100,
) -> np.array:
    df_dx = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
    df_dy = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
    return np.array([df_dx, df_dy])
