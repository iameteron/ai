import numpy as np

from util import compare_methods
from test_functions import Rosenbrock_function, Rosenbrock_gradient

if __name__ == '__main__':

    initial_point = np.array([-0.5, 2.0])
    xlim = (-1.5, 2)
    ylim = (-0.5, 3)
    x_star = np.array([1, 1])

    cmap = "magma_r"

    compare_methods(
        function=Rosenbrock_function,
        grad_function=Rosenbrock_gradient,
        initial_point=initial_point,
        xlim=xlim,
        ylim=ylim,
        x_star=x_star,
        cmap=cmap,
    )