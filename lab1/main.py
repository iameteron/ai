import warnings

import numpy as np
from test_functions import *
from util import compare_methods

warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":

    initial_points = [
        np.array([-0.5, 2.0]),
        np.array([-0.5, 2.0]),
        np.array([-4, 3]),
        np.array([-4, 3]),
        np.array([-7, -8]),
        np.array([-4, 3]),
    ]

    xlims = [
        (-1.5, 2),
        (-1.5, 2),
        (-5, 5),
        (-5, 5),
        (-10, 10),
        (-5, 5),
    ]

    ylims = [
        (-0.5, 3),
        (-0.5, 3),
        (-5, 5),
        (-5, 5),
        (-10, 10),
        (-5, 5),
    ]

    x_stars = [
        np.array([1, 1]),
        np.array([1, 1]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([1, 3]),
        np.array([0, 0]),
    ]

    functions = [
        Rosenbrock_function,
        normalized_Rosenbrock_function,
        sphere_function,
        ellipse_function,
        Booths_function,
        Rastrigin_function,
    ]

    grad_functions = [
        Rosenbrock_gradient,
        normalized_Rosenbrock_gradient,
        sphere_gradient,
        ellipse_gradient,
        Booths_gradient,
        Rastrigin_gradient,
    ]

    for i in range(len(functions)):
        compare_methods(
            function=functions[i],
            grad_function=grad_functions[i],
            initial_point=initial_points[i],
            xlim=xlims[i],
            ylim=ylims[i],
            x_star=x_stars[i],
            animate=True
        )
        break
