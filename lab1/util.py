import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import matplotlib.animation as animation
from optimizer import GradientOptimizer


def solution_visualization(
    function: callable,
    x_seqs: list = None,
    x_star: np.array = None,
    xlim: tuple = (-5, 5),
    ylim: tuple = (-5, 5),
    step: float = 0.01,
    flat: bool = True,
    labels: list = [],
    cmap: str = "viridis",
    animate: bool = True,
):
    x1 = np.arange(xlim[0], xlim[1], step)
    x2 = np.arange(ylim[0], ylim[1], step)
    x = np.meshgrid(x1, x2)

    y = function(x)

    if flat:
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(10)

        im = ax.imshow(
            y,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )

        im = ax.imshow(
            y,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            cmap=cmap,
            norm=AsinhNorm(),
            aspect="equal",
        )

        fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        if x_star is not None:
            ax.scatter(x_star[0], x_star[1], marker=(5, 1), c="y")

        if animate:

            lines = {}
            for x_seq, label in zip(x_seqs, labels):
                if x_seq is not None:
                    lines[label] = ax.plot(x_seq[0, 0], x_seq[0, 1], label=label)
                    ax.scatter(x_seq[0, 0], x_seq[0, 1], c="black")
                    ax.scatter(x_seq[-1, 0], x_seq[-1, 1], c="r")

            def update(i):
                for x_seq, label in zip(x_seqs, labels):
                    if x_seq is not None:
                        lines[label].set_data(x_seq[i, 0], x_seq[i, 1])

            ani = animation.FuncAnimation(fig=fig, func=update, frames=500, interval=30)
            plt.legend()
            plt.show()
            ani.save('./animtaion.gif')

        else:
            for x_seq, label in zip(x_seqs, labels):
                if x_seq is not None:
                    ax.plot(x_seq[:, 0], x_seq[:, 1], label=label)
                    ax.scatter(x_seq[0, 0], x_seq[0, 1], c="black")
                    ax.scatter(x_seq[-1, 0], x_seq[-1, 1], c="r")

            plt.legend()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            x[0], x[1], y, facecolors=plt.cm.viridis(y), rstride=1, cstride=1
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()

    
def hyperparameter_optimization(optimizer: GradientOptimizer) -> None:
    alphas = np.linspace(1e-2, 1e-3, 100)
    min_value = +np.inf

    for alpha in alphas:
        optimizer.alpha = alpha
        x_seq, y_seq = optimizer.find_minimum()
        if y_seq[-1] < min_value:
            best_alpha = alpha
            min_value = y_seq[-1]

    optimizer.alpha = best_alpha
    

def compare_methods(
    function: callable,
    initial_point: np.array,
    xlim: tuple,
    ylim: tuple,
    x_star: np.array = None,
    grad_function: callable = None,
    cmap: str = "viridis",
) -> None:
    methods = ["GD", "Momentum", "NesterovMomentum", "RMSProp", "Adam"]

    x_seqs = []
    y_seqs = []

    for method in methods:
        optimizer = GradientOptimizer(function, initial_point, grad_function, method)
        hyperparameter_optimization(optimizer)

        x_seq, y_seq = optimizer.find_minimum()
        x_seqs.append(x_seq)
        y_seqs.append(y_seq)

        print(
            f"{method}: steps = {x_seq.shape[0] - 1}, value = {y_seq[-1]}, lr = {optimizer.alpha}"
        )

    solution_visualization(
        function, x_seqs, x_star, xlim, ylim, cmap=cmap, labels=methods
    )