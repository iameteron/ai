import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm
from optimizer import GradientOptimizer


def solution_visualization(
    function: callable,
    x_seqs: list = None,
    x_star: np.array = None,
    xlim: tuple = (-5, 5),
    ylim: tuple = (-5, 5),
    step: float = 0.1,
    flat: bool = True,
    labels: list = [],
    cmap: str = "viridis",
    animate: bool = True,
    path: str = "./gifs/test.gif",
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
            points = {}
            if len(x_seqs) > 0:
                for x_seq, label in zip(x_seqs, labels):
                    lines[label] = ax.plot(x_seq[0, 0], x_seq[0, 1], label=label)[0]
                    points[label] = ax.scatter(x_seq[0, 0], x_seq[0, 1], c="r", s=20)

            def update(i):
                for x_seq, label in zip(x_seqs, labels):
                    lines[label].set_data(x_seq[:i, 0], x_seq[:i, 1])
                    points[label].set_offsets((x_seq[i, 0], x_seq[i, 1]))


            ani = animation.FuncAnimation(
                fig=fig, func=update, frames=x_seqs[0][:, 0].size, interval=90
            )

            plt.legend()
            ani.save(path)

        else:
            if len(x_seqs) > 0:
                for x_seq, label in zip(x_seqs, labels):
                    ax.plot(x_seq[:, 0], x_seq[:, 1], label=label)
                    ax.scatter(x_seq[0, 0], x_seq[0, 1], c="black", s=10)
                    ax.scatter(x_seq[-1, 0], x_seq[-1, 1], c="r", s=10)

            plt.legend()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            x[0], x[1], y, facecolors=plt.cm.viridis(y), rstride=1, cstride=1
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()


def compare_methods(
    function: callable,
    initial_point: np.array,
    xlim: tuple,
    ylim: tuple,
    x_star: np.array = None,
    grad_function: callable = None,
    cmap: str = "magma_r",
    animate: bool = False,
) -> None:
    methods = ["GD", "Momentum", "NesterovMomentum", "RMSProp", "Adam"]

    x_seqs = []
    y_seqs = []

    for method in methods:
        optimizer = GradientOptimizer(function, initial_point, grad_function, method)
        optimizer.hyperparameter_optimization()

        x_seq, y_seq = optimizer.find_minimum()
        x_seqs.append(x_seq)
        y_seqs.append(y_seq)

        print(
            f"{method}: steps = {x_seq.shape[0] - 1}, value = {y_seq[-1]}, lr = {optimizer.alpha}, beta_1 = {optimizer.beta_1}, beta_2 = {optimizer.beta_2}"
        )

    solution_visualization(
        function,
        x_seqs,
        x_star,
        xlim,
        ylim,
        cmap=cmap,
        labels=methods,
        animate=animate,
        path=f"{function.__name__}.gif",
    )
