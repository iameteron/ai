import numpy as np


class GradientOptimizer:
    def __init__(
        self,
        function: callable,
        initial_point: np.array,
        grad_function: callable = None,
        method: str = "GD",
        grad_calculation: str = "",
        stop_criteria: str = "",
        max_iteration: int = 500,
        alpha: float = 1e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        eps: float = 1e-8,
        eps_stop: float = 1e-3,
    ):
        self.function = function
        self.initial_point = initial_point
        self.grad_function = grad_function
        self.method = method
        self.grad_calculation = grad_calculation
        self.stop_criteria = stop_criteria
        self.max_iteration = max_iteration
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.eps_stop = eps_stop

    def step(self):
        grad = self.grad_function(self.point_seq[-1])

        if self.method == "GD":
            self.next_point = self.point_seq[-1] - self.alpha * grad

        if self.method == "Momentum":
            if len(self.point_seq) > 1:
                self.next_point = (
                    self.point_seq[-1]
                    - self.alpha * grad
                    + self.beta_1 * (self.point_seq[-1] - self.point_seq[-2])
                )
            else:
                self.next_point = self.point_seq[-1] - self.alpha * grad

        if self.method == "NesterovMomentum":
            if len(self.point_seq) > 1:
                grad = self.grad_function(
                    self.point_seq[-1] + self.beta_1 * self.curr_momentum
                )
                self.curr_momentum = (
                    self.beta_1 * self.curr_momentum - self.alpha * grad
                )
            else:
                grad = self.grad_function(self.point_seq[-1])
                self.curr_momentum = -self.alpha * grad

            self.next_point = self.point_seq[-1] + self.curr_momentum

        if self.method == "RMSProp":
            if len(self.point_seq) > 1:
                self.curr_adaptation = (
                    self.beta_2 * self.curr_adaptation + (1 - self.beta_2) * grad**2
                )
                self.next_point = self.point_seq[-1] - self.alpha * grad / np.sqrt(
                    self.curr_adaptation + self.eps
                )
            else:
                self.curr_adaptation = 1
                self.next_point = self.point_seq[-1] - self.alpha * grad

        if self.method == "Adam":
            if len(self.point_seq) > 1:
                self.curr_momentum = (
                    self.beta_1 * self.curr_momentum + (1 - self.beta_1) * grad
                )
                self.curr_adaptation = (
                    self.beta_2 * self.curr_adaptation + (1 - self.beta_2) * grad**2
                )
                self.next_point = (self.point_seq[-1] -
                    self.alpha * self.curr_momentum / np.sqrt(
                    self.curr_adaptation + self.eps)
                )
            else:
                self.curr_adaptation = 1
                self.curr_momentum = -self.alpha * grad
                self.next_point = self.point_seq[-1] - self.alpha * grad

        self.next_value = self.function(self.next_point)

        self.done = False

        if self.stop_criteria == "point_norm":
            self.done = (
                np.linalg.norm(self.point_seq[-1] - self.next_point) < self.eps_stop
            )

        if self.stop_criteria == "value_norm":
            self.done = (
                np.linalg.norm(self.value_seq[-1] - self.next_value) < self.eps_stop
            )

        if self.stop_criteria == "grad_norm":
            self.done = np.linalg.norm(grad) < self.eps_stop

    def find_minimum(self):
        if self.grad_function is None:
            pass

        self.point_seq = []
        self.value_seq = []
        self.point_seq.append(self.initial_point)
        self.value_seq.append(self.function(self.initial_point))

        for _ in range(self.max_iteration):
            self.step()
            self.point_seq.append(self.next_point)
            self.value_seq.append(self.next_value)
            if self.done:
                break

        return np.array(self.point_seq), np.array(self.value_seq)