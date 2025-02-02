import numpy as np

class StepwiseNelderMead:
    def __init__(self, initial_point, step_sizes, bounds, alpha=1.0, gamma=2.0, beta=0.5, sigma=0.5):
        """
        Initialize the stepwise Nelder-Mead optimizer.

        :param initial_point: Initial guess (3D point as a list or numpy array).
        :param step_sizes: Initial step sizes for each dimension (list or numpy array).
        :param bounds: List of tuples specifying the domain for each variable.
        :param alpha: Reflection coefficient (default: 1.0).
        :param gamma: Expansion coefficient (default: 2.0).
        :param beta: Contraction coefficient (default: 0.5).
        :param sigma: Shrink coefficient (default: 0.5).
        """
        self.simplex = self._initialize_simplex(initial_point, step_sizes)
        self.bounds = bounds
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.values = None  # To store the objective function values of the simplex
        self.history = []  # To store the history of evaluations

    def _initialize_simplex(self, initial_point, step_sizes):
        """
        Initialize the simplex with the given initial point and step sizes.
        """
        n = len(initial_point)
        simplex = [np.array(initial_point)]
        for i in range(n):
            point = np.array(initial_point)
            point[i] += step_sizes[i]
            simplex.append(point)
        return simplex

    def _clip_to_bounds(self, point):
        """
        Clip the point to stay within the specified bounds.
        """
        return np.clip(point, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

    def _reflect(self, centroid, worst_point):
        """
        Reflect the worst point through the centroid.
        """
        reflected_point = centroid + self.alpha * (centroid - worst_point)
        return self._clip_to_bounds(reflected_point)

    def _expand(self, centroid, reflected_point):
        """
        Expand the reflected point further.
        """
        expanded_point = centroid + self.gamma * (reflected_point - centroid)
        return self._clip_to_bounds(expanded_point)

    def _contract(self, centroid, worst_point):
        """
        Contract the worst point towards the centroid.
        """
        contracted_point = centroid + self.beta * (worst_point - centroid)
        return self._clip_to_bounds(contracted_point)

    def _shrink(self, best_point):
        """
        Shrink the simplex towards the best point.
        """
        new_simplex = [best_point]
        for point in self.simplex[1:]:
            new_simplex.append(best_point + self.sigma * (point - best_point))
        return new_simplex

    def step(self, new_value=None):
        """
        Perform one step of the Nelder-Mead algorithm.

        :param new_value: The objective function value of the last suggested point.
        :return: The next point to evaluate (or None if the algorithm has converged).
        """
        if self.values is None:
            # First step: evaluate the initial simplex
            self.values = [None] * len(self.simplex)
            return self.simplex[0]  # Return the first point to evaluate

        # Update the value of the last evaluated point
        if new_value is not None:
            for i in range(len(self.simplex)):
                if self.values[i] is None:
                    self.values[i] = new_value
                    break

        # Check if all points in the simplex have been evaluated
        if None in self.values:
            # Still evaluating the initial simplex
            next_index = self.values.index(None)
            return self.simplex[next_index]

        # Sort the simplex based on function values
        sorted_indices = np.argsort(self.values)
        self.simplex = [self.simplex[i] for i in sorted_indices]
        self.values = [self.values[i] for i in sorted_indices]

        # Compute centroid of the n best points
        centroid = np.mean(self.simplex[:-1], axis=0)

        # Reflect the worst point
        reflected_point = self._reflect(centroid, self.simplex[-1])
        reflected_value = None  # Placeholder for the reflected value

        if self.values[0] <= reflected_value < self.values[-2]:
            # Replace the worst point with the reflected point
            self.simplex[-1] = reflected_point
            self.values[-1] = reflected_value
        elif reflected_value < self.values[0]:
            # Try to expand
            expanded_point = self._expand(centroid, reflected_point)
            expanded_value = None  # Placeholder for the expanded value
            if expanded_value < reflected_value:
                self.simplex[-1] = expanded_point
                self.values[-1] = expanded_value
            else:
                self.simplex[-1] = reflected_point
                self.values[-1] = reflected_value
        else:
            # Contract
            contracted_point = self._contract(centroid, self.simplex[-1])
            contracted_value = None  # Placeholder for the contracted value
            if contracted_value < self.values[-1]:
                self.simplex[-1] = contracted_point
                self.values[-1] = contracted_value
            else:
                # Shrink the simplex
                self.simplex = self._shrink(self.simplex[0])
                self.values = [None] * len(self.simplex)  # Reset values for the new simplex

        # Return the next point to evaluate
        if None in self.values:
            next_index = self.values.index(None)
            return self.simplex[next_index]
        else:
            return None  # Algorithm has converged

# Example usage
initial_point = [1.0, 2.0, 3.0]
step_sizes = [0.5, 0.5, 0.5]
bounds = [(-10, 10), (-10, 10), (-10, 10)]  # Domain for each variable

optimizer = StepwiseNelderMead(initial_point, step_sizes, bounds)

# Simulate a delayed objective function
def objective_function(x):
    # Example objective function with a delay (simulated by a sleep)
    import time
    # time.sleep(1)  # Simulate delay
    return x[0]**2 + x[1]**2 + x[2]**2  # Example: minimize sum of squares

# Stepwise optimization loop
for step in range(20):  # Maximum of 20 steps
    next_point = optimizer.step()
    if next_point is None:
        print("Optimization converged.")
        break
    print(f"Step {step + 1}: Evaluate point {next_point}")
    value = objective_function(next_point)
    print(f"Objective function value: {value}")
    optimizer.step(value)  # Provide the value to the optimizer