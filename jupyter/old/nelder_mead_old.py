import numpy as np

class NelderMead:
    def __init__(self, objective_function, initial_point, step_sizes, bounds, max_iter=100, tol=1e-4):
        """
        Initialize the Nelder-Mead optimizer.

        :param objective_function: The objective function to minimize.
        :param initial_point: Initial guess (3D point as a list or numpy array).
        :param step_sizes: Initial step sizes for each dimension (list or numpy array).
        :param bounds: List of tuples specifying the domain for each variable.
        :param max_iter: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        """
        self.objective_function = objective_function
        self.simplex = self._initialize_simplex(initial_point, step_sizes)
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
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

    def _evaluate_simplex(self):
        """
        Evaluate the objective function at all points in the simplex.
        """
        return [self.objective_function(point) for point in self.simplex]

    def _reflect(self, centroid, worst_point, alpha=1.0):
        """
        Reflect the worst point through the centroid.
        """
        reflected_point = centroid + alpha * (centroid - worst_point)
        return self._clip_to_bounds(reflected_point)

    def _expand(self, centroid, reflected_point, gamma=2.0):
        """
        Expand the reflected point further.
        """
        expanded_point = centroid + gamma * (reflected_point - centroid)
        return self._clip_to_bounds(expanded_point)

    def _contract(self, centroid, worst_point, beta=0.5):
        """
        Contract the worst point towards the centroid.
        """
        contracted_point = centroid + beta * (worst_point - centroid)
        return self._clip_to_bounds(contracted_point)

    def _shrink(self, best_point, sigma=0.5):
        """
        Shrink the simplex towards the best point.
        """
        new_simplex = [best_point]
        for point in self.simplex[1:]:
            new_simplex.append(best_point + sigma * (point - best_point))
        return new_simplex

    def _clip_to_bounds(self, point):
        """
        Clip the point to stay within the specified bounds.
        """
        return np.clip(point, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

    def optimize(self):
        """
        Run the Nelder-Mead optimization algorithm.
        """
        for iteration in range(self.max_iter):
            # Evaluate the simplex
            values = self._evaluate_simplex()
            self.history.append((self.simplex.copy(), values.copy()))

            # Sort the simplex based on function values
            sorted_indices = np.argsort(values)
            self.simplex = [self.simplex[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]

            # Check for convergence
            if np.std(values) < self.tol:
                break

            # Compute centroid of the n best points
            centroid = np.mean(self.simplex[:-1], axis=0)

            # Reflect the worst point
            reflected_point = self._reflect(centroid, self.simplex[-1])
            reflected_value = self.objective_function(reflected_point)

            if values[0] <= reflected_value < values[-2]:
                # Replace the worst point with the reflected point
                self.simplex[-1] = reflected_point
            elif reflected_value < values[0]:
                # Try to expand
                expanded_point = self._expand(centroid, reflected_point)
                expanded_value = self.objective_function(expanded_point)
                if expanded_value < reflected_value:
                    self.simplex[-1] = expanded_point
                else:
                    self.simplex[-1] = reflected_point
            else:
                # Contract
                contracted_point = self._contract(centroid, self.simplex[-1])
                contracted_value = self.objective_function(contracted_point)
                if contracted_value < values[-1]:
                    self.simplex[-1] = contracted_point
                else:
                    # Shrink the simplex
                    self.simplex = self._shrink(self.simplex[0])

        # Return the best point and its value
        best_index = np.argmin(values)
        return self.simplex[best_index], values[best_index]

# Example usage
def objective_function(x):
    # Example objective function with a delay (simulated by a sleep)
    import time
    # time.sleep(1)  # Simulate delay
    return x[0]**2 + x[1]**2 + x[2]**2  # Example: minimize sum of squares

initial_point = [1.0, 2.0, 3.0]
step_sizes = [0.5, 0.5, 0.5]
bounds = [(-10, 10), (-10, 10), (-10, 10)]  # Domain for each variable

optimizer = NelderMead(objective_function, initial_point, step_sizes, bounds)
best_point, best_value = optimizer.optimize()

print("Best point:", best_point)
print("Best value:", best_value)