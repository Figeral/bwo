import numpy as np

class BenchmarkFunctions:
    @staticmethod
    def sphere(x):
        return np.sum(x**2)

    @staticmethod
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @classmethod
    def get_function(cls, name):
        functions = {
            "Sphere": cls.sphere,
            "Rastrigin": cls.rastrigin,
            "Rosenbrock": cls.rosenbrock
        }
        return functions.get(name, cls.sphere)

    @staticmethod
    def get_bounds(name):
        bounds = {
            "Sphere": (-5.12, 5.12),
            "Rastrigin": (-5.12, 5.12),
            "Rosenbrock": (-2.048, 2.048)
        }
        return bounds.get(name, (-5.12, 5.12))

    @staticmethod
    def get_surface_data(func, bounds, res=100):
        """Generates X, Y, Z data for Plotly 3D Surface over the given bounds."""
        x = np.linspace(bounds[0], bounds[1], res)
        y = np.linspace(bounds[0], bounds[1], res)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                # We assume 2D for surface plotting
                pt = np.array([X[i, j], Y[i, j]])
                Z[i, j] = func(pt)
        return X, Y, Z
