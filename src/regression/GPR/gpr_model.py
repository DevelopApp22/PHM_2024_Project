import numpy as np
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class GPRModel:
    def __init__(self, random_state=42):
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e-1))
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=10,
            alpha=1e-10,
            random_state=random_state
        )
        self.is_fitted = False

    def check_fitted(self):
        if not getattr(self, "is_fitted", False):
            raise RuntimeError("Modello non addestrato: chiama .fit(X, y) prima di .predict/.evaluate.")

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X, return_std=True):
        self.check_fitted()
        return self.model.predict(X, return_std=return_std)

    def predict_mu_std(self, X):
        mu, std = self.predict(X, return_std=True)
        return np.asarray(mu).ravel(), np.asarray(std).ravel()

    def save(self, path):
        self.check_fitted()
        joblib.dump(self.model, path)

    def sample_y(self, x_test, n_sample=5000):
        self.check_fitted()
        return self.model.sample_y(x_test, n_sample)

    def evaluate(self, X, y):
        y = np.asarray(y).ravel()

        y_pred = self.predict(X, return_std=False)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))

        out = {
            f"rmse": rmse,
            f"mae": mae,
            f"r2": r2
        }
        return out