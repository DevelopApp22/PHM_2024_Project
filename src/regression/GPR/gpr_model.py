import numpy as np
import joblib
import pandas as pd

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

    def predict_mu_std_all(self, X, y_true=None, FEATURES=None, keep_cols=None,scaler=None):
        """
        GPR prediction with uncertainty propagation to trq_margin.

        Output:
          - id
          - trq_target_pred
          - trq_margin_pred
          - trq_margin_std
          - (optional) trq_margin_true
        """
        self.check_fitted()

        # --- assicurati DataFrame ---
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n = len(X_df)

        # --- FEATURES obbligatorie ---
        if FEATURES is None:
            raise ValueError("FEATURES must be provided.")

        FEATURES = list(FEATURES)

        # --- controlli ---
        missing = [c for c in FEATURES if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing columns in X: {missing}")

        if "trq_measured" not in X_df.columns:
            raise ValueError("Column 'trq_measured' is required.")

        # --- input modello ---
        X_num = scaler.fit_transform(X_df[FEATURES])

        # --- GPR prediction ---
        mu, std = self.model.predict(X_num, return_std=True)
        mu = np.asarray(mu).ravel()
        std = np.asarray(std).ravel()

        mu = np.clip(mu, 1e-6, None)

        # --- output ---
        out = pd.DataFrame(index=X_df.index)
        out["id"] = np.arange(n)
        out["trq_target_pred"] = mu

        measured = X_df["trq_measured"].to_numpy()

        # --- margin ---
        margin_pred = (measured / mu - 1) * 100
        out["trq_margin_pred"] = margin_pred

        # 🔥 --- propagazione incertezza (FORMULA GIUSTA) ---
        deriv = np.abs(-100 * measured / (mu ** 2))
        margin_std = deriv * std
        margin_std = np.clip(margin_std, 1e-12, None)

        out["trq_margin_std"] = margin_std

        # --- true opzionale ---
        if y_true is not None:
            y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
            if len(y_arr) != n:
                raise ValueError(f"y_true length ({len(y_arr)}) != X length ({n})")
            out["trq_margin_true"] = y_arr

        # --- colonne extra opzionali ---
        if keep_cols is not None:
            keep_cols = list(keep_cols)
            missing_keep = [c for c in keep_cols if c not in X_df.columns]
            if missing_keep:
                raise ValueError(f"keep_cols not found in X: {missing_keep}")
            for c in keep_cols:
                out[c] = X_df[c].values

        return out

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