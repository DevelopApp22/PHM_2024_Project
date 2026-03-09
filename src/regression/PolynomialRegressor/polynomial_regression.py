import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
class PolynomialRegression:

    def __init__(self,degree=2):
        self.degree = degree
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self.is_fitted = False

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X_train, y_train):
        X_train = self._ensure_2d(X_train)
        y_train = np.asarray(y_train).ravel()
        Z_train = self.scaler.fit_transform(X_train)
        Z_poly = self.poly.fit_transform(Z_train)
        self.model.fit(Z_poly, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._ensure_2d(X)
        Z = self.scaler.transform(X)
        Z_poly = self.poly.transform(Z)
        return self.model.predict(Z_poly)

    def predictAll(self, X, y_true=None, FEATURES=None, keep_cols=None):
        """
        Restituisce un DataFrame con:
          - id (0..n-1)
          - trq_target_pred
          - trq_margin_pred
          - (opzionale) trq_margin_true
          - (opzionale) colonne originali keep_cols

        NOTE:
          - Il modello predice trq_target_pred.
          - trq_margin_pred = (trq_measured / trq_target_pred - 1) * 100
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # --- FEATURES obbligatorie ---
        if FEATURES is None:
            raise ValueError("FEATURES must be provided (list of feature column names).")
        FEATURES = list(FEATURES)

        # --- check: NaN dentro FEATURES ---
        if any(pd.isna(FEATURES)):
            raise ValueError(f"FEATURES contains NaN: {[f for f in FEATURES if pd.isna(f)]}")

        # --- check: colonne mancanti ---
        missing = [c for c in FEATURES if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing columns in X: {missing}")

        # --- check: trq_measured serve per il margin ---
        if "trq_measured" not in X_df.columns:
            raise ValueError("Column 'trq_measured' is required to compute trq_margin_pred.")

        # --- preprocessing per il modello ---
        X_num = X_df[FEATURES].to_numpy()
        X_2d = self._ensure_2d(X_num)
        Z = self.scaler.transform(X_2d)
        Z_poly = self.poly.transform(Z)

        # --- predizione (torque target) ---
        y_pred = self.model.predict(Z_poly)

        if np.ndim(y_pred) == 2 and y_pred.shape[1] >= 1:
            trq_target_pred = y_pred[:, 0]
        else:
            trq_target_pred = np.asarray(y_pred).ravel()

        # --- sicurezza numerica ---
        trq_target_pred = np.clip(trq_target_pred, 1e-6, None)

        # --- output ---
        n = len(X_df)
        out = pd.DataFrame(index=X_df.index)

        out["id"] = np.arange(n)
        out["trq_target_pred"] = trq_target_pred

        measured = X_df["trq_measured"].to_numpy()
        out["trq_margin_pred"] = (measured / trq_target_pred - 1) * 100

        # --- true (opzionale) ---
        if y_true is not None:
            y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
            if len(y_arr) != n:
                raise ValueError(f"y_true length ({len(y_arr)}) != X length ({n})")
            out["trq_margin_true"] = y_arr

        # --- colonne originali extra (opzionale) ---
        if keep_cols is not None:
            keep_cols = list(keep_cols)
            missing_keep = [c for c in keep_cols if c not in X_df.columns]
            if missing_keep:
                raise ValueError(f"keep_cols not found in X: {missing_keep}")
            for c in keep_cols:
                out[c] = X_df[c].values

        return out

    def compute_residuals(self, X_train, y_true, path):

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_train)
        residuals = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true
        })

        if not os.path.exists(path):
            os.makedirs(path)

        residuals.to_csv(path+"/residui.csv", index=False)

    def evaluate_regression(self, X_test, y_test):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae}")
        print(f"MSE: {mse:}")
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")

    def tune_degree(
        self,
        X_train,
        y_train,
        degrees=(2, 3),
        alpha=None,
        cv=5,
        metric="rmse",
        random_state=42,
        shuffle=True,
        verbose=True
    ):
        """
           Sceglie il miglior degree (es. 2 vs 3) con Cross-Validation.

           Per ogni degree d:
           - fa CV (KFold)
           - a ogni fold:
               1) fit scaler sul train fold
               2) crea feature polinomiali di grado d
               3) fit Ridge (con alpha fissato)
               4) valuta su validation fold (RMSE o MAE)
           - fa la media sui fold
           - sceglie il degree con errore medio MINORE

           NOTE:
           - alpha: se None usa self.alpha, altrimenti usa quello che passi.
             (Consigliato: prima fare tune_alpha e poi tune_degree usando quell'alpha)
           - metric: "rmse" o "mae"
           """
        X_train = self._ensure_2d(X_train)
        y_train = np.asarray(y_train).ravel()

        metric = metric.lower().strip()
        if metric not in ("rmse", "mae"):
            raise ValueError("metric must be 'rmse' or 'mae'")

        kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

        best_degree = None
        best_score = np.inf
        scores_by_degree = {}

        for d in degrees:
            fold_scores = []

            for tr_idx, va_idx in kf.split(X_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                scaler = StandardScaler()
                poly = PolynomialFeatures(degree=int(d), include_bias=False)
                model = LinearRegression()

                Z_tr = scaler.fit_transform(X_tr)
                Z_tr_poly = poly.fit_transform(Z_tr)
                model.fit(Z_tr_poly, y_tr)

                Z_va = scaler.transform(X_va)
                Z_va_poly = poly.transform(Z_va)
                y_pred = model.predict(Z_va_poly)

                if metric == "rmse":
                    score = float(np.sqrt(mean_squared_error(y_va, y_pred)))
                else:
                    score = float(mean_absolute_error(y_va, y_pred))

                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            scores_by_degree[int(d)] = mean_score

            if mean_score < best_score:
                best_score = mean_score
                best_degree = int(d)

        if verbose:
            print(f"[tune_degree] best_degree={best_degree} | best_{metric}={best_score} | alpha={alpha}")

        return best_degree, best_score, scores_by_degree





