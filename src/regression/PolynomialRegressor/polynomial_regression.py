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





