import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RandomForestRegressorModel:

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,

        n_jobs=-1,
    ):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.is_fitted = False


    def fit(self, X, y):
        """
        Addestra la Random Forest.
        - X: (n_samples, n_features)
        - y: (n_samples,)
        """
        self.model.fit(X, y)
        self.is_fitted = True


    def predict_mean(self, X):
        """
        Predizione standard della Random Forest.
        In regressione, sklearn restituisce la media delle predizioni dei singoli alberi.
        Output: array (n_samples,)
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_mean_all(self, X, y_true=None, FEATURES=None, keep_cols=None):
        """
        Random Forest prediction (mean of trees) + conversion to trq_margin.

        Output columns:
          - id
          - trq_target_pred
          - trq_margin_pred
          - (optional) trq_margin_true
          - (optional) keep_cols
        """
        self._check_fitted()

        import pandas as pd
        import numpy as np

        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n = len(X_df)

        if FEATURES is None:
            raise ValueError("FEATURES must be provided.")

        FEATURES = list(FEATURES)

        missing = [c for c in FEATURES if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing columns in X: {missing}")

        if "trq_measured" not in X_df.columns:
            raise ValueError("Column 'trq_measured' is required.")

        X_num = X_df[FEATURES].to_numpy()

        trq_target_pred = self.model.predict(X_num)

        trq_target_pred = np.clip(trq_target_pred, 1e-6, None)

        out = pd.DataFrame(index=X_df.index)
        out["id"] = np.arange(n)
        out["trq_target_pred"] = trq_target_pred

        measured = X_df["trq_measured"].to_numpy()
        out["trq_margin_pred"] = (measured / trq_target_pred - 1) * 100

        if y_true is not None:
            y_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
            if len(y_arr) != n:
                raise ValueError(f"y_true length ({len(y_arr)}) != X length ({n})")
            out["trq_margin_true"] = y_arr

        if keep_cols is not None:
            keep_cols = list(keep_cols)
            missing_keep = [c for c in keep_cols if c not in X_df.columns]
            if missing_keep:
                raise ValueError(f"keep_cols not found in X: {missing_keep}")
            for c in keep_cols:
                out[c] = X_df[c].values

        return out

    def predict_trees(self, X):
        """
        Restituisce le predizioni di ogni albero separatamente
        come DataFrame (n_samples, n_trees).

        - riga i = un campione
        - colonna t = predizione dell'albero t
        """
        self._check_fitted()
        X = np.asarray(X)

        tree_preds = np.array(
            [tree.predict(X) for tree in self.model.estimators_],
            dtype=np.float64
        )


        tree_preds_flat = tree_preds.reshape(-1)

        # DataFrame finale
        trq_target_prediction = pd.DataFrame({
            "trq_target_prediction": tree_preds_flat
        })

        return trq_target_prediction


    def predict_distribution_stats(self, X):
        """
        Calcola μ e σ usando la distribuzione delle predizioni dei singoli alberi.

        μ(x) = media delle predizioni dei tree
        σ(x) = std delle predizioni dei tree

        Output:
        - mean: (n_samples,)
        - std : (n_samples,)
        """
        tree_preds = self.predict_trees(X)
        mean = np.mean(tree_preds, axis=1)
        std = np.std(tree_preds, axis=1)
        return mean, std


    def evaluate(self, X, y, set_name="set", verbose=True):
        """
        Valuta il modello come regressore "puntuale" (predice un numero).
        Metriche: MAE, RMSE, R2.
        """
        self._check_fitted()

        y_pred = self.predict_mean(X)

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = float(np.sqrt(mse))

        r2 = r2_score(y, y_pred)

        metrics = {
            f"{set_name}_MAE": mae,
            f"{set_name}_RMSE": rmse,
            f"{set_name}_R2": r2,
        }

        if verbose:
            print(f"\nMetrics on {set_name}")
            print(f"MAE  : {mae:.6f}")
            print(f"RMSE : {rmse:.6f}")
            print(f"R²   : {r2:.6f}")

        return metrics
