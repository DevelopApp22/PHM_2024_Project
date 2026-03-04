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
        X_num = X_df[FEATURES].to_numpy()

        # --- RF prediction (media alberi) ---
        trq_target_pred = self.model.predict(X_num)

        trq_target_pred = np.clip(trq_target_pred, 1e-6, None)

        # --- output ---
        out = pd.DataFrame(index=X_df.index)
        out["id"] = np.arange(n)
        out["trq_target_pred"] = trq_target_pred

        measured = X_df["trq_measured"].to_numpy()
        out["trq_margin_pred"] = (measured / trq_target_pred - 1) * 100

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

        # appiattiamo → (n_trees,)
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


    def cross_validate(self, X, y, n_splits=5, shuffle=True, random_state=42, verbose=True):
        """
        K-Fold CV:
        - Divide il dataset in K fold
        - Ogni fold a turno diventa validation
        - Gli altri fold diventano training
        - Addestra un modello nuovo per ogni fold

        Ritorna:
        - fold_results: lista di dict (uno per fold)
        - summary: media e std delle metriche sui fold
        """
        X = np.asarray(X)
        y = np.asarray(y)

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        fold_results = []
        params = self.model.get_params()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            m = RandomForestRegressorModel(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=params["random_state"],
                n_jobs=params["n_jobs"],
            )
            m.fit(X_tr, y_tr)

            y_hat = m.predict_mean(X_va)

            mae = mean_absolute_error(y_va, y_hat)
            mse = mean_squared_error(y_va, y_hat)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_va, y_hat)

            res = {"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2}
            fold_results.append(res)

            if verbose:
                print(f"Fold {fold}: MAE={mae:.6f} | RMSE={rmse:.6f} | R2={r2:.6f}")

        mae_vals = np.array([r["MAE"] for r in fold_results], dtype=float)
        rmse_vals = np.array([r["RMSE"] for r in fold_results], dtype=float)
        r2_vals = np.array([r["R2"] for r in fold_results], dtype=float)

        summary = {
            "MAE_mean": float(mae_vals.mean()),
            "MAE_std": float(mae_vals.std(ddof=1)),
            "RMSE_mean": float(rmse_vals.mean()),
            "RMSE_std": float(rmse_vals.std(ddof=1)),
            "R2_mean": float(r2_vals.mean()),
            "R2_std": float(r2_vals.std(ddof=1)),
        }

        if verbose:
            print("\nCV Summary (mean ± std)")
            print(f"MAE : {summary['MAE_mean']:.6f} ± {summary['MAE_std']:.6f}")
            print(f"RMSE: {summary['RMSE_mean']:.6f} ± {summary['RMSE_std']:.6f}")
            print(f"R2  : {summary['R2_mean']:.6f} ± {summary['R2_std']:.6f}")

        return fold_results, summary


    def _check_fitted(self):
        """
        Impedisce di chiamare predict/evaluate prima del fit.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")


    def tune_n_estimators_cv(
        self,
        X,
        y,
        n_estimators_list=None,
        n_splits=5,
        shuffle=True,
        random_state=42,
        verbose=True
    ):
        """
        Prova diversi n_estimators, fa CV per ciascuno e ritorna:
        - results: lista di dict con metriche (mean±std)
        - best: dict migliore (min RMSE_mean)
        """
        if n_estimators_list is None:
            n_estimators_list = [300, 500, 750, 1000, 1250]

        base_params = self.model.get_params()

        results = []
        best = None

        for n in n_estimators_list:
            m = RandomForestRegressorModel(
                n_estimators=n,
                max_depth=base_params["max_depth"],
                min_samples_leaf=base_params["min_samples_leaf"],
                max_features=base_params["max_features"],
                random_state=base_params["random_state"],
                n_jobs=base_params["n_jobs"],
            )

            _, summary = m.cross_validate(
                X, y,
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
                verbose=False
            )

            row = {
                "n_estimators": int(n),
                **summary
            }
            results.append(row)

            if verbose:
                print(
                    f"n={n:4d} | RMSE={row['RMSE_mean']:.6f} ± {row['RMSE_std']:.6f} "
                    f"| MAE={row['MAE_mean']:.6f} | R2={row['R2_mean']:.6f}"
                )

            if (best is None) or (row["RMSE_mean"] < best["RMSE_mean"]):
                best = row

        if verbose and best is not None:
            print("\n Best n_estimators (min RMSE_mean)")
            print(
                f"n={best['n_estimators']} | RMSE={best['RMSE_mean']:.6f} ± {best['RMSE_std']:.6f} "
                f"| MAE={best['MAE_mean']:.6f} | R2={best['R2_mean']:.6f}"
            )

        return results, best

    def tune_min_samples_leaf_cv(
            self,
            X,
            y,
            min_samples_list=None,
            n_splits=5,
            shuffle=True,
            random_state=42,
            verbose=True
    ):
        """
        Cerca il miglior min_samples_leaf usando Cross Validation.

        Ritorna:
        - results: lista con metriche per ogni valore
        - best: configurazione con RMSE minimo
        """

        if min_samples_list is None:
            min_samples_list = [1, 2, 5, 10, 20, 30]

        base_params = self.model.get_params()

        results = []
        best = None

        for msl in min_samples_list:

            model = RandomForestRegressorModel(
                n_estimators=base_params["n_estimators"],  # fisso
                max_depth=base_params["max_depth"],
                min_samples_leaf=msl,
                max_features=base_params["max_features"],
                random_state=base_params["random_state"],
                n_jobs=base_params["n_jobs"],
            )

            _, summary = model.cross_validate(
                X, y,
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
                verbose=False
            )

            row = {
                "min_samples_leaf": msl,
                **summary
            }
            results.append(row)

            if verbose:
                print(
                    f"leaf={msl:2d} | RMSE={row['RMSE_mean']:.6f} ± {row['RMSE_std']:.6f} "
                    f"| MAE={row['MAE_mean']:.6f} | R2={row['R2_mean']:.6f}"
                )

            if (best is None) or (row["RMSE_mean"] < best["RMSE_mean"]):
                best = row

        if verbose:
            print("\n✅ BEST min_samples_leaf")
            print(
                f"leaf={best['min_samples_leaf']} | "
                f"RMSE={best['RMSE_mean']:.6f} ± {best['RMSE_std']:.6f} "
                f"| MAE={best['MAE_mean']:.6f} | R2={best['R2_mean']:.6f}"
            )

        return results, best
