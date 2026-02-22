import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


class MLPInterpreter:
    """
    Interprete minimale per classificazione binaria.

    Include SOLO i 2 metodi più importanti:
    1) Permutation Importance → importanza globale robusta
    2) SHAP Summary → interpretazione globale con direzione effetto

    Compatibile con modelli sklearn che espongono predict_proba().
    """

    def __init__(self, model, feature_names=None, positive_class=1):
        """
        Parameters
        ----------
        model : estimator già fit()
            Es. sklearn.neural_network.MLPClassifier.
        feature_names : list[str] | None
            Nomi feature se X è numpy.
        positive_class : int
            Indice classe positiva nella predict_proba.
        """
        self.model = model
        self.feature_names = feature_names
        self.positive_class = positive_class

    # -------------------------------------------------
    # Utils
    # -------------------------------------------------
    def _to_dataframe(self, X):
        """Garantisce che X sia un DataFrame con nomi colonna."""
        if isinstance(X, pd.DataFrame):
            return X

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X deve essere 2D.")

        cols = self.feature_names
        if cols is None:
            cols = [f"x{i}" for i in range(X.shape[1])]

        return pd.DataFrame(X, columns=cols)

    # =================================================
    # ⭐ METODO 1 — Permutation Importance (GLOBALE)
    # =================================================
    def permutation_importance(self, X, y, scoring="roc_auc", n_repeats=20):
        """
        Ranking globale delle feature.

        LOGICA:
        - Permuta una feature
        - misura quanto peggiora la performance
        - più peggiora → più la feature è importante

        Returns
        -------
        DataFrame ordinato per importanza.
        """
        Xdf = self._to_dataframe(X)

        r = permutation_importance(
            self.model,
            Xdf,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        imp_df = pd.DataFrame(
            {
                "feature": Xdf.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        return imp_df

    def plot_permutation_importance(self, imp_df, top_k=15):
        """Plot delle top-k feature più importanti."""
        df = imp_df.head(top_k).iloc[::-1]

        plt.figure(figsize=(8, max(3, 0.35 * len(df))))
        plt.barh(df["feature"], df["importance_mean"])
        plt.xlabel("Permutation importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    # =================================================
    # ⭐ METODO 2 — SHAP Summary (GLOBALE + DIREZIONE)
    # =================================================
    def shap_summary(self, X_background, X_explain, nsamples_bg=100, nsamples=200, class_index=None):
        """
        Calcola e mostra il summary plot SHAP (globale) per la classe positiva.

        Gestisce output shap_values sia come:
        - list (una matrice per classe)
        - ndarray 3D (n_samples, n_features, n_classes)
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("Installa shap: pip install shap") from e

        if class_index is None:
            class_index = self.positive_class

        Xbg = self._to_dataframe(X_background)
        Xexp = self._to_dataframe(X_explain)

        # Riduci background per velocità
        if len(Xbg) > nsamples_bg:
            Xbg = Xbg.sample(nsamples_bg, random_state=42)

        def f(X_np):
            Xdf = pd.DataFrame(X_np, columns=Xbg.columns)
            return self.model.predict_proba(Xdf)

        explainer = shap.KernelExplainer(f, Xbg.values)

        shap_values = explainer.shap_values(Xexp.values, nsamples=nsamples)

        # ---- FIX compatibilità: list vs ndarray 3D ----
        if isinstance(shap_values, list):
            sv = shap_values[class_index]  # (n_samples, n_features)
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                sv = sv[:, :, class_index]  # (n_samples, n_features)
            elif sv.ndim == 2:
                # caso raro: già (n_samples, n_features)
                pass
            else:
                raise ValueError(f"Formato shap_values non gestito: shape={sv.shape}")

        # Controllo shape (debug utile)
        if sv.shape[1] != Xexp.shape[1]:
            raise ValueError(
                f"SHAP feature mismatch: sv ha {sv.shape[1]} feature, Xexp ne ha {Xexp.shape[1]}. "
                "Probabile mismatch colonne tra background e explain."
            )

        shap.summary_plot(sv, Xexp)
        return shap_values
