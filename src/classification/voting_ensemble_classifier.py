import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, accuracy_score, f1_score, classification_report, \
    ConfusionMatrixDisplay

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from score_PHM import get_classification_score

class VotingEnsembleClassifier:
    """
    Soft voting ensemble based on Random Forest, XGBoost, and LightGBM.
    Binary classification. If weights=None, they are estimated via cross-validation.
    """

    def __init__(self, weights=None, random_state=42,
                 auto_weight_metric="logloss", cv_splits=5):
        self.weights = weights
        self.random_state = random_state
        self.auto_weight_metric = auto_weight_metric
        self.cv_splits = cv_splits
        self.model = None
        self.fitted_weights_ = None
        self.classes_ = None
        self.base_models_ = {}

    def _build_base_models(self):
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
        )

        lgbm = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
        )

        return {"rf": rf, "xgb": xgb, "lgbm": lgbm}

    def _compute_auto_weights(self, X, y, models):
        skf = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        scores = {}
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError("This class is configured for binary classification only.")

        self.classes_ = classes

        for name, model in models.items():
            proba_oof = cross_val_predict(
                model,
                X,
                y,
                cv=skf,
                method="predict_proba",
                n_jobs=-1
            )

            if self.auto_weight_metric == "auc":
                score = roc_auc_score(y, proba_oof[:, 1])
            else:
                eps = 1e-15
                proba_oof = np.clip(proba_oof, eps, 1 - eps)
                ll = log_loss(y, proba_oof, labels=classes)
                score = 1.0 / ll

            scores[name] = score

        raw = np.array([scores["rf"], scores["xgb"], scores["lgbm"]], dtype=float)
        weights = raw / raw.sum()

        self.fitted_weights_ = {
            "rf": float(weights[0]),
            "xgb": float(weights[1]),
            "lgbm": float(weights[2]),
            "metric_used": self.auto_weight_metric
        }

        return weights

    def fit(self, X, y):
        models = self._build_base_models()

        if self.weights is None:
            self.weights = self._compute_auto_weights(X, y, models)

        self.base_models_ = {}
        for name, model in models.items():
            model.fit(X, y)
            self.base_models_[name] = model

        self.model = VotingClassifier(
            estimators=[
                ("rf", self.base_models_["rf"]),
                ("xgb", self.base_models_["xgb"]),
                ("lgbm", self.base_models_["lgbm"]),
            ],
            voting="soft",
            weights=self.weights,
            n_jobs=-1
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_weights(self):
        return self.weights

    def evaluate(self, X, y, plot_cm=True, evaluate_base=True):
        results = {}

        # =========================
        # ENSEMBLE
        # =========================
        print("\n==============================")
        print("ENSEMBLE (VotingClassifier)")
        print("==============================")

        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        cm = confusion_matrix(y, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 macro: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y, y_pred))

        if plot_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix - Ensemble")
            plt.show()

        results["ensemble"] = {
            "accuracy": acc,
            "f1_macro": f1,
            "confusion_matrix": cm,
        }

        # =========================
        # BASE MODELS
        # =========================
        if evaluate_base:
            results["base_models"] = {}

            for name, model in self.base_models_.items():
                print("\n==============================")
                print(f"BASE MODEL: {name.upper()}")
                print("==============================")

                y_pred_base = model.predict(X)

                acc_b = accuracy_score(y, y_pred_base)
                f1_b = f1_score(y, y_pred_base, average="macro")
                cm_b = confusion_matrix(y, y_pred_base)

                print(f"Accuracy: {acc_b:.4f}")
                print(f"F1 macro: {f1_b:.4f}")
                print("\nConfusion Matrix:")
                print(cm_b)
                print("\nClassification report:")
                print(classification_report(y, y_pred_base))

                if plot_cm:
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_b)
                    disp.plot(cmap="Blues")
                    plt.title(f"Confusion Matrix - {name.upper()}")
                    plt.show()

                results["base_models"][name] = {
                    "accuracy": acc_b,
                    "f1_macro": f1_b,
                    "confusion_matrix": cm_b,
                }

        return results

    def evaluate_phm_score(self, X, y_true):

        y_pred = self.predict(X)
        proba = self.model.predict_proba(X)
        confidence = np.max(proba, axis=1)

        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()

        if y_true_arr.size == 1 and y_pred_arr.size > 1:
            y_true_arr = np.full_like(y_pred_arr, fill_value=y_true_arr.item())

        if y_true_arr.size != y_pred_arr.size:
            raise ValueError(
                f"Size mismatch: y_true has {y_true_arr.size}, y_pred has {y_pred_arr.size}. "
                "Pass y_true with same number of samples as X."
            )

        scores_ens = [
            get_classification_score(int(t), int(p), float(c))
            for t, p, c in zip(y_true_arr, y_pred_arr, confidence)
        ]
        mean_score_ens = float(np.mean(scores_ens))

        results = {
            "ensemble_mean": mean_score_ens,
            "ensemble_scores": scores_ens,
            "base_models": {}
        }

        print(f"PHM score Ensemble (mean): {mean_score_ens:.4f}")

        for name, model in self.base_models_.items():
            y_pred_m = model.predict(X)
            proba_m = model.predict_proba(X)
            conf_m = np.max(proba_m, axis=1)

            scores_m = [
                get_classification_score(int(t), int(p), float(c))
                for t, p, c in zip(y_true_arr, y_pred_m, conf_m)
            ]
            mean_score_m = float(np.mean(scores_m))

            results["base_models"][name] = {
                "mean": mean_score_m,
                "scores": scores_m
            }

            print(f"PHM score {name.upper()} (mean): {mean_score_m:.4f}")

        return results