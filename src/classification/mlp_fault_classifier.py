import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from score_PHM import get_classification_score


class MLPFaultClassifier:

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        learning_rate_init=0.001,
        learning_rate="adaptive",
        alpha=0.0001,
        max_iter=500,
        random_state=42
    ):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate_init,
                learning_rate=learning_rate,
                alpha=alpha,
                max_iter=max_iter,
                early_stopping=True,
                random_state=random_state
            ))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, plot_cm=True):
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 macro: {f1:.4f}")

        cm = confusion_matrix(y, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification report:")
        print(classification_report(y, y_pred))

        if plot_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix - MLP")
            plt.show()

        return {
            "accuracy": acc,
            "f1_macro": f1,
            "confusion_matrix": cm
        }

    def evaluate_phm_score(self, X, y_true):
        y_pred = self.predict(X)
        proba = self.model.predict_proba(X)

        confidence = np.max(proba, axis=1)

        scores = [get_classification_score(t, p, c) for t, p, c in zip(y_true, y_pred, confidence)]
        mean_score = float(np.mean(scores))

        print(f"PHM score (mean): {mean_score:.4f}")
        return mean_score, scores
