import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Ambil tracking URI dari GitHub Secrets / Environment
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

DATASET_PATH = "heart_disease_preprocessing.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2
    )

    return parser.parse_args()


def save_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Tidak Sakit", "Sakit"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():

    args = parse_args()

    # Dataset
    df = pd.read_csv(DATASET_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MLflow Project sudah membuat run otomatis
    active_run = mlflow.active_run()

    if active_run:
        print(
            f"Using MLflow Run ID: {active_run.info.run_id}"
        )

    # Logging parameter
    mlflow.log_param(
        "n_estimators",
        args.n_estimators
    )

    mlflow.log_param(
        "max_depth",
        args.max_depth
    )

    mlflow.log_param(
        "min_samples_split",
        args.min_samples_split
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", auc)

    # Artifact
    save_confusion_matrix(
        y_test,
        y_pred,
        "confusion_matrix.png"
    )

    mlflow.log_artifact(
        "confusion_matrix.png"
    )

    # Model Artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        conda_env="conda.yaml"
    )

    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"ROC AUC   : {auc:.4f}")

    print("Training selesai!")


if __name__ == "__main__":
    main()