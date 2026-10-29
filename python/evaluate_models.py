import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------------------------------------------
# REPRODUCIBILITY SEED
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_splits(data_dir="data/processed"):
    """
    Load pre-split train, val, and test datasets.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing split file: {p}. Please run python/prepare_data.py first."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    target_col = "target_severity"
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset splits.")

    return train_df, val_df, test_df, target_col

def build_preprocessor(feature_df):
    """
    Build a ColumnTransformer that scales continuous features with StandardScaler
    and passes through binary/dummy encoded area features.
    """
    # Core continuous features to scale
    candidate_num = ['hour_sin', 'hour_cos', 'trailing_30d_density']
    num_cols = [c for c in candidate_num if c in feature_df.columns]

    # Categorical raw area column if present
    cat_cols = [c for c in ['Area', 'Location', 'District'] if c in feature_df.columns]

    transformers = []
    if num_cols:
        transformers.append(('num_scaler', StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(('cat_encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    return preprocessor

def get_feature_columns(df, target_col):
    """
    Select engineered feature columns, excluding raw text metadata and targets.
    """
    exclude_cols = {
        target_col, 'timestamp', 'Time', 'Date', 'Severity', 'Severity_Level',
        'Crime_Type', 'target_severity', 'Unnamed: 0'
    }
    features = [c for c in df.columns if c not in exclude_cols]
    return features

def evaluate_predictions(y_true, y_pred, y_prob=None, split_name="Test"):
    """
    Computes accuracy, balanced accuracy, macro F1, and per-class precision/recall.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    class_names = {0: "Low", 1: "Medium", 2: "High"}
    per_class = {}
    for c_id in [0, 1, 2]:
        str_id = str(c_id)
        if str_id in report:
            per_class[class_names[c_id]] = {
                "precision": round(report[str_id]["precision"], 4),
                "recall": round(report[str_id]["recall"], 4),
                "f1_score": round(report[str_id]["f1-score"], 4),
                "support": int(report[str_id]["support"])
            }
        else:
            per_class[class_names[c_id]] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class
    }

def plot_and_save_confusion_matrix(y_true, y_pred, output_path="results/confusion_matrix.png", model_name="Best Model"):
    """
    Save a normalized confusion matrix plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = [0, 1, 2]
    target_names = ["Low (0)", "Medium (1)", "High (2)"]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title=f"Normalized Confusion Matrix ({model_name})\nEvaluated on Held-out Test Set",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    fmt = '.2f'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Normalized confusion matrix saved to: {output_path}")

def run_evaluation_pipeline():
    print("=" * 70)
    print("STARTING AUDIT-RIGOROUS MODEL EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load chronologically split data
    train_df, val_df, test_df, target_col = load_splits()
    feature_cols = get_feature_columns(train_df, target_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols]
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    print(f"Features ({len(feature_cols)}): {feature_cols[:8]} ...")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # 2. Build Preprocessor Pipeline
    preprocessor = build_preprocessor(X_train)

    # -------------------------------------------------------------------------
    # 3. BASELINES AND CANDIDATE MODELS
    # -------------------------------------------------------------------------
    baselines = {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000
        )
    }

    # Candidates: Tune hyperparameters strictly on the VALIDATION set
    print("\n[Phase 1] Tuning Candidate Classifiers on Validation Set...")

    candidate_tuning = {
        "Decision Tree": [
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_SEED),
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_SEED),
            DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_SEED),
            DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=RANDOM_SEED)
        ],
        "K-Nearest Neighbors": [
            KNeighborsClassifier(n_neighbors=3),
            KNeighborsClassifier(n_neighbors=5),
            KNeighborsClassifier(n_neighbors=7),
            KNeighborsClassifier(n_neighbors=11)
        ],
        "Random Forest": [
            RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", random_state=RANDOM_SEED),
            RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED),
            RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=RANDOM_SEED)
        ]
    }

    best_candidates = {}
    val_performance_log = {}

    for name, variants in candidate_tuning.items():
        best_f1 = -1.0
        best_clf = variants[0]
        for variant in variants:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', variant)
            ])
            pipe.fit(X_train, y_train)
            val_preds = pipe.predict(X_val)
            val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_clf = variant

        best_candidates[name] = best_clf
        val_performance_log[name] = {
            "selected_params": str(best_clf),
            "val_macro_f1": round(best_f1, 4)
        }
        print(f"  * {name:20} -> Best Val Macro-F1: {best_f1:.4f} | Config: {best_clf}")

    # Build Voting Ensemble from best tuned candidates
    best_candidates["Voting Classifier (Soft)"] = VotingClassifier(
        estimators=[
            ('rf', best_candidates["Random Forest"]),
            ('dt', best_candidates["Decision Tree"]),
            ('knn', best_candidates["K-Nearest Neighbors"])
        ],
        voting='soft'
    )

    # -------------------------------------------------------------------------
    # 4. SELECT OVERALL BEST MODEL BASED ON VALIDATION SET
    # -------------------------------------------------------------------------
    all_models = {**baselines, **best_candidates}
    all_pipelines = {}
    val_results = {}

    print("\n[Phase 2] Evaluating All Models on Validation Set for Selection...")
    best_val_f1 = -1.0
    best_model_name = None

    for name, clf in all_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        all_pipelines[name] = pipe

        val_preds = pipe.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_preds, split_name="Validation")
        val_results[name] = val_metrics
        print(f"  {name:26} | Val Acc: {val_metrics['accuracy']:.4f} | Bal Acc: {val_metrics['balanced_accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # Selection criterion: Validation Macro-F1 (penalizes majority-class bias)
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_model_name = name

    print(f"\n[Validation Winner] '{best_model_name}' achieved highest Val Macro-F1 ({best_val_f1:.4f}).")

    # -------------------------------------------------------------------------
    # 5. SINGLE FINAL EVALUATION ON UNSEEN TEST SET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION (HELD-OUT 15%)")
    print("=" * 70)

    test_results = {}
    best_test_preds = None

    for name, pipe in all_pipelines.items():
        test_preds = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, test_preds, split_name="Test")
        test_results[name] = metrics

        if name == best_model_name:
            best_test_preds = test_preds

        print(f"  {name:26} | Test Acc: {metrics['accuracy']:.4f} | Bal Acc: {metrics['balanced_accuracy']:.4f} | Test Macro-F1: {metrics['macro_f1']:.4f}")

    # Plot and save normalized confusion matrix for the winning model
    plot_and_save_confusion_matrix(
        y_test,
        best_test_preds,
        output_path="results/confusion_matrix.png",
        model_name=best_model_name
    )

    # -------------------------------------------------------------------------
    # 6. PLAIN STATEMENT: COMPARISON TO MAJORITY-CLASS BASELINE
    # -------------------------------------------------------------------------
    majority_baseline = test_results["Dummy (Most Frequent)"]
    best_model_test = test_results[best_model_name]

    delta_f1 = best_model_test["macro_f1"] - majority_baseline["macro_f1"]
    delta_bal_acc = best_model_test["balanced_accuracy"] - majority_baseline["balanced_accuracy"]

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON STATEMENT")
    print("=" * 70)

    if delta_f1 >= 0.05:
        statement = (
            f"YES: The selected model ({best_model_name}) meaningfully beats the majority-class baseline. "
            f"It achieves a Macro-F1 of {best_model_test['macro_f1']:.4f} compared to the baseline's "
            f"{majority_baseline['macro_f1']:.4f} (an improvement of +{delta_f1:.4f}), and a Balanced Accuracy "
            f"of {best_model_test['balanced_accuracy']:.4f} vs {majority_baseline['balanced_accuracy']:.4f} (+{delta_bal_acc:.4f})."
        )
    elif delta_f1 > 0:
        statement = (
            f"MARGINAL: The selected model ({best_model_name}) achieves only a marginal gain over the majority-class baseline "
            f"(Macro-F1 {best_model_test['macro_f1']:.4f} vs baseline {majority_baseline['macro_f1']:.4f}, delta +{delta_f1:.4f}). "
            f"It cannot be claimed to substantially outperform the naive baseline."
        )
    else:
        statement = (
            f"NO: The selected model ({best_model_name}) does NOT beat the majority-class baseline on held-out test data. "
            f"Model Macro-F1: {best_model_test['macro_f1']:.4f} vs Baseline Macro-F1: {majority_baseline['macro_f1']:.4f} "
            f"(delta: {delta_f1:+.4f})."
        )

    print(statement)

    # -------------------------------------------------------------------------
    # 7. SAVE BEST PIPELINE AND METRICS JSON
    # -------------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib_path = "models/crime_pipeline.joblib"
    joblib.dump(all_pipelines[best_model_name], joblib_path)
    print(f"\n[Model Saved] Best pipeline exported to: {joblib_path}")

    full_output = {
        "validation_selection": {
            "winner": best_model_name,
            "val_macro_f1": best_val_f1,
            "all_val_metrics": val_results
        },
        "test_evaluation": test_results,
        "baseline_comparison_statement": statement,
        "best_model_per_class": best_model_test["per_class"],
        "metadata": {
            "random_seed": RANDOM_SEED,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "features_used": feature_cols
        }
    }

    metrics_json_path = "results/crime_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"[Metrics Saved] Full evaluation metrics written to: {metrics_json_path}")

    return full_output

if __name__ == "__main__":
    try:
        run_evaluation_pipeline()
    except FileNotFoundError as e:
        print(f"\n[Prerequisite Error] {e}")
        sys.exit(1)
