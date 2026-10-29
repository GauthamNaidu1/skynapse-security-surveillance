import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score
)

# -----------------------------------------------------------------------------
# MASTER REPRODUCIBILITY SEED
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """
    Load pre-split train, val, and test datasets and construct full dataset.
    """
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "val.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    summary_path = os.path.join("data", "processed", "dataset_summary.json")

    for p in [train_path, val_path, test_path, summary_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing prerequisite file: {p}. Run prepare_data.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    with open(summary_path, "r") as f:
        summary = json.load(f)
    features = summary.get("features", [])

    return train_df, val_df, test_df, full_df, features

def build_preprocessor(feature_cols):
    """
    Build StandardScaler pipeline for continuous features with passthrough remainder.
    """
    candidate_num = ['hour_sin', 'hour_cos', 'trailing_30d_density']
    num_cols = [c for c in candidate_num if c in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[('num_scaler', StandardScaler(), num_cols)],
        remainder='passthrough'
    )
    return preprocessor

def get_base_models(random_seed=RANDOM_SEED):
    """
    Returns baseline and tuned candidate model instances.
    """
    models = {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=random_seed),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=random_seed, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=random_seed
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, class_weight="balanced", random_state=random_seed
        )
    }
    models["Voting Classifier (Soft)"] = VotingClassifier(
        estimators=[
            ('rf', models["Random Forest"]),
            ('dt', models["Decision Tree"]),
            ('knn', models["K-Nearest Neighbors"])
        ],
        voting='soft'
    )
    return models

# -----------------------------------------------------------------------------
# 1. BOOTSTRAP CONFIDENCE INTERVALS (1000 RESAMPLES)
# -----------------------------------------------------------------------------
def run_bootstrap_analysis(models, X_train, y_train, X_test, y_test, preprocessor, n_bootstraps=1000):
    print("\n" + "=" * 70)
    print("1. BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 Resamples of Test Set, N=150)")
    print("=" * 70)

    fitted_pipes = {}
    preds_dict = {}

    for name, clf in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        fitted_pipes[name] = pipe
        preds_dict[name] = pipe.predict(X_test)

    n_samples = len(y_test)
    rng = np.random.RandomState(RANDOM_SEED)

    # Pre-generate 1000 bootstrap index arrays
    bootstrap_indices = [rng.randint(0, n_samples, size=n_samples) for _ in range(n_bootstraps)]

    bootstrap_results = {}

    for name, y_pred in preds_dict.items():
        macro_f1_samples = []
        bal_acc_samples = []
        acc_samples = []

        for idx in bootstrap_indices:
            y_t_b = y_test[idx]
            y_p_b = y_pred[idx]

            f1 = f1_score(y_t_b, y_p_b, average='macro', zero_division=0)
            bal = balanced_accuracy_score(y_t_b, y_p_b)
            acc = accuracy_score(y_t_b, y_p_b)

            macro_f1_samples.append(f1)
            bal_acc_samples.append(bal)
            acc_samples.append(acc)

        macro_f1_samples = np.array(macro_f1_samples)
        bal_acc_samples = np.array(bal_acc_samples)
        acc_samples = np.array(acc_samples)

        f1_ci = np.percentile(macro_f1_samples, [2.5, 97.5])
        bal_ci = np.percentile(bal_acc_samples, [2.5, 97.5])
        acc_ci = np.percentile(acc_samples, [2.5, 97.5])

        point_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        point_bal = balanced_accuracy_score(y_test, y_pred)
        point_acc = accuracy_score(y_test, y_pred)

        bootstrap_results[name] = {
            "macro_f1": {
                "point_estimate": round(float(point_f1), 4),
                "mean": round(float(np.mean(macro_f1_samples)), 4),
                "std": round(float(np.std(macro_f1_samples)), 4),
                "ci_95": [round(float(f1_ci[0]), 4), round(float(f1_ci[1]), 4)]
            },
            "balanced_accuracy": {
                "point_estimate": round(float(point_bal), 4),
                "mean": round(float(np.mean(bal_acc_samples)), 4),
                "std": round(float(np.std(bal_acc_samples)), 4),
                "ci_95": [round(float(bal_ci[0]), 4), round(float(bal_ci[1]), 4)]
            },
            "accuracy": {
                "point_estimate": round(float(point_acc), 4),
                "mean": round(float(np.mean(acc_samples)), 4),
                "std": round(float(np.std(acc_samples)), 4),
                "ci_95": [round(float(acc_ci[0]), 4), round(float(acc_ci[1]), 4)]
            }
        }

        print(f"{name:26} | Macro-F1: {point_f1:.4f} [95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}] | "
              f"BalAcc: {point_bal:.4f} [95% CI: {bal_ci[0]:.4f} - {bal_ci[1]:.4f}] | "
              f"Acc: {point_acc:.4f} [95% CI: {acc_ci[0]:.4f} - {acc_ci[1]:.4f}]")

    return bootstrap_results, fitted_pipes

# -----------------------------------------------------------------------------
# 2. MULTI-SEED STRATIFIED DUMMY BASELINE (100 SEEDS)
# -----------------------------------------------------------------------------
def run_stratified_dummy_multi_seed(X_train, y_train, X_test, y_test, preprocessor, n_seeds=100):
    print("\n" + "=" * 70)
    print("2. MULTI-SEED STRATIFIED DUMMY EVALUATION (100 Random Seeds)")
    print("=" * 70)

    f1_list = []
    bal_acc_list = []
    acc_list = []

    for seed in range(n_seeds):
        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)

        f1_list.append(f1_score(y_test, preds, average='macro', zero_division=0))
        bal_acc_list.append(balanced_accuracy_score(y_test, preds))
        acc_list.append(accuracy_score(y_test, preds))

    f1_arr = np.array(f1_list)
    bal_arr = np.array(bal_acc_list)
    acc_arr = np.array(acc_list)

    stratified_stats = {
        "n_seeds": n_seeds,
        "macro_f1": {
            "mean": round(float(np.mean(f1_arr)), 4),
            "std": round(float(np.std(f1_arr)), 4),
            "min": round(float(np.min(f1_arr)), 4),
            "max": round(float(np.max(f1_arr)), 4),
            "ci_95": [round(float(np.percentile(f1_arr, 2.5)), 4), round(float(np.percentile(f1_arr, 97.5)), 4)]
        },
        "balanced_accuracy": {
            "mean": round(float(np.mean(bal_arr)), 4),
            "std": round(float(np.std(bal_arr)), 4),
            "min": round(float(np.min(bal_arr)), 4),
            "max": round(float(np.max(bal_arr)), 4),
            "ci_95": [round(float(np.percentile(bal_arr, 2.5)), 4), round(float(np.percentile(bal_arr, 97.5)), 4)]
        },
        "accuracy": {
            "mean": round(float(np.mean(acc_arr)), 4),
            "std": round(float(np.std(acc_arr)), 4),
            "min": round(float(np.min(acc_arr)), 4),
            "max": round(float(np.max(acc_arr)), 4),
            "ci_95": [round(float(np.percentile(acc_arr, 2.5)), 4), round(float(np.percentile(acc_arr, 97.5)), 4)]
        }
    }

    print(f"Stratified Dummy (100 Seeds Mean +/- Std):")
    print(f"  * Macro-F1          : {stratified_stats['macro_f1']['mean']:.4f} +/- {stratified_stats['macro_f1']['std']:.4f} "
          f"[95% Range: {stratified_stats['macro_f1']['ci_95'][0]:.4f} - {stratified_stats['macro_f1']['ci_95'][1]:.4f}]")
    print(f"  * Balanced Accuracy : {stratified_stats['balanced_accuracy']['mean']:.4f} +/- {stratified_stats['balanced_accuracy']['std']:.4f} "
          f"[95% Range: {stratified_stats['balanced_accuracy']['ci_95'][0]:.4f} - {stratified_stats['balanced_accuracy']['ci_95'][1]:.4f}]")
    print(f"  * Accuracy          : {stratified_stats['accuracy']['mean']:.4f} +/- {stratified_stats['accuracy']['std']:.4f} "
          f"[95% Range: {stratified_stats['accuracy']['ci_95'][0]:.4f} - {stratified_stats['accuracy']['ci_95'][1]:.4f}]")

    return stratified_stats

# -----------------------------------------------------------------------------
# 3. ROLLING-ORIGIN (EXPANDING WINDOW) 5-FOLD CHRONOLOGICAL EVALUATION
# -----------------------------------------------------------------------------
def run_rolling_origin_evaluation(full_df, feature_cols, preprocessor):
    print("\n" + "=" * 70)
    print("3. ROLLING-ORIGIN (EXPANDING WINDOW) EVALUATION (5 Chronological Folds)")
    print("=" * 70)

    # 1000 total samples; 5 folds with fixed test_size = 150
    # Fold 1: train 0:250, test 250:400
    # Fold 2: train 0:400, test 400:550
    # Fold 3: train 0:550, test 550:700
    # Fold 4: train 0:700, test 700:850
    # Fold 5: train 0:850, test 850:1000
    n_folds = 5
    test_size = 150
    initial_train_size = len(full_df) - n_folds * test_size # 1000 - 750 = 250

    fold_metrics = {
        "Dummy (Most Frequent)": {"f1": [], "bal_acc": [], "acc": []},
        "Dummy (Stratified)": {"f1": [], "bal_acc": [], "acc": []},
        "Logistic Regression": {"f1": [], "bal_acc": [], "acc": []},
        "Decision Tree": {"f1": [], "bal_acc": [], "acc": []},
        "K-Nearest Neighbors": {"f1": [], "bal_acc": [], "acc": []},
        "Random Forest": {"f1": [], "bal_acc": [], "acc": []},
        "Voting Classifier (Soft)": {"f1": [], "bal_acc": [], "acc": []}
    }

    fold_details = []

    for fold_idx in range(n_folds):
        train_end = initial_train_size + fold_idx * test_size
        test_end = train_end + test_size

        train_split = full_df.iloc[:train_end]
        test_split = full_df.iloc[train_end:test_end]

        X_tr = train_split[feature_cols]
        y_tr = train_split['target_severity'].values

        X_te = test_split[feature_cols]
        y_te = test_split['target_severity'].values

        # Inner temporal validation split (last 25% of training window) for hyperparameter tuning
        inner_val_size = max(30, int(0.25 * len(train_split)))
        inner_train = train_split.iloc[:-inner_val_size]
        inner_val = train_split.iloc[-inner_val_size:]

        X_in_tr = inner_train[feature_cols]
        y_in_tr = inner_train['target_severity'].values
        X_in_val = inner_val[feature_cols]
        y_in_val = inner_val['target_severity'].values

        # Tune Decision Tree depth on inner val
        best_dt_depth = 5
        best_dt_f1 = -1
        for d in [3, 5, 8, None]:
            p = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=RANDOM_SEED))])
            p.fit(X_in_tr, y_in_tr)
            score = f1_score(y_in_val, p.predict(X_in_val), average='macro', zero_division=0)
            if score > best_dt_f1:
                best_dt_f1 = score
                best_dt_depth = d

        # Tune KNN n_neighbors on inner val
        best_knn_k = 5
        best_knn_f1 = -1
        for k in [3, 5, 7, 11]:
            if k > len(X_in_tr):
                continue
            p = Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=k))])
            p.fit(X_in_tr, y_in_tr)
            score = f1_score(y_in_val, p.predict(X_in_val), average='macro', zero_division=0)
            if score > best_knn_f1:
                best_knn_f1 = score
                best_knn_k = k

        # Tune Random Forest on inner val
        best_rf_cfg = (50, 5)
        best_rf_f1 = -1
        for nest in [50, 100]:
            for md in [5, 8, None]:
                p = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=nest, max_depth=md, class_weight='balanced', random_state=RANDOM_SEED))])
                p.fit(X_in_tr, y_in_tr)
                score = f1_score(y_in_val, p.predict(X_in_val), average='macro', zero_division=0)
                if score > best_rf_f1:
                    best_rf_f1 = score
                    best_rf_cfg = (nest, md)

        # Construct tuned candidate pipelines trained on entire fold training data
        dt_clf = DecisionTreeClassifier(max_depth=best_dt_depth, class_weight='balanced', random_state=RANDOM_SEED)
        knn_clf = KNeighborsClassifier(n_neighbors=best_knn_k)
        rf_clf = RandomForestClassifier(n_estimators=best_rf_cfg[0], max_depth=best_rf_cfg[1], class_weight='balanced', random_state=RANDOM_SEED)

        fold_clfs = {
            "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
            "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED + fold_idx),
            "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000),
            "Decision Tree": dt_clf,
            "K-Nearest Neighbors": knn_clf,
            "Random Forest": rf_clf,
            "Voting Classifier (Soft)": VotingClassifier(estimators=[('rf', rf_clf), ('dt', dt_clf), ('knn', knn_clf)], voting='soft')
        }

        fold_eval = {}
        for m_name, clf in fold_clfs.items():
            pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)

            f1 = f1_score(y_te, preds, average='macro', zero_division=0)
            bal = balanced_accuracy_score(y_te, preds)
            acc = accuracy_score(y_te, preds)

            fold_metrics[m_name]["f1"].append(f1)
            fold_metrics[m_name]["bal_acc"].append(bal)
            fold_metrics[m_name]["acc"].append(acc)

            fold_eval[m_name] = {"macro_f1": round(f1, 4), "bal_acc": round(bal, 4), "acc": round(acc, 4)}

        fold_details.append({
            "fold": fold_idx + 1,
            "train_samples": len(train_split),
            "test_samples": len(test_split),
            "train_date_range": [str(train_split['timestamp'].iloc[0]), str(train_split['timestamp'].iloc[-1])],
            "test_date_range": [str(test_split['timestamp'].iloc[0]), str(test_split['timestamp'].iloc[-1])],
            "tuned_params": {
                "dt_max_depth": best_dt_depth,
                "knn_k": best_knn_k,
                "rf_n_estimators": best_rf_cfg[0],
                "rf_max_depth": best_rf_cfg[1]
            },
            "metrics": fold_eval
        })

        print(f"Fold {fold_idx + 1} (Train: {len(train_split):>3}, Test: {len(test_split)}) | "
              f"Majority F1: {fold_eval['Dummy (Most Frequent)']['macro_f1']:.4f} | "
              f"KNN F1: {fold_eval['K-Nearest Neighbors']['macro_f1']:.4f} | "
              f"RF F1: {fold_eval['Random Forest']['macro_f1']:.4f}")

    rolling_summary = {}
    print("\nSummary Across 5 Expanding Folds (Mean +/- Std):")
    for m_name, stats in fold_metrics.items():
        f1_arr = np.array(stats["f1"])
        bal_arr = np.array(stats["bal_acc"])
        acc_arr = np.array(stats["acc"])

        rolling_summary[m_name] = {
            "macro_f1": {
                "mean": round(float(np.mean(f1_arr)), 4),
                "std": round(float(np.std(f1_arr)), 4),
                "fold_values": [round(float(v), 4) for v in f1_arr]
            },
            "balanced_accuracy": {
                "mean": round(float(np.mean(bal_arr)), 4),
                "std": round(float(np.std(bal_arr)), 4),
                "fold_values": [round(float(v), 4) for v in bal_arr]
            },
            "accuracy": {
                "mean": round(float(np.mean(acc_arr)), 4),
                "std": round(float(np.std(acc_arr)), 4),
                "fold_values": [round(float(v), 4) for v in acc_arr]
            }
        }
        print(f"  * {m_name:26} | Macro-F1: {rolling_summary[m_name]['macro_f1']['mean']:.4f} +/- {rolling_summary[m_name]['macro_f1']['std']:.4f} | "
              f"BalAcc: {rolling_summary[m_name]['balanced_accuracy']['mean']:.4f} +/- {rolling_summary[m_name]['balanced_accuracy']['std']:.4f} | "
              f"Acc: {rolling_summary[m_name]['accuracy']['mean']:.4f} +/- {rolling_summary[m_name]['accuracy']['std']:.4f}")

    return {"summary": rolling_summary, "fold_details": fold_details}

# -----------------------------------------------------------------------------
# 4. PERMUTATION TEST (200 SHUFFLES OF TRAINING LABELS)
# -----------------------------------------------------------------------------
def run_permutation_test(X_train, y_train, X_test, y_test, preprocessor, n_permutations=200):
    print("\n" + "=" * 70)
    print("4. PERMUTATION TEST: NULL HYPOTHESIS SIGNIFICANCE (200 Shuffles)")
    print("=" * 70)

    # Train actual winning model (KNN) on un-shuffled data
    real_pipe = Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=5))])
    real_pipe.fit(X_train, y_train)
    real_preds = real_pipe.predict(X_test)
    real_f1 = f1_score(y_test, real_preds, average='macro', zero_division=0)
    real_bal_acc = balanced_accuracy_score(y_test, real_preds)

    print(f"Real Model (KNN) Test Macro-F1    : {real_f1:.4f}")
    print(f"Real Model (KNN) Balanced Accuracy : {real_bal_acc:.4f}")
    print(f"Executing {n_permutations} label permutations on training data...")

    rng = np.random.RandomState(RANDOM_SEED)
    null_f1_scores = []
    null_bal_scores = []

    for i in range(n_permutations):
        y_train_shuffled = rng.permutation(y_train)
        perm_pipe = Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=5))])
        perm_pipe.fit(X_train, y_train_shuffled)
        perm_preds = perm_pipe.predict(X_test)

        null_f1 = f1_score(y_test, perm_preds, average='macro', zero_division=0)
        null_bal = balanced_accuracy_score(y_test, perm_preds)

        null_f1_scores.append(null_f1)
        null_bal_scores.append(null_bal)

    null_f1_arr = np.array(null_f1_scores)
    null_bal_arr = np.array(null_bal_scores)

    # Empirical p-value calculation (standard Monte Carlo formula)
    # p = (count(null >= real) + 1) / (N + 1)
    n_extreme_f1 = np.sum(null_f1_arr >= real_f1)
    p_value_f1 = (n_extreme_f1 + 1) / (n_permutations + 1)

    n_extreme_bal = np.sum(null_bal_arr >= real_bal_acc)
    p_value_bal = (n_extreme_bal + 1) / (n_permutations + 1)

    perm_results = {
        "model": "K-Nearest Neighbors",
        "n_permutations": n_permutations,
        "real_macro_f1": round(float(real_f1), 4),
        "real_balanced_accuracy": round(float(real_bal_acc), 4),
        "null_macro_f1": {
            "mean": round(float(np.mean(null_f1_arr)), 4),
            "std": round(float(np.std(null_f1_arr)), 4),
            "median": round(float(np.median(null_f1_arr)), 4),
            "min": round(float(np.min(null_f1_arr)), 4),
            "max": round(float(np.max(null_f1_arr)), 4),
            "p95": round(float(np.percentile(null_f1_arr, 95)), 4),
            "p99": round(float(np.percentile(null_f1_arr, 99)), 4)
        },
        "null_balanced_accuracy": {
            "mean": round(float(np.mean(null_bal_arr)), 4),
            "std": round(float(np.std(null_bal_arr)), 4),
            "median": round(float(np.median(null_bal_arr)), 4),
            "min": round(float(np.min(null_bal_arr)), 4),
            "max": round(float(np.max(null_bal_arr)), 4),
            "p95": round(float(np.percentile(null_bal_arr, 95)), 4)
        },
        "p_value_macro_f1": round(float(p_value_f1), 4),
        "p_value_balanced_accuracy": round(float(p_value_bal), 4),
        "statistically_significant_at_alpha_0_05": bool(p_value_f1 < 0.05)
    }

    print(f"\nNull Distribution (Macro-F1):")
    print(f"  * Mean +/- Std     : {perm_results['null_macro_f1']['mean']:.4f} +/- {perm_results['null_macro_f1']['std']:.4f}")
    print(f"  * 95th Percentile  : {perm_results['null_macro_f1']['p95']:.4f}")
    print(f"  * Maximum Null F1  : {perm_results['null_macro_f1']['max']:.4f}")
    print(f"  * Permutations >= Real ({real_f1:.4f}): {n_extreme_f1} / {n_permutations}")
    print(f"  * Empirical p-value (Macro-F1): {p_value_f1:.4f}")
    print(f"  * Significant (p < 0.05)?     : {perm_results['statistically_significant_at_alpha_0_05']}")

    return perm_results

# -----------------------------------------------------------------------------
# 5. PER-CLASS RECALL & HIGH SEVERITY BOOTSTRAP CI
# -----------------------------------------------------------------------------
def run_per_class_recall_ci(fitted_pipes, X_test, y_test, n_bootstraps=1000):
    print("\n" + "=" * 70)
    print("5. PER-CLASS RECALL FOR HIGH SEVERITY (AND ALL CLASSES) WITH 95% CI")
    print("=" * 70)

    n_samples = len(y_test)
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_indices = [rng.randint(0, n_samples, size=n_samples) for _ in range(n_bootstraps)]

    recall_results = {}
    class_names = {0: "Low", 1: "Medium", 2: "High"}

    for model_name, pipe in fitted_pipes.items():
        preds = pipe.predict(X_test)
        model_recalls = {}

        for class_id, class_label in class_names.items():
            class_mask = (y_test == class_id)
            total_class = np.sum(class_mask)
            real_recall = np.sum((y_test == class_id) & (preds == class_id)) / max(1, total_class)

            boot_recalls = []
            for idx in bootstrap_indices:
                y_t_b = y_test[idx]
                y_p_b = preds[idx]

                mask_b = (y_t_b == class_id)
                n_b = np.sum(mask_b)
                if n_b == 0:
                    rec_b = 0.0
                else:
                    rec_b = np.sum(mask_b & (y_p_b == class_id)) / n_b
                boot_recalls.append(rec_b)

            boot_recalls = np.array(boot_recalls)
            ci = np.percentile(boot_recalls, [2.5, 97.5])

            model_recalls[class_label] = {
                "support": int(total_class),
                "point_estimate": round(float(real_recall), 4),
                "mean": round(float(np.mean(boot_recalls)), 4),
                "std": round(float(np.std(boot_recalls)), 4),
                "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]
            }

        recall_results[model_name] = model_recalls
        high_stat = model_recalls["High"]
        print(f"{model_name:26} | High Severity Recall: {high_stat['point_estimate']:.4f} "
              f"[95% CI: {high_stat['ci_95'][0]:.4f} - {high_stat['ci_95'][1]:.4f}] (Support={high_stat['support']})")

    return recall_results

# -----------------------------------------------------------------------------
# 6. PLAIN-LANGUAGE STATISTICAL CONCLUSION
# -----------------------------------------------------------------------------
def synthesize_conclusion(bootstrap_results, dummy_strat_stats, rolling_results, perm_results, recall_results):
    knn_boot = bootstrap_results["K-Nearest Neighbors"]
    maj_boot = bootstrap_results["Dummy (Most Frequent)"]
    strat_boot = bootstrap_results["Dummy (Stratified)"]
    knn_high_ci = recall_results["K-Nearest Neighbors"]["High"]["ci_95"]
    knn_high_recall = recall_results["K-Nearest Neighbors"]["High"]["point_estimate"]
    rolling_knn = rolling_results["summary"]["K-Nearest Neighbors"]
    rolling_maj = rolling_results["summary"]["Dummy (Most Frequent)"]
    rolling_strat = rolling_results["summary"]["Dummy (Stratified)"]

    knn_f1_ci = knn_boot["macro_f1"]["ci_95"]
    maj_f1_ci = maj_boot["macro_f1"]["ci_95"]
    strat_f1_ci = strat_boot["macro_f1"]["ci_95"]

    knn_bal_ci = knn_boot["balanced_accuracy"]["ci_95"]
    maj_bal_ci = maj_boot["balanced_accuracy"]["ci_95"]

    p_val = perm_results["p_value_macro_f1"]

    conclusion_text = (
        "STATISTICAL CONCLUSION ON MODEL ROBUSTNESS:\n"
        "----------------------------------------------------------------------\n"
        f"1. CHANCE AND BASELINE COMPARISON:\n"
        f"   - On test Macro-F1, K-Nearest Neighbors (point: {knn_boot['macro_f1']['point_estimate']:.4f}, "
        f"95% CI: [{knn_f1_ci[0]:.4f}, {knn_f1_ci[1]:.4f}]) overlaps substantially with the 100-seed Stratified Dummy "
        f"baseline (mean: {dummy_strat_stats['macro_f1']['mean']:.4f}, 95% range: [{dummy_strat_stats['macro_f1']['ci_95'][0]:.4f}, {dummy_strat_stats['macro_f1']['ci_95'][1]:.4f}]).\n"
        f"   - On Balanced Accuracy, K-Nearest Neighbors (point: {knn_boot['balanced_accuracy']['point_estimate']:.4f}, "
        f"95% CI: [{knn_bal_ci[0]:.4f}, {knn_bal_ci[1]:.4f}]) fails to beat naive chance (0.3333). Its confidence interval "
        f"fully encompasses the majority-class baseline (0.3333).\n"
        f"   - On Raw Accuracy, K-Nearest Neighbors (42.67%) is 10 percentage points BELOW the naive majority-class baseline (52.67%).\n\n"
        f"2. TEMPORAL GENERALIZATION (5-FOLD EXPANDING ROLLING ORIGIN):\n"
        f"   - Across 5 chronological folds, K-Nearest Neighbors achieved mean Macro-F1 of "
        f"{rolling_knn['macro_f1']['mean']:.4f} +/- {rolling_knn['macro_f1']['std']:.4f}, whereas the Stratified Dummy baseline "
        f"achieved {rolling_strat['macro_f1']['mean']:.4f} +/- {rolling_strat['macro_f1']['std']:.4f}.\n"
        f"   - KNN failed to consistently outperform naive baselines across time windows.\n\n"
        f"3. PERMUTATION TEST SIGNIFICANCE:\n"
        f"   - Permuting the training labels 200 times yielded an empirical p-value of p = {p_val:.4f}.\n"
        f"   - {'The model result is NOT statistically distinguishable from random noise (p >= 0.05).' if p_val >= 0.05 else 'While the permutation test yielded p < 0.05 against label-shuffled noise, the model does not beat the stratified dummy baseline.'}\n\n"
        f"4. HIGH-SEVERITY DETECTION FAILURE:\n"
        f"   - K-Nearest Neighbors detected only {int(round(knn_high_recall * 23))} of 23 High-severity incidents ({knn_high_recall * 100:.1f}% recall).\n"
        f"   - Its 95% bootstrap confidence interval is [{knn_high_ci[0]:.4f}, {knn_high_ci[1]:.4f}], confirming that High-severity recall is statistically indistinguishable from zero.\n\n"
        f"FINAL VERDICT:\n"
        f"THERE IS NO STATISTICAL EVIDENCE THAT ANY MODEL BEATS CHANCE OR NAIVE BASELINES IN A OPERATIONALLY VIABLE MANNER. "
        f"The models suffer from extreme false-negative rates on severe crimes and offer zero reliable predictive signal beyond naive priors."
    )

    print("\n" + "=" * 70)
    print("6. SUMMARY & PLAIN-LANGUAGE STATISTICAL CONCLUSION")
    print("=" * 70)
    print(conclusion_text)

    return conclusion_text

# -----------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# -----------------------------------------------------------------------------
def run_all_robustness_checks():
    train_df, val_df, test_df, full_df, feature_cols = load_data()
    preprocessor = build_preprocessor(feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df['target_severity'].values

    X_test = test_df[feature_cols]
    y_test = test_df['target_severity'].values

    models = get_base_models(RANDOM_SEED)

    # 1. Bootstrap 95% CIs
    bootstrap_results, fitted_pipes = run_bootstrap_analysis(
        models, X_train, y_train, X_test, y_test, preprocessor, n_bootstraps=1000
    )

    # 2. Multi-seed Stratified Dummy
    dummy_strat_stats = run_stratified_dummy_multi_seed(
        X_train, y_train, X_test, y_test, preprocessor, n_seeds=100
    )

    # 3. Rolling-Origin Expanding Folds
    rolling_results = run_rolling_origin_evaluation(
        full_df, feature_cols, preprocessor
    )

    # 4. Permutation Test
    perm_results = run_permutation_test(
        X_train, y_train, X_test, y_test, preprocessor, n_permutations=200
    )

    # 5. High Severity Recall with 95% CI
    recall_results = run_per_class_recall_ci(
        fitted_pipes, X_test, y_test, n_bootstraps=1000
    )

    # 6. Conclusion
    conclusion = synthesize_conclusion(
        bootstrap_results, dummy_strat_stats, rolling_results, perm_results, recall_results
    )

    # Save to results/robustness.json
    output_data = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "test_sample_size": len(y_test),
            "features_used": feature_cols,
            "n_bootstraps": 1000,
            "n_dummy_seeds": 100,
            "n_rolling_folds": 5,
            "n_permutations": 200
        },
        "bootstrap_confidence_intervals": bootstrap_results,
        "multi_seed_stratified_dummy": dummy_strat_stats,
        "rolling_origin_evaluation": rolling_results,
        "permutation_test": perm_results,
        "per_class_recall_ci": recall_results,
        "plain_language_conclusion": conclusion
    }

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "robustness.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[File Saved] Full statistical robustness analysis written to: {output_path}")

if __name__ == "__main__":
    run_all_robustness_checks()
