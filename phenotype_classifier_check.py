import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, permutation_test_score
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5
RANDOM_STATE = 42
TOP_N_FEATURES_TO_PLOT = 15
N_PERMUTATIONS = 200


def load_data(path: Path):
    df = pd.read_csv(path)
    feature_cols = [
        c for c in df.columns
        if c not in ("patient_id", "phenotype_class")
    ]
    X = df[feature_cols].values
    y = df["phenotype_class"].values
    return df, X, y, feature_cols


def run_cross_validated_classifier(X, y, n_classes: int):
    X_scaled = StandardScaler().fit_transform(X)

    n_splits = min(N_FOLDS, pd.Series(y).value_counts().min())
    if n_splits < 2:
        raise ValueError(
            f"Smallest phenotype class has fewer than 2 members - cannot run "
            f"cross-validation. Class counts: {pd.Series(y).value_counts().to_dict()}"
        )
    if n_splits < N_FOLDS:
        print(f"[warning] smallest class only supports {n_splits}-fold CV "
              f"(requested {N_FOLDS}); reducing fold count accordingly. "
              f"This is a direct consequence of small/imbalanced class sizes - "
              f"flag this explicitly if reporting these numbers.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")

    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)

    balanced_acc = balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # Fit on full data for feature importances (not for the reported CV metrics)
    clf.fit(X_scaled, y)

    return {
        "n_splits_used": n_splits,
        "balanced_accuracy": balanced_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "fitted_model": clf,
        "X_scaled": X_scaled,
        "cv": cv,
    }


def run_permutation_test(clf, X_scaled, y, cv):
    """
    Answers: is the observed classifier performance actually better than what
    you'd get from randomly-shuffled labels? Runs N_PERMUTATIONS shuffles and
    reports a p-value - the fraction of shuffled runs that matched or beat
    the real score. With small/imbalanced data this is the honest way to
    check "above chance" rather than eyeballing a single accuracy number.
    """
    balanced_scorer = make_scorer(balanced_accuracy_score)
    score, permutation_scores, p_value = permutation_test_score(
        clf, X_scaled, y, cv=cv, scoring=balanced_scorer,
        n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return {
        "true_score": score,
        "permutation_scores": permutation_scores,
        "p_value": p_value,
    }


def run_logistic_regression_comparison(X_scaled, y, cv):
    """
    Comparison classifier: if random forest and logistic regression agree on
    roughly how separable the classes are, that's more convincing than a
    single model's number - a single classifier's accuracy can reflect that
    model's own quirks rather than genuine signal in the data.
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring=make_scorer(balanced_accuracy_score))
    return {"balanced_accuracy_mean": scores.mean(), "balanced_accuracy_std": scores.std()}


def plot_permutation_histogram(perm_results: dict, out_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(perm_results["permutation_scores"], bins=30, alpha=0.7, label="Shuffled-label scores")
    plt.axvline(perm_results["true_score"], color="red", linestyle="--",
                label=f"Observed score ({perm_results['true_score']:.3f})")
    plt.xlabel("Balanced accuracy")
    plt.ylabel("Count")
    plt.title(f"Permutation test (n={N_PERMUTATIONS}), p = {perm_results['p_value']:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "permutation_test.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, out_dir: Path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted phenotype class")
    plt.ylabel("True phenotype class (from clustering)")
    plt.title("Cross-validated classifier confusion matrix\n(robustness check, not a clinical model)")
    plt.tight_layout()
    plt.savefig(out_dir / "phenotype_classifier_confusion_matrix.png", dpi=200)
    plt.close()


def plot_feature_importances(clf, feature_cols, out_dir: Path):
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=False).head(TOP_N_FEATURES_TO_PLOT)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=top.values, y=top.index, color="steelblue")
    plt.xlabel("Feature importance")
    plt.title(f"Top {TOP_N_FEATURES_TO_PLOT} radiomic features\ndriving phenotype separation")
    plt.tight_layout()
    plt.savefig(out_dir / "phenotype_feature_importances.png", dpi=200)
    plt.close()

    top.to_csv(out_dir / "top_feature_importances.csv", header=["importance"])
    return top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_with_labels", type=Path,
                         default=Path("../results/patients_with_phenotype_class.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("../results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y, feature_cols = load_data(args.features_with_labels)
    n_classes = len(set(y))

    print(f"Loaded {len(df)} patients, {len(feature_cols)} features, {n_classes} phenotype classes")
    print(f"Class sizes: {pd.Series(y).value_counts().sort_index().to_dict()}")

    results = run_cross_validated_classifier(X, y, n_classes)

    print(f"\n{results['n_splits_used']}-fold cross-validated results:")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.3f}")
    print("\nPer-class classification report:")
    print(results["classification_report"])

    plot_confusion_matrix(results["confusion_matrix"], args.out_dir)
    top_features = plot_feature_importances(results["fitted_model"], feature_cols, args.out_dir)

    print(f"\nTop {TOP_N_FEATURES_TO_PLOT} features driving phenotype separation:")
    print(top_features.to_string())

    print(f"\nRunning permutation test ({N_PERMUTATIONS} shuffles - this may take a few minutes)...")
    perm_results = run_permutation_test(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"),
        results["X_scaled"], y, results["cv"],
    )
    print(f"Permutation test: observed balanced accuracy = {perm_results['true_score']:.3f}, "
          f"p = {perm_results['p_value']:.4f} (fraction of {N_PERMUTATIONS} label-shuffles "
          f"matching or beating the real score)")
    plot_permutation_histogram(perm_results, args.out_dir)

    logreg_results = run_logistic_regression_comparison(results["X_scaled"], y, results["cv"])
    print(f"\nLogistic regression comparison: balanced accuracy = "
          f"{logreg_results['balanced_accuracy_mean']:.3f} +/- {logreg_results['balanced_accuracy_std']:.3f} "
          f"(random forest was {results['balanced_accuracy']:.3f})")

    # Save a small summary for the manuscript
    with open(args.out_dir / "classifier_robustness_summary.txt", "w") as f:
        f.write(f"Cross-validated ({results['n_splits_used']}-fold) random forest phenotype classifier\n")
        f.write(f"Balanced accuracy: {results['balanced_accuracy']:.3f}\n")
        f.write(f"Permutation test p-value: {perm_results['p_value']:.4f} (n={N_PERMUTATIONS} shuffles)\n")
        f.write(f"Logistic regression comparison: {logreg_results['balanced_accuracy_mean']:.3f} "
                f"+/- {logreg_results['balanced_accuracy_std']:.3f}\n\n")
        f.write("Per-class report (random forest):\n")
        f.write(results["classification_report"])
        f.write("\nNOTE: this validates that phenotype clusters are learnable from the\n")
        f.write("radiomic feature space (technical soundness check). It is NOT a\n")
        f.write("clinical predictive model and makes no claim about patient outcomes.\n")
        f.write("The permutation p-value indicates whether performance exceeds chance\n")
        f.write("given this specific small/imbalanced sample - a non-significant p-value\n")
        f.write("here would mean the phenotype classes are not yet demonstrably learnable\n")
        f.write("from radiomics alone, which is itself a valid, reportable finding.\n")

    print(f"\nSummary written to {args.out_dir / 'classifier_robustness_summary.txt'}")


if __name__ == "__main__":
    main()
