import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.9
N_BOOTSTRAP_DEFAULT = 1000


def reduce_redundant_features(df, feature_cols):
    """Identical rule to the main pipeline: for |Spearman r| > 0.9, drop the
    later-ordered feature and keep the earlier-ordered one."""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in feature_cols if X[c].notna().all()]
    X = X[valid_cols]
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    retained = [c for c in valid_cols if c not in dropped]
    return retained, dropped


def fit_cluster_labels(df, feature_cols, k, random_state):
    """Exact-pipeline preprocessing: standardize, then k-means directly
    (no PCA truncation -- see module docstring for why this is equivalent
    to the untruncated-PCA step actually used to produce the reported
    labels)."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    return model.fit_predict(X_scaled)


def best_label_mapping(reference_labels, bootstrap_labels, k):
    contingency = np.zeros((k, k), dtype=int)
    for r, b in zip(reference_labels, bootstrap_labels):
        contingency[int(r), int(b)] += 1
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return np.array([mapping[int(x)] for x in bootstrap_labels], dtype=int)


def jaccard_for_cluster(reference_labels, bootstrap_labels, cluster):
    ref = reference_labels == cluster
    boot = bootstrap_labels == cluster
    intersection = np.logical_and(ref, boot).sum()
    union = np.logical_or(ref, boot).sum()
    return np.nan if union == 0 else intersection / union


def bootstrap_stability(df, feature_cols, k, original_labels, n_bootstrap):
    rng = np.random.default_rng(RANDOM_STATE)
    per_cluster = {c: [] for c in range(k)}
    n = len(df)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        boot_labels = fit_cluster_labels(boot_df, feature_cols, k, RANDOM_STATE + b + 1)
        ref_labels = original_labels[idx]
        mapped = best_label_mapping(ref_labels, boot_labels, k)
        for c in range(k):
            score = jaccard_for_cluster(ref_labels, mapped, c)
            if not np.isnan(score):
                per_cluster[c].append(float(score))
    means = {c: float(np.mean(v)) if v else np.nan for c, v in per_cluster.items()}
    return means, per_cluster


def interpret(score):
    if score > 0.85:
        return "highly stable"
    if score >= 0.60:
        return "reasonably stable"
    return "unstable"


def main():
    parser = argparse.ArgumentParser(
        description="Exact-pipeline bootstrap stability (no PCA truncation)."
    )
    parser.add_argument("--features", type=Path, default=Path("results/radiomics_features.csv"))
    parser.add_argument("--labels", type=Path,
                         default=Path("results_v4_k2/v4_patients_with_phenotype_class.csv"),
                         help="CSV with patient_id,phenotype_class from the main pipeline run.")
    parser.add_argument("--out_dir", type=Path, default=Path("results_sensitivity"))
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)

    all_cols = [c for c in features.columns if c != "patient_id"]
    retained, dropped = reduce_redundant_features(features, all_cols)

    merged = features[["patient_id"] + retained].merge(
        labels[["patient_id", "phenotype_class"]], on="patient_id",
        how="inner", validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError("No patients matched between feature and label files.")

    original_labels = merged["phenotype_class"].to_numpy(dtype=int)
    classes = sorted(np.unique(original_labels).tolist())
    if classes != list(range(args.k)):
        raise ValueError(f"Labels contain classes {classes}, but --k {args.k} was requested.")

    print("Exact-pipeline bootstrap stability audit (Item A)")
    print(f"Patients: {len(merged)}")
    print(f"Raw radiomic features: {len(all_cols)}")
    print(f"Retained features: {len(retained)}")
    print(f"Dropped features: {len(dropped)} at |Spearman r| > {CORRELATION_THRESHOLD}")
    print("Preprocessing: standardization -> k-means directly")
    print("(equivalent to the untruncated-PCA step used for the reported labels;")
    print(" see module docstring. This removes the 95%-variance-PCA inconsistency.)")
    print(f"Clusters tested: k={args.k}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print()
    print("Phenotype class sizes:")
    for c in classes:
        print(f"  class {c}: {(original_labels == c).sum()}")

    means, scores = bootstrap_stability(merged, retained, args.k, original_labels, args.n_bootstrap)

    print()
    print("Cluster stability (mean Jaccard similarity across bootstraps):")
    for c in classes:
        n = int((original_labels == c).sum())
        print(f"  class {c} (n={n}): {means[c]:.3f} - {interpret(means[c])}")

    summary_path = args.out_dir / "exact_pipeline_bootstrap_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Exact-pipeline bootstrap stability audit (n={args.n_bootstrap}, k={args.k})\n")
        f.write("Preprocessing: Spearman redundancy filtering, standardization, k-means\n")
        f.write("(no PCA truncation -- identical preprocessing to the reported phenotype labels).\n")
        f.write(f"Patients: {len(merged)}\n")
        f.write(f"Raw features: {len(all_cols)}\n")
        f.write(f"Retained features: {len(retained)}\n\n")
        for c in classes:
            n = int((original_labels == c).sum())
            f.write(f"class {c} (n={n}): mean Jaccard = {means[c]:.3f} - {interpret(means[c])}\n")
        f.write("\nInterpretation: >0.85 highly stable, 0.60-0.85 reasonably stable, <0.60 unstable.\n")

    max_len = max(len(v) for v in scores.values())
    boot_df = pd.DataFrame({
        f"class_{c}_jaccard": scores[c] + [np.nan] * (max_len - len(scores[c]))
        for c in classes
    })
    boot_df.to_csv(args.out_dir / "exact_pipeline_bootstrap_scores.csv", index=False)

    print()
    print(f"Summary written to {summary_path}")
    print(f"Bootstrap scores written to {args.out_dir / 'exact_pipeline_bootstrap_scores.csv'}")


if __name__ == "__main__":
    main()
