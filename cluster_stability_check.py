#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

N_BOOTSTRAP = 200
CORRELATION_THRESHOLD = 0.9
PCA_VARIANCE = 0.95
RANDOM_STATE = 42

def reduce_redundant_features(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in feature_cols if X[c].notna().all()]
    X = X[valid_cols]
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    retained = [c for c in valid_cols if c not in dropped]
    return retained, dropped

def prepare_matrix(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    if X.isna().any().any():
        raise ValueError("NaN values remain after median imputation.")
    return X.to_numpy(dtype=float)

def fit_cluster_labels(df, feature_cols, k, random_state):
    X = prepare_matrix(df, feature_cols)
    X_scaled = StandardScaler().fit_transform(X)
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=random_state)
    X_pca_full = pca.fit_transform(X_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, PCA_VARIANCE) + 1)
    n_components = max(1, min(n_components, max_components))
    X_pca = X_pca_full[:, :n_components]
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    return model.fit_predict(X_pca)

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
        boot_labels = fit_cluster_labels(
            boot_df, feature_cols, k, RANDOM_STATE + b + 1
        )
        ref_labels = original_labels[idx]
        mapped = best_label_mapping(ref_labels, boot_labels, k)
        for c in range(k):
            score = jaccard_for_cluster(ref_labels, mapped, c)
            if not np.isnan(score):
                per_cluster[c].append(float(score))
    means = {c: float(np.mean(v)) if v else np.nan
             for c, v in per_cluster.items()}
    return means, per_cluster

def interpret(score):
    if score > 0.85:
        return "highly stable"
    if score >= 0.60:
        return "reasonably stable"
    return "unstable"

def main():
    parser = argparse.ArgumentParser(
        description="V4 compatible bootstrap stability audit."
    )
    parser.add_argument("--features", type=Path,
                        default=Path("results/radiomics_features.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_v4"))
    parser.add_argument("--k", type=int, required=True,
                        help="Cluster count to test; use --k 3 for V4.")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    label_path = args.out_dir / "v4_patients_with_phenotype_class.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"{label_path} not found.")
    labels = pd.read_csv(label_path)

    all_cols = [c for c in features.columns if c != "patient_id"]
    retained, dropped = reduce_redundant_features(features, all_cols)

    merged = features[["patient_id"] + retained].merge(
        labels[["patient_id", "phenotype_class"]],
        on="patient_id", how="inner", validate="one_to_one"
    )
    if len(merged) == 0:
        raise ValueError("No patients matched between feature and V4 label files.")

    original_labels = merged["phenotype_class"].to_numpy(dtype=int)
    classes = sorted(np.unique(original_labels).tolist())
    if classes != list(range(args.k)):
        raise ValueError(
            f"V4 labels contain classes {classes}, but --k {args.k} was requested."
        )

    print("V4 cluster stability audit")
    print(f"Patients: {len(merged)}")
    print(f"Raw radiomic features: {len(all_cols)}")
    print(f"Retained features: {len(retained)}")
    print(f"Dropped features: {len(dropped)} at |Spearman r| > {CORRELATION_THRESHOLD}")
    print("Clustering: standardization -> PCA -> K-means")
    print(f"Clusters tested: k={args.k}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print()
    print("V4 phenotype class sizes:")
    for c in classes:
        print(f"  class {c}: {(original_labels == c).sum()}")

    means, scores = bootstrap_stability(
        merged, retained, args.k, original_labels, args.n_bootstrap
    )

    print()
    print("Cluster stability (mean Jaccard similarity across bootstraps):")
    for c in classes:
        n = int((original_labels == c).sum())
        print(f"  class {c} (n={n}): {means[c]:.3f} - {interpret(means[c])}")

    summary = args.out_dir / "v4_cluster_stability_summary.txt"
    with open(summary, "w") as f:
        f.write(f"V4 bootstrap cluster stability audit (n={args.n_bootstrap}, k={args.k})\n")
        f.write("Preprocessing: Spearman redundancy filtering, standardization, PCA, K-means.\n")
        f.write(f"Patients: {len(merged)}\n")
        f.write(f"Raw features: {len(all_cols)}\n")
        f.write(f"Retained features: {len(retained)}\n\n")
        for c in classes:
            n = int((original_labels == c).sum())
            f.write(f"class {c} (n={n}): mean Jaccard = {means[c]:.3f} - {interpret(means[c])}\n")
        f.write("\nInterpretation: >0.85 highly stable, 0.60-0.85 reasonably stable, <0.60 unstable.\n")
        f.write("Minority clusters should be interpreted cautiously because small clusters are more sensitive to bootstrap resampling.\n")

    max_len = max(len(v) for v in scores.values())
    boot_df = pd.DataFrame({
        f"class_{c}_jaccard": scores[c] + [np.nan] * (max_len - len(scores[c]))
        for c in classes
    })
    boot_df.to_csv(args.out_dir / "v4_cluster_stability_bootstrap_scores.csv", index=False)

    print()
    print(f"Summary written to {summary}")
    print(f"Bootstrap scores written to {args.out_dir / 'v4_cluster_stability_bootstrap_scores.csv'}")

if __name__ == "__main__":
    main()
