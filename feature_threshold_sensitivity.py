import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

RANDOM_STATE = 42
K_CANDIDATES = range(2, 6)


def reduce_redundant_features(df, feature_cols, threshold):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in feature_cols if X[c].notna().all()]
    X = X[valid_cols]
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > threshold)]
    retained = [c for c in valid_cols if c not in dropped]
    return retained, dropped


def cluster_all_k(X_scaled, random_state):
    """Return {k: (labels, silhouette)} for k in K_CANDIDATES."""
    out = {}
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        out[k] = (labels, score)
    return out


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


def bootstrap_jaccard_k2(X_full, k2_labels, n_bootstrap, random_state):
    rng = np.random.default_rng(random_state)
    n = X_full.shape[0]
    per_cluster = {0: [], 1: []}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_boot = StandardScaler().fit_transform(X_full[idx])
        model = KMeans(n_clusters=2, random_state=random_state + b + 1, n_init=20)
        boot_labels = model.fit_predict(X_boot)
        ref_labels = k2_labels[idx]
        mapped = best_label_mapping(ref_labels, boot_labels, 2)
        for c in (0, 1):
            score = jaccard_for_cluster(ref_labels, mapped, c)
            if not np.isnan(score):
                per_cluster[c].append(float(score))
    return {c: float(np.mean(v)) if v else np.nan for c, v in per_cluster.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Feature-redundancy threshold sensitivity analysis (Item D)."
    )
    parser.add_argument("--features", type=Path, default=Path("results/radiomics_features.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_sensitivity"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--reference_threshold", type=float, default=0.90,
                         help="Threshold whose k=2 labels are used as the reference for "
                              "ARI/NMI comparison (default 0.90, matching the reported analysis).")
    parser.add_argument("--n_bootstrap", type=int, default=200)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    all_cols = [c for c in features.columns if c != "patient_id"]

    print("Feature-redundancy threshold sensitivity analysis (Item D)")
    print(f"Patients: {len(features)}")
    print(f"Raw radiomic features: {len(all_cols)}")
    print(f"Thresholds tested: {args.thresholds}")
    print(f"Reference threshold: {args.reference_threshold}")
    print(f"Bootstrap resamples per threshold: {args.n_bootstrap}")
    print()

    # First pass: get k=2 labels at the reference threshold.
    ref_retained, _ = reduce_redundant_features(features, all_cols, args.reference_threshold)
    ref_X = features[ref_retained].apply(pd.to_numeric, errors="coerce")
    ref_X = ref_X.fillna(ref_X.median(numeric_only=True)).to_numpy(dtype=float)
    ref_X_scaled = StandardScaler().fit_transform(ref_X)
    ref_model = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)
    reference_k2_labels = ref_model.fit_predict(ref_X_scaled)

    rows = []
    for threshold in args.thresholds:
        retained, dropped = reduce_redundant_features(features, all_cols, threshold)
        X = features[retained].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
        X_scaled = StandardScaler().fit_transform(X)

        results_by_k = cluster_all_k(X_scaled, RANDOM_STATE)
        best_k = max(results_by_k, key=lambda k: (results_by_k[k][1] if not np.isnan(results_by_k[k][1]) else -np.inf))
        best_silhouette = results_by_k[best_k][1]

        k2_labels, k2_silhouette = results_by_k[2]

        # Align this threshold's k=2 labels to the reference before ARI/NMI
        # (ARI/NMI are permutation-invariant to label naming already, so no
        # explicit relabeling is required, but we report the raw scores).
        ari = adjusted_rand_score(reference_k2_labels, k2_labels)
        nmi = normalized_mutual_info_score(reference_k2_labels, k2_labels)

        sizes = pd.Series(k2_labels).value_counts().sort_index().to_dict()

        jaccard = bootstrap_jaccard_k2(X, k2_labels, args.n_bootstrap, RANDOM_STATE)

        print(f"Threshold {threshold:.2f}: {len(retained)} features retained, "
              f"best k={best_k} (S={best_silhouette:.3f}), "
              f"k=2 S={k2_silhouette:.3f}, sizes={sizes}, "
              f"ARI vs ref={ari:.3f}, NMI vs ref={nmi:.3f}, "
              f"Jaccard(0/1)={jaccard.get(0, np.nan):.3f}/{jaccard.get(1, np.nan):.3f}")

        rows.append({
            "threshold": threshold,
            "n_features_retained": len(retained),
            "best_k": best_k,
            "silhouette_best_k": best_silhouette,
            "silhouette_k2": k2_silhouette,
            "phenotype0_size": sizes.get(0, np.nan),
            "phenotype1_size": sizes.get(1, np.nan),
            "ARI_vs_reference": ari,
            "NMI_vs_reference": nmi,
            "bootstrap_jaccard_class0": jaccard.get(0, np.nan),
            "bootstrap_jaccard_class1": jaccard.get(1, np.nan),
        })

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "feature_threshold_sensitivity.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Summary table written to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
