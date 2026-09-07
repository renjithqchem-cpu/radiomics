import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.9
K_CANDIDATES = range(2, 6)
N_PERMUTATIONS_DEFAULT = 1000


def reduce_redundant_features(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in feature_cols if X[c].notna().all()]
    X = X[valid_cols]
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    retained = [c for c in valid_cols if c not in dropped]
    return retained, dropped


def max_silhouette_over_k(X_scaled, random_state):
    """Return the maximum silhouette coefficient over k=2..5 for a given
    (already standardized) feature matrix."""
    best = -np.inf
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best:
            best = score
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Null permutation test for clustering structure (Item B)."
    )
    parser.add_argument("--features", type=Path, default=Path("results/radiomics_features.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_sensitivity"))
    parser.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS_DEFAULT)
    parser.add_argument("--observed_max_silhouette", type=float, default=0.450,
                         help="Observed max silhouette over k=2-5 on the real, unpermuted data "
                              "(default 0.450, the reported k=3 value; pass your own value if it "
                              "differs when you re-run the real pipeline).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    all_cols = [c for c in features.columns if c != "patient_id"]
    retained, dropped = reduce_redundant_features(features, all_cols)

    X = features[retained].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)

    print("Clustering null permutation test (Item B)")
    print(f"Patients: {X.shape[0]}")
    print(f"Retained features (unpermuted, used as the permutation pool): {X.shape[1]}")
    print(f"Permutations: {args.n_permutations}")
    print(f"k values tested per permutation: {list(K_CANDIDATES)}")
    print(f"Observed max silhouette (real data): {args.observed_max_silhouette:.3f}")
    print()

    rng = np.random.default_rng(RANDOM_STATE)
    null_scores = np.empty(args.n_permutations)
    n = X.shape[0]

    for i in range(args.n_permutations):
        X_perm = X.copy()
        for j in range(X_perm.shape[1]):
            X_perm[:, j] = X_perm[rng.permutation(n), j]
        X_perm_scaled = StandardScaler().fit_transform(X_perm)
        null_scores[i] = max_silhouette_over_k(X_perm_scaled, RANDOM_STATE + i + 1)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.n_permutations} permutations complete")

    p_value = float(np.mean(null_scores >= args.observed_max_silhouette))

    print()
    print(f"Null distribution: mean={null_scores.mean():.3f}, sd={null_scores.std():.3f}, "
          f"max={null_scores.max():.3f}, min={null_scores.min():.3f}")
    print(f"Empirical p-value (fraction of null >= observed): p = {p_value:.4f}")

    pd.DataFrame({"null_max_silhouette": null_scores}).to_csv(
        args.out_dir / "cluster_null_permutation_scores.csv", index=False
    )

    with open(args.out_dir / "cluster_null_permutation_summary.txt", "w") as f:
        f.write("Clustering null permutation test (Item B)\n")
        f.write(f"Permutations: {args.n_permutations}\n")
        f.write("Method: each of the retained features independently permuted across patients,\n")
        f.write("        standardized, k-means run for k=2..5, maximum silhouette recorded.\n")
        f.write(f"Observed max silhouette (real data, unpermuted): {args.observed_max_silhouette:.4f}\n")
        f.write(f"Null distribution: mean={null_scores.mean():.4f}, sd={null_scores.std():.4f}, "
                f"max={null_scores.max():.4f}, min={null_scores.min():.4f}\n")
        f.write(f"Empirical p-value: p = {p_value:.4f}\n")
        f.write("\nInterpretation: p is the fraction of permuted (structure-free) datasets whose\n")
        f.write("best-of-k=2-5 silhouette meets or exceeds the real, unpermuted value. A small p\n")
        f.write("indicates the observed clustering structure is unlikely to arise from a feature\n")
        f.write("space with no real multivariate structure.\n")

    print()
    print(f"Scores written to {args.out_dir / 'cluster_null_permutation_scores.csv'}")
    print(f"Summary written to {args.out_dir / 'cluster_null_permutation_summary.txt'}")


if __name__ == "__main__":
    main()
