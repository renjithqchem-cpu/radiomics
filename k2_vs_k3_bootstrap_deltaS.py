import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.9
N_BOOTSTRAP_DEFAULT = 1000


def reduce_redundant_features(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in feature_cols if X[c].notna().all()]
    X = X[valid_cols]
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    retained = [c for c in valid_cols if c not in dropped]
    return retained, dropped


def silhouette_at_k(X_scaled, k, random_state):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = model.fit_predict(X_scaled)
    if len(set(labels)) < 2:
        return np.nan
    return silhouette_score(X_scaled, labels)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap comparison of k=2 vs k=3 silhouette coefficients (Item C)."
    )
    parser.add_argument("--features", type=Path, default=Path("results/radiomics_features.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results_sensitivity"))
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    all_cols = [c for c in features.columns if c != "patient_id"]
    retained, dropped = reduce_redundant_features(features, all_cols)

    X_full = features[retained].apply(pd.to_numeric, errors="coerce")
    X_full = X_full.fillna(X_full.median(numeric_only=True)).to_numpy(dtype=float)
    n = X_full.shape[0]

    X_full_scaled = StandardScaler().fit_transform(X_full)
    s2_observed = silhouette_at_k(X_full_scaled, 2, RANDOM_STATE)
    s3_observed = silhouette_at_k(X_full_scaled, 3, RANDOM_STATE)

    print("k=2 vs k=3 bootstrap silhouette-difference comparison (Item C)")
    print(f"Patients: {n}")
    print(f"Retained features: {len(retained)}")
    print(f"Observed (full-cohort) silhouette: k=2: {s2_observed:.3f}, k=3: {s3_observed:.3f}")
    print(f"Observed Delta S = S(k=3) - S(k=2) = {s3_observed - s2_observed:.4f}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print()

    rng = np.random.default_rng(RANDOM_STATE)
    s2_boot = np.empty(args.n_bootstrap)
    s3_boot = np.empty(args.n_bootstrap)

    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_boot = X_full[idx]
        X_boot_scaled = StandardScaler().fit_transform(X_boot)
        s2_boot[b] = silhouette_at_k(X_boot_scaled, 2, RANDOM_STATE + b + 1)
        s3_boot[b] = silhouette_at_k(X_boot_scaled, 3, RANDOM_STATE + b + 1)
        if (b + 1) % 100 == 0:
            print(f"  ... {b + 1}/{args.n_bootstrap} bootstrap resamples complete")

    delta_s = s3_boot - s2_boot
    delta_s = delta_s[~np.isnan(delta_s)]

    ci_low, ci_high = np.percentile(delta_s, [2.5, 97.5])
    mean_delta = float(np.mean(delta_s))
    contains_zero = ci_low <= 0 <= ci_high

    print()
    print(f"Bootstrap Delta S: mean = {mean_delta:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"95% CI contains zero: {contains_zero}")
    print(f"P(Delta S > 0) across bootstraps: {float(np.mean(delta_s > 0)):.3f}")

    pd.DataFrame({"S_k2": s2_boot, "S_k3": s3_boot, "delta_S": s3_boot - s2_boot}).to_csv(
        args.out_dir / "k2_vs_k3_bootstrap_deltaS.csv", index=False
    )

    with open(args.out_dir / "k2_vs_k3_bootstrap_summary.txt", "w") as f:
        f.write("k=2 vs k=3 bootstrap silhouette-difference comparison (Item C)\n")
        f.write(f"Bootstrap resamples: {args.n_bootstrap}\n")
        f.write(f"Observed (full-cohort) silhouette: k=2: {s2_observed:.4f}, k=3: {s3_observed:.4f}\n")
        f.write(f"Observed Delta S = S(k=3) - S(k=2) = {s3_observed - s2_observed:.4f}\n")
        f.write(f"Bootstrap Delta S: mean = {mean_delta:.4f}\n")
        f.write(f"Bootstrap Delta S: 95% CI = [{ci_low:.4f}, {ci_high:.4f}]\n")
        f.write(f"95% CI contains zero: {contains_zero}\n")
        f.write(f"P(Delta S > 0) across bootstraps: {float(np.mean(delta_s > 0)):.4f}\n")
        f.write("\nInterpretation: if the 95% CI contains zero, the nominally higher k=3\n")
        f.write("silhouette is not distinguishable from the k=2 silhouette under resampling,\n")
        f.write("providing a formal basis (alongside the separately reported bootstrap Jaccard\n")
        f.write("stability and minority-class size) for retaining k=2.\n")

    print()
    print(f"Bootstrap scores written to {args.out_dir / 'k2_vs_k3_bootstrap_deltaS.csv'}")
    print(f"Summary written to {args.out_dir / 'k2_vs_k3_bootstrap_summary.txt'}")


if __name__ == "__main__":
    main()
