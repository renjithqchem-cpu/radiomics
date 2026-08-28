#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.9
K_CANDIDATES = range(2, 6)
SUB_EXCITATION_THRESHOLD_EV = 7.4


def hu_to_red(mean_hu):
    return max(0.01, 1.0 + float(mean_hu) / 1000.0)


def load_features(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Feature file is empty: {path}")
    return df


def identify_feature_columns(df):
    excluded = {
        "patient_id", "patient", "id", "phenotype_class",
        "phenotype", "label", "diagnosis", "outcome"
    }
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).lower() not in excluded:
            cols.append(c)
    if not cols:
        raise ValueError("No numeric radiomics feature columns found.")
    return cols


def reduce_redundant_features(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    retained = [c for c in feature_cols if c not in dropped]
    return X[retained], retained, dropped


def cluster_phenotypes(df):
    all_cols = identify_feature_columns(df)
    X, retained, dropped = reduce_redundant_features(df, all_cols)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(random_state=RANDOM_STATE).fit_transform(X_scaled)

    scores = {}
    best_k = None
    best_score = -np.inf
    best_labels = None

    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        scores[k] = round(float(score), 3)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    clustered = df.copy()
    clustered["phenotype_class"] = best_labels
    return clustered, X_pca, best_k, retained, dropped, scores


def compute_red_weights(df):
    candidates = [c for c in df.columns if str(c).endswith("firstorder_Mean")]
    if not candidates:
        raise ValueError("No PyRadiomics first-order Mean feature found.")
    hu_col = candidates[0]

    mean_hu = df.groupby("phenotype_class")[hu_col].mean().sort_index()
    red = mean_hu.apply(hu_to_red)
    # Preserve V3.1 normalization: each phenotype RED divided by the
    # unweighted mean of phenotype-specific RED values.
    weights = (red / red.mean()).to_dict()
    return weights, mean_hu.to_dict(), red.to_dict(), hu_col


def load_lee_distribution(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LEE distribution file not found: {path}")

    df = pd.read_csv(path, comment="#")
    required = {"energy_eV", "P_E"}
    if not required.issubset(df.columns):
        raise ValueError("LEE CSV must contain energy_eV,P_E.")

    df = df[["energy_eV", "P_E"]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("energy_eV").reset_index(drop=True)
    df = df[df["P_E"] >= 0].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("LEE distribution requires at least two valid bins.")

    # The source-derived reconstruction is a 1-eV histogram.
    # Normalize probability mass explicitly.
    total_probability = float(df["P_E"].sum())
    if total_probability <= 0:
        raise ValueError("LEE distribution has non-positive total probability.")
    df["P_bin"] = df["P_E"] / total_probability

    # Verify equally spaced 1-eV bin centres, because the bin-assignment
    # method depends on the supplied histogram representation.
    centres = df["energy_eV"].to_numpy(dtype=float)
    diffs = np.diff(centres)
    if not np.allclose(diffs, 1.0, atol=1e-8):
        raise ValueError(
            "V4 expects 1-eV-spaced histogram centres (e.g. 0.5, 1.5, ...)."
        )

    widths = np.diff(centres)
    # For the expected 0.5,1.5,... grid, the first/last edges are 0 and 25 eV.
    lower_edges = centres - 0.5
    upper_edges = centres + 0.5
    if not np.isclose(lower_edges[0], 0.0, atol=1e-8):
        raise ValueError("LEE histogram must start at the 0-1 eV bin.")

    df["bin_lower_eV"] = lower_edges
    df["bin_upper_eV"] = upper_edges
    return df


def load_verified_database(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Verified DEA database not found: {path}")

    df = pd.read_csv(path)
    required = {
        "compound", "channel", "peak_energy_eV", "peak_cross_section_cm2",
        "evidence", "source"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Verified DEA database is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["peak_energy_eV"] = pd.to_numeric(df["peak_energy_eV"], errors="coerce")
    df["peak_cross_section_cm2"] = pd.to_numeric(
        df["peak_cross_section_cm2"], errors="coerce"
    )
    df["peak_cross_section_Mb"] = df["peak_cross_section_cm2"] / 1.0e-18

    usable = df[
        df["peak_energy_eV"].notna()
        & df["peak_cross_section_cm2"].notna()
        & (df["peak_cross_section_cm2"] > 0)
    ].copy()

    if usable.empty:
        raise ValueError("Verified DEA database contains no quantitative records.")
    return usable


def energy_compatibility(peak_energy_eV):
    return float(np.exp(-float(peak_energy_eV) / SUB_EXCITATION_THRESHOLD_EV))


def assign_lee_bin(lee, peak_energy_eV):
    """Assign a DEA peak to the 1-eV histogram bin containing its energy."""
    E = float(peak_energy_eV)
    # Histogram is defined over [0, 25) eV.
    if E < 0 or E >= float(lee["bin_upper_eV"].iloc[-1]):
        return None

    idx = int(np.floor(E))
    # For the standard 0.5,1.5,... grid, floor(E) is the bin index.
    if idx < 0 or idx >= len(lee):
        return None
    row = lee.iloc[idx]
    # Explicit containment check avoids silently accepting an incompatible grid.
    if not (row["bin_lower_eV"] <= E < row["bin_upper_eV"]):
        return None
    return {
        "bin_index": idx,
        "bin_lower_eV": float(row["bin_lower_eV"]),
        "bin_upper_eV": float(row["bin_upper_eV"]),
        "bin_center_eV": float(row["energy_eV"]),
        "bin_probability": float(row["P_bin"]),
    }


def parse_spectrum_map(path):
    if path is None:
        return {}
    df = pd.read_csv(path)
    if not {"compound", "spectrum_file"}.issubset(df.columns):
        raise ValueError("Spectrum map must contain compound,spectrum_file.")
    return dict(zip(df["compound"], df["spectrum_file"]))


def integrate_histogram_spectrum(lee, spectrum_path):
    
    spec = pd.read_csv(spectrum_path)
    if not {"energy_eV", "sigma_Mb"}.issubset(spec.columns):
        raise ValueError(f"{spectrum_path} must contain energy_eV,sigma_Mb.")
    spec = spec[["energy_eV", "sigma_Mb"]].apply(pd.to_numeric, errors="coerce").dropna()
    spec = spec.sort_values("energy_eV")
    spec = spec[spec["sigma_Mb"] >= 0]
    if len(spec) < 2:
        raise ValueError(f"{spectrum_path} has fewer than two valid spectrum points.")

    total = 0.0
    for _, row in lee.iterrows():
        lo = float(row["bin_lower_eV"])
        hi = float(row["bin_upper_eV"])
        E = spec["energy_eV"].to_numpy()
        S = spec["sigma_Mb"].to_numpy()
        mask = (E >= lo) & (E <= hi)
        local_E = E[mask]
        local_S = S[mask]

        # Include bin boundaries by interpolation if the spectrum spans them.
        if lo >= E.min() and lo <= E.max():
            local_E = np.append(local_E, lo)
            local_S = np.append(local_S, np.interp(lo, E, S))
        if hi >= E.min() and hi <= E.max():
            local_E = np.append(local_E, hi)
            local_S = np.append(local_S, np.interp(hi, E, S))

        order = np.argsort(local_E)
        local_E = local_E[order]
        local_S = local_S[order]
        unique_E, unique_idx = np.unique(local_E, return_index=True)
        unique_S = local_S[unique_idx]

        if len(unique_E) >= 2:
            mean_sigma = float(np.trapz(unique_S, unique_E) / (hi - lo))
        else:
            # No spectrum coverage in this bin means zero contribution.
            mean_sigma = 0.0

        total += float(row["P_bin"]) * mean_sigma

    return float(total)


def score_candidates(red_weights, lee, database, spectrum_map):
    rows = []

    for _, cand in database.iterrows():
        compound = cand["compound"]
        peak = float(cand["peak_energy_eV"])
        sigma_max = float(cand["peak_cross_section_Mb"])

        energy_score = energy_compatibility(peak)
        assignment = assign_lee_bin(lee, peak)
        if assignment is None:
            raise ValueError(
                f"DEA peak energy {peak} eV for {compound} is outside the 0-25 eV LEE histogram."
            )

        p_bin = assignment["bin_probability"]
        lee_peak_sigma = sigma_max * p_bin

        absolute_integral = np.nan
        integral_method = "not_calculated_peak_only"
        if compound in spectrum_map:
            absolute_integral = integrate_histogram_spectrum(
                lee, spectrum_map[compound]
            )
            integral_method = "absolute_spectrum_histogram_expectation"

        for phenotype, weight in red_weights.items():
            rows.append({
                "compound": compound,
                "phenotype_class": phenotype,
                "channel": cand["channel"],
                "peak_energy_eV": peak,
                "peak_cross_section_Mb": sigma_max,
                "energy_compatibility_score_V3_1": energy_score,
                "LEE_bin_lower_eV": assignment["bin_lower_eV"],
                "LEE_bin_upper_eV": assignment["bin_upper_eV"],
                "LEE_bin_center_eV": assignment["bin_center_eV"],
                "LEE_bin_probability": p_bin,
                "LEE_weighted_peak_sigma_Mb": lee_peak_sigma,
                "red_weight": weight,
                "RED_weighted_LEE_peak_sigma_Mb": lee_peak_sigma * weight,
                "RED_weighted_energy_score_V3_1": energy_score * weight,
                "LEE_weighted_absolute_sigma_Mb": (
                    absolute_integral * weight if np.isfinite(absolute_integral) else np.nan
                ),
                "score_method": "LEE_weighted_peak_sigma" if not np.isfinite(absolute_integral)
                                else "absolute_spectrum_histogram_expectation",
                "evidence": cand["evidence"],
                "source": cand["source"],
                "source_note": cand.get("source_note", ""),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="V4 radiomics phenotype + histogram based LEE-weighted peak DEA scoring."
    )
    ap.add_argument("--features", required=True)
    ap.add_argument("--lee_distribution", required=True)
    ap.add_argument("--dea_database", required=True)
    ap.add_argument("--spectrum_map", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        choices=K_CANDIDATES,
        help=(
            "Override the silhouette-selected number of phenotype classes. "
            "If omitted, the highest-silhouette k is used."
        ),
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = load_features(args.features)
    clustered, X_pca, silhouette_selected_k, retained, dropped, sil = cluster_phenotypes(features)

    print(
        f"Feature redundancy filter: {len(retained)+len(dropped)} raw features -> "
        f"{len(retained)} after dropping {len(dropped)} pairwise-correlated "
        f"(|Spearman r| > {CORRELATION_THRESHOLD}) duplicates."
    )
    print("Silhouette scores by k:", sil)
    print(
        f"Silhouette-selected k={silhouette_selected_k} "
        f"(silhouette={sil[silhouette_selected_k]})"
    )

    best_k = silhouette_selected_k
    if args.k is not None:
        best_k = args.k
        print(
            f"Manual k override requested: k={best_k} "
            f"(silhouette={sil[best_k]})"
        )
        if best_k != silhouette_selected_k:
            km = KMeans(
                n_clusters=best_k,
                random_state=RANDOM_STATE,
                n_init=20,
            )
            clustered = features.copy()
            clustered["phenotype_class"] = km.fit_predict(X_pca)
    else:
        print(f"Final model uses silhouette-selected k={best_k}")

    print("\nChosen number of phenotype classes:", best_k)
    print("Phenotype cluster sizes:")
    print(clustered["phenotype_class"].value_counts().sort_index())

    weights, mean_hu, red, hu_col = compute_red_weights(clustered)

    print("\nHU column used for RED:", hu_col)
    print("\nMean HU by phenotype class:")
    for c in sorted(mean_hu):
        print(f"  class {c}: mean HU = {mean_hu[c]:.3f}")

    print("\nRelative electron density:")
    for c in sorted(red):
        print(f"  class {c}: RED = {red[c]:.6f}")

    print("\nRED weights:")
    for c in sorted(weights):
        print(f"  class {c}: {weights[c]:.3f}")

    lee = load_lee_distribution(args.lee_distribution)
    print(
        f"\nLEE histogram range: {lee.bin_lower_eV.min():.3f} - "
        f"{lee.bin_upper_eV.max():.3f} eV"
    )
    print(f"LEE probability sum (histogram bins): {lee['P_bin'].sum():.6f}")
    print(
        f"LEE mean energy from bin centres: "
        f"{np.sum(lee.energy_eV * lee.P_bin):.3f} eV"
    )
    print("LEE representation: 1-eV histogram bins; no interpolation between centres.")

    database = load_verified_database(args.dea_database)
    print(f"\nVerified quantitative DEA records loaded: {len(database)}")
    print("Compounds:", ", ".join(database["compound"].astype(str)))

    spectrum_map = parse_spectrum_map(args.spectrum_map)
    scores = score_candidates(weights, lee, database, spectrum_map)

    scores.to_csv(out / "v4_phenotype_dea_scores.csv", index=False)

    # One row per compound/phenotype for ranking.
    rank = scores.drop_duplicates(subset=["compound", "phenotype_class"]).copy()
    rank["LEE_peak_rank_within_phenotype"] = (
        rank.groupby("phenotype_class")["RED_weighted_LEE_peak_sigma_Mb"]
        .rank(ascending=False, method="min")
    )
    rank["V3_1_energy_rank_within_phenotype"] = (
        rank.groupby("phenotype_class")["RED_weighted_energy_score_V3_1"]
        .rank(ascending=False, method="min")
    )
    rank = rank.sort_values(["phenotype_class", "LEE_peak_rank_within_phenotype", "compound"])
    rank.to_csv(out / "v4_candidate_ranking.csv", index=False)

    comp = scores.drop_duplicates(subset=["compound"]).copy()
    comp["V3_1_rank"] = comp["energy_compatibility_score_V3_1"].rank(
        ascending=False, method="min"
    )
    comp["V4_LEE_peak_rank"] = comp["LEE_weighted_peak_sigma_Mb"].rank(
        ascending=False, method="min"
    )
    comp["rank_change_V4_minus_V3_1"] = comp["V4_LEE_peak_rank"] - comp["V3_1_rank"]
    comp = comp.sort_values(["V4_LEE_peak_rank", "compound"])
    comp.to_csv(out / "v4_model_comparison.csv", index=False)

    # Explicit audit of which LEE bin each DEA peak entered.
    assignment_rows = []
    for _, cand in database.iterrows():
        a = assign_lee_bin(lee, float(cand["peak_energy_eV"]))
        assignment_rows.append({
            "compound": cand["compound"],
            "peak_energy_eV": cand["peak_energy_eV"],
            "bin_lower_eV": a["bin_lower_eV"],
            "bin_upper_eV": a["bin_upper_eV"],
            "bin_center_eV": a["bin_center_eV"],
            "bin_probability": a["bin_probability"],
            "peak_cross_section_Mb": cand["peak_cross_section_Mb"],
            "LEE_weighted_peak_sigma_Mb": cand["peak_cross_section_Mb"] * a["bin_probability"],
            "evidence": cand["evidence"],
            "source": cand["source"],
        })
    pd.DataFrame(assignment_rows).to_csv(out / "v4_lee_bin_assignment.csv", index=False)

    patient_cols = [c for c in ["patient_id", "phenotype_class"] if c in clustered.columns]
    clustered[patient_cols].to_csv(out / "v4_patients_with_phenotype_class.csv", index=False)

    pd.DataFrame([
        {
            "phenotype_class": c,
            "n_patients": int((clustered.phenotype_class == c).sum()),
            "mean_HU": mean_hu[c],
            "RED": red[c],
            "RED_weight": weights[c],
        }
        for c in sorted(mean_hu)
    ]).to_csv(out / "v4_phenotype_summary.csv", index=False)

    with open(out / "v4_model_metadata.txt", "w", encoding="utf-8") as f:
        f.write("V4 LEE-WEIGHTED PEAK DEA COMPATIBILITY MODEL\n")
        f.write("=============================================\n")
        f.write(f"Silhouette-selected k: {silhouette_selected_k}\n")
        f.write(f"Silhouette-selected k score: {sil[silhouette_selected_k]}\n")
        f.write(f"Final k used: {best_k}\n")
        f.write(f"Final k silhouette score: {sil[best_k]}\n")
        f.write(f"Manual k override: {args.k if args.k is not None else 'None'}\n")
        f.write(f"Silhouette scores: {sil}\n")
        f.write(f"Raw features: {len(retained)+len(dropped)}\n")
        f.write(f"Retained features: {len(retained)}\n")
        f.write(f"Dropped correlated features: {len(dropped)}\n")
        f.write(f"HU feature: {hu_col}\n")
        f.write("RED formula: max(0.01, 1 + HU/1000)\n")
        f.write("RED weight normalization: phenotype RED / unweighted mean phenotype RED\n")
        f.write("V3.1 benchmark energy descriptor: exp(-E_peak/7.4 eV)\n")
        f.write("V4 principal descriptor: sigma_peak * P_bin(E_peak)\n")
        f.write("LEE source representation: 1-eV histogram bins, centres 0.5 to 24.5 eV\n")
        f.write("LEE bin assignment: containing interval [0,1), [1,2), ..., [24,25) eV\n")
        f.write("No interpolation between LEE histogram centres is performed.\n")
        f.write("V4 metric is a peak-weighted compatibility metric, not a true sigma(E) integral.\n")
        f.write("No Gaussian/FWHM reconstruction, synthetic sigma(E), or missing-value imputation.\n")
        f.write(f"Verified quantitative DEA records: {len(database)}\n")
        f.write("Pimblott & LaVerne source: Radiat. Phys. Chem. 76, 1244-1247 (2007), DOI 10.1016/j.radphyschem.2007.02.012\n")

    print("\nV4 complete.")
    print(f"Outputs written to: {out}")
    print("Principal new score: sigma_peak * LEE bin probability")
    print("The V3.1 exponential score is retained only as a benchmark.")


if __name__ == "__main__":
    main()

