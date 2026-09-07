import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_lee_production(path, prob_col=None):
    """Read the production LEE CSV after its # metadata block."""
    path = Path(path)

    # Find the first non-comment, non-empty line and use it as CSV header.
    header_line = None
    with open(path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s and not s.startswith("#"):
                header_line = i
                break

    if header_line is None:
        raise ValueError(f"No CSV header/data found in {path}")

    lee = pd.read_csv(path, skiprows=header_line)

    # Production V4 uses P_E. Also allow equivalent names without changing
    # the production file.
    if prob_col is not None:
        if prob_col not in lee.columns:
            raise ValueError(
                f"Requested LEE probability column '{prob_col}' not found. "
                f"Available columns: {list(lee.columns)}"
            )
        chosen = prob_col
    else:
        candidates = [
            "P_E", "P_bin", "P", "probability",
            "probability_bin", "bin_probability",
            "LEE_probability", "LEE_bin_probability",
        ]
        chosen = next((c for c in candidates if c in lee.columns), None)

    if chosen is None:
        raise ValueError(
            "Could not identify LEE probability column. "
            f"Available columns: {list(lee.columns)}"
        )

    return lee, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--v4_dir", required=True)
    ap.add_argument("--dea_database", required=True)
    ap.add_argument("--lee_distribution", required=True)
    ap.add_argument("--lee_prob_col", default=None)
    args = ap.parse_args()

    v4 = Path(args.v4_dir)

    feat = pd.read_csv(args.features)
    summ = pd.read_csv(v4 / "v4_phenotype_summary.csv")
    assign = pd.read_csv(v4 / "v4_lee_bin_assignment.csv")
    score = pd.read_csv(v4 / "v4_phenotype_dea_scores.csv")
    dea = pd.read_csv(args.dea_database)

    lee, prob_col = read_lee_production(
        args.lee_distribution, args.lee_prob_col
    )

    print("FINAL V4-k2 TECHNICAL AUDIT")
    print("=" * 70)

    checks = []

    # ------------------------------------------------------------
    # Dataset and feature checks
    # ------------------------------------------------------------
    raw = [c for c in feat.columns if c != "patient_id"]

    checks.append(("Patient count", len(feat) == 418, f"{len(feat)}"))
    checks.append(("Raw feature count", len(raw) == 107, f"{len(raw)}"))

    counts = summ.set_index("phenotype_class")["n_patients"].to_dict()
    checks.append(
        ("Phenotype classes are 2",
         set(counts) == {0, 1},
         str(counts))
    )
    checks.append(
        ("Phenotype counts sum to patients",
         sum(counts.values()) == len(feat),
         str(sum(counts.values())))
    )

    # ------------------------------------------------------------
    # RED checks
    # ------------------------------------------------------------
    expected_red = 1 + summ["mean_HU"] / 1000
    red_ok = np.allclose(
        summ["RED"], expected_red, rtol=0, atol=1e-9
    )

    checks.append(
        ("RED = 1 + HU/1000",
         red_ok,
         "max abs diff %.3g" %
         np.max(np.abs(summ["RED"] - expected_red)))
    )

    expected_weights = summ["RED"] / summ["RED"].mean()
    weights_ok = np.allclose(
        summ["RED_weight"],
        expected_weights,
        rtol=0,
        atol=1e-9
    )

    checks.append(
        ("RED weight normalization",
         weights_ok,
         "mean weight %.12f" %
         summ["RED_weight"].mean())
    )

    # ------------------------------------------------------------
    # COMPLETE LEE histogram checks
    # ------------------------------------------------------------
    lee[prob_col] = pd.to_numeric(
        lee[prob_col], errors="coerce"
    )

    valid_prob = lee[prob_col].notna().all()
    nonnegative = (
        valid_prob and
        (lee[prob_col] >= 0).all()
    )
    lee_sum = lee[prob_col].sum()

    checks.append(
        ("Full LEE histogram probabilities sum to 1",
         np.isclose(lee_sum, 1.0, atol=1e-8),
         f"{lee_sum:.12f} (column: {prob_col})")
    )

    checks.append(
        ("Full LEE histogram probabilities nonnegative",
         nonnegative,
         f"minimum {lee[prob_col].min():.12g}"
         if valid_prob else "non-numeric/missing values")
    )

    # Optional check of energy range / bin count if the columns exist.
    energy_col = next(
        (c for c in ["energy_eV", "energy", "E_eV", "E"] if c in lee.columns),
        None
    )

    if energy_col is not None:
        energy = pd.to_numeric(lee[energy_col], errors="coerce")
        checks.append(
            ("LEE energy values are numeric",
             energy.notna().all(),
             f"{len(energy)} bins")
        )

    # ------------------------------------------------------------
    # Five DEA assignment checks
    # ------------------------------------------------------------
    assignment_probs = pd.to_numeric(
        assign["bin_probability"], errors="coerce"
    )

    assignment_ok = (
        assignment_probs.notna().all()
        and (assignment_probs >= 0).all()
        and (assignment_probs <= 1).all()
    )

    checks.append(
        ("Assigned LEE bin probabilities are valid",
         assignment_ok,
         f"{len(assign)} DEA peak assignments checked")
    )

    # IMPORTANT: do not sum assignment_probs.
    # Repeated 0-1 eV assignments are the same histogram bin.

    contained = (
        (assign["peak_energy_eV"] >= assign["bin_lower_eV"]) &
        (assign["peak_energy_eV"] < assign["bin_upper_eV"])
    ).all()

    checks.append(
        ("DEA peak lies in assigned LEE bin",
         contained,
         "all rows")
    )

    # ------------------------------------------------------------
    # Exact V4 score audit
    # ------------------------------------------------------------
    calc = (
        score["peak_cross_section_Mb"] *
        score["LEE_bin_probability"]
    )

    score_ok = np.allclose(
        score["LEE_weighted_peak_sigma_Mb"],
        calc,
        rtol=1e-10,
        atol=1e-12
    )

    checks.append(
        ("V4 score = sigma_peak * P_bin",
         score_ok,
         "all score rows")
    )

    calc_red = (
        score["LEE_weighted_peak_sigma_Mb"] *
        score["red_weight"]
    )

    redscore_ok = np.allclose(
        score["RED_weighted_LEE_peak_sigma_Mb"],
        calc_red,
        rtol=1e-10,
        atol=1e-12
    )

    checks.append(
        ("RED-weighted score = V4 score * RED weight",
         redscore_ok,
         "all score rows")
    )

    # ------------------------------------------------------------
    # DEA unit / provenance checks
    # ------------------------------------------------------------
    unit_ok = (
        np.all(pd.to_numeric(
            dea["peak_cross_section_cm2"],
            errors="coerce"
        ).notna())
    )

    checks.append(
        ("DEA cross sections are numeric in cm2",
         unit_ok,
         "all DEA records")
    )

    evidence_ok = (
        dea["evidence"].astype(str).str.len() > 0
    ).all()

    source_ok = (
        dea["source"].astype(str).str.len() > 0
    ).all()

    checks.append(
        ("DEA provenance populated",
         evidence_ok and source_ok,
         "evidence and source present for all DEA records")
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    for name, ok, detail in checks:
        print(
            f"[{'PASS' if ok else 'FAIL'}] "
            f"{name}: {detail}"
        )

    failures = [x for x in checks if not x[1]]

    print(
        "\nOverall:",
        "PASS" if not failures else f"FAIL ({len(failures)} checks)"
    )

    print("\nLEE audit detail:")
    print(f"  Production file: {args.lee_distribution}")
    print(f"  Probability column: {prob_col}")
    print(f"  Number of histogram rows: {len(lee)}")
    print(f"  Probability sum: {lee_sum:.12f}")

    print("\nImportant methodological limitations:")
    print(
        "  - No full continuous sigma(E) curve was supplied; "
        "V4 uses the 1-eV LEE histogram bin containing E_peak."
    )
    print(
        "  - v4_lee_bin_assignment.csv contains only the five DEA "
        "peak assignments; those rows are not the complete histogram."
    )
    print(
        "  - 5-FU and catechin sulfate have no numerical DEA uncertainty "
        "in the supplied database and are fixed in the quantitative "
        "Monte Carlo analysis."
    )
    print(
        "  - Order-of-magnitude uncertainty was represented as a "
        "log-uniform sensitivity model, not as a claimed source "
        "probability distribution."
    )
    print(
        "  - The DEA score is a compatibility/prioritization score, "
        "not a measured clinical radiosensitization endpoint."
    )

    report = v4 / "final_technical_audit.txt"

    with open(report, "w", encoding="utf-8") as f:
        for name, ok, detail in checks:
            f.write(
                f"[{'PASS' if ok else 'FAIL'}] "
                f"{name}: {detail}\n"
            )

        f.write(
            "\nOverall: " +
            ("PASS\n" if not failures else
             f"FAIL ({len(failures)} checks)\n")
        )

        f.write(
            f"\nLEE probability column: {prob_col}\n"
        )
        f.write(
            f"Complete LEE probability sum: {lee_sum:.12f}\n"
        )

        f.write(
            "\nImportant: v4_lee_bin_assignment.csv contains only "
            "the five DEA peak assignments and is not expected to "
            "sum to one.\n"
        )

    print("\nAudit report:", report)


if __name__ == "__main__":
    main()
