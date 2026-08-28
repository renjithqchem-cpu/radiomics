#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED = 20260826
N_SIM_DEFAULT = 10000

RED_WEIGHTS = {0: 1.0429424357648776, 1: 0.9570575642351223}

def load_inputs(dea_path, assignment_path):
    dea = pd.read_csv(dea_path).drop_duplicates("compound").copy()
    lee = pd.read_csv(assignment_path).drop_duplicates("compound").copy()

    req1 = {"compound","peak_cross_section_cm2","cross_section_uncertainty"}
    req2 = {"compound","bin_probability"}
    if not req1.issubset(dea.columns):
        raise ValueError(f"DEA database missing: {sorted(req1-set(dea.columns))}")
    if not req2.issubset(lee.columns):
        raise ValueError(f"LEE assignment missing: {sorted(req2-set(lee.columns))}")

    df = dea.merge(
        lee[["compound","bin_probability"]],
        on="compound", how="inner", validate="one_to_one"
    )
    if len(df) != 5:
        raise ValueError(f"Expected 5 compounds; found {len(df)}")

    df["peak_cross_section_cm2"] = pd.to_numeric(df["peak_cross_section_cm2"], errors="coerce")
    df["bin_probability"] = pd.to_numeric(df["bin_probability"], errors="coerce")
    if df[["peak_cross_section_cm2","bin_probability"]].isna().any().any():
        raise ValueError("Missing numeric cross section or LEE probability.")
    return df.sort_values("compound").reset_index(drop=True)

def sample_cross_sections(df, n_sim, rng):
    samples = np.tile(df["peak_cross_section_cm2"].to_numpy(float), (n_sim,1))
    notes = {}

    for j,row in df.iterrows():
        name = row["compound"]
        s0 = float(row["peak_cross_section_cm2"])

        if name == "5-bromouracil":
            samples[:,j] = s0 * 10.0**rng.uniform(-1,1,n_sim)
            notes[name] = "log-uniform 0.1x-10x (approximately one-order uncertainty)"
        elif name == "2-nitrofuran":
            samples[:,j] = s0 * 10.0**rng.uniform(-1,1,n_sim)
            notes[name] = "log-uniform 0.1x-10x (order-of-magnitude uncertainty)"
        elif name == "5-iodouridine":
            sd = 1.9e-14/1.96
            vals = rng.normal(s0, sd, n_sim)
            while np.any(vals <= 0):
                bad = vals <= 0
                vals[bad] = rng.normal(s0, sd, bad.sum())
            samples[:,j] = vals
            notes[name] = "truncated normal; 95% half-width 1.9e-14 cm2"
        elif name == "5-fluorouridine":
            notes[name] = "fixed; no numerical uncertainty supplied"
        elif name == "catechin_sulfate":
            notes[name] = "fixed; no numerical uncertainty supplied"
        else:
            notes[name] = "fixed; no recognized numerical uncertainty"
    return samples, notes

def score_rank(df, sigma_cm2, red_weight):
    p = df["bin_probability"].to_numpy(float)
    score_cm2 = sigma_cm2 * p[None,:] * red_weight
    score_mb = score_cm2 * 1e18
    order = np.argsort(-score_mb, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:,None], order] = np.arange(1, len(df)+1)
    return score_mb, ranks

def summary(df, scores, ranks, phenotype):
    out=[]
    for j,name in enumerate(df["compound"]):
        s=scores[:,j]; r=ranks[:,j]
        out.append({
            "compound":name, "phenotype_class":phenotype,
            "median_score_Mb":np.median(s), "mean_score_Mb":np.mean(s),
            "p2_5_score_Mb":np.percentile(s,2.5),
            "p97_5_score_Mb":np.percentile(s,97.5),
            "median_rank":np.median(r), "mean_rank":np.mean(r),
            "rank1_probability":np.mean(r==1),
            "rank2_probability":np.mean(r==2),
            "rank3_probability":np.mean(r==3),
            "rank4_probability":np.mean(r==4),
            "rank5_probability":np.mean(r==5),
        })
    return pd.DataFrame(out)

def pairwise(df, scores, phenotype):
    rows=[]
    names=df["compound"].tolist()
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a=np.mean(scores[:,i]>scores[:,j])
            rows.append({
                "phenotype_class":phenotype,
                "compound_A":names[i],"compound_B":names[j],
                "P(A_score_gt_B)":a,"P(B_score_gt_A)":np.mean(scores[:,j]>scores[:,i])
            })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dea_database",required=True)
    ap.add_argument("--lee_assignment",required=True)
    ap.add_argument("--out_dir",required=True)
    ap.add_argument("--n_sim",type=int,default=N_SIM_DEFAULT)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=load_inputs(args.dea_database,args.lee_assignment)
    rng=np.random.default_rng(RNG_SEED)
    sigma,notes=sample_cross_sections(df,args.n_sim,rng)

    summaries=[]; pairs=[]
    for phenotype,w in RED_WEIGHTS.items():
        scores,ranks=score_rank(df,sigma,w)
        summaries.append(summary(df,scores,ranks,phenotype))
        pairs.append(pairwise(df,scores,phenotype))

    summary_df=pd.concat(summaries,ignore_index=True)
    pair_df=pd.concat(pairs,ignore_index=True)
    summary_df.to_csv(out/"dea_sensitivity_score_summary.csv",index=False)
    pair_df.to_csv(out/"dea_sensitivity_pairwise_probability.csv",index=False)

    # Molecular rank probabilities: RED is a common positive multiplier within a phenotype,
    # so the molecular ordering is identical for both phenotypes.
    scores,ranks=score_rank(df,sigma,1.0)
    rows=[]
    for j,name in enumerate(df["compound"]):
        r=ranks[:,j]
        rows.append({
            "compound":name,
            "P_rank_1":np.mean(r==1),"P_rank_2":np.mean(r==2),
            "P_rank_3":np.mean(r==3),"P_rank_4":np.mean(r==4),
            "P_rank_5":np.mean(r==5)
        })
    pd.DataFrame(rows).to_csv(out/"dea_sensitivity_rank_probability.csv",index=False)

    with open(out/"dea_sensitivity_metadata.txt","w",encoding="utf-8") as f:
        f.write("DEA UNCERTAINTY SENSITIVITY ANALYSIS - FINAL UNIT-CORRECTED VERSION\n")
        f.write("===============================================================\n")
        f.write(f"Monte Carlo simulations: {args.n_sim}\nRandom seed: {RNG_SEED}\n")
        f.write("Score: sigma_peak(cm2) * P_LEE_bin * RED_weight, converted to Mb by x1e18.\n")
        f.write("5-bromouracil: log-uniform 0.1x-10x.\n")
        f.write("2-nitrofuran: log-uniform 0.1x-10x.\n")
        f.write("5-iodouridine: truncated normal using reported 95% half-width 1.9e-14 cm2.\n")
        f.write("5-fluorouridine: fixed because no numerical uncertainty supplied.\n")
        f.write("catechin_sulfate: fixed because no numerical uncertainty supplied.\n")
        f.write("LEE bin probabilities are held fixed; no uncertainty distribution was supplied.\n")
        f.write("The log-uniform distributions are sensitivity models, not claimed source probability laws.\n")
    print("Unit-corrected sensitivity analysis complete.")
    print("Outputs written to:",out)

if __name__=="__main__":
    main()
