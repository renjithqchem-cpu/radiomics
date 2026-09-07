# A transparent radiomic DEA framework for prioritizing low energy electron radiosensitizer candidates

Code accompanying the manuscript "Radiomic--DEA framework for
prioritizing low energy electron radiosensitizer candidates" (Renjith Bhaskaran,
submitted to *Physics in Medicine & Biology*).

This repository implements a transparent computational framework that bridges
two normally separate domains -- CT-based tumour radiomics and molecular
dissociative electron attachment (DEA) physics -- to prioritize candidate
radiosensitizer molecules for radiotherapy. Pretreatment CT scans from the
NSCLC-Radiomics cohort (418 patients) are used to discover two CT-derived
tumour phenotypes via unsupervised clustering, validated through bootstrap
stability and supervised reproducibility checks. Each phenotype is weighted
by a CT-derived relative electron density (RED) descriptor and combined with
a reconstructed low energy electron (LEE) spectrum to score five
literature verified DEA cross-section records, with full Monte Carlo
uncertainty propagation over the resulting candidate ranking. The framework
is a screening/prioritization tool, not a clinical prediction model.

## Pipeline order

The complete computational workflow is organized into 11 scripts, progressing from
patient-level CT/segmentation conversion through radiomics extraction, phenotype
discovery and validation, DEA/LEE scoring, uncertainty analysis, and final technical
auditing.

1. `dicom_to_nifti.py` -- converts per-patient TCIA DICOM CT series and segmentation
   objects into standardized NIfTI image/mask pairs (`image.nii.gz` and
   `mask.nii.gz`).

2. `extract_radiomics.py` -- runs PyRadiomics feature extraction on the converted
   NIfTI image/mask pairs, producing a patient-level radiomics feature table
   containing 107 descriptors per patient.

3. `correlate_analysis_lee_peak.py` -- performs Spearman redundancy filtering,
   feature standardization, k-means phenotype discovery, REDproxy weighting,
   LEE-bin assignment, and DEA-based candidate scoring. The script combines the
   radiomics phenotype results with the source-controlled DEA database and LEE
   distribution to produce phenotype labels, phenotype summaries, and candidate
   ranking tables.

4. `exact_pipeline_bootstrap.py` -- evaluates cluster stability using 1000 bootstrap
   resamples. The clustering is refitted on the fixed 47-feature representation
   selected in the primary analysis, and clusterwise Jaccard stability is calculated
   after optimal label matching.

5. `phenotype_classifier_check.py` -- performs a supervised reproducibility check
   of the unsupervised phenotype labels using random forest and logistic regression
   with five-fold cross-validation, together with a label-permutation significance
   test. The outputs include balanced accuracy and permutation-based significance
   measures.

6. `cluster_null_permutation_test.py` -- evaluates whether the observed clustering
   structure exceeds a feature-permutation null distribution by calculating the
   maximum silhouette coefficient over k = 2--5 across 1000 feature permutations.
   The output includes the null silhouette distribution and empirical p-value.

7. `k2_vs_k3_bootstrap_deltaS.py` -- compares the k = 2 and k = 3 clustering
   solutions using 1000 bootstrap resamples. The script generates the bootstrap
   distribution of the silhouette difference (ΔS), its 95% percentile interval,
   and the probability that ΔS is positive.

8. `feature_threshold_sensitivity.py` -- evaluates the sensitivity of the phenotype
   partition to alternative Spearman redundancy thresholds of 0.80, 0.85, 0.90,
   and 0.95. The analysis reports retained feature counts, preferred k, cluster
   sizes, ARI, NMI, and bootstrap stability measures.

9. `dea_resonance_profiles.py` -- constructs the source-controlled DEA candidate
   database, retaining the reported resonance energy, cross section, molecular
   species/channel, evidence type, and provenance for each candidate. The script
   produces `DEA_verified_scoring_data.csv`, which serves as the controlled input
   to the DEA scoring and uncertainty analyses.

10. `dea_uncertainty_sensitivity.py` -- performs Monte Carlo uncertainty propagation
    using 10,000 simulations over the DEA scoring inputs. The analysis produces
    candidate rank probabilities and pairwise comparison probabilities, allowing
    the robustness of the candidate prioritization to be assessed.

11. `final_technical_audit.py` -- performs final integrity checks of the complete
    analysis, including patient counts, radiomics feature counts, REDproxy
    weighting, LEE-distribution normalization, DEA energy-bin assignment, score
    construction, and DEA data provenance. The script produces the final technical
    audit report.

## Data inputs

- Imaging: NSCLC-Radiomics collection, TCIA,
  <https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI> (not redistributed here;
  download via the TCIA/NBIA Data Retriever).
- `DEA_verified_scoring_data.csv` -- the five quantitative DEA records used
  in the manuscript, with peak energy, peak cross section, uncertainty
  information, and full source citation per record.
- `lee_energy_distribution.csv` -- 1 eV-binned reconstruction of the
  secondary-electron energy distribution of Pimblott & LaVerne (2007),
  *Radiat. Phys. Chem.* **76**, 1244-1247.

## Requirements

See `requirements.txt` / `environment.yml`. Developed with Python 3.10,
PyRadiomics, SimpleITK, scikit-learn, pandas, numpy, matplotlib, seaborn.

## Citation

If you use this code, please cite the associated manuscript (full citation
to be added on publication) and, where relevant, the original data sources
listed above.

## License

Released under the GNU General Public License v3.0 (GPLv3) -- see `LICENSE`.

## Contact

Renjith Bhaskaran, Department of Chemistry, Madanapalle Institute of
Technology & Science (MITS), India. drrenjithb@mits.ac.in
