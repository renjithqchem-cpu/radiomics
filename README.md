# A transparent radiomic-DEA framework for prioritizing low-energy-electron radiosensitizer candidates

Code accompanying the manuscript "A transparent radiomic--DEA framework for
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
a reconstructed low-energy-electron (LEE) spectrum to score five
literature-verified DEA cross-section records, with full Monte Carlo
uncertainty propagation over the resulting candidate ranking. The framework
is a screening/prioritization tool, not a clinical prediction model.

## Pipeline order

1. `dicom_to_nifti.py` -- converts TCIA DICOM CT + SEG downloads into
   `image.nii.gz` / `mask.nii.gz` pairs.
2. `extract_radiomics.py` -- runs PyRadiomics feature extraction on the
   converted images, producing `radiomics_features.csv` (107 features per
   patient).
3. `correlate_analysis_v4_lee_peak.py` -- Spearman redundancy filtering,
   standardization, PCA, k-means phenotype discovery, RED weighting, and
   LEE-bin assignment of the DEA database; produces the candidate ranking
   and phenotype summary tables.
4. `cluster_stability_check_v4.py` -- bootstrap (200-resample) cluster-wise
   Jaccard stability audit of the phenotype partition.
5. `phenotype_classifier_check.py` -- supervised reproducibility check
   (random forest / logistic regression, 5-fold CV, permutation test) of the
   unsupervised phenotype labels.
6. `dea_uncertainty_sensitivity.py` -- 10,000-simulation Monte Carlo
   uncertainty propagation over the DEA database and RED weights, producing
   rank probabilities and pairwise comparison probabilities.
7. `dea_resonance_profiles.py` -- source-controlled database of candidate
   DEA resonance profiles, with per-record evidence-type tagging
   (`computed` / `verified_primary` / `verified_secondary` / `unverified`).

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
