import argparse
import logging
from pathlib import Path

import pandas as pd
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)  # quiet the very verbose default logging


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Configure PyRadiomics with a sensible default feature set for tumor phenotyping."""
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": None,  # set to e.g. [1, 1, 1] once you check native spacing
        "interpolator": "sitkBSpline",
        "verbose": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("shape")
    return extractor


def extract_all(data_dir: Path) -> pd.DataFrame:
    extractor = build_extractor()
    rows = []

    patient_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(
            f"No patient subdirectories found under {data_dir}. "
            "Expected data/<patient_id>/image.nii.gz + mask.nii.gz."
        )

    for patient_dir in patient_dirs:
        image_path = patient_dir / "image.nii.gz"
        mask_path = patient_dir / "mask.nii.gz"
        if not image_path.exists() or not mask_path.exists():
            print(f"[skip] {patient_dir.name}: missing image.nii.gz or mask.nii.gz")
            continue

        try:
            features = extractor.execute(str(image_path), str(mask_path))
        except Exception as exc:  # noqa: BLE001 - report and continue over a large cohort
            print(f"[error] {patient_dir.name}: {exc}")
            continue

        # PyRadiomics returns diagnostic keys (prefixed "diagnostics_") alongside the
        # actual features - keep only the feature columns for the modeling stage.
        feature_row = {"patient_id": patient_dir.name}
        feature_row.update(
            {k: v for k, v in features.items() if not k.startswith("diagnostics_")}
        )
        rows.append(feature_row)
        print(f"[ok] {patient_dir.name}: {len(feature_row) - 1} features extracted")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from a tumor cohort.")
    parser.add_argument("--data_dir", type=Path, default=Path("../data"))
    parser.add_argument("--out", type=Path, default=Path("../results/radiomics_features.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = extract_all(args.data_dir)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} patient feature rows to {args.out}")


if __name__ == "__main__":
    main()
