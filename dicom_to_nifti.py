import argparse
from pathlib import Path

import pydicom
import SimpleITK as sitk


def find_series_by_modality(patient_dir: Path):
    """
    Walk a patient's downloaded folder tree and group DICOM files into series
    by SeriesInstanceUID, tagging each series with its Modality (CT or SEG).
    Returns (ct_series: dict[uid -> list[Path]], seg_files: list[Path]).
    """
    ct_series = {}
    seg_files = []

    for dcm_path in patient_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        except Exception as exc:  # noqa: BLE001 - skip unreadable/non-DICOM files
            print(f"  [skip] {dcm_path.name}: {exc}")
            continue

        modality = getattr(ds, "Modality", None)
        if modality == "CT":
            uid = ds.SeriesInstanceUID
            ct_series.setdefault(uid, []).append(dcm_path)
        elif modality == "SEG":
            seg_files.append(dcm_path)
        # Other modalities (RTSTRUCT, RTDOSE, etc.) are ignored here - this
        # script targets collections that ship DICOM SEGMENTATION objects,
        # like the current version of NSCLC-Radiomics.

    return ct_series, seg_files


def convert_ct_series_to_nifti(dcm_files: list, out_path: Path):
    reader = sitk.ImageSeriesReader()
    # Re-sort by SimpleITK's own file-order detection rather than trusting
    # filesystem order, since DICOM instance order isn't guaranteed to match.
    series_dir = str(dcm_files[0].parent)
    sorted_filenames = reader.GetGDCMSeriesFileNames(series_dir)
    reader.SetFileNames(sorted_filenames)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))
    return image


def convert_seg_to_nifti(seg_path: Path, reference_image: sitk.Image, out_path: Path):
    """
    Reads a DICOM SEGMENTATION object and resamples it onto the reference CT
    image's grid, so the mask lines up voxel-for-voxel with image.nii.gz.
    """
    seg_image = sitk.ReadImage(str(seg_path))

    # DICOM SEG objects can encode multiple labeled segments as separate
    # frames/channels. For a single primary tumor volume (GTV-1 in
    # NSCLC-Radiomics), this collapses everything to one binary mask.
    if seg_image.GetNumberOfComponentsPerPixel() > 1:
        seg_image = sitk.VectorIndexSelectionCast(seg_image, 0)

    seg_resampled = sitk.Resample(
        seg_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        seg_image.GetPixelID(),
    )
    binary_mask = sitk.BinaryThreshold(seg_resampled, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)
    sitk.WriteImage(binary_mask, str(out_path))


def convert_patient(patient_dir: Path, out_dir: Path) -> bool:
    patient_id = patient_dir.name
    print(f"\n[{patient_id}]")

    ct_series, seg_files = find_series_by_modality(patient_dir)

    if not ct_series:
        print("  [error] no CT series found - skipping")
        return False
    if not seg_files:
        print("  [error] no SEG object found - skipping")
        return False

    # If multiple CT series exist (rare - e.g. a repeat scan), take the one
    # with the most slices as the primary series.
    best_uid = max(ct_series, key=lambda uid: len(ct_series[uid]))
    ct_files = ct_series[best_uid]
    if len(ct_series) > 1:
        print(f"  [note] {len(ct_series)} CT series found, using the largest ({len(ct_files)} slices)")

    patient_out_dir = out_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    image_out = patient_out_dir / "image.nii.gz"
    mask_out = patient_out_dir / "mask.nii.gz"

    reference_image = convert_ct_series_to_nifti(ct_files, image_out)
    print(f"  [ok] wrote {image_out}")

    # If more than one SEG object is present, take the first - inspect
    # manually if your cohort has patients with multiple structure sets.
    convert_seg_to_nifti(seg_files[0], reference_image, mask_out)
    print(f"  [ok] wrote {mask_out}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert TCIA DICOM CT+SEG downloads to NIfTI image/mask pairs.")
    parser.add_argument("--raw_dir", type=Path, default=Path("../raw_downloads"),
                         help="Top-level folder from the NBIA Data Retriever, containing one subfolder per patient.")
    parser.add_argument("--out_dir", type=Path, default=Path("../data"),
                         help="Where to write patient_id/image.nii.gz + mask.nii.gz pairs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in args.raw_dir.iterdir() if p.is_dir())
    if not patient_dirs:
        raise FileNotFoundError(f"No patient subdirectories found under {args.raw_dir}")

    succeeded, failed = 0, 0
    for patient_dir in patient_dirs:
        if convert_patient(patient_dir, args.out_dir):
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. Converted {succeeded} patients, skipped {failed}.")
    print(f"Output ready for: python extract_radiomics.py --data_dir {args.out_dir}")


if __name__ == "__main__":
    main()
