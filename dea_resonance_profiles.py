from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class ResonanceProfile:
    compound: str
    resonance_energy_eV: Optional[float]
    resonance_width_eV: Optional[float]
    cross_section_window_eV: Optional[Tuple[float, float]]
    peak_cross_section_cm2: Optional[float]
    evidence_type: str
    source: str = ""
    notes: str = ""


CANDIDATE_PROFILES = [

    ResonanceProfile(
        compound="catechin_sulfate",
        resonance_energy_eV=1.1128,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=4.61e-18,
        evidence_type="computed",
        source="Your CAP-EOM-EA-CCSD/LCP-TDWP calculation (doorway-vs-DEA-max production run)",
        notes=(
            "Dominant DEA cross-section maximum from the outgoing-flux spectrum, v=0. "
            "The 0.4512 eV doorway/autocorrelation resonance is a DIFFERENT feature and "
            "is NOT used here. FWHM and 50%-sigma window still need extraction from the "
            "sigma(E) curve of the same run - not yet available."
        ),
    ),

    ResonanceProfile(
        compound="nimorazole",
        resonance_energy_eV=3.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=3e-18,
        evidence_type="verified_primary",
        source="Meissner et al., Nat. Commun. 2019, 10, 2388, DOI:10.1038/s41467-019-10340-8",
        notes=(
            "NO2- forms abundantly in the 2-4 eV region (confirmed directly from the primary "
            "paper); reported cross section ~3e-18 cm2 indicates S-wave electron attachment "
            "at near-0 eV for the PARENT anion channel, which is actually dominant overall - "
            "NO2- itself is a secondary, shape-resonance-mediated channel in the 2-4 eV range. "
            "The precise decimal '2.97 eV' from an earlier compiled table is not traceable to "
            "a specific number in the source; use '~3 eV' rather than false precision."
        ),
    ),

    ResonanceProfile(
        compound="misonidazole",
        resonance_energy_eV=3.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="verified_secondary",
        source="Meissner et al., Int. J. Mol. Sci. 2019, 20, 3496; cross-referenced via "
               "Saqib et al., Int. J. Mol. Sci. 2020, 21, 8906",
        notes=(
            "The 2-nitrofuran paper directly states misonidazole's NO2- anion efficiency curve "
            "has the 'same peak shape' as 2-nitrofuran's own confirmed 3.1 eV NO2- peak, "
            "supporting a ~3 eV assignment. IMPORTANT: misonidazole's actual DOMINANT channel "
            "overall is associative attachment (non-dissociative parent anion) at ~0 eV, not "
            "this NO2- DEA channel - do not present 3 eV as 'the' resonance without this caveat."
        ),
    ),

    ResonanceProfile(
        compound="metronidazole",
        resonance_energy_eV=None,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="unverified",
        source="Qualitatively mentioned (NO2- and OH- channels exist) in Sedmidubska & Kocisek, "
               "Phys. Chem. Chem. Phys. 2024, 26, 9112 (Section 3.3), but no numeric peak energy given.",
        notes=(
            "Existence of an NO2- DEA channel is confirmed, but no specific resonance energy "
            "has been located in any source checked so far. Left as None - do not reintroduce "
            "the earlier ~3.0 eV guess without a numeric citation."
        ),
    ),

    ResonanceProfile(
        compound="2-nitrofuran",
        resonance_energy_eV=3.1,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=7.8e-16,
        evidence_type="verified_primary",
        source="Saqib, Arthur-Baidoo, Oncak & Denifl, Int. J. Mol. Sci. 2020, 21, 8906, "
               "DOI:10.3390/ijms21238906 (checked against full text, Table 1 and Fig. 2b)",
        notes=(
            "Main NO2- resonance confirmed at 3.1 eV (weaker secondary resonance at 1.5 eV "
            "excluded per the single-resonance rule). Cross section confirmed at "
            "7.8e-20 m^2 = 7.8e-16 cm^2."
        ),
    ),

    ResonanceProfile(
        compound="5-chlorouracil",
        resonance_energy_eV=None,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="unverified",
        source="No primary or secondary source located yet.",
        notes="Earlier compiled values (0.23 eV; 5e-14 cm^2) could not be independently confirmed.",
    ),

    ResonanceProfile(
        compound="5-bromouracil",
        resonance_energy_eV=0.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=4.0e-14,
        evidence_type="verified_secondary",
        source="RSC Advances review on fluoro-substituted nucleoside DNA radiosensitization, "
               "DOI:10.1039/C3RA46735J",
        notes="Br- formed by electron attachment at zero eV; cross section confirmed at 4e-14 cm^2.",
    ),

    ResonanceProfile(
        compound="5-iodouracil",
        resonance_energy_eV=0.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=9.0e-14,
        evidence_type="verified_secondary",
        source="Same RSC Advances review as 5-bromouracil, DOI:10.1039/C3RA46735J",
        notes="I- formed by electron attachment at zero eV; cross section confirmed at 9e-14 cm^2.",
    ),

    ResonanceProfile(
        compound="5-iodouridine",
        resonance_energy_eV=None,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="unverified",
        source="No primary or secondary source located yet.",
        notes="Earlier compiled value ((2.7+/-1.9)e-14 cm^2) could not be independently confirmed.",
    ),

    ResonanceProfile(
        compound="temozolomide",
        resonance_energy_eV=0.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="verified_primary",
        source="Denifl group crossed electron-molecular beam TMZ study; see also Arthur-Baidoo, "
               "Izadi, Guerra, Garcia, Oncak & Denifl, Front. Phys. 2022, 10, 880689",
        notes="Most abundant fragment anion confirmed at 0 eV resonance energy.",
    ),

    ResonanceProfile(
        compound="sanazole",
        resonance_energy_eV=4.0,
        resonance_width_eV=None,
        cross_section_window_eV=None,
        peak_cross_section_cm2=None,
        evidence_type="verified_primary",
        source="Izadi, Mahmoodi-Darian, Luxford, Kocisek, Denifl & Oncak, ChemPlusChem 2025, "
               "90, e202500120, DOI:10.1002/cplu.202500120 (Table 1, Fig. 1e)",
        notes=(
            "Confirmed: 'the peak at 4.0 eV represents a characteristic peak in NO2- anion "
            "yields' for sanazole - this is the NO2--specific channel, matching the earlier "
            "compiled table's value exactly with a real primary-source citation now attached. "
            "IMPORTANT CAVEAT: NO2- is NOT sanazole's most abundant fragment overall - that is "
            "(NTR-yl)- at m/z 113, formed near 0 eV with ~40x higher intensity, in an "
            "exothermic reaction (-0.30 eV calculated). If 'principal DEA channel' should mean "
            "'most abundant fragment' rather than 'the NO2- channel specifically' (as tracked "
            "for the other nitroimidazole-type compounds in this table), consider using 0 eV "
            "instead of 4.0 eV - this is a real judgment call to make explicit in the manuscript."
        ),
    ),
]


def as_dataframe():
    """Convenience accessor for Stage-C correlation/scoring."""
    return pd.DataFrame([p.__dict__ for p in CANDIDATE_PROFILES])


def verified_only():
    """Return only compounds safe to use in Stage C / cite in the manuscript."""
    return [
        p for p in CANDIDATE_PROFILES
        if p.evidence_type in ("computed", "verified_primary", "verified_secondary")
        and p.resonance_energy_eV is not None
    ]


def validate_profiles():
    errors = []
    for p in CANDIDATE_PROFILES:
        if p.resonance_energy_eV is not None and p.resonance_energy_eV < 0:
            errors.append(f"{p.compound}: resonance energy < 0 eV")
        if p.resonance_width_eV is not None and p.resonance_width_eV <= 0:
            errors.append(f"{p.compound}: FWHM must be > 0")
        if p.cross_section_window_eV is not None:
            low, high = p.cross_section_window_eV
            if low >= high:
                errors.append(f"{p.compound}: invalid cross-section window")
            if low < 0:
                errors.append(f"{p.compound}: cross-section window < 0 eV")
        if p.peak_cross_section_cm2 is not None and p.peak_cross_section_cm2 <= 0:
            errors.append(f"{p.compound}: cross section must be > 0")
    return errors


if __name__ == "__main__":
    df = as_dataframe()

    print("=" * 80)
    print("STAGE B: DOMINANT DEA RESONANCE DATABASE")
    print("=" * 80)
    print("\nTotal profiles:", len(df))

    errors = validate_profiles()
    print("\nData-integrity check:", "PASS" if not errors else errors)

    print("\nBy evidence type:")
    print(df["evidence_type"].value_counts().to_string())

    verified = verified_only()
    print(f"\nCompounds safe to use in Stage C ({len(verified)}):")
    for p in verified:
        print(f"  {p.compound}: {p.resonance_energy_eV} eV ({p.evidence_type})")

    unverified = df[df["evidence_type"] == "unverified"]
    if not unverified.empty:
        print(f"\nStill unverified, excluded from Stage C ({len(unverified)}):")
        print(unverified["compound"].tolist())
