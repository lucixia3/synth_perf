import argparse
from pathlib import Path

import SimpleITK as sitk

from registra_mni import llegeix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isles", type=Path, required=True)
    ap.add_argument("--vasos_dir", type=Path, default=None)
    ap.add_argument("--casos", type=Path, required=True)
    ap.add_argument("--mni_dir", type=Path, required=True)
    a = ap.parse_args()

    ref = sitk.ReadImage(str(a.mni_dir / "atles_mni.nii.gz"), sitk.sitkFloat32)
    for c in [c for c in a.casos.read_text().split() if c]:
        T = sitk.ReadTransform(str(a.mni_dir / f"{c}_mni_a_pacient.tfm"))
        cta = llegeix(a.isles / "derivatives" / c / "ses-01" / f"{c}_ses-01_space-ncct_cta.nii.gz")
        sitk.WriteImage(sitk.Resample(cta, ref, T, sitk.sitkLinear, 0.0), str(a.mni_dir / f"{c}_cta.nii.gz"))
        if a.vasos_dir is None:
            print(c, flush=True)
            continue
        seg = sitk.ReadImage(str(a.vasos_dir / f"{c}.nii.gz"))
        vasos = sitk.Cast(seg > 0, sitk.sitkFloat32)
        sitk.WriteImage(sitk.Resample(vasos, ref, T, sitk.sitkLinear, 0.0), str(a.mni_dir / f"{c}_vasos.nii.gz"))
        print(c, flush=True)


if __name__ == "__main__":
    main()
