import csv
import json
import argparse
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from dades import TallsMNI, ESCALA_TMAX, etiquetes_atles, vas_a_prompt
from model import Generador

LLINDAR = 6.0
ENSEMBLES = {"ens_cta": ["cta", "cta_lesio"]}


def carrega_model(d, dev):
    cfg = json.loads((d / "config.json").read_text())
    entrada = cfg["entrada"]
    n = 0 if entrada == "cap" else len(entrada.split("+"))
    taula = None if cfg["taula"] == "None" else np.load(cfg["taula"])
    G = Generador(etiquetes_atles(cfg["mni_dir"]), c_in=(int(cfg.get("n_talls", 3)) + 1) * n, taula_inicial=taula,
                  dropout=float(cfg.get("dropout", 0.0)),
                  canals=tuple(int(c) for c in cfg.get("canals", "32,64,128,256").split(",")),
                  n_clin=2 if cfg.get("nihss") == "True" else 0,
                  simetria=cfg.get("simetria") == "True").to(dev)
    G.load_state_dict(torch.load(d / "millor.pt", map_location=dev, weights_only=False)["G"])
    return G.eval(), cfg


@torch.no_grad()
def prediu_mni(G, ds, cas, dev, tta):
    forma = ds._volums(cas)[1].shape
    vol = np.zeros(forma, np.float32)
    cobert = np.zeros(forma, bool)
    idx = [i for i, (c, _) in enumerate(ds.index) if c == cas]
    for k in range(0, len(idx), 16):
        bs = [ds[i] for i in idx[k:k + 16]]
        x = torch.stack([b["ncct"] for b in bs]).to(dev)
        z = torch.stack([b["z"] for b in bs]).to(dev)
        seg = torch.stack([b["seg"] for b in bs]).to(dev)
        costat = torch.stack([b["costat"] for b in bs]).to(dev)
        clin = torch.stack([b["clin"] for b in bs]).to(dev)
        p = G(x, z, seg, costat, clin=clin)[0]
        if tta:
            pf = G(torch.flip(x, dims=[2]), z, seg, torch.tensor([1, 0, 2], device=dev)[costat], clin=clin)[0]
            p = (p + torch.flip(pf, dims=[2])) / 2
        p = F.interpolate(p, size=forma[1:], mode="bilinear", align_corners=False)[:, 0]
        for b, pi in zip(bs, p.cpu().numpy()):
            vol[int(b["z"])] = pi * ESCALA_TMAX
            cobert[int(b["z"])] = True
    return vol, cobert


def a_pacient(vol_zxy, mni_dir, cas, ref_pac, tmp, interp):
    canon = nib.as_closest_canonical(nib.load(str(Path(mni_dir) / "atles_mni.nii.gz")))
    ruta = Path(tmp) / "v.nii.gz"
    nib.save(nib.Nifti1Image(vol_zxy.transpose(1, 2, 0).astype(np.float32), canon.affine), str(ruta))
    im = sitk.ReadImage(str(ruta), sitk.sitkFloat32)
    T = sitk.ReadTransform(str(Path(mni_dir) / f"{cas}_mni_a_pacient.tfm")).GetInverse()
    return sitk.GetArrayFromImage(sitk.Resample(im, ref_pac, T, interp, 0.0))


def metriques(pred, real, mascara, ml_voxel):
    m = mascara & (real > 0)
    les = m & (real > LLINDAR)
    pr, rl = (pred > LLINDAR) & m, les
    v_pr, v_rl = pr.sum() * ml_voxel, rl.sum() * ml_voxel
    return {"mae_encefal": float(np.abs(pred - real)[m].mean()),
            "mae_lesio": float(np.abs(pred - real)[les].mean()) if les.any() else np.nan,
            "dice_t6": float(2 * (pr & rl).sum() / (pr.sum() + rl.sum())) if (pr.sum() + rl.sum()) else np.nan,
            "vol_real_ml": float(v_rl), "vol_pred_ml": float(v_pr), "err_vol_ml": float(abs(v_pr - v_rl))}


def bootstrap(a, b, n=5000, llavor=0):
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    r = np.random.default_rng(llavor)
    mostres = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(mostres, 2.5), np.percentile(mostres, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, default=Path("exp"))
    ap.add_argument("--models", default="cta,cta_lesio")
    ap.add_argument("--isles", type=Path, required=True)
    ap.add_argument("--vasos", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--ref", default="cta_tta")
    ap.add_argument("--out", type=Path, default=Path("exp/avaluacio"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    llegeix = lambda p: [c for c in p.read_text().split() if c]
    vasos = json.loads(a.vasos.read_text(encoding="utf-8"))
    casos = [c for c in llegeix(a.val) if vas_a_prompt(vasos.get(c, {}).get("vas")) is not None]

    mni_ref = json.loads((a.exp / a.models.split(",")[0] / "config.json").read_text())["mni_dir"]
    dtr = TallsMNI(mni_ref, a.vasos, llegeix(a.train), entrada="cap")
    mostra = np.concatenate([dtr._volums(c)[1][dtr._volums(c)[2] > 0][::50] for c in dtr.casos])
    constant = float(np.median(mostra.astype(np.float32))) * ESCALA_TMAX
    print(f"nul constant (mediana train): {constant:.2f} s", flush=True)

    noms = [n for n in a.models.split(",") if (a.exp / n / "millor.pt").exists()]
    print("models sense millor.pt, s'ometen:", [n for n in a.models.split(",") if n not in noms], flush=True)
    membres = {m for ms in ENSEMBLES.values() for m in ms}
    cfgs = {n: json.loads((a.exp / n / "config.json").read_text()) for n in noms}
    pacient, mascara, guardades, files = {}, {}, {}, []
    afegeix = lambda n, c, p: files.append({"model": n, "cas": c, "vas": vasos[c]["vas"],
                                            **metriques(p, pacient[c][1], mascara[c], pacient[c][2])})
    with tempfile.TemporaryDirectory() as tmp:
        for c in casos:
            ref = sitk.ReadImage(str(a.isles / "raw_data" / c / "ses-01" / f"{c}_ses-01_ncct.nii.gz"), sitk.sitkFloat32)
            real = sitk.ReadImage(str(a.isles / "derivatives" / c / "ses-01" / "perfusion-maps" /
                                      f"{c}_ses-01_space-ncct_tmax.nii.gz"), sitk.sitkFloat32)
            r = np.nan_to_num(sitk.GetArrayFromImage(real), nan=0.0, posinf=0.0, neginf=0.0)
            pacient[c] = (ref, np.clip(r, 0, ESCALA_TMAX), np.prod(ref.GetSpacing()) / 1000)
        for d in sorted({cfg["mni_dir"] for cfg in cfgs.values()}):
            ds = TallsMNI(d, a.vasos, casos, entrada="cap")
            for c in casos:
                cob = np.zeros(ds._volums(c)[1].shape, np.float32)
                for cc, z in ds.index:
                    if cc == c:
                        cob[z] = ds._volums(c)[2][z]
                m = a_pacient(cob, d, c, pacient[c][0], tmp, sitk.sitkLinear) > 0.5
                mascara[c] = m if c not in mascara else mascara[c] & m
        for n in noms:
            G, cfg = carrega_model(a.exp / n, dev)
            ds = TallsMNI(cfg["mni_dir"], a.vasos, casos, entrada=cfg["entrada"], n_talls=int(cfg.get("n_talls", 3)))
            for c in casos:
                for tta in (False, True):
                    vol, _ = prediu_mni(G, ds, c, dev, tta)
                    p = a_pacient(vol, cfg["mni_dir"], c, pacient[c][0], tmp, sitk.sitkLinear)
                    afegeix(n + ("_tta" if tta else ""), c, p)
                    if tta and n in membres:
                        guardades[(n, c)] = p.astype(np.float16)
            print(f"{n} fet", flush=True)
            del G
    for e, ms in ENSEMBLES.items():
        if all(m in noms for m in ms):
            for c in casos:
                afegeix(e + "_tta", c, np.mean([guardades[(m, c)].astype(np.float32) for m in ms], 0))
    for c in casos:
        afegeix("nul_constant", c, np.full_like(pacient[c][1], constant))

    camps = list(files[0])
    with open(a.out / "per_cas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, camps)
        w.writeheader()
        w.writerows(files)

    taula = lambda n, k: np.array([f[k] for f in files if f["model"] == n], dtype=float)
    mets = ["mae_encefal", "mae_lesio", "dice_t6", "err_vol_ml"]
    linies = [f"{len(casos)} casos de val, espai pacient, màscara comuna a tots els models\n",
              f"{'model':26s}" + "".join(f"{k:>14s}" for k in mets)]
    models = list(dict.fromkeys(f["model"] for f in files))
    for n in models:
        linies.append(f"{n:26s}" + "".join(f"{np.nanmean(taula(n, k)):14.3f}" for k in mets))
    linies.append(f"\ndiferència pareada contra {a.ref} (mitjana [IC95 bootstrap]), negatiu = millor en MAE/volum\n")
    for n in models:
        if n == a.ref:
            continue
        cel = []
        for k in mets:
            m, lo, hi = bootstrap(taula(n, k), taula(a.ref, k))
            cel.append(f"{k} {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        linies.append(f"{n:26s} " + "  ".join(cel))
    text = "\n".join(linies)
    (a.out / "resum.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
