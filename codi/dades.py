import os
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import SEGMENTS, COSTATS

GRUP = {"ICA-C1-C5": "caròtida_cervical", "ICA-C6-C7": "caròtida_terminal", "M1": "M1", "M2": "M2", "M3": "M2",
        "A1A2": "ACA", "A3": "ACA", "P1P2": "ACP", "P3P4": "ACP", "BA": "basilar"}
ESCALA_TMAX = 30.0
MIDA = 256
MIN_CERVELL = 2000
FINESTRA = {"cta": (0.0, 400.0), "vasos": (0.0, 1.0)}
FENOTIP = Path(os.environ.get("ISLES_FENOTIP", "phenotype"))


def nihss(casos):
    import pandas as pd
    out = {}
    for c in casos:
        f = FENOTIP / c / "ses-01" / f"{c}_ses-01_demographic_baseline.xlsx"
        v = pd.to_numeric(pd.read_excel(f)["NIHSS at admission"], errors="coerce").iloc[0] if f.exists() else np.nan
        out[c] = torch.tensor([0.0, 1.0]) if np.isnan(v) else torch.tensor([float(v) / 42.0, 0.0])
    return out


def vas_a_prompt(vas):
    if not vas:
        return None
    if vas[:2] in ("R-", "L-"):
        costat, seg = ("dreta" if vas[0] == "R" else "esquerra"), vas[2:]
    else:
        costat, seg = "mig", vas
    if seg not in GRUP:
        return None
    return SEGMENTS.index(GRUP[seg]), COSTATS.index(costat)


def carrega(p):
    a = np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj).astype(np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def a_mida(t, mode):
    return F.interpolate(t[None], size=(MIDA, MIDA), mode=mode,
                         **({} if mode == "nearest" else {"align_corners": False}))[0]


def etiquetes_atles(mni_dir):
    a = torch.from_numpy(carrega(Path(mni_dir) / "atles_mni.nii.gz").round()).permute(2, 0, 1)
    return a_mida(a, "nearest").long()


def talls_valids(mni_dir):
    a = carrega(Path(mni_dir) / "atles_mni.nii.gz")
    n = (a > 0).sum((0, 1))
    return [z for z in range(1, a.shape[2] - 1) if n[z] > MIN_CERVELL]


class TallsMNI(Dataset):
    def __init__(self, mni_dir, vasos_json, casos, augmenta=False, entrada="ncct", aug_fort=False, n_talls=3):
        self.dir = Path(mni_dir)
        self.nt = n_talls
        self.bloc = n_talls + 1
        self.mig = n_talls // 2
        self.augmenta = augmenta
        self.aug_fort = aug_fort
        self.entrada = [] if entrada == "cap" else entrada.split("+")
        vasos = json.loads(Path(vasos_json).read_text(encoding="utf-8"))
        self.prompt = {c: vas_a_prompt(vasos.get(c, {}).get("vas")) for c in casos}
        self.casos = [c for c in casos if self.prompt[c] is not None]
        self.clin = nihss(self.casos)
        zs = talls_valids(self.dir)
        self.cau = self.dir / "cau"
        self.cau.mkdir(exist_ok=True)
        self.index = []
        for c in self.casos:
            ruta = self.cau / f"{c}.npy"
            if not ruta.exists():
                ncct = np.clip(carrega(self.dir / f"{c}_ncct.nii.gz"), 0, 80) / 80.0
                tmax = np.clip(carrega(self.dir / f"{c}_tmax.nii.gz"), 0, ESCALA_TMAX) / ESCALA_TMAX
                cob = carrega(self.dir / f"{c}_cob.nii.gz") > 0
                np.save(ruta, np.stack([ncct, tmax, cob]).transpose(0, 3, 1, 2).astype(np.float16))
            for m in self.entrada:
                rm = self.cau / f"{c}_{m}.npy"
                if m != "ncct" and not rm.exists():
                    lo, hi = FINESTRA[m]
                    v = (np.clip(carrega(self.dir / f"{c}_{m}.nii.gz"), lo, hi) - lo) / (hi - lo)
                    np.save(rm, v.transpose(2, 0, 1).astype(np.float16))
            cob = np.load(ruta, mmap_mode="r")[2].sum((1, 2))
            self.index += [(c, z) for z in zs if cob[z] > 0]

    @property
    def canals(self):
        return self.bloc * len(self.entrada)

    def _volums(self, c):
        v = np.load(self.cau / f"{c}.npy", mmap_mode="r")
        return v[0], v[1], v[2]

    def _imatge(self, c, m):
        if m == "ncct":
            return np.load(self.cau / f"{c}.npy", mmap_mode="r")[0]
        return np.load(self.cau / f"{c}_{m}.npy", mmap_mode="r")

    def _deforma(self, x, y, cb):
        ang = (torch.rand(1).item() * 2 - 1) * 8 * np.pi / 180
        esc = 1 + (torch.rand(1).item() * 2 - 1) * 0.08
        tx, ty = ((torch.rand(2) * 2 - 1) * 6 / MIDA * 2).tolist()
        m = torch.tensor([[np.cos(ang) / esc, -np.sin(ang) / esc, tx],
                          [np.sin(ang) / esc, np.cos(ang) / esc, ty]], dtype=torch.float32)[None]
        g = F.affine_grid(m, (1, 1, MIDA, MIDA), align_corners=False)
        mou = lambda t, mode: F.grid_sample(t[None], g, mode=mode, align_corners=False)[0]
        y, cb = mou(y, "bilinear"), mou(cb, "nearest")
        self._theta = m[0]
        if len(x):
            x = mou(x, "bilinear")
            for k in range(len(self.entrada)):
                i0 = self.bloc * k
                b = x[i0:i0 + self.nt]
                gamma = float(np.exp((torch.rand(1).item() * 2 - 1) * 0.2))
                b = b.clamp(0, 1) ** gamma + torch.randn_like(b) * 0.02
                x[i0:i0 + self.nt] = b.clamp(0, 1)
                x[i0 + self.nt] = torch.flip(x[i0 + self.mig], dims=[0])
        return x, y, cb

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, z = self.index[i]
        _, tmax, cob = self._volums(c)
        blocs = []
        for m in self.entrada:
            vol = self._imatge(c, m)
            zs = np.clip(np.arange(z - self.mig, z + self.mig + 1), 0, vol.shape[0] - 1)
            talls = torch.from_numpy(np.array(vol[zs], dtype=np.float32))
            blocs.append(torch.cat([talls, torch.flip(talls[self.mig:self.mig + 1], dims=[1])], 0))
        x = a_mida(torch.cat(blocs, 0), "bilinear") if blocs else torch.zeros(0, MIDA, MIDA)
        y = a_mida(torch.from_numpy(np.array(tmax[z][None], dtype=np.float32)), "bilinear")
        cb = a_mida(torch.from_numpy(np.array(cob[z][None], dtype=np.float32)), "nearest")
        seg, costat = self.prompt[c]
        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        if self.augmenta and self.aug_fort:
            x, y, cb = self._deforma(x, y, cb)
            theta = self._theta
        if self.augmenta and torch.rand(1) < 0.5:
            x, y, cb = (torch.flip(t, dims=[1]) for t in (x, y, cb))
            for k in range(len(self.entrada)):
                x[self.bloc * k + self.nt] = torch.flip(x[self.bloc * k + self.mig], dims=[0])
            costat = {0: 1, 1: 0, 2: 2}[costat]
            theta = theta * torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
        return {"ncct": x, "tmax": y, "cob": cb, "z": torch.tensor(z),
                "seg": torch.tensor(seg), "costat": torch.tensor(costat), "theta": theta, "clin": self.clin[c], "cas": c}
