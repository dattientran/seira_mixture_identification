import hashlib
from pathlib import Path

import numpy as np
from pybaselines.whittaker import airpls

COMPOUNDS = ["NH2", "NMe2", "CHO", "COOH", "CN", "Cl", "CO2Me", "OH", "NO2", "F"]

GRID = np.arange(650.0, 3501.0, 2.0).astype(np.float32)

HELD_OUT = {
    "NH2": [5, 9, 10, 12, 14, 25, 30, 42, 46],
    "NMe2": [3, 5, 7, 13, 14, 15, 19, 28, 40, 42, 45, 47, 60, 69, 71, 75, 92, 93, 106, 112, 121, 129, 132, 134, 137,
             146, 157, 159, 162, 163, 164, 168, 170, 178, 182, 183, 187, 188, 193],
    "CHO": [3, 4, 8, 14, 23, 28, 37, 42, 45, 53, 59, 62, 63],
    "COOH": [3, 4, 10, 17, 19, 40, 41, 44, 45, 50, 51],
    "CN": [15, 16, 17, 21, 24, 34, 38, 40, 41, 49, 54],
    "Cl": [1, 2, 3, 20, 35, 41, 46, 52, 53, 54, 56],
    "CO2Me": [3, 6, 19, 20, 23, 27, 29, 31, 34, 43, 49],
    "OH": [15, 18, 22, 30, 34, 38, 43, 45, 48, 49],
    "NO2": [2, 13, 17, 18, 33, 38, 41, 43, 44],
    "F": [5, 6, 18, 21, 44, 45, 46, 48, 50, 51, 52],
}

VARIATION_REGIONS = [(650.0, 1800.0), (2800.0, 3500.0)]


def read_sample(folder):
    scans = [np.loadtxt(path, delimiter=",", skiprows=1) for path in sorted(Path(folder).glob("scan_*.csv"))]
    if not scans:
        raise FileNotFoundError(f"No scan files in {folder}")
    wavenumbers = scans[0][:, 0].astype(np.float32)
    for scan in scans:
        if not np.array_equal(scan[:, 0].astype(np.float32), wavenumbers):
            raise ValueError(f"Scans in {folder} do not share one wavenumber axis")
    return wavenumbers, np.stack([scan[:, 1] for scan in scans]).astype(np.float32)


def read_spectra(spectra_dir):
    spectra_dir = Path(spectra_dir)
    compounds = {name: read_sample(spectra_dir / "single_compounds" / name) for name in COMPOUNDS}
    mixtures = []
    folders = sorted((spectra_dir / "mixtures").iterdir(),
                     key=lambda f: [int(n) for n in f.name.split("_")[0].split("-")])
    for folder in folders:
        mixture_id, groups = folder.name.split("_", 1)
        label = np.array([name in groups.split("+") for name in COMPOUNDS])
        wavenumbers, spectra = read_sample(folder)
        mixtures.append({"id": mixture_id, "label": label, "wavenumbers": wavenumbers, "spectra": spectra})
    return compounds, mixtures


def load_inputs(spectra_dir):
    compounds, mixtures = read_spectra(spectra_dir)
    pools = [mixing_pool(*compounds[name], HELD_OUT[name]) for name in COMPOUNDS]
    single = {name: model_input(*compounds[name]) for name in COMPOUNDS}
    for m in mixtures:
        m["x"] = model_input(m["wavenumbers"], m["spectra"])
    return pools, single, mixtures


def remove_baseline(spectrum):
    baseline = airpls(spectrum.astype(np.float64), lam=1e4, diff_order=2, max_iter=50, tol=1e-3)[0]
    return spectrum - baseline.astype(np.float32)


def to_grid(wavenumbers, spectrum):
    return np.interp(GRID, wavenumbers, spectrum).astype(np.float32)


def unit_length(spectra):
    return spectra / np.linalg.norm(spectra, axis=-1, keepdims=True)


def model_input(wavenumbers, spectra):
    return unit_length(np.stack([to_grid(wavenumbers, remove_baseline(s)) for s in spectra]))


def mixing_pool(wavenumbers, spectra, held_out):
    pool = np.stack([to_grid(wavenumbers, s) for k, s in enumerate(spectra, start=1) if k not in held_out])
    return np.round(pool.astype(np.float64), 8).astype(np.float32)


def variation_seed(seed):
    digest = hashlib.sha256(f"nuisance:{seed}:seira".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 2**32


def synthetic_mixtures(pools, seed, n_compositions, n_repeats=8):
    rng = np.random.default_rng(seed)
    variation_rng = np.random.default_rng(variation_seed(seed))
    n_compounds = np.arange(1, 7)
    p_compounds = np.full(6, 1 / 6)
    p_compounds = p_compounds / p_compounds.sum()
    spectra = np.empty((n_compositions, n_repeats, len(GRID)), dtype=np.float32)
    labels = np.zeros((n_compositions, len(pools)), dtype=np.float32)
    for g in range(n_compositions):
        k = int(rng.choice(n_compounds, p=p_compounds))
        chosen = rng.choice(len(pools), size=k, replace=False)
        labels[g, chosen] = 1.0
        proportions = np.array([1.0], dtype=np.float32)
        if k > 1:
            raw = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=k))
            proportions = (raw / raw.sum()).astype(np.float32)
        for c in chosen:
            rng.integers(0, len(pools[c]))
        for m in range(n_repeats):
            weights = proportions.copy()
            if k > 1:
                jittered = proportions.astype(np.float64) * np.exp(rng.normal(0.0, 0.3, size=k))
                weights = (jittered / jittered.sum()).astype(np.float32)
            mixture = np.zeros(len(GRID), dtype=np.float64)
            for w, c in zip(weights, chosen):
                mixture += float(w) * pools[c][int(rng.integers(0, len(pools[c])))].astype(np.float64)
            spectra[g, m] = mixture.astype(np.float32)
        for m in range(n_repeats):
            spectra[g, m] = vary(spectra[g, m], variation_rng)
    corrected = np.stack([remove_baseline(s) for s in spectra.reshape(-1, len(GRID))]).astype(np.float64)
    return (corrected / np.linalg.norm(corrected, axis=1, keepdims=True)).astype(np.float32), labels


def vary(spectrum, rng):
    shift = float(rng.normal(0.0, 2.0))
    x = np.interp(GRID - shift, GRID, spectrum, left=spectrum[0], right=spectrum[-1]).astype(np.float32)
    t = np.linspace(-1.0, 1.0, len(x), dtype=np.float32)
    quadratic = t**2 - np.mean(t**2)
    x = x * np.clip(1.0 + float(rng.normal(0.0, 0.015)) * t + float(rng.normal(0.0, 0.015)) * quadratic, 0.94, 1.06)
    amplitude = float(rng.uniform(0.0, 0.006)) * float(np.abs(x).max())
    shape = float(rng.normal()) + float(rng.normal()) * t + float(rng.normal()) * quadratic
    x = x + shape / float(np.abs(shape).max()) * amplitude
    sigma = float(rng.uniform(0.001, 0.005)) * float(np.abs(x).max())
    x = x + rng.normal(0.0, sigma, size=len(x)).astype(np.float32)
    if float(rng.uniform()) < 0.35:
        x = x + fano_features(rng, float(np.abs(x).max()))
    return suppress_bands(x, rng)


def fano_features(rng, scale):
    residual = np.zeros(len(GRID), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        lo, hi = VARIATION_REGIONS[int(rng.integers(0, len(VARIATION_REGIONS)))]
        center, width = float(rng.uniform(lo, hi)), float(rng.uniform(15.0, 35.0))
        q = float(rng.uniform(0.5, 3.0)) * (-1 if float(rng.uniform()) < 0.5 else 1)
        epsilon = (GRID - center) / (width / 2)
        feature = (q + epsilon) ** 2 / (1 + epsilon**2) - 1
        residual += feature / float(np.abs(feature).max())
    return residual / float(np.abs(residual).max()) * (scale * float(rng.uniform(0.01, 0.05)))


def suppress_bands(x, rng):
    if float(rng.uniform()) >= 0.2:
        return x
    attenuation = np.ones(len(GRID), dtype=np.float32)
    for _ in range(int(rng.integers(1, 3))):
        lo, hi = VARIATION_REGIONS[int(rng.integers(0, len(VARIATION_REGIONS)))]
        center, fwhm, depth = float(rng.uniform(lo, hi)), float(rng.uniform(20.0, 60.0)), float(rng.uniform(0.05, 0.25))
        attenuation *= 1 - depth * np.exp(-0.5 * ((GRID - center) / (fwhm / 2.354820045)) ** 2)
    return x * np.clip(attenuation, 0.5, 1.0)
