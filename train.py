"""Synthetic data generation and model fitting."""
from pathlib import Path

import numpy as np
import pandas as pd

from src import models, spectra

RESULTS = Path("results")
PAPER_SET = 2026092815
TRAINING_SETS = [2026052411] + list(range(2026092801, 2026092821))
VALIDATION_SET = 2026052412
N_COMPOSITIONS = 2500
NETWORK_SEEDS = range(20260534, 20260539)
MASKED = ["4-2", "6-2", "6-5"]
NH2 = spectra.COMPOUNDS.index("NH2")


def labels_of(mixtures):
    return np.concatenate([np.repeat(m["label"][None].astype(float), len(m["x"]), axis=0) for m in mixtures])


def combined_folds(syn_x, syn_y, mixtures, start, ratio=10.0):
    folds, scores = [], []
    for held, mixture in enumerate(mixtures):
        train = [m for m in mixtures if m is not mixture]
        exp_x = np.concatenate([m["x"] for m in train])
        weight = models.combined_weights(len(syn_x), len(exp_x), ratio)
        theta = models.fit_lc(np.concatenate([syn_x, exp_x]), np.concatenate([syn_y, labels_of(train)]), weight,
                              start=start)
        folds.append(theta)
        scores.append(models.lc_scores(theta, [mixture["x"]])[0])
        print(f"    Combined fold {held + 1}/{len(mixtures)}", flush=True)
    return np.stack(folds), np.stack(scores)


def masked_nh2_scores(syn_x, syn_y, mixtures, start):
    scores = []
    for mixture in mixtures:
        train = [m for m in mixtures if m is not mixture]
        used = sum(len(m["x"]) for m in train if m["id"] not in MASKED)
        weight = [np.full(len(m["x"]), 0.0 if m["id"] in MASKED else 10.0 * len(syn_x) / used) for m in train]
        x = np.concatenate([syn_x] + [m["x"] for m in train])
        y = np.concatenate([syn_y[:, [NH2]], labels_of(train)[:, [NH2]]])
        theta = models.fit_lc(x, y, np.concatenate([np.ones(len(syn_x))] + weight), start=start[:, [NH2]])
        scores.append(models.lc_scores(theta, [mixture["x"]])[0, 0])
    return np.array(scores)


def network_scores(syn_x, syn_y, valid_x, valid_y, mixtures):
    out = []
    for seed in NETWORK_SEEDS:
        network = models.train_network(syn_x, syn_y, valid_x, valid_y, seed)
        out.append(models.network_scores(network, [m["x"] for m in mixtures]))
    return np.stack(out)


def train_set(seed, pools, mixtures, valid_x, valid_y):
    path = RESULTS / "training_sets" / f"{seed}.npz"
    if path.exists():
        return
    print(f"training set {seed}", flush=True)
    syn_x, labels = spectra.synthetic_mixtures(pools, seed, N_COMPOSITIONS)
    syn_y = np.repeat(labels, 8, axis=0)
    theta_ref = models.fit_lc(syn_x, syn_y)
    theta_folds, scores_comb = combined_folds(syn_x, syn_y, mixtures, theta_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, theta_ref=theta_ref, scores_ref=models.lc_scores(theta_ref, [m["x"] for m in mixtures]),
             theta_folds=theta_folds, scores_comb=scores_comb,
             masked_scores=masked_nh2_scores(syn_x, syn_y, mixtures, theta_ref),
             network_scores=network_scores(syn_x, syn_y, valid_x, valid_y, mixtures))


def experimental_only(mixtures):
    path = RESULTS / "experimental_only.npz"
    if path.exists():
        return
    print("Experimental-only LC", flush=True)
    scores = []
    for mixture in mixtures:
        train = [m for m in mixtures if m is not mixture]
        x, y, theta = np.concatenate([m["x"] for m in train]), labels_of(train), None
        for C in (100.0, 1e4, 2e5):
            theta = models.fit_lc(x, y, C=C, start=theta, max_iter=200)
        scores.append(models.lc_scores(theta, [mixture["x"]])[0])
    np.savez(path, scores=np.stack(scores))


def comparison_methods(pools, mixtures, valid_x, valid_y):
    path = RESULTS / "comparison_methods.npz"
    if path.exists():
        return
    print("spectral reconstruction and peak matching", flush=True)
    groups = valid_x.reshape(len(valid_y), 8, -1)
    references = models.medoid_spectra(pools)
    tune = np.random.default_rng(20260551).permutation(len(valid_y))[:1000]
    threshold = models.reconstruction_threshold(models.reconstruction_scores(references, groups[tune]), valid_y[tune])
    signatures = models.peak_signatures(pools)
    valid_peaks = models.peak_matching_scores(signatures, valid_x).reshape(len(valid_y), 8, -1).mean(axis=1)
    np.savez(path, reconstruction=models.reconstruction_scores(references, [m["x"] for m in mixtures]),
             reconstruction_threshold=threshold,
             peak_matching=np.stack([models.peak_matching_scores(signatures, m["x"]).mean(axis=0) for m in mixtures]),
             peak_thresholds=models.peak_matching_thresholds(valid_peaks, valid_y))


def weight_ratios(syn_x, syn_y, theta_ref, mixtures):
    path = RESULTS / "weight_ratios.npz"
    if path.exists():
        return
    print("weight ratios 3:1 and 30:1", flush=True)
    scores = {f"ratio_{r:g}": combined_folds(syn_x, syn_y, mixtures, theta_ref, ratio=r)[1] for r in (3.0, 30.0)}
    np.savez(path, **scores)


def experimental_mixture_curve(syn_x, syn_y, theta_ref, mixtures):
    path = RESULTS / "experimental_mixture_curve.npz"
    if path.exists():
        return
    print("learning curve over experimental mixtures (600 fits)", flush=True)
    sizes, rng = (1, 2, 4, 8, 12, 16), np.random.default_rng(20260920)
    scores = np.zeros((len(sizes), 5, len(mixtures), len(spectra.COMPOUNDS)))
    for held, mixture in enumerate(mixtures):
        others = np.array([i for i in range(len(mixtures)) if i != held])
        per_spectrum = 10.0 * len(syn_x) / sum(len(mixtures[i]["x"]) for i in others)
        for a, size in enumerate(sizes):
            for b in range(5):
                train = [mixtures[i] for i in rng.choice(others, size, replace=False)]
                exp_x = np.concatenate([m["x"] for m in train])
                weight = np.concatenate([np.ones(len(syn_x)), np.full(len(exp_x), per_spectrum)])
                theta = models.fit_lc(np.concatenate([syn_x, exp_x]), np.concatenate([syn_y, labels_of(train)]), weight,
                                      start=theta_ref)
                scores[a, b, held] = models.lc_scores(theta, [mixture["x"]])[0]
        print(f"    held-out mixture {held + 1}/{len(mixtures)}", flush=True)
    np.savez(path, sizes=np.array(sizes), scores=scores)


def synthetic_size_curve(pools, mixtures):
    path = RESULTS / "synthetic_size_curve.npz"
    if path.exists():
        return
    print("learning curve over synthetic set size (35 fits)", flush=True)
    sizes = [2000, 5000, 10000, 20000, 40000, 80000, 160000]
    scores = np.zeros((len(sizes), 5, len(mixtures), len(spectra.COMPOUNDS)))
    for a, n in enumerate(sizes):
        for b in range(5):
            x, labels = spectra.synthetic_mixtures(pools, 2026093011 + 10 * a + b, n // 8)
            scores[a, b] = models.lc_scores(models.fit_lc(x, np.repeat(labels, 8, axis=0)), [m["x"] for m in mixtures])
        print(f"    {n} spectra", flush=True)
    np.savez(path, sizes=np.array(sizes), scores=scores)


def save_paper_models():
    with np.load(RESULTS / "training_sets" / f"{PAPER_SET}.npz") as run:
        for name, theta in (("lc_synthetic_only", run["theta_ref"]), ("lc_combined", run["theta_folds"].mean(axis=0))):
            index = pd.Index([f"{w:g}" for w in spectra.GRID] + ["intercept"], name="wavenumber_cm-1")
            pd.DataFrame(theta, index=index, columns=spectra.COMPOUNDS).to_csv(RESULTS / f"{name}.csv")


def main():
    RESULTS.mkdir(exist_ok=True)
    pools, _, mixtures = spectra.load_inputs("data/spectra")
    valid_x, valid_labels = spectra.synthetic_mixtures(pools, VALIDATION_SET, 2000)
    valid_y = np.repeat(valid_labels, 8, axis=0)
    comparison_methods(pools, mixtures, valid_x, valid_labels)
    experimental_only(mixtures)
    for seed in [PAPER_SET] + [s for s in TRAINING_SETS if s != PAPER_SET]:
        train_set(seed, pools, mixtures, valid_x, valid_y)
    save_paper_models()
    syn_x, labels = spectra.synthetic_mixtures(pools, PAPER_SET, N_COMPOSITIONS)
    syn_y = np.repeat(labels, 8, axis=0)
    with np.load(RESULTS / "training_sets" / f"{PAPER_SET}.npz") as run:
        theta_ref = run["theta_ref"]
    weight_ratios(syn_x, syn_y, theta_ref, mixtures)
    experimental_mixture_curve(syn_x, syn_y, theta_ref, mixtures)
    synthetic_size_curve(pools, mixtures)
    print("done: results/", flush=True)


if __name__ == "__main__":
    main()
