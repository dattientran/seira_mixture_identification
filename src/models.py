import copy

import numpy as np
import torch
from scipy.optimize import nnls
from scipy.signal import find_peaks, peak_prominences
from sklearn.metrics import f1_score, precision_recall_curve

from src.spectra import GRID, remove_baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_lc(x, y, weight=None, C=10.0, start=None, max_iter=60):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=DEVICE)
    n, p = x.shape
    k = y.shape[1]
    weight = torch.ones(n, dtype=torch.float64, device=DEVICE) if weight is None else \
        torch.as_tensor(np.asarray(weight), dtype=torch.float64, device=DEVICE)
    xa = torch.cat([x, torch.ones(n, 1, dtype=torch.float64, device=DEVICE)], 1)
    xa32 = xa.float() if n > 5000 else None
    ridge = torch.zeros(p + 1, dtype=torch.float64, device=DEVICE)
    ridge[:p] = 1.0 / C
    theta = torch.zeros(p + 1, k, dtype=torch.float64, device=DEVICE) if start is None else \
        torch.as_tensor(np.asarray(start), dtype=torch.float64, device=DEVICE).clone()
    total = float(weight.sum())
    diagonal = torch.arange(p + 1, device=DEVICE)

    def objective(t):
        z = xa @ t
        loss = (weight[:, None] * (torch.nn.functional.softplus(z) - y * z)).sum()
        return float(loss + 0.5 * (ridge[:, None] * t**2).sum())

    for _ in range(max_iter):
        p_hat = torch.sigmoid(xa @ theta)
        gradient = xa.T @ (weight[:, None] * (p_hat - y)) + ridge[:, None] * theta
        if float(gradient.abs().max()) / total < 1e-9:
            break
        curvature = weight[:, None] * p_hat * (1 - p_hat)
        step = torch.empty_like(theta)
        for j in range(k):
            if xa32 is not None:
                hessian = (xa32.T @ (xa32 * curvature[:, j:j + 1].float())).double()
            else:
                hessian = xa.T @ (xa * curvature[:, j:j + 1])
            hessian[diagonal, diagonal] += ridge
            step[:, j] = torch.linalg.solve(hessian, gradient[:, j])
        current, slope, scale = objective(theta), float((gradient * step).sum()), 1.0
        while scale > 1e-6 and objective(theta - scale * step) > current - 1e-4 * scale * slope:
            scale *= 0.5
        theta = theta - scale * step
    else:
        raise RuntimeError(f"LC fit did not converge in {max_iter} Newton steps")
    return theta.cpu().numpy()


def lc_scores(theta, mixtures):
    return np.stack([(1 / (1 + np.exp(-(np.asarray(x, dtype=np.float64) @ theta[:-1] + theta[-1])))).mean(0)
                     for x in mixtures])


def combined_weights(n_synthetic, n_experimental, ratio=10.0):
    return np.concatenate([np.ones(n_synthetic), np.full(n_experimental, ratio * n_synthetic / n_experimental)])


def train_network(train_x, train_y, valid_x, valid_y, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(128, train_y.shape[1])).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    train_x, train_y = (torch.as_tensor(a, dtype=torch.float32, device=DEVICE) for a in (train_x, train_y))
    valid_x, valid_y = (torch.as_tensor(a, dtype=torch.float32, device=DEVICE) for a in (valid_x, valid_y))
    best, best_state, stale = float("inf"), None, 0
    for _ in range(50):
        model.train()
        order = torch.randperm(len(train_x), device=DEVICE, generator=generator)
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(train_x[batch]), train_y[batch]).backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            loss = float(np.mean([loss_fn(model(valid_x[s:s + 8192]), valid_y[s:s + 8192]).item()
                                  for s in range(0, len(valid_x), 8192)]))
        if loss < best - 1e-6:
            best, best_state, stale = loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 20:
                break
    model.load_state_dict(best_state)
    return model.eval()


def network_scores(model, mixtures):
    scores = []
    with torch.no_grad():
        for x in mixtures:
            outputs = model(torch.as_tensor(x, dtype=torch.float32, device=DEVICE))
            scores.append(torch.sigmoid(outputs).mean(0).cpu().numpy())
    return np.stack(scores)


def medoid_spectra(pools):
    columns = []
    for pool in pools:
        rows = np.stack([remove_baseline(s) for s in pool]).astype(np.float64)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        columns.append(rows[np.argmin((1 - rows @ rows.T).sum(axis=1))])
    return np.stack(columns, axis=1)


def reconstruction_scores(references, mixtures):
    return np.stack([np.mean([nnls(references, x)[0] for x in spectra], axis=0) for spectra in mixtures])


def coefficient_fractions(coefficients):
    return coefficients / np.maximum(coefficients.sum(axis=1, keepdims=True), 1e-12)


def reconstruction_threshold(coefficients, labels):
    fractions = coefficient_fractions(coefficients)
    candidates = np.unique(np.concatenate([[fractions.min()], np.quantile(fractions, np.linspace(0, 1, 401)),
                                           [fractions.max() + np.finfo(float).eps]]))
    return max((f1_score(labels, fractions >= c, average="samples"), float(c)) for c in candidates)[1]


def peak_signatures(pools, n_peaks=20):
    signatures = []
    for pool in pools:
        prototype = np.median(np.stack([remove_baseline(s) for s in pool]), axis=0).astype(np.float32)
        prototype = (prototype / np.linalg.norm(prototype)).astype(np.float32)
        detect = prototype.astype(np.float64)
        detect = (detect - detect.min()) / (detect.max() - detect.min())
        peaks = find_peaks(detect, prominence=0.03)[0]
        peaks = peaks[prototype[peaks] > 0.0]
        if len(peaks) > n_peaks:
            peaks = np.sort(peaks[np.argsort(peak_prominences(detect, peaks)[0])[::-1][:n_peaks]])
        intensities = prototype[peaks]
        signatures.append((GRID[peaks], (intensities / np.max(np.abs(intensities))).astype(np.float32)))
    return signatures


def peak_matching_scores(signatures, spectra, tolerance=15.0):
    x = (spectra / np.linalg.norm(spectra, axis=1, keepdims=True)).astype(np.float32)
    scores = np.zeros((len(x), len(signatures)), dtype=np.float32)
    for j, (positions, intensities) in enumerate(signatures):
        pooled = np.stack([x[:, np.abs(GRID - float(p)) <= tolerance].max(axis=1) for p in positions], axis=1)
        scores[:, j] = pooled.astype(np.float64) @ intensities.astype(np.float64)
    return scores


def peak_matching_thresholds(scores, labels):
    thresholds = np.zeros(scores.shape[1], dtype=np.float32)
    for j in range(scores.shape[1]):
        precision, recall, candidates = precision_recall_curve(labels[:, j] >= 0.5, scores[:, j])
        total = precision[:-1] + recall[:-1]
        f1 = np.divide(2 * precision[:-1] * recall[:-1], total, out=np.zeros(len(candidates)), where=total > 1e-12)
        thresholds[j] = candidates[np.flatnonzero(f1 == f1.max())[-1]]
    return thresholds
