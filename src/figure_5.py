import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import t as t_distribution

from src import models, spectra, stats

GRID = spectra.GRID.astype(float)
COOH = spectra.COMPOUNDS.index("COOH")
C = 10.0
CARBONYL, SHIFTED, ORIGINAL = (1682, 1704), (1388, 1400), (1404, 1416)
GAINS = {"carbonyl weight gain": np.array([1.0, 0.0, 0.0]),
         "gain near the shifted carboxylate band": np.array([0.0, 1.0, -1.0])}
CARBOXYLATE = (1370, 1430)
CO_BAND, COO_BAND = (1670, 1720), (1380, 1440)
OFFSET = 1.7


def window(lo, hi):
    return (GRID >= lo) & (GRID <= hi)


def cosine(u, v):
    return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


def held_out_means(data):
    return np.stack([data.single[name][data.held_out[name]].astype(float).mean(axis=0) for name in spectra.COMPOUNDS])


def distinctive_parts(means):
    parts = []
    for k, spectrum in enumerate(means):
        others = np.delete(means, k, axis=0).T
        parts.append(spectrum - others @ np.linalg.lstsq(others, spectrum, rcond=None)[0])
    return np.stack(parts)


def mixture_mean(data, mixture_id):
    return data.mixtures[data.ids.index(mixture_id)]["x"].astype(float).mean(axis=0)


def lc_weights(run):
    return run["theta_ref"][:-1], run["theta_folds"][:, :-1].mean(axis=0)


def similarities(weights, means, parts):
    full = np.array([cosine(weights[:, k], means[k]) for k in range(len(means))])
    distinct = np.array([cosine(weights[:, k], parts[k]) for k in range(len(means))])
    return full, distinct


def weight_changes(run):
    synthetic, combined = lc_weights(run)
    return np.array([1 - cosine(synthetic[:, k], combined[:, k]) for k in range(len(spectra.COMPOUNDS))])


def compound_f1s(run, truth):
    return stats.compound_f1(run["scores_ref"] >= 0.5, truth), stats.compound_f1(run["scores_comb"] >= 0.5, truth)


def change_f1_correlation(run, truth):
    before, after = compound_f1s(run, truth)
    return stats.pearson(weight_changes(run), after - before)


def band_position(spectrum, lo, hi):
    inside = np.flatnonzero(window(lo, hi))
    return float(GRID[inside[np.argmax(spectrum[inside])]])


def carboxylate_maximum(spectrum):
    inside = np.flatnonzero(window(*CARBOXYLATE))
    i = inside[np.argmax(spectrum[inside])]
    left, top, right = spectrum[i - 1], spectrum[i], spectrum[i + 1]
    return float(GRID[i] + (GRID[1] - GRID[0]) * 0.5 * (left - right) / (left - 2 * top + right))


def height_ratio(spectrum):
    return float(spectrum[window(*CO_BAND)].max() / spectrum[window(*COO_BAND)].max())


def positive_weight_center(weights):
    inside = window(*CARBOXYLATE)
    positive = np.clip(weights[inside], 0, None)
    return float(GRID[inside] @ positive / positive.sum())


def window_sums():
    sums = np.zeros((len(GRID) + 1, 3))
    for j, (lo, hi) in enumerate((CARBONYL, SHIFTED, ORIGINAL)):
        sums[:-1, j] = window(lo, hi)
    return sums


def fit_weights(n_synthetic, counts, held):
    if held is None:
        return np.concatenate([np.ones(n_synthetic), np.zeros(counts.sum())])
    training = np.repeat(np.arange(len(counts)) != held, counts)
    weight = models.combined_weights(n_synthetic, training.sum())
    return np.concatenate([weight[:n_synthetic], np.where(training, weight[-1], 0.0)])


def fixed_total(scores, sizes):
    return scores - sizes[:, None] / sizes.sum() * scores.sum(axis=0)


def cluster_influence(x, y, theta, counts, held, sums):
    n_synthetic = len(x) - counts.sum()
    weight = fit_weights(n_synthetic, counts, held)
    p = expit(x @ theta)
    hessian = x.T @ (x * (weight * p * (1 - p))[:, None])
    hessian[np.diag_indices(len(theta) - 1)] += 1 / C
    scores = (x @ cho_solve(cho_factor(hessian), sums)) * (weight * (y - p))[:, None]
    synthetic = scores[:n_synthetic].reshape(-1, 8, sums.shape[1]).sum(axis=1)
    experimental = np.zeros((len(counts), sums.shape[1]))
    if held is not None:
        training = np.arange(len(counts)) != held
        per_mixture = np.add.reduceat(scores[n_synthetic:], np.cumsum(counts) - counts)
        experimental[training] = fixed_total(per_mixture[training], counts[training])
    return np.vstack([fixed_total(synthetic, np.full(len(synthetic), 8)), experimental])


def cluster_covariance(influence, n_compositions):
    covariance = 0.0
    for part in (influence[:n_compositions], influence[n_compositions:]):
        covariance = covariance + len(part) / (len(part) - 1) * part.T @ part
    return covariance


def regional_wald_tests(data, seed):
    syn_x, labels = spectra.synthetic_mixtures(data.pools, seed, data.n_compositions)
    counts = np.array([len(m["x"]) for m in data.mixtures])
    x = np.column_stack([np.vstack([syn_x] + [m["x"] for m in data.mixtures]), np.ones(len(syn_x) + counts.sum())])
    y = np.concatenate([np.repeat(labels[:, COOH], 8), np.repeat(data.truth[:, COOH], counts)]).astype(float)
    run, sums = data.sets[seed], window_sums()
    synthetic = cluster_influence(x, y, run["theta_ref"][:, COOH], counts, None, sums)
    combined = np.mean([cluster_influence(x, y, theta, counts, held, sums)
                        for held, theta in enumerate(run["theta_folds"][:, :, COOH])], axis=0)
    covariance = cluster_covariance(combined - synthetic, len(labels))
    before, after = run["theta_ref"][:, COOH] @ sums, run["theta_folds"][:, :, COOH].mean(axis=0) @ sums
    tests = {}
    for name, contrast in GAINS.items():
        estimate, error = (after - before) @ contrast, np.sqrt(contrast @ covariance @ contrast)
        tests[name] = estimate, error, estimate / error, 2 * t_distribution.sf(abs(estimate / error), len(counts) - 1)
    return before, after, tests


def figure_5a(data):
    shown = window(650, 1800)
    means = held_out_means(data)
    fold = data.paper["theta_folds"][data.ids.index("2-6"), :-1]
    columns = {"Mixture 2-6 spectrum": mixture_mean(data, "2-6")}
    for name in ("CHO", "CO2Me"):
        columns[f"-{name} weight, Combined LC without 2-6"] = fold[:, spectra.COMPOUNDS.index(name)]
        columns[f"-{name} single-compound spectrum"] = means[spectra.COMPOUNDS.index(name)]
    table = pd.DataFrame({"Wavenumber (cm-1)": GRID[shown]})
    for name, values in columns.items():
        table[f"{name} (scaled)"] = values[shown] / np.abs(values[shown]).max()
    return table


def figure_5b(data):
    means = held_out_means(data)
    full, distinct = similarities(lc_weights(data.paper)[1], means, distinctive_parts(means))
    return pd.DataFrame({"Compound": spectra.COMPOUNDS, "Similarity to spectrum": full,
                         "Similarity to distinctive part": distinct})


def figure_5c(data):
    shown = window(1330, 1760)
    mixture, single = mixture_mean(data, "2-5")[shown], held_out_means(data)[COOH, shown]
    synthetic, combined = (weights[shown, COOH] for weights in lc_weights(data.paper))
    scale = max(np.abs(synthetic).max(), np.abs(combined).max())
    return pd.DataFrame({"Wavenumber (cm-1)": GRID[shown], "Mixture 2-5 spectrum (scaled)": mixture / mixture.max(),
                         "-COOH single-compound spectrum (scaled)": single / single.max(),
                         "-COOH weight, Synthetic-only LC (scaled)": synthetic / scale,
                         "-COOH weight, mean Combined LC (scaled)": combined / scale})


def figure_5d(data):
    before, after = compound_f1s(data.paper, data.truth)
    return pd.DataFrame({"Compound": spectra.COMPOUNDS, "Weight change": weight_changes(data.paper),
                         "F1, Synthetic-only LC": before, "F1, Combined LC": after})


def statistics_5a(data):
    means = held_out_means(data)
    return [stats.statistics_row("a", f"-{name} C=O band", "Position (cm-1)",
                                 band_position(means[spectra.COMPOUNDS.index(name)], *CO_BAND))
            for name in ("CO2Me", "CHO")]


def statistics_5b(data):
    means = held_out_means(data)
    full, distinct = similarities(lc_weights(data.paper)[1], means, distinctive_parts(means))
    w, p = stats.signed_rank(distinct, full)
    return [stats.statistics_row("b", "Combined LC weights, distinctive part vs spectrum", "W", w, p, len(full),
                                 "Wilcoxon signed-rank")]


def statistics_5c(data):
    mixture, single = mixture_mean(data, "2-5"), held_out_means(data)[COOH]
    bands = [("Mixture 2-5 C=O band", mixture, CO_BAND), ("-COOH carboxylate band", single, COO_BAND),
             ("Mixture 2-5 carboxylate band", mixture, COO_BAND)]
    rows = [stats.statistics_row("c", label, "Position (cm-1)", band_position(y, *band)) for label, y, band in bands]
    for label, weights in zip(("Synthetic-only LC", "mean Combined LC"), lc_weights(data.paper)):
        rows.append(stats.statistics_row("c", f"{label}, positive carboxylate weight", "Center (cm-1)",
                                         positive_weight_center(weights[:, COOH])))
    return rows


def statistics_5c_tests(data):
    before, after, tests = regional_wald_tests(data, data.paper_set)
    rows = []
    for (lo, hi), summed in zip((CARBONYL, SHIFTED, ORIGINAL), zip(before, after)):
        rows += [stats.statistics_row("c", f"{label}, {lo}-{hi} cm-1", "Summed -COOH weight", value)
                 for label, value in zip(("Synthetic-only LC", "mean Combined LC"), summed)]
    n = len(data.mixtures)
    for name, (estimate, error, t, p) in tests.items():
        rows += [stats.statistics_row("c", f"-COOH {name}", "Estimate", estimate),
                 stats.statistics_row("c", f"-COOH {name}", "SE", error),
                 stats.statistics_row("c", f"-COOH {name}", "t", t, p, n, f"Wald t-test (df {n - 1})")]
    return rows


def statistics_5d(data):
    r, p = change_f1_correlation(data.paper, data.truth)
    return [stats.statistics_row("d", "Weight change vs F1 gain", "r", r, p, len(spectra.COMPOUNDS), "Pearson")]


def figure_5(data):
    statistics = (statistics_5a(data) + statistics_5b(data) + statistics_5c(data) + statistics_5c_tests(data)
                  + statistics_5d(data))
    return {"Fig. 5a": figure_5a(data), "Fig. 5b": figure_5b(data), "Fig. 5c": figure_5c(data),
            "Fig. 5d": figure_5d(data), "Fig. 5 statistics": pd.DataFrame(statistics)}


def figure_s3(data):
    samples = {"-COOH": data.single["COOH"], "2-5": data.mixtures[data.ids.index("2-5")]["x"],
               "2-8": data.mixtures[data.ids.index("2-8")]["x"]}
    shown = window(1370, 1450) | window(1640, 1760)
    panel_a, panel_bc = pd.DataFrame({"Wavenumber (cm-1)": GRID[shown]}), []
    for i, (name, x) in enumerate(samples.items()):
        x = x.astype(float)
        mean, offset = x.mean(axis=0), OFFSET * (len(samples) - 1 - i)
        label = f"{name} (scaled, offset {offset:g})"
        panel_a[label] = mean[shown] / mean[window(*COO_BAND)].max() + offset
        panel_bc += [{"Sample": name, "Spectrum": j + 1, "C=O / COO- height ratio": height_ratio(spectrum),
                      "Carboxylate maximum (cm-1)": carboxylate_maximum(spectrum)} for j, spectrum in enumerate(x)]
    return {"Fig. S3a": panel_a, "Fig. S3b,c": pd.DataFrame(panel_bc)}


def set_weight_comparisons(data, seed, means, parts):
    run, rows = data.sets[seed], []
    for label, weights in zip(("Synthetic-only", "Combined"), lc_weights(run)):
        full, distinct = similarities(weights, means, parts)
        rows.append({"Panel": "5b", "Comparison": f"{label} LC weights, distinctive part vs spectrum",
                     "Statistic": "Compounds closer to distinctive part", "Value": int((distinct > full).sum()),
                     "P": stats.signed_rank(distinct, full)[1]})
    for name, (_, _, t, p) in regional_wald_tests(data, seed)[2].items():
        rows.append({"Panel": "5c", "Comparison": f"-COOH {name}", "Statistic": "t", "Value": t, "P": p})
    r, p = change_f1_correlation(run, data.truth)
    rows.append({"Panel": "5d", "Comparison": "Weight change vs F1 gain", "Statistic": "r", "Value": r, "P": p})
    return rows


def table_s1_rows(data):
    means = held_out_means(data)
    parts = distinctive_parts(means)
    return pd.DataFrame([{"Training set": seed, "Main text": run is data.paper, **row}
                         for seed, run in data.sets.items()
                         for row in set_weight_comparisons(data, seed, means, parts)])
