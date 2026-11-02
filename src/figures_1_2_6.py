import itertools

import numpy as np
import pandas as pd
from scipy.optimize import brentq, nnls

from src import stats
from src.spectra import COMPOUNDS, GRID, remove_baseline, unit_length

WAVENUMBER = "Wavenumber (cm-1)"
WAVENUMBERS = GRID.astype(float)
FINGERPRINT = GRID <= 1800
FIGURE_ORDER = ["NO2", "F", "OH", "CO2Me", "CHO", "CN", "COOH", "Cl", "NH2", "NMe2"]
MIXTURE_PANELS = {"Fig. 1c": ("2-2", ["NO2", "NH2"]), "Fig. 1d": ("4-5", ["NH2", "NMe2", "CN", "F"]),
                  "Fig. 1e": ("6-1", ["NO2", "COOH", "CHO", "CO2Me", "OH", "Cl"])}
EXAMPLE_PROPORTIONS = {"NO2": 0.24, "F": 0.44, "CO2Me": 0.15, "CN": 0.17}
EQUAL_RMSE = "RMSE increase, equal contribution instead of zero (%)"


def find_mixture(data, mixture_id):
    return data.mixtures[data.ids.index(mixture_id)]


def constituents(mixture):
    return [name for name, present in zip(COMPOUNDS, mixture["label"]) if present]


def mean_spectra(data):
    return {name: data.single[name].astype(float).mean(axis=0) for name in COMPOUNDS}


def mean_spectrum(mixture):
    return mixture["x"].astype(float).mean(axis=0)


def additive_fit(spectrum, references):
    coefficients = nnls(references.T, spectrum)[0]
    fit = coefficients @ references
    residual = spectrum - fit
    return coefficients, fit, 100 * (1 - residual @ residual / (spectrum @ spectrum))


def contributions(coefficients):
    return coefficients / coefficients.sum()


def mixture_fit(mixture, means):
    return additive_fit(mean_spectrum(mixture), np.stack([means[name] for name in constituents(mixture)]))


def fitted_contributions(data, means):
    rows = []
    for mixture in data.mixtures:
        for name, value in zip(constituents(mixture), contributions(mixture_fit(mixture, means)[0])):
            rows.append({"Mixture": mixture["id"], "Compound": name, "Contribution (%)": 100 * value})
    return pd.DataFrame(rows)


def fixed_contribution_error(spectrum, references, k, contribution):
    design = (1 - contribution) * np.delete(references, k, axis=0) + contribution * references[k]
    coefficients = nnls(design.T, spectrum)[0]
    residual = spectrum - coefficients @ design
    return residual @ residual


def contribution_in_each_spectrum(data, means, mixture_id="6-4", compound="COOH"):
    mixture = find_mixture(data, mixture_id)
    names = constituents(mixture)
    references, k = np.stack([means[name] for name in names]), names.index(compound)
    rows = []
    for number, spectrum in enumerate(mixture["x"].astype(float), start=1):
        at_zero = fixed_contribution_error(spectrum, references, k, 0.0)
        at_equal = fixed_contribution_error(spectrum, references, k, 1 / len(names))
        rows.append({"Mixture": mixture_id, "Compound": compound, "Spectrum": number,
                     "Contribution (%)": 100 * contributions(additive_fit(spectrum, references)[0])[k],
                     EQUAL_RMSE: 100 * (np.sqrt(at_equal / at_zero) - 1)})
    return pd.DataFrame(rows)


def spectra_table(columns, shown=slice(None)):
    return pd.DataFrame({WAVENUMBER: WAVENUMBERS[shown], **{label: values[shown] for label, values in columns.items()}})


def presence_score(logit):
    return 1 / (1 + np.exp(-logit))


def lc_logit(theta, x):
    return x @ theta[:-1] + theta[-1]


def fold_weights(data, mixture_id, compound):
    theta, k = data.paper["theta_folds"][data.ids.index(mixture_id)], COMPOUNDS.index(compound)
    return theta[:-1, k], theta[-1, k]


def partner_mean(means, partners):
    return np.mean([means[name] for name in partners], axis=0)


def normalized_sum_logit(contribution, target, partners, w, b):
    x = contribution * target + (1 - contribution) * partners
    return x @ w / np.linalg.norm(x, axis=-1) + b


def detection_threshold(target, partners, w, b):
    return brentq(normalized_sum_logit, 0.0, 1.0, args=(target, partners, w, b))


def individual_combinations(data):
    rng = np.random.default_rng(20260927)
    combinations = {}
    for mixture in data.mixtures:
        sizes = [len(data.single[name]) for name in constituents(mixture)]
        if np.prod(sizes) <= 20000:
            combinations[mixture["id"]] = np.array(list(itertools.product(*[range(n) for n in sizes])))
        else:
            combinations[mixture["id"]] = np.stack([rng.choice(n, size=4000) for n in sizes], axis=1)
    return combinations


def combination_spectra(data, mixture_id, compound):
    names = constituents(find_mixture(data, mixture_id))
    picks = individual_combinations(data)[mixture_id]
    spectra = [data.single[name][picks[:, j]].astype(float) for j, name in enumerate(names)]
    partners = sum(s for s, name in zip(spectra, names) if name != compound) / (len(names) - 1)
    return spectra[names.index(compound)], partners


def mixture_spectra_panel(data, mixture_id, compounds):
    samples = [(f"Mixture {mixture_id}", find_mixture(data, mixture_id)["x"])]
    samples += [(f"-{name}", data.single[name]) for name in compounds]
    columns = {}
    for label, spectra in samples:
        spectra = spectra.astype(float)
        columns.update({f"{label} spectrum {k} (a.u.)": spectrum for k, spectrum in enumerate(spectra, start=1)})
        columns[f"{label} mean spectrum (a.u.)"] = spectra.mean(axis=0)
    return spectra_table(columns, (GRID >= 760) & (GRID <= 1800))


def principal_components(spectra):
    centred = spectra - spectra.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ components.T
    scores *= np.sign(scores[np.abs(scores).argmax(axis=0), np.arange(scores.shape[1])])
    return scores, singular_values**2 / np.sum(singular_values**2)


def principal_component_panel(data):
    spectra = np.concatenate([data.single[name][data.held_out[name]] for name in COMPOUNDS]).astype(float)
    scores, variance = principal_components(spectra)
    table = pd.DataFrame([(name, number) for name in COMPOUNDS for number in np.flatnonzero(data.held_out[name]) + 1],
                         columns=["Compound", "Spectrum"])
    table["Principal component 1 (a.u.)"], table["Principal component 2 (a.u.)"] = scores[:, 0], scores[:, 1]
    return table, variance


def additive_fit_panel(data, means, mixture_id="6-2"):
    mixture = find_mixture(data, mixture_id)
    spectrum = mean_spectrum(mixture)
    _, fit, accounted = mixture_fit(mixture, means)
    table = spectra_table({f"Mixture {mixture_id} mean spectrum (a.u.)": spectrum, "Additive fit (a.u.)": fit,
                           "Residual (a.u.)": spectrum - fit}, FINGERPRINT)
    return table, accounted


def additivity_of_every_spectrum(data, means):
    rows = []
    for mixture in data.mixtures:
        references = np.stack([means[name] for name in constituents(mixture)])
        for number, spectrum in enumerate(mixture["x"].astype(float), start=1):
            rows.append({"Mixture": mixture["id"], "Spectrum": number, "Compounds in the mixture": len(references),
                         "Accounted for by additive fit (%)": additive_fit(spectrum, references)[2]})
    return pd.DataFrame(rows)


def box_statistics(values):
    rows = []
    for size, group in values.groupby("Compounds in the mixture"):
        x = group["Accounted for by additive fit (%)"].to_numpy()
        p5, q1, median, q3, p95 = np.percentile(x, [5, 25, 50, 75, 95])
        rows.append({"Compounds in the mixture": size, "Spectra": len(x), "Lower whisker (%)": x[x >= p5].min(),
                     "First quartile (%)": q1, "Median (%)": median, "Third quartile (%)": q3,
                     "Upper whisker (%)": x[x <= p95].max()})
    return pd.DataFrame(rows)


def figure_1(data):
    means = mean_spectra(data)
    sheets = {name: mixture_spectra_panel(data, *panel) for name, panel in MIXTURE_PANELS.items()}
    sheets["Fig. 1f"], variance = principal_component_panel(data)
    sheets["Fig. 1g"], accounted = additive_fit_panel(data, means)
    sheets["Fig. 1h"] = additivity_of_every_spectrum(data, means)
    sheets["Fig. 1h boxes"] = box_statistics(sheets["Fig. 1h"])
    n = len(sheets["Fig. 1f"])
    rows = [stats.statistics_row("f", f"Principal component {k + 1}", "Variance explained (%)", 100 * variance[k], n=n)
            for k in range(2)]
    rows += [stats.statistics_row("f", "Principal components 1 and 2", "Variance explained (%)",
                                  100 * variance[:2].sum(), n=n),
             stats.statistics_row("g", "Mixture 6-2", "Accounted for by additive fit (%)", accounted)]
    sheets["Fig. 1 statistics"] = pd.DataFrame(rows)
    return sheets


def synthetic_example(data):
    singles = {name: unit_length(data.pools[COMPOUNDS.index(name)].astype(float).mean(axis=0))
               for name in EXAMPLE_PROPORTIONS}
    weighted_sum = unit_length(sum(p * singles[name] for name, p in EXAMPLE_PROPORTIONS.items()))
    corrected = remove_baseline(weighted_sum)
    return singles, weighted_sum, weighted_sum - corrected, unit_length(corrected)


def shifted(spectrum, shift):
    return np.interp(WAVENUMBERS - shift, WAVENUMBERS, spectrum, left=spectrum[0], right=spectrum[-1])


def fano_feature(center, width, q):
    epsilon = (WAVENUMBERS - center) / (width / 2)
    feature = (q + epsilon) ** 2 / (1 + epsilon**2) - 1
    return feature / np.abs(feature).max()


def suppression(center, fwhm, depth):
    return np.clip(1 - depth * np.exp(-0.5 * ((WAVENUMBERS - center) / (fwhm / 2.354820045)) ** 2), 0.5, 1)


def spectral_variations(synthetic):
    shown = (GRID >= 1292) & (GRID <= 1382)
    height = synthetic[shown].max()
    return spectra_table({"Before (a.u.)": synthetic,
                          "Shift by 7 cm-1 (a.u.)": shifted(synthetic, 7.0),
                          "Line shape (a.u.)": synthetic + 0.6 * height * fano_feature(1338.0, 12.0, -0.8),
                          "Masking (a.u.)": synthetic * suppression(1336.0, 34.0, 0.5)}, shown)


def scaled_weights(data, held_out_id, compounds):
    models = {"Synthetic-only LC": data.paper["theta_ref"],
              f"Combined LC without {held_out_id}": data.paper["theta_folds"][data.ids.index(held_out_id)]}
    weights = {}
    for model, theta in models.items():
        for name in compounds:
            w = theta[:-1, COMPOUNDS.index(name)]
            weights[f"-{name} weight, {model} (scaled)"] = w / np.abs(w).max()
    return weights


def presence_scores(data, mixture_id):
    i = data.ids.index(mixture_id)
    rows = []
    for name in FIGURE_ORDER:
        k = COMPOUNDS.index(name)
        row = {"Compound": name, "Present": bool(data.truth[i, k])}
        for label, scores in (("Synthetic-only", data.paper["scores_ref"]), ("Combined", data.paper["scores_comb"])):
            row[f"{label} presence score"], row[f"{label} called present"] = scores[i, k], bool(scores[i, k] >= 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def figure_2(data):
    singles, weighted_sum, baseline, synthetic = synthetic_example(data)
    steps = {f"-{name} weight {EXAMPLE_PROPORTIONS[name]} (a.u.)": spectrum
             for name, spectrum in singles.items()}
    steps.update({"Weighted sum (a.u.)": weighted_sum, "Estimated baseline (a.u.)": baseline,
                  "Synthetic mixture (a.u.)": synthetic})
    examples = {f"Mixture {i} mean spectrum (a.u.)": mean_spectrum(find_mixture(data, i)) for i in ("2-3", "4-4")}
    examples["Synthetic mixture (a.u.)"] = synthetic
    return {"Fig. 2a": spectra_table(steps, FINGERPRINT), "Fig. 2a close-ups": spectral_variations(synthetic),
            "Fig. 2b spectra": spectra_table(examples, FINGERPRINT),
            "Fig. 2b LC weights": spectra_table(scaled_weights(data, "4-4", ["NO2", "CN", "COOH"])),
            "Fig. 2b scores": presence_scores(data, "4-4")}


def missed_compound_panel(data, means, mixture_id="6-4", compound="COOH"):
    mixture = find_mixture(data, mixture_id)
    spectrum, fit = mean_spectrum(mixture), mixture_fit(mixture, means)[1]
    single, scale = means[compound], np.abs(spectrum[FINGERPRINT]).max()
    return spectra_table({f"-{compound} single-compound spectrum (scaled)": single / np.abs(single[FINGERPRINT]).max(),
                          f"Mixture {mixture_id} mean spectrum (scaled)": spectrum / scale,
                          "Additive fit (scaled)": fit / scale}, FINGERPRINT)


def threshold_curve(data, means, mixture_id="6-4", compound="COOH"):
    w, b = fold_weights(data, mixture_id, compound)
    partners = partner_mean(means, [name for name in constituents(find_mixture(data, mixture_id)) if name != compound])
    targets, partner_spectra = combination_spectra(data, mixture_id, compound)
    percent = np.arange(121) * 0.25
    line = [presence_score(normalized_sum_logit(c, means[compound], partners, w, b)) for c in percent / 100]
    band = np.array([presence_score(normalized_sum_logit(c, targets, partner_spectra, w, b)) for c in percent / 100])
    table = pd.DataFrame({f"-{compound} contribution (%)": percent, "Presence score, mean spectra": line,
                          "Lowest presence score, individual spectra": band.min(axis=1),
                          "Highest presence score, individual spectra": band.max(axis=1)})
    individual = np.array([detection_threshold(t, p, w, b) for t, p in zip(targets, partner_spectra)])
    return table, detection_threshold(means[compound], partners, w, b), individual


def thresholds_by_mixture_size(data, means):
    theta = data.paper["theta_folds"].mean(axis=0)
    rows = []
    for name in FIGURE_ORDER:
        k, others = COMPOUNDS.index(name), [other for other in COMPOUNDS if other != name]
        for n in range(2, 7):
            thresholds = [detection_threshold(means[name], partner_mean(means, partners), theta[:-1, k], theta[-1, k])
                          for partners in itertools.combinations(others, n - 1)]
            rows.append({"Compound": name, "Compounds in the mixture": n, "Partner sets": len(thresholds),
                         "Contribution needed (%)": 100 * np.mean(thresholds)})
    return pd.DataFrame(rows)


def fit_against_experiment(data, means):
    rows = []
    for i, mixture in enumerate(data.mixtures):
        theta = data.paper["theta_folds"][i]
        fit_logits = lc_logit(theta, unit_length(mixture_fit(mixture, means)[1]))
        experimental_logits = lc_logit(theta, mean_spectrum(mixture))
        for name in constituents(mixture):
            k = COMPOUNDS.index(name)
            rows.append({"Mixture": mixture["id"], "Compound": name, "Additive fit (logit)": fit_logits[k],
                         "Experimental mixture (logit)": experimental_logits[k],
                         "Found": bool(data.paper["scores_comb"][i, k] >= 0.5)})
    return pd.DataFrame(rows)


def nh2_contributions(contribution_table):
    table = contribution_table[contribution_table["Compound"] == "NH2"].sort_values("Contribution (%)", ascending=False)
    table = table.rename(columns={"Contribution (%)": "-NH2 contribution (%)"})[["Mixture", "-NH2 contribution (%)"]]
    table["Label masked"] = table["-NH2 contribution (%)"] < 5
    return table.reset_index(drop=True)


def call_change(present, before, after):
    if before == after:
        return ""
    return "Corrected" if after == present else "Introduced error"


def masking_calls(data, nh2_mixtures):
    nh2, order = COMPOUNDS.index("NH2"), list(nh2_mixtures) + [m for m in data.ids if m not in set(nh2_mixtures)]
    rows = []
    for mixture_id in order:
        i = data.ids.index(mixture_id)
        present = bool(data.truth[i, nh2])
        before, after = data.paper["scores_comb"][i, nh2], data.paper["masked_scores"][i]
        rows.append({"Mixture": mixture_id, "-NH2 prepared": present, "Combined presence score": before,
                     "Combined call": stats.OUTCOMES[present, bool(before >= 0.5)],
                     "Labels masked presence score": after,
                     "Labels masked call": stats.OUTCOMES[present, bool(after >= 0.5)],
                     "Change": call_change(present, bool(before >= 0.5), bool(after >= 0.5))})
    return pd.DataFrame(rows)


def figure_6_statistics(contribution_table, per_spectrum, threshold, individual, fits):
    cooh = contribution_table[(contribution_table["Mixture"] == "6-4") & (contribution_table["Compound"] == "COOH")]
    low = contribution_table["Contribution (%)"] < 5
    each, rmse = per_spectrum["Contribution (%)"], per_spectrum[EQUAL_RMSE]
    r, p = stats.pearson(fits["Additive fit (logit)"].to_numpy(), fits["Experimental mixture (logit)"].to_numpy())
    return pd.DataFrame([
        stats.statistics_row("a", "Mixture 6-4", "-COOH contribution (%)", cooh["Contribution (%)"].iloc[0]),
        stats.statistics_row("a", "All mixtures", "Constituents below 5%", int(low.sum()), n=len(low)),
        stats.statistics_row("a", "Mixture 6-4 spectra", "-COOH at zero", int((each == 0).sum()), n=len(each)),
        stats.statistics_row("a", "Mixture 6-4 spectra", "Largest -COOH contribution (%)", each.max(), n=len(each)),
        stats.statistics_row("a", "Mixture 6-4 spectra, -COOH at 1/6", "Spectra fitted worse", int((rmse > 0).sum()),
                             n=len(rmse)),
        stats.statistics_row("a", "Mixture 6-4 spectra, -COOH at 1/6", "Median RMSE increase (%)", rmse.median(),
                             n=len(rmse)),
        stats.statistics_row("b", "Mixture 6-4, mean spectra", "Detection threshold (%)", 100 * threshold),
        stats.statistics_row("b", "Mixture 6-4, individual spectra", "Lowest threshold (%)", 100 * individual.min(),
                             n=len(individual)),
        stats.statistics_row("b", "Mixture 6-4, individual spectra", "Highest threshold (%)", 100 * individual.max(),
                             n=len(individual)),
        stats.statistics_row("d", "Additive fit vs experimental logits", "r", r, p, len(fits), "Pearson")])


def figure_6(data):
    means = mean_spectra(data)
    contribution_table = fitted_contributions(data, means)
    curve, threshold, individual = threshold_curve(data, means)
    fits, nh2 = fit_against_experiment(data, means), nh2_contributions(contribution_table)
    per_spectrum = contribution_in_each_spectrum(data, means)
    return {"Fig. 6a": missed_compound_panel(data, means), "Fig. 6b": curve,
            "Fig. 6c": thresholds_by_mixture_size(data, means), "Fig. 6d": fits, "Fig. 6e": nh2,
            "Fig. 6f": masking_calls(data, nh2["Mixture"]),
            "Fig. 6 statistics": figure_6_statistics(contribution_table, per_spectrum, threshold, individual, fits)}


def masking_by_set(data):
    nh2 = COMPOUNDS.index("NH2")
    present = data.truth[:, nh2]
    rows = []
    for seed, run in data.sets.items():
        before, after = run["scores_comb"][:, nh2] >= 0.5, run["masked_scores"] >= 0.5
        rows.append({"Training set": seed, "Main text": run is data.paper,
                     "-NH2 false detections left": int((after & ~present).sum()),
                     "Correct -NH2 detections lost": int((before & ~after & present).sum())})
    return pd.DataFrame(rows)


def table_s1_rows(data):
    rows = []
    for _, row in masking_by_set(data).iterrows():
        for quantity in ("-NH2 false detections left", "Correct -NH2 detections lost"):
            rows.append({"Training set": row["Training set"], "Main text": row["Main text"], "Panel": "6f",
                         "Comparison": quantity, "Statistic": "Count", "Value": row[quantity], "P": float("nan")})
    return pd.DataFrame(rows)
