import numpy as np
import pandas as pd

from src import models, spectra, stats

PAPER_SIZE = 20000
BOOTSTRAP = "paired bootstrap"
SIGNED_RANK = "Wilcoxon signed-rank"
SYNTHETIC_SETS = "Synthetic-only, 5 sets"


def peak_matching_scores(data):
    return data.peak_matching - data.peak_thresholds + 0.5


def reconstruction_calls(data):
    return models.coefficient_fractions(data.reconstruction) >= data.reconstruction_threshold


def methods_3a(data):
    lc, networks, peaks = data.paper["scores_ref"], list(data.paper["network_scores"]), peak_matching_scores(data)
    return {"Logistic classifier": ([lc], [lc >= 0.5]),
            "Neural network": (networks, [scores >= 0.5 for scores in networks]),
            "Spectral reconstruction": ([data.reconstruction], [reconstruction_calls(data)]),
            "Peak matching": ([peaks], [peaks >= 0.5])}


def conditions_4a(data):
    return {"Combined": data.paper["scores_comb"], "Synthetic-only": data.paper["scores_ref"],
            "Experimental-only": data.experimental_only}


def measures(truth, fits, calls, mixtures):
    return (np.mean([stats.auprc(truth[mixtures], scores[mixtures]) for scores in fits]),
            np.mean([stats.mixture_f1(c[mixtures], truth[mixtures]).mean() for c in calls]))


def performance(truth, fits, calls):
    auprc, f1 = measures(truth, fits, calls, np.arange(len(truth)))
    sd = np.array([measures(truth, fits, calls, resample) for resample in stats.RESAMPLES]).std(axis=0, ddof=1)
    return {"AUPRC": auprc, "AUPRC SD": sd[0], "Mean F1": f1, "Mean F1 SD": sd[1]}


def precision_recall_by_size(truth, calls):
    rows = []
    for size in np.unique(truth.sum(axis=1)):
        group = truth.sum(axis=1) == size
        found, called, present = int((calls & truth)[group].sum()), int(calls[group].sum()), int(truth[group].sum())
        rows.append({"Compounds in the mixture": int(size), "Found": found, "Positive calls": called,
                     "Present compounds": present, "Precision": found / called, "Recall": found / present})
    return pd.DataFrame(rows)


def call_table(data, scores, calls):
    rows = []
    for m, mixture in enumerate(data.ids):
        for k, compound in enumerate(spectra.COMPOUNDS):
            present, called = bool(data.truth[m, k]), bool(calls[m, k])
            rows.append({"Mixture": mixture, "Compound": compound, "Present": present, "Presence score": scores[m, k],
                         "Called present": called, "Outcome": stats.OUTCOMES[present, called]})
    return pd.DataFrame(rows)


def combined_calls(data):
    table = call_table(data, data.paper["scores_comb"], data.paper["scores_comb"] >= 0.5)
    before = ((data.paper["scores_ref"] >= 0.5) == data.truth).ravel()
    after = (table["Called present"] == table["Present"]).to_numpy()
    change = np.select([after & ~before, before & ~after], ["Corrected", "Introduced error"], "")
    table["Change"] = change
    return table


def curve_table(x_name, curve, truth):
    rows = []
    for x, training, fits in curve:
        summaries = [stats.summary(truth, scores) for scores in fits]
        for measure in ("AUPRC", "Mean F1"):
            values = [s[measure] for s in summaries]
            rows.append({x_name: x, "Training": training, "Measure": measure, "Mean": np.mean(values),
                         "Minimum": min(values), "Maximum": max(values)})
    return pd.DataFrame(rows)


def synthetic_size_curve(data):
    sizes, scores = data.synthetic_curve["sizes"], data.synthetic_curve["scores"]
    curve = [(int(n), SYNTHETIC_SETS, fits) for n, fits in zip(sizes, scores)]
    curve.append((PAPER_SIZE, "Synthetic-only, main text", [data.paper["scores_ref"]]))
    return curve_table("Synthetic training spectra", curve, data.truth)


def experimental_mixture_curve(data):
    sizes, scores = data.experimental_curve["sizes"], data.experimental_curve["scores"]
    curve = [(0, "Synthetic-only", [data.paper["scores_ref"]])]
    curve += [(int(n), "Combined, 5 subsets", fits) for n, fits in zip(sizes, scores)]
    curve.append((len(data.ids) - 1, "Combined", [data.paper["scores_comb"]]))
    return curve_table("Experimental mixtures added", curve, data.truth)


def paired_bootstrap_f1(truth, calls_a, calls_b):
    difference = stats.mixture_f1(calls_a, truth) - np.mean([stats.mixture_f1(c, truth) for c in calls_b], axis=0)
    resampled = np.array([difference[resample].mean() for resample in stats.RESAMPLES])
    return float(difference.mean()), float(min(1.0, 2 * min((resampled <= 0).mean(), (resampled >= 0).mean())))


def figure_3_statistics(data):
    methods = methods_3a(data)
    lc, lc_calls = data.paper["scores_ref"], data.paper["scores_ref"] >= 0.5
    others = ["Spectral reconstruction", "Peak matching", "Neural network"]
    rows = []
    for name in others:
        fits, _ = methods[name]
        difference, p = stats.paired_bootstrap(data.truth, lc, fits)
        rows.append(stats.statistics_row("a", f"LC minus {name.lower()}", "AUPRC difference", difference, p,
                                         len(data.truth), BOOTSTRAP))
    for name in others:
        _, calls = methods[name]
        difference, p = paired_bootstrap_f1(data.truth, lc_calls, calls)
        rows.append(stats.statistics_row("a", f"LC minus {name.lower()}", "Mean F1 difference", difference, p,
                                         len(data.truth), BOOTSTRAP))
    return pd.DataFrame(rows)


def synthetic_gain(curve):
    means = curve[curve["Training"] == SYNTHETIC_SETS].set_index(["Synthetic training spectra", "Measure"])["Mean"]
    largest = int(curve["Synthetic training spectra"].max())
    comparison = f"{largest:,} vs {PAPER_SIZE:,} synthetic spectra"
    rows = []
    for measure in ("AUPRC", "Mean F1"):
        before, after = means[PAPER_SIZE, measure], means[largest, measure]
        rows += [stats.statistics_row("b", comparison, f"{measure} increase", after - before, n=5),
                 stats.statistics_row("b", comparison, f"{measure} increase (%)", 100 * (after - before) / before, n=5)]
    return rows


def mixture_trend(curve):
    rows = []
    for measure in ("AUPRC", "Mean F1"):
        points = curve[curve["Measure"] == measure]
        r = stats.pearson(points["Experimental mixtures added"].to_numpy(float), points["Mean"].to_numpy(float))[0]
        rows.append(stats.statistics_row("c", f"{measure} vs experimental mixtures added", "r", r, n=len(points)))
    return rows


def figure_4_statistics(data, synthetic_curve, experimental_curve):
    scores = conditions_4a(data)
    rows = []
    for a, b in (("Combined", "Synthetic-only"), ("Combined", "Experimental-only"),
                 ("Synthetic-only", "Experimental-only")):
        difference, p = stats.paired_bootstrap(data.truth, scores[a], scores[b])
        rows.append(stats.statistics_row("a", f"{a} minus {b}", "AUPRC difference", difference, p, len(data.truth),
                                         BOOTSTRAP))
    combined_f1 = stats.mixture_f1(scores["Combined"] >= 0.5, data.truth)
    for b in ("Synthetic-only", "Experimental-only"):
        w, p = stats.signed_rank(combined_f1, stats.mixture_f1(scores[b] >= 0.5, data.truth))
        rows.append(stats.statistics_row("a", f"Combined vs {b}, per-mixture F1", "W", w, p, len(data.truth),
                                         SIGNED_RANK))
    return pd.DataFrame(rows + synthetic_gain(synthetic_curve) + mixture_trend(experimental_curve))


def figure_3(data):
    panel_a = pd.DataFrame([{"Method": name, **performance(data.truth, fits, calls)}
                            for name, (fits, calls) in methods_3a(data).items()])
    lc = data.paper["scores_ref"]
    return {"Fig. 3a": panel_a, "Fig. 3b": precision_recall_by_size(data.truth, lc >= 0.5),
            "Fig. 3c": call_table(data, lc, lc >= 0.5), "Fig. 3 statistics": figure_3_statistics(data)}


def figure_4(data):
    panel_a = pd.DataFrame([{"Training": name, **performance(data.truth, [scores], [scores >= 0.5])}
                            for name, scores in conditions_4a(data).items()])
    panel_b, panel_c = synthetic_size_curve(data), experimental_mixture_curve(data)
    return {"Fig. 4a": panel_a, "Fig. 4b": panel_b, "Fig. 4c": panel_c, "Fig. 4d": combined_calls(data),
            "Fig. 4 statistics": figure_4_statistics(data, panel_b, panel_c)}


def figure_s2(data):
    rows = []
    for seed, run in data.sets.items():
        for lc, key in (("Synthetic-only", "scores_ref"), ("Combined", "scores_comb")):
            summary = stats.summary(data.truth, run[key])
            rows.append({"Training set": seed, "Main text": run is data.paper, "LC": lc,
                         "AUPRC": summary["AUPRC"], "Mean F1": summary["Mean F1"],
                         "Correct calls": summary["Correct calls"]})
    return {"Fig. S2": pd.DataFrame(rows)}


def set_comparisons(data, run):
    lc, combined, experimental = run["scores_ref"], run["scores_comb"], data.experimental_only
    rows = []
    for panel, comparison, a, b in (("3a", "LC minus spectral reconstruction", lc, data.reconstruction),
                                    ("3a", "LC minus peak matching", lc, peak_matching_scores(data)),
                                    ("3a", "LC minus neural network", lc, list(run["network_scores"])),
                                    ("4a", "Combined minus Synthetic-only", combined, lc),
                                    ("4a", "Combined minus Experimental-only", combined, experimental),
                                    ("4a", "Synthetic-only minus Experimental-only", lc, experimental)):
        difference, p = stats.paired_bootstrap(data.truth, a, b)
        rows.append({"Panel": panel, "Comparison": comparison, "Statistic": "AUPRC difference", "Value": difference,
                     "P": p})
    combined_f1 = stats.mixture_f1(combined >= 0.5, data.truth)
    for name, b in (("Synthetic-only", lc), ("Experimental-only", experimental)):
        p = stats.wilcoxon_p(combined_f1, stats.mixture_f1(b >= 0.5, data.truth))
        rows.append({"Panel": "4a", "Comparison": f"Combined vs {name}, per-mixture F1", "Statistic": "P",
                     "Value": p, "P": p})
    return rows


def table_s1_rows(data):
    return pd.DataFrame([{"Training set": seed, "Main text": run is data.paper, **row}
                         for seed, run in data.sets.items() for row in set_comparisons(data, run)])


def table_s2(data):
    rows = []
    for ratio, scores in data.weight_ratios.items():
        summary = stats.summary(data.truth, scores)
        rows.append({"Weight ratio": f"{ratio}:1" + (" (main text)" if ratio == 10 else ""),
                     "AUPRC": summary["AUPRC"], "Mean F1": summary["Mean F1"],
                     "Correct calls": summary["Correct calls"]})
    return {"Table S2": pd.DataFrame(rows)}
