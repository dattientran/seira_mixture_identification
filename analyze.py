"""Analyses and tables."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import figure_5, figures_1_2_6, figures_3_4, spectra
from train import N_COMPOSITIONS, PAPER_SET, RESULTS, TRAINING_SETS


def load_data():
    pools, single, mixtures = spectra.load_inputs("data/spectra")
    data = SimpleNamespace(pools=pools, single=single, mixtures=mixtures, ids=[m["id"] for m in mixtures],
                           truth=np.stack([m["label"] for m in mixtures]), paper_set=PAPER_SET,
                           n_compositions=N_COMPOSITIONS)
    data.held_out = {name: np.isin(np.arange(1, len(single[name]) + 1), spectra.HELD_OUT[name])
                     for name in spectra.COMPOUNDS}
    data.sets = {}
    for seed in TRAINING_SETS:
        with np.load(RESULTS / "training_sets" / f"{seed}.npz") as run:
            data.sets[seed] = dict(run)
    data.paper = data.sets[PAPER_SET]
    data.experimental_only = np.load(RESULTS / "experimental_only.npz")["scores"]
    with np.load(RESULTS / "comparison_methods.npz") as z:
        data.reconstruction, data.reconstruction_threshold = z["reconstruction"], float(z["reconstruction_threshold"])
        data.peak_matching, data.peak_thresholds = z["peak_matching"], z["peak_thresholds"]
    with np.load(RESULTS / "weight_ratios.npz") as z:
        data.weight_ratios = {3: z["ratio_3"], 10: data.paper["scores_comb"], 30: z["ratio_30"]}
    with np.load(RESULTS / "experimental_mixture_curve.npz") as z:
        data.experimental_curve = {"sizes": z["sizes"], "scores": z["scores"]}
    with np.load(RESULTS / "synthetic_size_curve.npz") as z:
        data.synthetic_curve = {"sizes": z["sizes"], "scores": z["scores"]}
    return data


def table_s1(data):
    parts = [figures_3_4.table_s1_rows(data), figure_5.table_s1_rows(data), figures_1_2_6.table_s1_rows(data)]
    by_set = pd.concat(parts, ignore_index=True)
    rows = []
    for (panel, comparison, statistic), group in by_set.groupby(["Panel", "Comparison", "Statistic"], sort=False):
        counts = statistic == "Count"
        rows.append({"Panel": panel, "Comparison": comparison, "Statistic": statistic,
                     "Minimum": group["Value"].min(), "Maximum": group["Value"].max(),
                     "Sets with P < 0.05": None if counts else int((group["P"] < 0.05).sum()),
                     "Sets with 0": int((group["Value"] == 0).sum()) if counts else None, "Sets": len(group)})
    summary = pd.DataFrame(rows).astype({"Sets with P < 0.05": "Int64", "Sets with 0": "Int64"})
    return {"Table S1": summary, "Table S1 by set": by_set}


def main():
    data = load_data()
    sheets = {}
    for make in (figures_1_2_6.figure_1, figures_1_2_6.figure_2, figures_3_4.figure_3, figures_3_4.figure_4,
                 figure_5.figure_5, figures_1_2_6.figure_6, figures_3_4.figure_s2, figure_5.figure_s3, table_s1,
                 figures_3_4.table_s2):
        print(make.__name__, flush=True)
        sheets.update(make(data))
    with pd.ExcelWriter(RESULTS / "source_data.xlsx") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    print(f"written {RESULTS / 'source_data.xlsx'} ({len(sheets)} sheets)")


if __name__ == "__main__":
    main()
