import numpy as np
from scipy.stats import pearsonr, wilcoxon
from sklearn.metrics import average_precision_score

RESAMPLES = [np.random.default_rng(1000 + d).integers(0, 20, 20) for d in range(5000)]


def auprc(truth, scores):
    return float(average_precision_score(np.ravel(truth), np.ravel(scores)))


def mixture_f1(calls, truth):
    return 2 * (calls & truth).sum(1) / np.maximum(calls.sum(1) + truth.sum(1), 1)


def compound_f1(calls, truth):
    tp, fp, fn = (calls & truth).sum(0), (calls & ~truth).sum(0), (~calls & truth).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def summary(truth, scores, calls=None):
    calls = scores >= 0.5 if calls is None else calls
    return {"AUPRC": auprc(truth, scores), "Mean F1": float(mixture_f1(calls, truth).mean()),
            "Correct calls": int((calls == truth).sum()), "Found": int((calls & truth).sum()),
            "False detections": int((calls & ~truth).sum())}


def paired_bootstrap(truth, a, b):
    b = b if isinstance(b, list) else [b]
    diff = np.array([auprc(truth[i], a[i]) - np.mean([auprc(truth[i], s[i]) for s in b]) for i in RESAMPLES])
    observed = auprc(truth, a) - np.mean([auprc(truth, s) for s in b])
    return observed, float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


def wilcoxon_p(x, y):
    return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def signed_rank(x, y):
    return float(wilcoxon(x, y).statistic), wilcoxon_p(x, y)


def pearson(x, y):
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


OUTCOMES = {(True, True): "Found", (True, False): "Missed", (False, True): "False detection",
            (False, False): "Correct rejection"}


def statistics_row(panel, comparison, statistic, value, p=np.nan, n=None, test=None):
    return {"Panel": panel, "Comparison": comparison, "Statistic": statistic, "Value": value, "P": p, "n": n,
            "Test": test}
