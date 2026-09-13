"""Linear probes for exponent vectors, with the controls the design requires.

- ridge regression residual -> exponent vector (continuous parameterisation);
- multinomial logistic residual -> lattice-point class (categorical parameterisation);
- control task (Hewitt & Liang): labels re-assigned by a random but *consistent* map from
  unit lexeme to a lattice point, so a probe can only succeed by memorising lexemes;
- cross-lexeme folds (group k-fold by unit lexeme) and lattice-holdout folds (held-out points).
Metrics: per-axis R², exact-vector accuracy after rounding, nearest-point accuracy (among the
label set), and selectivity = task − control.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler


def _prep(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_ridge(Xtr, Ytr, Xte, alpha: float | None = None):
    """Ridge with alpha chosen by efficient leave-one-out CV on the training fold (alpha=None)."""
    Xtr_, Xte_ = _prep(Xtr, Xte)
    m = (RidgeCV(alphas=ALPHAS) if alpha is None else Ridge(alpha=alpha)).fit(Xtr_, Ytr)
    return m.predict(Xte_), m


def nearest_point(pred: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the nearest lattice point (rows of `points`) for each predicted vector."""
    d = ((pred[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def regression_metrics(pred: np.ndarray, Y: np.ndarray, points: np.ndarray | None = None) -> dict:
    Y = np.asarray(Y, dtype=float)
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    R = np.rint(pred)
    out = dict(r2_per_axis=r2.tolist(), r2_mean=float(np.nanmean(r2)),
               exact_acc=float((R == Y).all(1).mean()), mae=float(np.abs(pred - Y).mean()),
               axis_acc=float((R == Y).mean()),                      # per-axis rounded accuracy
               sign_acc=float((np.sign(R) == np.sign(Y)).mean()),    # per-axis sign (incl. zero) accuracy
               nonzero_axis_acc=float((R[Y != 0] == Y[Y != 0]).mean()) if (Y != 0).any() else np.nan)
    if points is not None:
        idx = nearest_point(pred, points)
        out["nearest_acc"] = float((points[idx] == Y).all(1).mean())
    return out


def fit_logistic(Xtr, ytr, Xte, C: float = 1.0):
    Xtr_, Xte_ = _prep(Xtr, Xte)
    m = LogisticRegression(C=C, max_iter=2000).fit(Xtr_, ytr)
    return m.predict(Xte_), m


def control_labels(unit_ids: list[str], Y: np.ndarray, seed: int) -> np.ndarray:
    """Consistent random re-assignment of lattice points to unit lexemes (same label marginal)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit_ids))
    # sample, for each lexeme, a random label from the empirical label set
    labels = np.unique(Y, axis=0)
    assign = {u: labels[rng.integers(len(labels))] for u in uniq}
    return np.stack([assign[u] for u in unit_ids])


def group_folds(groups: list[str], k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    parts = np.array_split(uniq, k)
    g = np.asarray(groups)
    folds = []
    for p in parts:
        te = np.isin(g, p)
        folds.append((np.where(~te)[0], np.where(te)[0]))
    return folds


def lattice_holdout_folds(Y: np.ndarray, k: int, seed: int, *, keep_spanning: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out whole lattice points. With keep_spanning, the base points (unit vectors) always stay in train."""
    rng = np.random.default_rng(seed)
    pts = np.unique(Y, axis=0)
    is_base = (np.abs(pts).sum(1) == 1)
    cand = pts[~is_base] if keep_spanning else pts
    idx = np.arange(len(cand))
    rng.shuffle(idx)
    parts = np.array_split(idx, k)
    folds = []
    for p in parts:
        held = cand[p]
        te = np.array([any((y == h).all() for h in held) for y in Y])
        folds.append((np.where(~te)[0], np.where(te)[0]))
    return folds


def run_probe_suite(X: np.ndarray, Y: np.ndarray, unit_ids: list[str], *, seed: int = 0, alpha: float | None = None,
                    k_lex: int = 5, k_lat: int = 5) -> dict:
    """Task and control probes under random / cross-lexeme / lattice-holdout splits."""
    Y = np.asarray(Y, dtype=float)
    points = np.unique(Y, axis=0)
    Yc = control_labels(unit_ids, Y, seed)
    res = {}
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    rand_folds = [(perm[: int(0.8 * n)], perm[int(0.8 * n):])]
    for split_name, folds in (("random", rand_folds), ("cross_lexeme", group_folds(unit_ids, k_lex, seed)),
                              ("lattice_holdout", lattice_holdout_folds(Y, k_lat, seed))):
        for lab_name, YY in (("task", Y), ("control", Yc)):
            preds = np.zeros_like(YY)
            mask = np.zeros(n, dtype=bool)
            for tr, te in folds:
                if len(te) == 0 or len(tr) == 0:
                    continue
                p, _ = fit_ridge(X[tr], YY[tr], X[te], alpha=alpha)
                preds[te] = p
                mask[te] = True
            m = regression_metrics(preds[mask], YY[mask], points)
            res[f"{split_name}/{lab_name}"] = m
        res[f"{split_name}/selectivity_nearest"] = res[f"{split_name}/task"]["nearest_acc"] - res[f"{split_name}/control"]["nearest_acc"]
        res[f"{split_name}/selectivity_r2"] = res[f"{split_name}/task"]["r2_mean"] - res[f"{split_name}/control"]["r2_mean"]
    return res
