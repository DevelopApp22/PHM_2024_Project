import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import simpson
import matplotlib.colors as mcolors

MODEL_PDFS = {
    "norm": stats.norm,
    "t": stats.t,
    "laplace": stats.laplace,
    "logistic": stats.logistic,
    "cauchy": stats.cauchy,
}

def compute_x_range(samples, pad_std=4.0):
    """Calcola (xmin, xmax) usando min/max campioni e media ± pad_std*std."""
    s = np.asarray(samples, dtype=float)
    mu = float(np.mean(s))
    sd = float(np.std(s))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1e-6

    xmin = min(float(np.min(s)), mu - pad_std * sd)
    xmax = max(float(np.max(s)), mu + pad_std * sd)

    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0

    return xmin, xmax


def make_x_grid(xmin, xmax, n_points=2000):
    """Genera una griglia lineare di punti tra xmin e xmax."""
    return np.linspace(float(xmin), float(xmax), int(n_points))


def resolve_pdf_names(model_pdfs, pdf_list=None):
    """Ritorna la lista dei nomi PDF da valutare (default: tutte)."""
    return list(model_pdfs.keys()) if pdf_list is None else list(pdf_list)


def fit_scipy_distribution(dist, samples):
    """Esegue dist.fit(samples) e restituisce dict con args/loc/scale."""
    params = dist.fit(samples)
    n_shapes = len(dist.shapes.split()) if dist.shapes else 0
    shape_params = tuple(float(p) for p in params[:n_shapes])
    loc = float(params[n_shapes])
    scale = float(params[n_shapes + 1])
    return {"args": shape_params, "loc": loc, "scale": scale}


def fit_and_compute_loglik(dist, samples, fit_fn=fit_scipy_distribution, eps=1e-12):
    """Fitta una distribuzione e calcola la log-likelihood sui campioni."""
    out = {"fit_ok": False, "pdf_args": None, "loglik": -np.inf, "error": ""}

    try:
        pdf_args = fit_fn(dist, samples)
        shape_args = tuple(pdf_args.get("args", ()))
        loc = float(pdf_args.get("loc", 0.0))
        scale = float(pdf_args.get("scale", 1.0))

        logpdf_vals = dist.logpdf(samples, *shape_args, loc=loc, scale=scale)
        logpdf_vals = np.where(np.isfinite(logpdf_vals), logpdf_vals, np.log(eps))
        loglik = float(np.sum(logpdf_vals))

        out.update({"fit_ok": True, "pdf_args": pdf_args, "loglik": loglik})
    except Exception as e:
        out["error"] = repr(e)

    return out


def plot_hist_and_fitted_pdf(dist, samples, x_grid, pdf_args, title, bins=40, x_range=None):
    """Plotta istogramma (density=True) e pdf fittata su x_grid."""
    shape_args = tuple(pdf_args.get("args", ()))
    loc = float(pdf_args.get("loc", 0.0))
    scale = float(pdf_args.get("scale", 1.0))

    y = dist.pdf(x_grid, *shape_args, loc=loc, scale=scale)
    y = np.where(np.isfinite(y), y, 0.0)
    BLUE = "#1f77b4"


    face_rgba = mcolors.to_rgba(BLUE, alpha=0.35)
    plt.figure()
    if x_range is not None:
        plt.hist(samples, bins=bins, range=x_range, density=True, alpha=0.35)
    else:
        plt.hist(samples, bins=bins, density=True, alpha=0.35,    facecolor=face_rgba, )
    plt.plot(x_grid, y, linewidth=2)
    plt.title(title)

    plt.xlabel("value")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.show()


def rank_by_loglik(results):
    """Ordina i risultati: prima i fit_ok per loglik decrescente, poi i falliti."""
    ok = [r for r in results if r.get("fit_ok", False)]
    bad = [r for r in results if not r.get("fit_ok", False)]
    ok.sort(key=lambda r: -r["loglik"])
    return ok + bad, ok, bad


def print_ranking(ok, bad):
    """Stampa classifica delle distribuzioni fittate e lista fallite."""
    print("\n=== CLASSIFICA (LOG-LIKELIHOOD) ===")
    for i, r in enumerate(ok, start=1):
        print(f"{i:02d}) {r['pdf_type']:12s} | loglik={r['loglik']:.2f}")

    if bad:
        print("\n=== FALLITE ===")
        for r in bad:
            print(f"- {r['pdf_type']}: {r.get('error','')}")



def fit_rank_pdfs_loglik(
    samples,
    pdf_list=None,
    eps=1e-12,
    fit_fn=fit_scipy_distribution,
    verbose=True,
):
    """
    Fitta più PDF, calcola log-likelihood, fa ranking.
    NON stampa errori a schermo (a meno che verbose=True).
    Ritorna: (ranked, best)
    """

    pdf_names = resolve_pdf_names(MODEL_PDFS, pdf_list)
    results = []

    for name in pdf_names:
        dist = MODEL_PDFS.get(name)
        if dist is None:
            continue

        row = {
            "pdf_type": name,
            "fit_ok": False,
            "pdf_args": None,
            "loglik": -np.inf,
            "error": "",
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                row_fit = fit_and_compute_loglik(
                    dist,
                    samples,
                    fit_fn=fit_fn,
                    eps=eps
                )

            row.update(row_fit)

        except Exception as e:
            # NON stampare nulla
            row["error"] = str(e)
            row["fit_ok"] = False

        results.append(row)

    ranked, ok, bad = rank_by_loglik(results)

    if verbose:
        print_ranking(ok, bad)

    best = ok[0] if ok else None
    return ranked, best

def fit_best_pdf(samples, pdf_names=("norm","t","laplace","logistic","cauchy"), eps=1e-12):
    """
    Input: samples (500,) -> predizioni dei 500 alberi per UNA riga
    Output:
      - best (dict): pdf scelta + pdf_args + loglik + AIC + BIC + k + n
      - results (list[dict]): tutte le pdf fittate con le stesse info

    Regola scelta:
      1) AIC minimo
      2) se ΔAIC < 2 tra top2 -> scegli il più semplice (k min)
      3) se un altro ha BIC migliore di >=10 -> scegli quello
    """
    s = samples
    n = len(s)

    results = []
    if n == 0:
        return None, []

    for name in pdf_names:
        dist = MODEL_PDFS.get(name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(s)  # SciPy: (*shape, loc, scale)

            shape_args = tuple(float(x) for x in params[:-2])
            loc = float(params[-2])
            scale = float(params[-1])

            pdf_args = {"args": shape_args, "loc": loc, "scale": scale}

            logpdf = dist.logpdf(s, *shape_args, loc=loc, scale=scale)
            logpdf = np.where(np.isfinite(logpdf), logpdf, np.log(eps))
            loglik = float(np.sum(logpdf))

            k = int(len(shape_args) + 2)   # shape + loc + scale
            aic = float(2 * k - 2 * loglik)
            bic = float(k * np.log(n) - 2 * loglik)

            results.append({
                "pdf_type": name,
                "fit_ok": True,
                "pdf_args": pdf_args,   # <-- quello che ti serve
                "loglik": loglik,
                "aic": aic,
                "bic": bic,
                "k": k,
                "n": n,
                "error": "",
            })

        except Exception as e:
            results.append({
                "pdf_type": name,
                "fit_ok": False,
                "pdf_args": None,
                "loglik": -np.inf,
                "aic": np.inf,
                "bic": np.inf,
                "k": None,
                "n": n,
                "error": repr(e),
            })

    ok = [r for r in results if r["fit_ok"]]
    bad = [r for r in results if not r["fit_ok"]]

    ok_sorted = sorted(ok, key=lambda r: r["aic"])
    results_sorted = ok_sorted + bad
    if not ok:
        # fallback: normal su mean/std
        mu = float(np.mean(s))
        sig = float(np.std(s, ddof=1) + 1e-9)
        best = {
            "pdf_type": "norm",
            "fit_ok": True,
            "pdf_args": {"args": (), "loc": mu, "scale": sig},
            "loglik": None,
            "aic": None,
            "bic": None,
            "k": 2,
            "n": n,
            "error": "",
            "note": "fallback_no_fit",
        }
        return best, results

    # 1) best AIC
    ok_sorted_aic = sorted(ok, key=lambda r: r["aic"])
    best = ok_sorted_aic[0]

    # 2) ΔAIC < 2: scegli più semplice tra top2
    if len(ok_sorted_aic) > 1:
        second = ok_sorted_aic[1]
        if (second["aic"] - best["aic"]) < 2.0:
            if second["k"] is not None and best["k"] is not None and second["k"] < best["k"]:
                best = second

    # 3) override BIC se molto migliore
    best_bic = min(ok, key=lambda r: r["bic"])
    if best_bic["pdf_type"] != best["pdf_type"]:
        if (best["bic"] - best_bic["bic"]) >= 10.0:
            best = best_bic

    return best, results_sorted



def plot_ranked_pdfs(
    ranked_results,
    samples,
    n_points=2000,
    pad_std=4.0,
    bins=30,
    top_k=5,  # default 4 come volevi
):
    """
    Plot unico con:
      - istogramma campioni
      - overlay delle prime top_k PDF (ordinate per AIC)
      - legenda con AIC
    """

    xmin, xmax = compute_x_range(samples, pad_std=pad_std)
    x_grid = make_x_grid(xmin, xmax, n_points=n_points)
    COLORS = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
    ]

    plt.figure()

    plt.hist(
        samples,
        bins=bins,
        range=(xmin, xmax),
        density=True,
        alpha=0.35,
        label="samples",
    )

    idx = 0

    for r in ranked_results:

        if idx >= top_k:
            break

        if not r.get("fit_ok", False):
            continue

        name = r["pdf_type"]
        dist = MODEL_PDFS.get(name)
        if dist is None:
            continue

        pdf_args = r["pdf_args"]
        shape_args = tuple(pdf_args.get("args", ()))
        loc = float(pdf_args.get("loc", 0.0))
        scale = float(pdf_args.get("scale", 1.0))
        aic = r.get("aic", np.nan)

        # calcolo pdf
        y = dist.pdf(x_grid, *shape_args, loc=loc, scale=scale)
        y = np.where(np.isfinite(y), y, 0.0)

        color = COLORS[idx % len(COLORS)]
        idx += 1

        plt.plot(
            x_grid,
            y,
            linewidth=2.5,
            color=color,
            label=f"{idx:02d}) {name} | AIC={aic:.2f}",
        )

    plt.title("Top ranked PDFs (AIC)")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_pdf_with_true_target(
    pdf_type,
    pdf_args,
    true_target,
    y_pred,
    n_points=4000,
    x_range=None,
    center_tightness=1.0,
    min_scale=1e-6,
    marker_size=30,
    line_width=2.5,
):
    """
    Plotta la curva "score" normalizzata e mostra:
    - true target (verde) con y = score(true_target)
    - prediction (rosso) con y = score(y_pred) SOLO se y_pred non è None
    """

    yt = float(true_target)
    yp = None if y_pred is None else float(y_pred)

    # --- modello + params ---
    model = MODEL_PDFS[pdf_type]
    shape_args = tuple(pdf_args.get("args", ()))
    kwargs = {k: v for k, v in pdf_args.items() if k != "args"}

    if not getattr(model, "shapes", None):
        shape_args = ()

    if "scale" in kwargs:
        kwargs["scale"] = float(max(float(kwargs["scale"]), min_scale))
    else:
        kwargs["scale"] = float(max(float(pdf_args.get("scale", 1.0)), min_scale))

    if "loc" in kwargs:
        kwargs["loc"] = float(kwargs["loc"])
    else:
        kwargs["loc"] = float(pdf_args.get("loc", 0.0))

    loc = float(kwargs.get("loc", 0.0))
    scale = float(kwargs.get("scale", 1.0))

    # --- definisci x_range (gestendo yp=None) ---
    if x_range is None:
        if yp is None:
            center = yt
            half = max(2.0 * scale, 0.25) / max(center_tightness, 1e-9)
        else:
            center = 0.5 * (yt + yp)
            dist = abs(yt - yp)
            half = max(0.75 * dist, 1.0 * scale, 0.25) / max(center_tightness, 1e-9)

        xmin, xmax = center - half, center + half
        if xmin == xmax:
            xmin, xmax = center - 1.0, center + 1.0
    else:
        xmin, xmax = map(float, x_range)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if xmin == xmax:
            xmin, xmax = xmin - 1.0, xmax + 1.0

    x_grid = np.linspace(xmin, xmax, int(n_points))

    # --- normalizzazione "score": pdf / area / max ---
    x_big = np.linspace(loc - 10 * scale, loc + 10 * scale, 100000)
    y_big = model.pdf(x_big, *shape_args, **kwargs)
    y_big = np.where(np.isfinite(y_big), y_big, 0.0)

    area = float(simpson(y_big, x_big))
    if (not np.isfinite(area)) or area <= min_scale:
        area = 1.0

    y_big = y_big / area
    y_max = float(np.max(y_big)) if y_big.size else 1.0
    if (not np.isfinite(y_max)) or y_max <= min_scale:
        y_max = 1.0

    def score_at(x):
        val = model.pdf(x, *shape_args, **kwargs)
        if not np.isfinite(val):
            val = 0.0
        return float(val / area / y_max)

    y_grid = model.pdf(x_grid, *shape_args, **kwargs)
    y_grid = np.where(np.isfinite(y_grid), y_grid, 0.0)
    y_grid = y_grid / area / y_max

    score_true = score_at(yt)
    score_pred = None if yp is None else score_at(yp)

    # --- plot ---
    plt.figure()
    plt.plot(x_grid, y_grid, linewidth=line_width)

    plt.scatter(
        [yt], [score_true],
        s=marker_size,
        color="green",
        zorder=6,
        label=f"true (score={score_true:.3g})"
    )
    plt.axvline(yt, linestyle="--", alpha=0.6, color="green")

    # ✅ pred SOLO se yp non è None
    if yp is not None:
        plt.scatter(
            [yp], [score_pred],
            s=marker_size,
            color="red",
            zorder=6,
            label=f"pred (score={score_pred:.3g})"
        )
        plt.axvline(yp, linestyle="--", alpha=0.6, color="red")

    y_max_plot = float(np.max(y_grid)) if y_grid.size else 1.0
    plt.ylim(0.0, max(y_max_plot * 1.15, 1e-9))
    plt.xlabel("value")
    plt.ylabel("score")
    plt.title(f"{pdf_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return {
        "pdf_type": pdf_type,
        "true_target": yt,
        "prediction": yp,
        "score_true": score_true,
        "score_pred": score_pred,
        "x_range": (xmin, xmax),
    }