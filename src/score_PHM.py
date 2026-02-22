import numpy as np
from scipy import stats
from scipy.integrate import simpson

MODEL_PDFS = {
    "norm": stats.norm,
    "expon": stats.expon,
    "uniform": stats.uniform,
    "gamma": stats.gamma,
    "beta": stats.beta,
    "lognorm": stats.lognorm,
    "chi2": stats.chi2,
    "weibull_min": stats.weibull_min,
    "t": stats.t,
    "f": stats.f,
    "cauchy": stats.cauchy,
    "laplace": stats.laplace,
    "rayleigh": stats.rayleigh,
    "pareto": stats.pareto,
    "gumbel_r": stats.gumbel_r,
    "logistic": stats.logistic,
    "erlang": stats.erlang,
    "powerlaw": stats.powerlaw,
    "nakagami": stats.nakagami,
    "betaprime": stats.betaprime,
}

def get_regression_score(pdf_type, pdf_args, true_target):

    model = MODEL_PDFS[pdf_type]

    shape_args = tuple(pdf_args.get("args", ()))
    kwargs = {k: v for k, v in pdf_args.items() if k != "args"}
    if not getattr(model, "shapes", None):
        shape_args = ()

    if "scale" in kwargs:
        kwargs["scale"] = float(max(kwargs["scale"], 1e-6))

    score = model.pdf(true_target, *shape_args, **kwargs)
    x = np.linspace(-100, 100, 100000)
    y = model.pdf(x, *shape_args, **kwargs)
    area = simpson(y, x)

    if area > 1:
        score /= area
    y_max = float(np.max(y))
    if y_max > 1:
        print(y_max)
        score /= y_max

    return float(score)

def get_classification_score(true_label, pred_label, confidence):
    if (confidence < 0) or (confidence > 1):
        return -100

    if pred_label not in (0, 1):
        return -100

    if pred_label != true_label:
        confidence = -confidence

    if true_label == 0:
        return confidence
    else:
        if confidence >= 0:
            return confidence
        else:
            return 4 * confidence ** 11 + confidence

def get_challange_score(regression_score,classification_score):
    return (regression_score+classification_score)/2