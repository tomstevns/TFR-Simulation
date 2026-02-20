# src/fit_post_stop.py
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from model import Params, simulate, bcr_abl_proxy_from_P
from load_qpcr import load_qpcr_csv, to_days_and_log10_is


# === SET THIS ONCE ===
STOP_DATE: date = date(2025, 2, 20)

# Generate a plausible (L*, P*, I*) at stop via a baseline run, then simulate post-stop only.
STOP_DAY_MODEL = 365.0

# Baseline initial condition for generating a plausible stop-state
Y0_BASELINE = (0.35, 18.0, 0.7)

# Data floor for log10 transform (use assay LOD if you know it)
FLOOR_IS = 1e-6


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(x - target)))


def _interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y, left=y[0], right=y[-1])


def _load_post_stop_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = load_qpcr_csv(csv_path)
    days, log10_is, _t0_used = to_days_and_log10_is(rows, t0_date=STOP_DATE, floor=FLOOR_IS)

    mask = days >= 0
    days_post = days[mask].astype(float)
    log10_post = log10_is[mask].astype(float)

    if len(days_post) == 0:
        raise ValueError(
            "No post-stop qPCR points found (t >= 0). "
            "Check STOP_DATE or your CSV dates."
        )

    order = np.argsort(days_post)
    return days_post[order], log10_post[order]


def _stop_state(params: Params) -> Tuple[float, float, float]:
    """
    Generate a plausible (L*, P*, I*) at stop by running baseline until STOP_DAY_MODEL.
    """
    t, y = simulate(stop_day=STOP_DAY_MODEL, t_end=STOP_DAY_MODEL, y0=Y0_BASELINE, params=params)
    L, P, I = y
    return float(L[-1]), float(P[-1]), float(I[-1])


def _simulate_post_stop(params: Params, *, L0: float, P0: float, I0: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-stop only:
      stop_day=0 => TKI off for all t>=0
      initial state at t=0: (L0, P0, I0)
    Returns: t (days since stop), P(t)
    """
    t, y = simulate(stop_day=0.0, t_end=float(t_end), y0=(L0, P0, I0), params=params)
    _L, P, _I = y
    return t, P


def _alpha_fit_from_t0(P0: float, days_post: np.ndarray, log10_post: np.ndarray) -> float:
    """
    Choose alpha so model matches qPCR level at the point closest to t=0.
    """
    idx0 = _nearest_index(days_post, 0.0)
    is0 = 10 ** float(log10_post[idx0])
    return float(is0 / max(P0, 1e-12))


def _predict_log10_at_days(
    days_post: np.ndarray,
    log10_post: np.ndarray,
    params: Params,
    *,
    P0_override: float | None = None,
    I0_override: float | None = None,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Returns:
      pred_at_days (log10),
      alpha,
      t_curve,
      log10_curve,
      (L0, P0, I0) used at t=0
    """
    Ls, Ps, Is = _stop_state(params)
    L0 = Ls
    P0 = float(P0_override) if P0_override is not None else Ps
    I0 = float(I0_override) if I0_override is not None else Is

    # Sim horizon: cover data + buffer
    t_end = float(max(days_post.max() + 60.0, 120.0))
    t_curve, P_curve = _simulate_post_stop(params, L0=L0, P0=P0, I0=I0, t_end=t_end)

    alpha = _alpha_fit_from_t0(P0, days_post, log10_post)
    log10_curve = bcr_abl_proxy_from_P(P_curve, alpha=alpha)
    pred_at_days = _interp(t_curve, log10_curve, days_post)
    return pred_at_days, alpha, t_curve, log10_curve, (L0, P0, I0)


# ---------- Fit objectives ----------

def _resid_fit_P0_I0(theta: np.ndarray, days_post: np.ndarray, log10_post: np.ndarray, params: Params) -> np.ndarray:
    """
    Fit P0 and I0 when n=2 (log-space):
      theta[0] = log10(P0)
      theta[1] = log10(I0)
    """
    P0 = 10 ** float(theta[0])
    I0 = 10 ** float(theta[1])
    pred, _alpha, _t, _curve, _y0 = _predict_log10_at_days(days_post, log10_post, params, P0_override=P0, I0_override=I0)
    return pred - log10_post


def _resid_fit_kip_klp(theta: np.ndarray, days_post: np.ndarray, log10_post: np.ndarray, params_base: Params) -> np.ndarray:
    """
    Fit kIP and kLP (log-space), post-stop only:
      theta[0] = log10(kIP)
      theta[1] = log10(kLP)
    P0,I0 are taken from the generated stop-state under these params.
    """
    kIP = 10 ** float(theta[0])
    kLP = 10 ** float(theta[1])
    params = replace(params_base, kIP=float(kIP), kLP=float(kLP))
    pred, _alpha, _t, _curve, _y0 = _predict_log10_at_days(days_post, log10_post, params)
    return pred - log10_post


def main() -> None:
    root = _project_root()
    csv_path = root / "data" / "qpcr.csv"

    days_post, log10_post = _load_post_stop_data(csv_path)
    n = len(days_post)

    print(f"[INFO] Found {n} post-stop qPCR points (t >= 0) using STOP_DATE={STOP_DATE}.")
    print(f"[INFO] Post-stop days range: {days_post.min():.1f} .. {days_post.max():.1f}")

    params0 = Params()

    if n >= 3:
        mode = "fit_kip_and_klp"
    elif n == 2:
        mode = "fit_P0_I0"
    else:
        mode = "no_fit"

    if mode == "fit_P0_I0":
        # Baseline stop-state to set reasonable starting guess
        Ls, Ps, Is = _stop_state(params0)

        # Fit log10(P0), log10(I0)
        x0 = np.array([np.log10(max(Ps, 1e-12)), np.log10(max(Is, 1e-12))], dtype=float)

        # Wide bounds but keep within sane numeric ranges
        # P0 in [1e-12, 1e2], I0 in [1e-6, 1e3]
        lb = np.array([np.log10(1e-12), np.log10(1e-6)], dtype=float)
        ub = np.array([np.log10(1e2),   np.log10(1e3)], dtype=float)

        res = least_squares(
            fun=lambda th: _resid_fit_P0_I0(th, days_post, log10_post, params0),
            x0=x0,
            bounds=(lb, ub),
            max_nfev=400,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        P0_fit = float(10 ** res.x[0])
        I0_fit = float(10 ** res.x[1])
        kIP_fit = params0.kIP
        kLP_fit = params0.kLP
        solver_ok = res.success
        solver_msg = res.message

        pred, alpha, t_curve, log10_curve, y0_used = _predict_log10_at_days(
            days_post, log10_post, params0, P0_override=P0_fit, I0_override=I0_fit
        )

    elif mode == "fit_kip_and_klp":
        x0 = np.array([np.log10(params0.kIP), np.log10(params0.kLP)], dtype=float)
        lb = np.array([np.log10(1e-4), np.log10(1e-5)], dtype=float)
        ub = np.array([np.log10(1.0),  np.log10(1.0)], dtype=float)

        res = least_squares(
            fun=lambda th: _resid_fit_kip_klp(th, days_post, log10_post, params0),
            x0=x0,
            bounds=(lb, ub),
            max_nfev=400,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        kIP_fit = float(10 ** res.x[0])
        kLP_fit = float(10 ** res.x[1])
        solver_ok = res.success
        solver_msg = res.message

        params_fit = replace(params0, kIP=kIP_fit, kLP=kLP_fit)
        pred, alpha, t_curve, log10_curve, y0_used = _predict_log10_at_days(days_post, log10_post, params_fit)

        # report implied P0/I0 from stop-state under fitted params
        _Ls, P0_fit, I0_fit = y0_used

    else:
        solver_ok = True
        solver_msg = "No fit performed (only 1 post-stop point)."
        kIP_fit = params0.kIP
        kLP_fit = params0.kLP
        pred, alpha, t_curve, log10_curve, y0_used = _predict_log10_at_days(days_post, log10_post, params0)
        _Ls, P0_fit, I0_fit = y0_used

    rmse = float(np.sqrt(np.mean((pred - log10_post) ** 2)))

    print("\n=== Fit summary (post-stop only) ===")
    print(f"Mode                 = {mode}")
    print(f"kIP (immune killing) = {kIP_fit:.6g}  (default {params0.kIP})")
    print(f"kLP (L->P seeding)   = {kLP_fit:.6g}  (default {params0.kLP})")
    print(f"P0 at stop           = {P0_fit:.6g}")
    print(f"I0 at stop           = {I0_fit:.6g}")
    print(f"alpha (scale factor) = {alpha:.6g}")
    print(f"RMSE (log10-space)   = {rmse:.6g}")
    print(f"Solver success       = {solver_ok} | message: {solver_msg}")

    plt.figure()
    plt.plot(t_curve, log10_curve, label="Model (post-stop, scaled to t≈0)")
    plt.scatter(days_post, log10_post, label="qPCR (post-stop)")
    plt.scatter(days_post, pred, marker="x", label="Model @ data times")
    plt.axvline(0.0, linestyle="--", label="Stop (t=0)")
    plt.xlim(0, max(float(days_post.max()) + 60.0, 120.0))
    plt.xlabel("Days since stop")
    plt.ylabel("log10(IS% or scaled proxy)")
    plt.title("TFR post-stop fit (v0.7: fit P0+I0 when n=2)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
