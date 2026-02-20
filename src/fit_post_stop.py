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

# Baseline run to generate a plausible stop-state (L*,P*,I*)
STOP_DAY_MODEL = 365.0
Y0_BASELINE = (0.35, 18.0, 0.7)

# Data floor for log10 transform (use assay LOD if known)
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
        raise ValueError("No post-stop qPCR points found (t >= 0). Check STOP_DATE or CSV dates.")

    order = np.argsort(days_post)
    return days_post[order], log10_post[order]


def _stop_state(params: Params) -> Tuple[float, float, float]:
    # run baseline up to stop day and read final state
    _t, y = simulate(stop_day=STOP_DAY_MODEL, t_end=STOP_DAY_MODEL, y0=Y0_BASELINE, params=params)
    L, P, I = y
    return float(L[-1]), float(P[-1]), float(I[-1])


def _simulate_post_stop(params: Params, *, L0: float, P0: float, I0: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    # post-stop only: stop_day=0 => TKI off for all t>=0
    t, y = simulate(stop_day=0.0, t_end=float(t_end), y0=(L0, P0, I0), params=params)
    _L, P, _I = y
    return t, P


def _alpha_fit_from_t0(P0: float, days_post: np.ndarray, log10_post: np.ndarray) -> float:
    idx0 = _nearest_index(days_post, 0.0)
    is0 = 10 ** float(log10_post[idx0])
    return float(is0 / max(P0, 1e-12))


def _predict_log10_at_days(
    days_post: np.ndarray,
    log10_post: np.ndarray,
    params: Params,
    *,
    P0: float,
    I0: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    # simulate long enough to cover data + buffer
    t_end = float(max(days_post.max() + 60.0, 120.0))

    # L0 is taken from stop-state under current params (kept consistent)
    Ls, _Ps, _Is = _stop_state(params)
    t_curve, P_curve = _simulate_post_stop(params, L0=Ls, P0=float(P0), I0=float(I0), t_end=t_end)

    alpha = _alpha_fit_from_t0(P0, days_post, log10_post)
    log10_curve = bcr_abl_proxy_from_P(P_curve, alpha=alpha)
    pred_at_days = _interp(t_curve, log10_curve, days_post)
    return pred_at_days, alpha, t_curve, log10_curve


# ---------------- Fit objectives ----------------

def _resid_fit_P0_I0(theta: np.ndarray, days_post: np.ndarray, log10_post: np.ndarray, params: Params) -> np.ndarray:
    # theta: [log10(P0), log10(I0)]
    P0 = 10 ** float(theta[0])
    I0 = 10 ** float(theta[1])
    pred, _alpha, _t, _curve = _predict_log10_at_days(days_post, log10_post, params, P0=P0, I0=I0)
    return pred - log10_post


def _resid_fit_kip_klp(theta: np.ndarray, days_post: np.ndarray, log10_post: np.ndarray, params_base: Params, P0_fix: float, I0_fix: float) -> np.ndarray:
    # theta: [log10(kIP), log10(kLP)], while P0 and I0 are frozen to baseline stop-state
    kIP = 10 ** float(theta[0])
    kLP = 10 ** float(theta[1])
    params = replace(params_base, kIP=float(kIP), kLP=float(kLP))
    pred, _alpha, _t, _curve = _predict_log10_at_days(days_post, log10_post, params, P0=P0_fix, I0=I0_fix)
    return pred - log10_post


def _sanity_report(days_post: np.ndarray, log10_post: np.ndarray) -> None:
    is_vals = 10 ** log10_post
    print(f"[INFO] IS%(t0≈0): {is_vals[_nearest_index(days_post, 0.0)]:.6g}")
    print(f"[INFO] IS%(last):  {is_vals[-1]:.6g} at day {days_post[-1]:.1f}")
    if len(days_post) >= 2:
        slope = (log10_post[-1] - log10_post[0]) / max(days_post[-1] - days_post[0], 1e-12)
        print(f"[INFO] Data slope: {slope:.6g} log10-units/day (first→last)")


def main() -> None:
    root = _project_root()
    csv_path = root / "data" / "qpcr.csv"

    days_post, log10_post = _load_post_stop_data(csv_path)
    n = len(days_post)

    print(f"[INFO] Found {n} post-stop qPCR points (t >= 0) using STOP_DATE={STOP_DATE}.")
    print(f"[INFO] Post-stop days range: {days_post.min():.1f} .. {days_post.max():.1f}")
    _sanity_report(days_post, log10_post)

    params0 = Params()

    # Baseline stop-state (used for freezing when n>=3)
    Ls0, P0_baseline, I0_baseline = _stop_state(params0)

    if n == 1:
        mode = "no_fit"
    elif n == 2:
        mode = "fit_P0_I0"
    else:
        mode = "fit_kip_klp_with_P0I0_frozen"

    # ---------------- Run chosen mode ----------------
    if mode == "fit_P0_I0":
        # Start guess from baseline stop-state, but clamp into bounds
        lb = np.array([np.log10(1e-12), np.log10(1e-6)], dtype=float)  # P0, I0
        ub = np.array([np.log10(1e2),   np.log10(1e3)], dtype=float)

        x0_raw = np.array([np.log10(max(P0_baseline, 1e-12)), np.log10(max(I0_baseline, 1e-6))], dtype=float)
        x0 = np.minimum(np.maximum(x0_raw, lb), ub)

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
        params_fit = params0
        kIP_fit, kLP_fit = params_fit.kIP, params_fit.kLP
        solver_ok, solver_msg = res.success, res.message

        pred, alpha, t_curve, log10_curve = _predict_log10_at_days(days_post, log10_post, params_fit, P0=P0_fit, I0=I0_fit)

    elif mode == "fit_kip_klp_with_P0I0_frozen":
        # Fit kIP,kLP but freeze P0,I0 to baseline stop-state
        P0_fix = float(P0_baseline)
        I0_fix = float(I0_baseline)

        # log-space bounds for kIP,kLP
        lb = np.array([np.log10(1e-4), np.log10(1e-5)], dtype=float)
        ub = np.array([np.log10(1.0),  np.log10(1.0)], dtype=float)
        x0 = np.array([np.log10(params0.kIP), np.log10(params0.kLP)], dtype=float)
        x0 = np.minimum(np.maximum(x0, lb), ub)

        res = least_squares(
            fun=lambda th: _resid_fit_kip_klp(th, days_post, log10_post, params0, P0_fix, I0_fix),
            x0=x0,
            bounds=(lb, ub),
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        kIP_fit = float(10 ** res.x[0])
        kLP_fit = float(10 ** res.x[1])
        params_fit = replace(params0, kIP=kIP_fit, kLP=kLP_fit)
        solver_ok, solver_msg = res.success, res.message

        # P0,I0 frozen:
        P0_fit, I0_fit = float(P0_fix), float(I0_fix)

        pred, alpha, t_curve, log10_curve = _predict_log10_at_days(days_post, log10_post, params_fit, P0=P0_fit, I0=I0_fit)

    else:
        # no fit
        params_fit = params0
        P0_fit, I0_fit = float(P0_baseline), float(I0_baseline)
        kIP_fit, kLP_fit = params_fit.kIP, params_fit.kLP
        solver_ok, solver_msg = True, "No fit performed (n=1)."
        pred, alpha, t_curve, log10_curve = _predict_log10_at_days(days_post, log10_post, params_fit, P0=P0_fit, I0=I0_fit)

    rmse = float(np.sqrt(np.mean((pred - log10_post) ** 2)))

    print("\n=== Fit summary (post-stop only) ===")
    print(f"Mode                 = {mode}")
    print(f"kIP (immune killing) = {kIP_fit:.6g}  (default {params0.kIP})")
    print(f"kLP (L->P seeding)   = {kLP_fit:.6g}  (default {params0.kLP})")
    print(f"P0 at stop           = {P0_fit:.6g}  (baseline stop-state {P0_baseline:.6g})")
    print(f"I0 at stop           = {I0_fit:.6g}  (baseline stop-state {I0_baseline:.6g})")
    print(f"alpha (scale factor) = {alpha:.6g}")
    print(f"RMSE (log10-space)   = {rmse:.6g}")
    print(f"Solver success       = {solver_ok} | message: {solver_msg}")

    # Plot post-stop only
    plt.figure()
    plt.plot(t_curve, log10_curve, label="Model (post-stop, scaled to t≈0)")
    plt.scatter(days_post, log10_post, label="qPCR (post-stop)")
    plt.scatter(days_post, pred, marker="x", label="Model @ data times")
    plt.axvline(0.0, linestyle="--", label="Stop (t=0)")
    plt.xlim(0, max(float(days_post.max()) + 60.0, 120.0))
    plt.xlabel("Days since stop")
    plt.ylabel("log10(IS% or scaled proxy)")
    plt.title("TFR post-stop fit (v0.8: auto mode switch)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
