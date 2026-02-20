# src/run_scenarios.py
from __future__ import annotations

from pathlib import Path
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from model import Params, simulate, bcr_abl_proxy_from_P
from load_qpcr import load_qpcr_csv, to_days_and_log10_is


# === SET THIS ONCE (recommended) ===
# Put your actual TKI stop date here. Example: date(2025, 2, 20)
STOP_DATE: date | None = None


def _project_root() -> Path:
    # src/run_scenarios.py -> parents[1] = project root
    return Path(__file__).resolve().parents[1]


def _nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(x - target)))


def main() -> None:
    p = Params()

    # --- Simulate scenarios (still parameterized with stop_day=365 in model time) ---
    STOP_DAY_MODEL = 365.0

    tA, yA = simulate(stop_day=STOP_DAY_MODEL, t_end=900, y0=(0.35, 18.0, 0.7), params=p)
    LA, PA, IA = yA

    tB, yB = simulate(stop_day=STOP_DAY_MODEL, t_end=900, y0=(0.35, 18.0, 0.2), params=p)
    LB, PB, IB = yB

    # --- Plot: Model only (absolute model time) ---
    plt.figure()
    plt.plot(tA, bcr_abl_proxy_from_P(PA, alpha=p.alpha), label="A: stronger immune control")
    plt.plot(tB, bcr_abl_proxy_from_P(PB, alpha=p.alpha), label="B: weaker immune control")
    plt.axvline(STOP_DAY_MODEL, linestyle="--", label="TKI stop day (model time)")
    plt.xlabel("Days (model time)")
    plt.ylabel("log10(BCR-ABL proxy)")
    plt.title("CML TFR toy model (absolute model time)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Load qPCR data robustly ---
    root = _project_root()
    csv_path = root / "data" / "qpcr.csv"

    rows = load_qpcr_csv(csv_path)  # auto delimiter + auto columns + auto date formats
    t0_date = STOP_DATE  # if None: loader function uses first measurement date
    days, log10_is, t0_used = to_days_and_log10_is(rows, t0_date=t0_date, floor=1e-6)

    # At this point: 'days' is days since STOP_DATE (if set) otherwise since first data date.

    # --- Align model time axis to stop: t=0 at stop ---
    tA_rel = tA - STOP_DAY_MODEL
    tB_rel = tB - STOP_DAY_MODEL

    # --- Calibrate alpha so model matches data level near stop (t≈0) ---
    # pick nearest model sample at t_rel=0
    idxA0 = _nearest_index(tA_rel, 0.0)
    P_stop = float(PA[idxA0])

    # pick nearest data sample at day=0 (or closest to it)
    idxD0 = _nearest_index(days, 0.0)
    log10_is_at0 = float(log10_is[idxD0])
    is_at0 = 10 ** log10_is_at0

    # alpha_fit maps P_stop -> IS scale at stop (or nearest measurement)
    alpha_fit = is_at0 / max(P_stop, 1e-12)

    # --- Plot: Overlay model vs qPCR on stop-aligned time axis ---
    plt.figure()
    plt.plot(tA_rel, bcr_abl_proxy_from_P(PA, alpha=alpha_fit), label="Model (scenario A, scaled)")
    plt.plot(tB_rel, bcr_abl_proxy_from_P(PB, alpha=alpha_fit), label="Model (scenario B, scaled)", alpha=0.8)
    plt.scatter(days, log10_is, label=f"qPCR (t0={t0_used})")
    plt.axvline(0.0, linestyle="--", label="TKI stop (t=0)")
    plt.xlabel("Days since stop (t=0 at stop)")
    plt.ylabel("log10(IS% or scaled proxy)")
    plt.title("Model vs qPCR overlay (stop-aligned + scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"[INFO] Using t0={t0_used} for qPCR alignment.")
    print(f"[INFO] Nearest qPCR to t=0: day={days[idxD0]:.1f}, log10(IS%)={log10_is_at0:.4f}")
    print(f"[INFO] Model P at stop (nearest): t_rel={tA_rel[idxA0]:.1f} days, P={P_stop:.6g}")
    print(f"[INFO] Fitted alpha (scale factor) = {alpha_fit:.6g}")


if __name__ == "__main__":
    main()
