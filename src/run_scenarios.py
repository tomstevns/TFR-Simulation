# src/run_scenarios.py
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from model import Params, simulate, bcr_abl_proxy_from_P
from load_qpcr import load_qpcr_csv, to_days_and_log10_is


def _project_root() -> Path:
    # src/run_scenarios.py -> parents[1] = project root
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = Params()

    # --- Simulate scenarios ---
    tA, yA = simulate(stop_day=365, t_end=900, y0=(0.35, 18.0, 0.7), params=p)
    LA, PA, IA = yA

    tB, yB = simulate(stop_day=365, t_end=900, y0=(0.35, 18.0, 0.2), params=p)
    LB, PB, IB = yB

    # --- Plot: Model only ---
    plt.figure()
    plt.plot(tA, bcr_abl_proxy_from_P(PA, alpha=p.alpha), label="A: stronger immune control")
    plt.plot(tB, bcr_abl_proxy_from_P(PB, alpha=p.alpha), label="B: weaker immune control")
    plt.axvline(365, linestyle="--", label="TKI stop day")
    plt.xlabel("Days")
    plt.ylabel("log10(BCR-ABL proxy)")
    plt.title("CML TFR toy model: stop TKI at day 365")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Load qPCR data robustly ---
    root = _project_root()
    csv_path = root / "data" / "qpcr.csv"

    rows = load_qpcr_csv(csv_path)  # auto delimiter + auto columns + auto date formats
    days, log10_is, t0 = to_days_and_log10_is(rows, t0_date=None, floor=1e-6)

    # --- Plot: Overlay model vs qPCR ---
    plt.figure()
    plt.plot(tA, bcr_abl_proxy_from_P(PA, alpha=p.alpha), label="Model (scenario A)")
    plt.scatter(days, log10_is, label=f"qPCR (t0={t0})")
    plt.axvline(365, linestyle="--", label="TKI stop day (model time)")
    plt.xlabel("Days")
    plt.ylabel("log10(IS% or proxy)")
    plt.title("Model vs qPCR overlay")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
