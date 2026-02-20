# src/run_scenarios.py
from __future__ import annotations

import matplotlib.pyplot as plt

from model import Params, simulate, bcr_abl_proxy_from_P


def main() -> None:
    p = Params()

    # Scenario A: stronger immune control -> TFR holds
    tA, yA = simulate(
        stop_day=365,
        t_end=900,
        y0=(0.35, 18.0, 0.7),  # higher initial I
        params=p,
    )
    LA, PA, IA = yA

    # Scenario B: weaker immune control -> relapse after stop
    tB, yB = simulate(
        stop_day=365,
        t_end=900,
        y0=(0.35, 18.0, 0.2),  # lower initial I
        params=p,
    )
    LB, PB, IB = yB

    # Plot: BCR-ABL proxy (log10)
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

    # Plot: immune control variable
    plt.figure()
    plt.plot(tA, IA, label="I(t) scenario A")
    plt.plot(tB, IB, label="I(t) scenario B")
    plt.axvline(365, linestyle="--", label="TKI stop day")
    plt.xlabel("Days")
    plt.ylabel("Immune control I (a.u.)")
    plt.title("Immune trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
