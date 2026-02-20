# src/model.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    # Leukemic stem-like compartment (L)
    rL: float = 0.005      # per day net growth of L
    KL: float = 1.0        # carrying capacity (scaled units)
    kLP: float = 0.02      # per day flux from L -> P

    # Proliferating leukemic compartment (P)
    rP: float = 0.08       # per day net growth of P
    KP: float = 50.0       # carrying capacity (scaled units)
    dP: float = 0.02       # per day baseline death of P

    # TKI effect (strong on P, weak on L)
    tkiP: float = 0.25     # per day extra removal of P when on TKI
    tkiL: float = 0.01     # per day extra removal of L when on TKI

    # Immune control variable (I)
    sI: float = 0.01       # per day immune stimulation by leukemic burden
    dI: float = 0.05       # per day immune decay
    KI: float = 10.0       # stimulation saturation scale

    # Immune-mediated killing of P
    kIP: float = 0.03      # per day killing coefficient

    # Mapping to "measured" BCR-ABL proxy
    alpha: float = 1.0     # proportionality constant for proxy


def u_tki(t: float, stop_day: float) -> float:
    """TKI schedule: on until stop_day, then off."""
    return 1.0 if t < stop_day else 0.0


def rhs(t: float, y: np.ndarray, p: Params, stop_day: float) -> list[float]:
    """
    State:
      L(t): leukemic stem-like reservoir
      P(t): proliferating leukemic burden (proxy for qPCR dynamics)
      I(t): immune-control variable (abstract)
    """
    L, P, I = y
    u = u_tki(t, stop_day)

    # L dynamics: slow logistic growth, leaks into P, weakly affected by TKI
    dL = p.rL * L * (1.0 - L / p.KL) - p.kLP * L - u * p.tkiL * L

    # P dynamics: logistic growth + influx from L - baseline death - TKI - immune kill
    immune_kill = p.kIP * I * P
    dP = (
        p.rP * P * (1.0 - P / p.KP)
        + p.kLP * L
        - p.dP * P
        - u * p.tkiP * P
        - immune_kill
    )

    # I dynamics: stimulation by leukemic burden (saturating), minus decay
    stim = p.sI * (P / (p.KI + P))
    dI = stim - p.dI * I

    return [dL, dP, dI]


def simulate(
    *,
    stop_day: float = 365.0,
    t_end: float = 900.0,
    y0: tuple[float, float, float] = (0.35, 18.0, 0.3),
    params: Params = Params(),
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t: shape (N,)
      y: shape (3, N) for [L, P, I]
    """
    if t_end <= 0:
        raise ValueError("t_end must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if stop_day < 0:
        raise ValueError("stop_day must be >= 0")

    t_eval = np.arange(0.0, t_end + dt, dt)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, stop_day),
        t_span=(0.0, float(t_end)),
        y0=np.array(y0, dtype=float),
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t, sol.y


def bcr_abl_proxy_from_P(P: np.ndarray, *, alpha: float = 1.0, floor: float = 1e-12) -> np.ndarray:
    """
    Convert proliferating burden P(t) to a log10 proxy resembling qPCR dynamics.

    This is NOT yet a calibrated mapping to IS%. It's a convenient plot quantity:
      proxy = log10(alpha * P)
    """
    P_safe = np.maximum(alpha * P, floor)
    return np.log10(P_safe)
