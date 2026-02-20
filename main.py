import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

@dataclass
class Params:
    # Leukemic stem-like compartment (L)
    rL: float = 0.005      # per day net growth of L
    KL: float = 1.0        # carrying capacity scale for L (dimensionless)
    kLP: float = 0.02      # per day flux from L -> P

    # Proliferating leukemic compartment (P)
    rP: float = 0.08       # per day net growth of P
    KP: float = 50.0       # carrying capacity scale for P
    dP: float = 0.02       # per day baseline death of P

    # TKI effect (kills/inhibits P, weakly affects L)
    tkiP: float = 0.25     # per day extra removal of P when on TKI
    tkiL: float = 0.01     # per day extra removal of L when on TKI (often small)

    # Immune control I
    sI: float = 0.01       # per day immune stimulation by leukemic load
    dI: float = 0.05       # per day immune decay
    KI: float = 10.0       # scale for stimulation saturation

    # Immune-mediated killing of P
    kIP: float = 0.03      # per day killing coefficient

    # Mapping to "measured" BCR-ABL proxy
    alpha: float = 1.0     # measurement proportionality for P (arbitrary)

def u_tki(t, stop_day=365.0):
    """TKI schedule: on until stop_day, then off."""
    return 1.0 if t < stop_day else 0.0

def cml_tfr_rhs(t, y, p: Params, stop_day):
    L, P, I = y
    u = u_tki(t, stop_day=stop_day)

    # L dynamics: slow logistic + (optional) weak TKI impact
    dL = p.rL * L * (1 - L / p.KL) - p.kLP * L - u * p.tkiL * L

    # P dynamics: logistic growth + influx from L - baseline death - TKI - immune kill
    immune_kill = p.kIP * I * P
    dP = p.rP * P * (1 - P / p.KP) + p.kLP * L - p.dP * P - u * p.tkiP * P - immune_kill

    # I dynamics: stimulated by leukemic burden with saturation, decays otherwise
    stim = p.sI * (P / (p.KI + P))
    dI = stim - p.dI * I

    return [dL, dP, dI]

def simulate(stop_day=365.0, t_end=800.0, y0=(0.3, 20.0, 0.2), params=Params()):
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], int(t_end) + 1)

    sol = solve_ivp(
        fun=lambda t, y: cml_tfr_rhs(t, y, params, stop_day),
        t_span=t_span,
        y0=list(y0),
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9
    )
    return sol

def bcr_abl_proxy(P, alpha=1.0, floor=1e-12):
    """Convert P to a log10 proxy resembling qPCR dynamics (arbitrary units)."""
    return np.log10(np.maximum(alpha * P, floor))

if __name__ == "__main__":
    p = Params()

    # Scenario A: good immune control -> TFR holds
    solA = simulate(stop_day=365, t_end=900, y0=(0.35, 18.0, 0.7), params=p)

    # Scenario B: weaker immune control -> relapse after stop
    solB = simulate(stop_day=365, t_end=900, y0=(0.35, 18.0, 0.2), params=p)

    tA = solA.t
    LA, PA, IA = solA.y
    tB = solB.t
    LB, PB, IB = solB.y

    plt.figure()
    plt.plot(tA, bcr_abl_proxy(PA, alpha=p.alpha), label="A: stronger immune control")
    plt.plot(tB, bcr_abl_proxy(PB, alpha=p.alpha), label="B: weaker immune control")
    plt.axvline(365, linestyle="--")
    plt.xlabel("Days")
    plt.ylabel("log10(BCR-ABL proxy)")
    plt.title("CML TFR toy model: stop TKI at day 365")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tA, IA, label="I (A)")
    plt.plot(tB, IB, label="I (B)")
    plt.axvline(365, linestyle="--")
    plt.xlabel("Days")
    plt.ylabel("Immune control I (a.u.)")
    plt.title("Immune trajectory")
    plt.legend()
    plt.show()
