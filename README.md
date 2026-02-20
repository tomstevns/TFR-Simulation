```markdown
# CML Treatment-Free Remission (TFR) Simulator (Python)

A small, research-style Python project that simulates **CML (Philadelphia chromosome / BCR-ABL1–driven disease)** dynamics under **TKI treatment** and after **treatment discontinuation** (TFR: *treatment-free remission*).

The goal is to build an interpretable, mechanistic “toy-but-useful” model that can reproduce common clinical trajectories:

- Deep molecular response under TKI (declining BCR-ABL1 signal)
- After stopping TKI: either  
  - **durable TFR** (BCR-ABL1 stays low), or  
  - **molecular relapse** (BCR-ABL1 rises again)

This repo is designed to be extended with personal qPCR time series (BCR-ABL1 IS %) and parameter fitting.

---

## Model Overview

We use a simple ODE system with three state variables:

- **L(t)** — leukemic stem-like / initiating compartment (slow, persistent reservoir)
- **P(t)** — proliferating leukemic compartment (drives measured BCR-ABL1 dynamics)
- **I(t)** — abstract immune-control variable (effective anti-leukemic pressure)

**TKI treatment** is represented by a time-dependent control **u(t)**:

- `u(t) = 1` while on TKI
- `u(t) = 0` after treatment stop (TFR phase)

### Intuition

- TKI strongly suppresses **P** (and weakly affects **L**).
- Residual disease can persist in **L** even when **P** is very low.
- After stopping TKI, the outcome depends on whether **immune control (I)** is strong enough to keep **P** suppressed while **L** slowly persists.

---

## Files / Structure (suggested)

You can keep it minimal, or expand into a small package.

```

cml-tfr-simulator/
README.md
requirements.txt
src/
model.py          # ODE definitions + simulation
run_scenarios.py  # example runs (TFR vs relapse)
fit.py            # (optional) parameter fitting to qPCR
data/
example_qpcr.csv  # (optional) your anonymized BCR-ABL1 IS%
notebooks/
exploration.ipynb # (optional) interactive analysis

```

---

## Requirements

- Python 3.10+ recommended
- `numpy`
- `scipy`
- `matplotlib`

Example `requirements.txt`:

```

numpy
scipy
matplotlib

````

Install:

```bash
pip install -r requirements.txt
````

---

## Quick Start

Create a script (e.g., `src/run_scenarios.py`) and paste in the model code (or import it from `src/model.py`).

Run:

```bash
python src/run_scenarios.py
```

You should see plots of:

* `log10(BCR-ABL proxy)` vs time, with a vertical line at the stop day
* Immune trajectory `I(t)` for different scenarios

---

## The Core ODE System (Concept)

A typical right-hand side (RHS) looks like:

* **L dynamics**: slow growth + flux into P + (weak) TKI effect
* **P dynamics**: faster growth + influx from L − baseline death − TKI effect − immune killing
* **I dynamics**: stimulation by leukemic burden (saturating) − decay

This gives a compact model that can exhibit:

* multi-phase decline under TKI
* rebound after stop if immune control is insufficient
* stable suppression after stop if immune control is sufficient

---

## Interpreting Output as “BCR-ABL1”

Clinical monitoring is usually based on **qPCR BCR-ABL1 (IS %)** and commonly plotted on a **log scale**.

In this project, we use a simple proxy:

* `BCRABL_proxy(t) = log10(alpha * P(t))`

Where:

* **P(t)** is the proliferating compartment
* **alpha** is a proportionality constant (used to scale the proxy)

When you add real data, you can map `P(t)` to the log10(IS %) scale more carefully (see below).

---

## Adding Your qPCR Data (BCR-ABL1 IS %)

### 1) Prepare a CSV

Put a file like `data/qpcr.csv`:

```csv
date,bcr_abl_is_percent
2024-02-01,0.008
2024-03-15,0.006
2024-05-01,0.004
...
```

### 2) Convert dates to “days since baseline”

Typical baseline choices:

* first measurement date
* TKI stop date (t = 0 at stop)
* start of treatment (if you have long history)

### 3) Fit parameters

A practical minimal fitting target:

* Fit to **pre-stop** data (while on TKI) to estimate TKI suppression and baseline growth.
* Validate on **post-stop** data to see if the immune-control term can explain durable TFR vs relapse.

Suggested parameters to fit first:

* `tkiP` (TKI impact on P)
* `rP` and/or `dP` (net dynamics of P)
* `kLP` (L → P flux)
* `kIP` and immune parameters (`sI`, `dI`) for post-stop behavior

> Note: many parameter combinations can fit similarly well (identifiability is a real issue).
> Use bounds, priors, and sensitivity analysis rather than trusting a single best-fit.

---

## What This Model Can and Cannot Claim

### It can illustrate

* Why some patients may maintain low BCR-ABL1 after stopping TKI (effective control)
* Why others relapse quickly (insufficient control or higher residual reservoir)
* How a persistent reservoir (L) can exist without overt relapse

### It does NOT prove

* Individual clinical outcomes
* Causal biology beyond the abstractions (immune variable is simplified)
* Mutation-specific TKI resistance (unless you explicitly add resistant clones)

---

## Extensions (Roadmap)

If you want to move closer to the literature and clinical reality, consider adding:

1. **Resistant subclone compartment**

   * Add `P_res(t)` with reduced TKI sensitivity
   * Model selection dynamics under TKI

2. **Multi-phase decline**

   * Add more compartments (stem/progenitor/mature)
   * Or introduce time-varying effective kill terms

3. **Explicit measurement model**

   * Noise, detection limits, log-normal assay error
   * Event definition: “loss of MMR” etc.

4. **Bayesian parameter inference**

   * Use `pymc` for posterior distributions over parameters
   * Better uncertainty handling than point estimates

---

## Safety / Medical Disclaimer

This code is for **education and research exploration** only.
It is **not** a diagnostic tool and must not be used to make treatment decisions.

Always discuss clinical questions and interpretation of BCR-ABL1 monitoring with your hematology team.

---

## License

Choose a license that fits your intent. Common options:

* MIT (permissive)
* Apache-2.0 (permissive + patent grant)
* GPL-3.0 (copyleft)

---

## Citation / Inspiration

This project is inspired by the general modeling literature on CML treatment response and treatment-free remission (TFR), including immune-control and residual disease dynamics.

If you add citations to specific papers/models you base parameters on, list them here.

---

## Contact / Notes

If you use this for your own monitoring data:

* keep data anonymized
* document how you processed the qPCR values
* write down the exact stop date and sampling schedule
* store assumptions in a `docs/assumptions.md`

```
::contentReference[oaicite:0]{index=0}
```
