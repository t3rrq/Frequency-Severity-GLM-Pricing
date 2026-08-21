# Frequency–severity GLM pricing (freMTPL2)

Technical motor TPL premium on the French CASdatasets `freMTPL2` files: a Poisson frequency GLM, a Gamma severity GLM, and pure premium \(E[N] \times E[X \mid \text{claim}]\), scored on a holdout against an exposure-weighted flat rate.

This is a small, complete baseline — not a rating engine.

## Specification

- **Frequency** — Poisson GLM, log link, `offset = log(Exposure)`. Claim counts capped at 4 (standard on this file).
- **Severity** — Gamma GLM, log link, claim-level amounts (0.995 quantile cap) joined to the policy’s rating factors.
- **Pure premium** — product of the two means on a 20% holdout, stratified on `ClaimNb > 0` so the severity sample is not emptied by chance.
- **Rating factors** — area, grouped vehicle power/age, driver age, bonus-malus, brand, fuel, region, log density.

## Holdout results

![Lift chart and Lorenz curve](outputs/lift_gini.png)

| | |
| --- | --- |
| Train / test policies | 542,410 / 135,603 |
| Train claims (severity fit) | 21,132 |
| Test actual loss | 11.06M |
| Test model premium | 12.27M |
| Gini (GLM rank) | −0.251 |
| Gini (exposure-only) | −0.168 |

In this Gini implementation, policies are ordered by **decreasing** score; a more negative value means more actual loss sits among the high-score policies. The GLM ranks better than a flat rate.

Decile 1 (highest predicted premium) has lift 1.54. The bottom decile is noisy: ranking is by **policy expected loss**, so short-exposure policies collect there and a few large claims inflate the empirical rate.

## Caveats

- On `freMTPL2`, frequency `ClaimNb` often does not equal the number of severity rows for the same `IDpol`. Frequency still uses `ClaimNb`; severity uses observed amounts. They are not forced to match.
- Severity factors are mostly weak; frequency (especially bonus-malus and driver age) carries most of the rate.
- Data: Dutang & Charpentier, CASdatasets (`freMTPL2freq` / `freMTPL2sev`). CSVs are not in this repo.

## Run

Place `freMTPL2freq.csv` in `data/raw/` (or `Downloads`). The first run fetches `freMTPL2sev.csv` from a Hugging Face mirror if it is missing.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # macOS / Linux
.venv/Scripts/python run.py                               # Windows
.venv/bin/python run.py
```

Use the venv interpreter, not a system `python` that never received the packages. Python 3.8 is enough (`matplotlib` 3.7).

Outputs: `outputs/frequency_coefs.csv`, `outputs/severity_coefs.csv`, `outputs/lift_table.csv`, `outputs/metrics.txt`, `outputs/lift_gini.png`.
