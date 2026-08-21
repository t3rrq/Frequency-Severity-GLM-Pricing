# Frequency–severity GLM pricing (freMTPL2)

Technical motor TPL premium on the French CASdatasets `freMTPL2` files: a Poisson frequency GLM, a Gamma severity GLM, and pure premium \(E[N] \times E[X \mid \text{claim}]\), scored on a holdout against an exposure-weighted flat rate.

This is a complete baseline, not a rating engine.

- **Not an actuary?** Start with [HOW-IT-WORKS.md](HOW-IT-WORKS.md).
- **Technical note:** [ANALYSIS.md](ANALYSIS.md).

## Results (20% holdout)

Ranking by policy expected loss concentrates actual loss better than a flat rate. Gini is **−0.251** vs **−0.168** when ranking by exposure only (more negative is better here: policies are ordered high score first). Top-decile lift is **1.54**.

Loss ratio is 0.90 on the holdout this reflects the severity cap and a known frequency/severity count mismatch in the source data (see [ANALYSIS.md](ANALYSIS.md#caveats)), not model miscalibration; ranking performance is unaffected.

![Lift chart and Lorenz curve](outputs/lift_gini.png)

| | |
| --- | --- |
| Train / test policies | 542,410 / 135,603 |
| Train claims (severity fit) | 21,132 |
| Test actual loss | 11.06M |
| Test model premium | 12.27M |
| Loss ratio | 0.90 |

Frequency (bonus-malus, driver age) carries most of the rate. Severity factors are mostly weak. Ranking by **annual** premium looks worse because policy expected loss mixes exposure length with risk.

## Specification

- **Frequency** — Poisson GLM, log link, `offset = log(Exposure)`. `ClaimNb` capped at 4.
- **Severity** — Gamma GLM, log link, claim amounts capped at the 0.995 quantile.
- **Split** — 80/20, stratified on `ClaimNb > 0`.
- **Factors** — area, grouped vehicle power/age, driver age, bonus-malus, brand, fuel, region, log density.

## Run

Place `freMTPL2freq.csv` in `data/raw/` (or `Downloads`). The first run fetches `freMTPL2sev.csv` from a Hugging Face mirror if it is missing. CSVs are not in this repo (Dutang & Charpentier, CASdatasets).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # macOS / Linux
.venv/Scripts/python run.py                               # Windows
.venv/bin/python run.py
```

Use the venv interpreter. Charts and tables land in `outputs/`.
