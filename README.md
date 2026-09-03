# Calibration and Market Efficiency in the English Premier League Betting Market

Replication code and data for:

> **Calibration and Market Efficiency in the English Premier League Betting Market:
> A Descriptive Analysis of Bookmaker Accuracy and Naive Strategy Returns, 2019–2026.**
> Agrim Sethi. *National High School Journal of Science* (under review).

The study asks how good bookmaker 1X2 odds are as probability forecasts. It converts
decimal odds to "no-vig" probabilities, scores them with the Brier score and log loss,
compares opening against closing prices, compares four bookmakers on a common set of
matches, and tests whether three naive fixed-stake strategies make money.

Everything in the paper — every figure, table and statistic — is reproduced by the two
scripts in `analysis/`.

## Main findings

- Closing no-vig probabilities are well calibrated across all four bookmakers.
- Judged on the 2,127 matches all four books priced, their accuracy is **statistically
  indistinguishable**. Pinnacle's distinction is its **margin** (2.66% overround against
  5.3–5.8%), not greater accuracy.
- Closing odds beat opening odds on both scoring rules, but the effect is small
  (Brier 0.5663 vs 0.5697; Cohen's d_z ≈ 0.07).
- No naive strategy is profitable, and all three lose on a risk-adjusted basis
  (Sharpe −0.02 to −0.05).

## Quick start

```bash
git clone https://github.com/<your-username>/epl-betting-calibration.git
cd epl-betting-calibration

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python analysis/main_results.py        # key numbers + all four figures
python analysis/revision_analysis.py   # paired bookmaker tests, cluster bootstraps, risk metrics
```

`main_results.py` prints every number quoted in the Results section and writes the four
figures to `figures/`. `revision_analysis.py` produces the analyses added during revision:
the intersection-restricted bookmaker comparison, the season- and matchweek-cluster
bootstraps, and the risk-adjusted strategy metrics.

## Repository layout

```
├── analysis/
│   ├── main_results.py        # headline numbers + Figures 1–4
│   └── revision_analysis.py   # paired bookmaker tests, cluster bootstraps, risk metrics
├── data/
│   ├── all_raw.csv            # EPL results + bookmaker odds (Football-Data.co.uk)
│   └── README.md              # provenance and column conventions
├── src/football_betting/
│   └── odds.py                # compute_no_vig_probs — the de-vigging utility
├── tests/
│   └── test_odds.py           # unit tests for the odds conversion
├── figures/                   # generated output
├── requirements.txt
└── LICENSE
```

## Method in brief

Decimal odds are converted to implied probabilities as `p = 1/odds`. Because these sum to
more than 1 by the bookmaker's margin, they are normalised by their own sum to give
no-vig probabilities. The overround is `(Σp − 1) × 100%`.

Forecasts are scored with two strictly proper rules, the multiclass Brier score and log
loss, so a bookmaker cannot improve its score by reporting probabilities it does not
believe. Confidence intervals come from a nonparametric bootstrap: per-match scores are
resampled with replacement 2,000 times and the middle 95% of the resampled means is taken.
Paired comparisons resample the per-match differences, and the cluster bootstraps resample
whole seasons or whole matchweeks to respect the fact that teams and seasons recur.

## Reproducibility notes

- Point estimates are deterministic and will match the paper exactly.
- Bootstrap confidence intervals are random. The seed is fixed (42), but intervals can
  still differ in the fourth decimal place depending on the order in which the scripts
  consume random numbers. This does not affect any conclusion.
- Tests: `python -m pytest tests/ -q`

## Data

Match results and odds come from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php),
which publishes historical football data free for academic use. See `data/README.md`.

## Licence

Code released under the MIT Licence (see `LICENSE`). The underlying match data remains the
property of Football-Data.co.uk and is included here only to make the results reproducible.
