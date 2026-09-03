"""
main_results.py — reproduces the headline numbers and all four figures.

Run from the repository root:
    python analysis/main_results.py

Outputs:
    figures/fig1_calibration_reliability.png
    figures/fig2_overround_timeseries.png
    figures/fig3_cumulative_profit.png
    figures/fig4_team_home_profitability.png

and prints every statistic quoted in the Results section.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA        = ROOT / "data" / "all_raw.csv"
FIG_DIR     = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

STAKE       = 100
PHASE_SPLIT = pd.Timestamp("2022-05-22")
DATA_START  = pd.Timestamp("2019-08-01")
DATA_END    = pd.Timestamp("2025-12-31")
ROLLING_N   = 50
BOOTSTRAP_N = 2000
SEED        = 42

# Okabe-Ito palette (colour-blind safe)
C_BLUE, C_RED, C_GREEN = "#0072B2", "#D55E00", "#009E73"
C_GRAY = "#999999"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


# ── HELPERS ──────────────────────────────────────────────────────────────────
def no_vig(h, d, a):
    """No-vig (fair) probabilities from decimal odds."""
    r = np.array([1 / h, 1 / d, 1 / a])
    return r / r.sum()


def overround_pct(h, d, a):
    return (1 / h + 1 / d + 1 / a - 1) * 100


OH = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}


def bootstrap_mean_ci(arr, n=BOOTSTRAP_N, ci=0.95):
    arr = np.asarray(arr)
    idx = np.random.randint(0, len(arr), (n, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = np.percentile(means, 100 * (1 - ci) / 2)
    hi = np.percentile(means, 100 * (1 + ci) / 2)
    return arr.mean(), lo, hi


def reliability_curve(probs, outcomes, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    mp, mo, ct = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 5:
            mp.append(probs[mask].mean())
            mo.append(outcomes[mask].mean())
            ct.append(mask.sum())
    return np.array(mp), np.array(mo), np.array(ct)


# ── DATA ─────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[df["Div"] == "E0"].dropna(subset=["Date"])
    df = df[(df["Date"] >= DATA_START) & (df["Date"] <= DATA_END)]
    df["FTR"] = df["FTR"].astype(str).str.strip()
    df = df[df["FTR"].isin(["H", "D", "A"])]
    df = df.sort_values("Date").reset_index(drop=True)
    df["season"] = (df["source_file"].astype(str)
                    .str.replace("E0_", "", regex=False)
                    .str.replace(".csv", "", regex=False))
    print(f"  Loaded {len(df):,} EPL matches "
          f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    return df


def score(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Per-match no-vig probabilities, Brier, log loss and overround."""
    h, d, a = cols
    sub = df[["Date", "HomeTeam", "FTR", "season", h, d, a]].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=list(cols)).reset_index(drop=True)
    P = np.vstack([no_vig(r[h], r[d], r[a]) for _, r in sub.iterrows()])
    Y = np.vstack([OH[f] for f in sub["FTR"]])
    sub[["pH", "pD", "pA"]] = P
    sub[["yH", "yD", "yA"]] = Y
    sub["brier"] = ((P - Y) ** 2).sum(1)
    sub["ll"] = -(Y * np.log(np.clip(P, 1e-9, 1 - 1e-9))).sum(1)
    sub["orr"] = [overround_pct(r[h], r[d], r[a]) for _, r in sub.iterrows()]
    sub["odds_H"], sub["odds_D"], sub["odds_A"] = sub[h], sub[d], sub[a]
    return sub


BOOKS = {
    "Pinnacle":     ("PSCH", "PSCD", "PSCA"),
    "Bet365":       ("B365CH", "B365CD", "B365CA"),
    "William Hill": ("WHCH", "WHCD", "WHCA"),
    "Betway":       ("BWCH", "BWCD", "BWCA"),
}
OPEN_B365 = ("B365H", "B365D", "B365A")


# ── KEY NUMBERS ──────────────────────────────────────────────────────────────
def key_numbers(df, bc, bo):
    print("\n" + "=" * 68)
    print("  KEY NUMBERS  (Results section)")
    print("=" * 68)

    print(f"\n  N matches (Bet365 open + close): {len(bc):,}")
    for lab, key in [("BRIER", "brier"), ("LOG LOSS", "ll")]:
        mo, lo_o, hi_o = bootstrap_mean_ci(bo[key].values)
        mc, lo_c, hi_c = bootstrap_mean_ci(bc[key].values)
        w, p = stats.wilcoxon(bo[key].values, bc[key].values)
        diff = bo[key].values - bc[key].values
        dz = diff.mean() / diff.std(ddof=1)
        print(f"\n  {lab}")
        print(f"    Opening : {mo:.4f}  (95% CI {lo_o:.4f}-{hi_o:.4f})")
        print(f"    Closing : {mc:.4f}  (95% CI {lo_c:.4f}-{hi_c:.4f})")
        print(f"    Delta   : {mo - mc:+.4f}   Wilcoxon W={w:,.1f}  p={p:.4f}"
              f"   Cohen d_z={dz:+.3f}")

    print(f"\n  OVERROUND")
    print(f"    Opening mean : {bo['orr'].mean():.2f}%")
    print(f"    Closing mean : {bc['orr'].mean():.2f}%")
    d = bc["orr"].values - bo["orr"].values
    print(f"    SD of per-match difference : {d.std(ddof=1):.4f}")
    m1 = bc["Date"] < PHASE_SPLIT
    print(f"    Phase 1 SD={d[m1.values].std(ddof=1):.3f} (n={m1.sum():,})   "
          f"Phase 2 SD={d[~m1.values].std(ddof=1):.3f} (n={(~m1).sum():,})")

    print(f"\n  NAIVE STRATEGIES  (${STAKE} flat stake)")
    print(f"    {'Strategy':<10}{'Total':>10}{'ROI%':>8}{'Sharpe':>9}"
          f"{'MaxDD':>11}{'Phase1':>11}{'Phase2':>11}")
    for lab, code, oc in [("Home", "H", "odds_H"), ("Draw", "D", "odds_D"),
                          ("Away", "A", "odds_A")]:
        win = (bc["FTR"] == code).values
        ret = np.where(win, (bc[oc].values - 1) * STAKE, -STAKE)
        cum = np.cumsum(ret)
        dd = (cum - np.maximum.accumulate(cum)).min()
        sharpe = ret.mean() / ret.std(ddof=1)
        p1 = ret[m1.values].sum()
        print(f"    {lab:<10}{cum[-1]:>+10,.0f}"
              f"{100 * cum[-1] / (STAKE * len(ret)):>+8.2f}{sharpe:>+9.3f}"
              f"{dd:>+11,.0f}{p1:>+11,.0f}{cum[-1] - p1:>+11,.0f}")

    # Table 1 — intersection of all four books
    print("\n" + "=" * 68)
    print("  TABLE 1  Bookmaker calibration on the common match set")
    print("=" * 68)
    allc = [c for cols in BOOKS.values() for c in cols]
    inter = df.dropna(subset=allc).copy()
    print(f"\n  Intersection: n={len(inter):,}  "
          f"seasons {inter['season'].min()} to {inter['season'].max()}")
    sc = {k: score(inter, v) for k, v in BOOKS.items()}
    print(f"\n  {'Bookmaker':<14}{'Brier':>8}{'  95% CI':>20}{'LogLoss':>10}{'OR%':>8}")
    for k in BOOKS:
        s = sc[k]
        m, lo, hi = bootstrap_mean_ci(s["brier"].values)
        print(f"  {k:<14}{m:>8.4f}  [{lo:.4f}-{hi:.4f}]{s['ll'].mean():>10.4f}"
              f"{s['orr'].mean():>8.2f}")
    print("\n  Paired tests vs Pinnacle (per-match Brier difference):")
    pin = sc["Pinnacle"]["brier"].values
    for k in ["Bet365", "William Hill", "Betway"]:
        v = sc[k]["brier"].values
        m, lo, hi = bootstrap_mean_ci(v - pin)
        _, p = stats.wilcoxon(v, pin)
        flag = "CI includes 0" if lo < 0 < hi else "CI EXCLUDES 0"
        print(f"    vs {k:<13} d={m:+.5f}  95% CI [{lo:+.5f},{hi:+.5f}]  "
              f"p={p:.3f}  -> {flag}")
    print("\n  Full-coverage head-to-head (Pinnacle vs Bet365, all matches):")
    pe, be = score(df, BOOKS["Pinnacle"]), score(df, BOOKS["Bet365"])
    m, lo, hi = bootstrap_mean_ci(be["brier"].values - pe["brier"].values)
    print(f"    Pinnacle {pe['brier'].mean():.4f}   Bet365 {be['brier'].mean():.4f}"
          f"   d={m:+.5f}  95% CI [{lo:+.5f},{hi:+.5f}]  (n={len(pe):,})")
    print()


# ── FIGURES ──────────────────────────────────────────────────────────────────
def fig1(bc, pin):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (label, pc, yc, letter) in zip(axes, [
            ("Home Win", "pH", "yH", "A"), ("Draw", "pD", "yD", "B"),
            ("Away Win", "pA", "yA", "C")]):
        mp, mo, _ = reliability_curve(bc[pc].values, bc[yc].values)
        ax.plot(mp, mo, "o-", color=C_BLUE, lw=1.8, ms=6, label="Bet365 closing", zorder=3)
        mp2, mo2, _ = reliability_curve(pin[pc].values, pin[yc].values)
        ax.plot(mp2, mo2, "s--", color=C_RED, lw=1.8, ms=6, label="Pinnacle closing", zorder=3)
        ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("No-vig predicted probability")
        ax.set_ylabel("Observed outcome frequency")
        ax.set_title(f"({letter}) {label}")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Figure 1.  Reliability diagrams for Bet365 and Pinnacle closing "
                 "no-vig probabilities (EPL 2019/20-2025/26)", fontsize=10, y=1.01)
    plt.tight_layout(); fig.savefig(FIG_DIR / "fig1_calibration_reliability.png"); plt.close()
    print("  saved fig1_calibration_reliability.png")


def fig2(bc, bo):
    ds = bc.copy()
    ds["or_o"] = bo["orr"].values
    ds["or_c"] = bc["orr"].values
    diff = ds["or_c"] - ds["or_o"]
    roll = diff.rolling(ROLLING_N, min_periods=10).std()
    sd1 = diff[ds["Date"] < PHASE_SPLIT].std()
    sd2 = diff[ds["Date"] >= PHASE_SPLIT].std()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7),
                                 gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    a1.plot(ds["Date"], ds["or_o"].rolling(ROLLING_N, min_periods=10).mean(),
            color=C_BLUE, lw=1.8, label="Opening odds overround")
    a1.plot(ds["Date"], ds["or_c"].rolling(ROLLING_N, min_periods=10).mean(),
            color=C_RED, lw=1.8, label="Closing odds overround")
    a1.axvline(PHASE_SPLIT, color=C_GRAY, lw=1.4, ls="--")
    a1.set_ylabel("Overround (%)"); a1.legend(fontsize=9)
    a1.set_title(f"(A) Bet365 overround: {ROLLING_N}-match rolling mean", fontsize=10)
    a2.fill_between(ds["Date"], roll, color=C_GREEN, alpha=0.4)
    a2.plot(ds["Date"], roll, color=C_GREEN, lw=1.5)
    a2.axvline(PHASE_SPLIT, color=C_GRAY, lw=1.4, ls="--")
    a2.set_ylabel("Rolling SD\n(close - open, %)"); a2.set_xlabel("Date")
    a2.set_title("(B) Per-match volatility between opening and closing overround", fontsize=10)
    a2.text(0.02, 0.85, f"Phase 1 SD = {sd1:.3f}    Phase 2 SD = {sd2:.3f}",
            transform=a2.transAxes, fontsize=9)
    fig.suptitle("Figure 2.  Overround dynamics between opening and closing Bet365 "
                 "1X2 odds (EPL 2019/20-2025/26)", fontsize=10, y=1.01)
    plt.tight_layout(); fig.savefig(FIG_DIR / "fig2_overround_timeseries.png"); plt.close()
    print("  saved fig2_overround_timeseries.png")


def fig3(bc):
    strat = [("Always home win", "H", "odds_H", C_BLUE),
             ("Always draw", "D", "odds_D", C_GREEN),
             ("Always away win", "A", "odds_A", C_RED)]
    fig, (ts, bar) = plt.subplots(1, 2, figsize=(14, 5.5),
                                  gridspec_kw={"width_ratios": [3, 1]})
    idx = int((bc["Date"] < PHASE_SPLIT).sum())
    labels, p1v, p2v = [], [], []
    for lab, code, oc, col in strat:
        win = (bc["FTR"] == code).values
        cum = np.cumsum(np.where(win, (bc[oc].values - 1) * STAKE, -STAKE))
        ts.plot(bc["Date"], cum, color=col, lw=1.8, label=lab)
        labels.append(lab.replace("Always ", "")); p1v.append(cum[idx - 1]); p2v.append(cum[-1] - cum[idx - 1])
    ts.axhline(0, color="black", lw=0.8)
    ts.axvline(PHASE_SPLIT, color=C_GRAY, lw=1.4, ls="--", label="Phase boundary (May 2022)")
    ts.set_xlabel("Date"); ts.set_ylabel("Cumulative profit ($)")
    ts.set_title("(A) Cumulative profit: $100 fixed stake per match"); ts.legend(fontsize=9)
    x = np.arange(len(labels)); w = 0.38
    b1 = bar.bar(x - w / 2, p1v, w, color=[C_BLUE if v >= 0 else C_RED for v in p1v],
                 alpha=0.9, label="Phase 1 (2019-22)")
    b2 = bar.bar(x + w / 2, p2v, w, color=[C_BLUE if v >= 0 else C_RED for v in p2v],
                 alpha=0.5, label="Phase 2 (2022-26)")
    bar.axhline(0, color="black", lw=0.8); bar.set_xticks(x)
    bar.set_xticklabels(labels, fontsize=9); bar.set_ylabel("Cumulative profit ($)")
    bar.set_title("(B) Profit by phase"); bar.legend(fontsize=8)
    for b in list(b1) + list(b2):
        h = b.get_height()
        bar.text(b.get_x() + b.get_width() / 2, h + (300 if h >= 0 else -300),
                 f"${h:,.0f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=7.5)
    fig.suptitle("Figure 3.  Cumulative profit from three naive fixed-stake strategies, "
                 "Bet365 closing odds (EPL 2019/20-2025/26, $100 stake per match)",
                 fontsize=10, y=1.01)
    plt.tight_layout(); fig.savefig(FIG_DIR / "fig3_cumulative_profit.png"); plt.close()
    print("  saved fig3_cumulative_profit.png")


def fig4(bc):
    rows = []
    for team, g in bc.groupby("HomeTeam"):
        if len(g) < 20:
            continue
        win = (g["FTR"] == "H").values
        pnl = np.where(win, (g["odds_H"].values - 1) * STAKE, -STAKE).sum()
        rows.append({"team": team, "profit": pnl, "n": len(g),
                     "hw": 100 * win.mean(), "avg": g["odds_H"].mean()})
    t = pd.DataFrame(rows).sort_values("profit")
    fig, ax = plt.subplots(figsize=(10, max(6, len(t) * 0.33)))
    bars = ax.barh(t["team"], t["profit"],
                   color=[C_GREEN if p >= 0 else C_RED for p in t["profit"]],
                   edgecolor="white", lw=0.5)
    ax.axvline(0, color="black", lw=0.9)
    off = max(abs(t["profit"].max()) * 0.02, 80)
    for b, (_, r) in zip(bars, t.iterrows()):
        xw = b.get_width()
        ax.text(xw + off if xw >= 0 else xw - off, b.get_y() + b.get_height() / 2,
                f"HW: {r['hw']:.0f}%  |  avg odds: {r['avg']:.2f}",
                va="center", ha="left" if xw >= 0 else "right", fontsize=7.5, color="#333333")
    ax.set_xlabel("Total cumulative profit ($)")
    ax.set_title("Figure 4.  Cumulative profit from backing each EPL team at home\n"
                 "(Bet365 closing odds, 2019/20-2025/26, $100 stake; >=20 home matches)",
                 fontsize=10)
    plt.tight_layout(); fig.savefig(FIG_DIR / "fig4_team_home_profitability.png"); plt.close()
    print("  saved fig4_team_home_profitability.png")


def main():
    np.random.seed(SEED)
    print("\nLoading data...")
    df = load()
    bc = score(df, BOOKS["Bet365"])
    bo = score(df, OPEN_B365)
    pin = score(df, BOOKS["Pinnacle"])
    key_numbers(df, bc, bo)
    print("Generating figures...")
    fig1(bc, pin); fig2(bc, bo); fig3(bc); fig4(bc)
    print(f"\nDone. Figures written to {FIG_DIR}/\n")


if __name__ == "__main__":
    main()
