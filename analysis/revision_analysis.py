#!/usr/bin/env python3
"""
revision_analysis.py - the three analyses added in revision.
Addresses reviewer comments M11/M19/M27 (paired bookmaker tests on intersection),
M15/M23 (block-bootstrap CIs + effect sizes, open vs close), M24 (risk metrics).
Run from the repository root:  python analysis/revision_analysis.py
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA = Path(__file__).resolve().parents[1] / "data" / "all_raw.csv"
B = 5000  # bootstrap reps

def no_vig(h,d,a):
    r=np.array([1/h,1/d,1/a]); return r/r.sum()
def brier_row(p,y): return float(np.sum((p-y)**2))
def ll_row(p,y,eps=1e-9): return float(-np.sum(y*np.log(np.clip(p,eps,1-eps))))
OH={"H":(1,0,0),"D":(0,1,0),"A":(0,0,1)}

df=pd.read_csv(DATA,low_memory=False)
df["D"]=pd.to_datetime(df["Date"],dayfirst=True,errors="coerce")
df=df[(df["Div"]=="E0")&(df["D"]>=pd.Timestamp("2019-08-01"))&(df["D"]<=pd.Timestamp("2025-12-31"))].copy()
df=df.sort_values("D").reset_index(drop=True)
df["season"]=df["source_file"].str.replace("E0_","",regex=False).str.replace(".csv","",regex=False)
df["FTR"]=df["FTR"].astype(str).str.strip()
df=df[df["FTR"].isin(["H","D","A"])].copy()
# matchweek block id: within season, every 10 matches by date
df["mw"]=df.groupby("season").cumcount()//10
df["block"]=df["season"]+"_"+df["mw"].astype(str)

def per_match_scores(frame,cols):
    h,d,a=cols; sub=frame[[h,d,a,"FTR","season","block"]].copy()
    for c in cols: sub[c]=pd.to_numeric(sub[c],errors="coerce")
    sub=sub.dropna(subset=cols)
    P=np.vstack([no_vig(r[h],r[d],r[a]) for _,r in sub.iterrows()])
    Y=np.vstack([OH[f] for f in sub["FTR"]])
    sub["brier"]=((P-Y)**2).sum(1)
    sub["ll"]=-(Y*np.log(np.clip(P,1e-9,1-1e-9))).sum(1)
    return sub

def boot_ci(x,reps=B):
    x=np.asarray(x); idx=np.random.randint(0,len(x),(reps,len(x)))
    m=x[idx].mean(1); return x.mean(),np.percentile(m,2.5),np.percentile(m,97.5)

def cluster_boot_ci(vals,groups,reps=B):
    g=pd.Series(vals).groupby(groups); gl=[v.values for _,v in g]; k=len(gl)
    means=[]
    for _ in range(reps):
        pick=np.random.randint(0,k,k); means.append(np.concatenate([gl[i] for i in pick]).mean())
    return np.mean(vals),np.percentile(means,2.5),np.percentile(means,97.5)

print("="*78); print("RERUN A — Bookmaker paired comparison on INTERSECTION (M11/M19/M27)"); print("="*78)
bms={"Pinnacle":["PSCH","PSCD","PSCA"],"Bet365":["B365CH","B365CD","B365CA"],
     "WilliamHill":["WHCH","WHCD","WHCA"],"Betway":["BWCH","BWCD","BWCA"]}
allcols=sum(bms.values(),[])
for c in allcols: df[c]=pd.to_numeric(df[c],errors="coerce")
inter=df.dropna(subset=allcols).copy()
print(f"All-4 intersection: n={len(inter)}  seasons {inter['season'].min()}..{inter['season'].max()}")
sc={}
for name,cols in bms.items():
    s=per_match_scores(inter,cols); sc[name]=s.reset_index(drop=True)
print(f"\n{'Bookmaker':<13}{'Brier':>9}{'  95% CI':>20}{'LogLoss':>10}{'  OR%':>8}")
for name,cols in bms.items():
    s=sc[name]; bm,bl,bh=boot_ci(s['brier'].values)
    orr=( (1/inter[cols[0]]+1/inter[cols[1]]+1/inter[cols[2]]-1)*100 ).mean()
    print(f"{name:<13}{bm:>9.4f}  [{bl:.4f},{bh:.4f}]{s['ll'].mean():>10.4f}{orr:>8.2f}")
print("\nPaired vs Pinnacle (per-match Brier diff = X - Pinnacle; >0 means Pinnacle better):")
pin=sc["Pinnacle"]["brier"].values
for name in ["Bet365","WilliamHill","Betway"]:
    diff=sc[name]["brier"].values-pin
    m,lo,hi=boot_ci(diff); w,p=stats.wilcoxon(sc[name]["brier"].values,pin)
    rbc=1-2*((sc[name]['brier'].values<pin).sum()+0.5*(sc[name]['brier'].values==pin).sum())/len(pin)
    sig="" if (lo<0<hi) else "  *CI excludes 0*"
    print(f"  Pinnacle vs {name:<11} Δ={m:+.5f} 95%CI[{lo:+.5f},{hi:+.5f}] Wilcoxon p={p:.3f} rank-biserial={rbc:+.3f}{sig}")

print("\n-- Head-to-head Pinnacle vs Bet365 on FULL coverage (n=2430) --")
pe=per_match_scores(df,bms["Pinnacle"]).reset_index(drop=True)
be=per_match_scores(df,bms["Bet365"]).reset_index(drop=True)
for metric in ["brier","ll"]:
    diff=be[metric].values-pe[metric].values
    m,lo,hi=boot_ci(diff); w,p=stats.wilcoxon(be[metric].values,pe[metric].values)
    print(f"  {metric.upper():5} Pinnacle={pe[metric].mean():.4f} Bet365={be[metric].mean():.4f} Δ(B365-Pin)={m:+.5f} CI[{lo:+.5f},{hi:+.5f}] p={p:.3f}")

print("\n"+"="*78); print("RERUN B — Open vs Close: block bootstrap + effect sizes (M15/M23)"); print("="*78)
bo=per_match_scores(df,["B365H","B365D","B365A"]).reset_index(drop=True)
bc=per_match_scores(df,["B365CH","B365CD","B365CA"]).reset_index(drop=True)
for metric,lab in [("brier","BRIER"),("ll","LOG LOSS")]:
    o=bo[metric].values; c=bc[metric].values; diff=o-c   # open-close; >0 => closing better
    md=diff.mean()
    _,il,ih=boot_ci(diff); _,sl,sh=cluster_boot_ci(diff,df["season"].values); _,ml,mh=cluster_boot_ci(diff,df["block"].values)
    w,p=stats.wilcoxon(o,c)
    dz=md/diff.std(ddof=1); rbc=1-2*((o<c).sum()+0.5*(o==c).sum())/len(diff)
    print(f"\n{lab}: open={o.mean():.4f} close={c.mean():.4f}  mean Δ(open-close)={md:+.5f}")
    print(f"   iid bootstrap 95% CI    : [{il:+.5f}, {ih:+.5f}]")
    print(f"   season-cluster 95% CI   : [{sl:+.5f}, {sh:+.5f}]   ({df['season'].nunique()} clusters)")
    print(f"   matchweek-cluster 95% CI: [{ml:+.5f}, {mh:+.5f}]   ({df['block'].nunique()} clusters)")
    print(f"   Wilcoxon p={p:.4f}  Cohen d_z={dz:+.3f}  rank-biserial={rbc:+.3f}  (relative Δ={100*md/o.mean():.2f}%)")

print("\n"+"="*78); print("RERUN C — Naive strategy risk-adjusted metrics (M24)"); print("="*78)
STAKE=100; PHASE=pd.Timestamp("2022-05-22")
print(f"{'Strategy':<12}{'Total$':>10}{'ROI%':>8}{'meanRet':>9}{'SD':>8}{'Sharpe':>9}{'MaxDD$':>10}{'MaxDD%':>8}")
for lab,code,col in [("Home","H","B365CH"),("Draw","D","B365CD"),("Away","A","B365CA")]:
    sub=df.dropna(subset=[col]).copy(); odds=pd.to_numeric(sub[col],errors="coerce")
    win=(sub["FTR"]==code).values; ret=np.where(win,(odds-1)*STAKE,-STAKE)  # $ per match
    cum=np.cumsum(ret); peak=np.maximum.accumulate(cum); dd=cum-peak; mdd=dd.min()
    total=cum[-1]; roi=100*total/(STAKE*len(ret)); sharpe=ret.mean()/ret.std(ddof=1)
    mdd_pct=100*mdd/(STAKE*len(ret))
    print(f"{lab:<12}{total:>+10.0f}{roi:>+8.2f}{ret.mean():>+9.2f}{ret.std():>8.1f}{sharpe:>+9.3f}{mdd:>+10.0f}{mdd_pct:>+8.2f}")
# Away strategy phase split (COVID context, M28)
sub=df.dropna(subset=["B365CA"]).copy(); odds=pd.to_numeric(sub["B365CA"],errors="coerce")
win=(sub["FTR"]=="A").values; ret=np.where(win,(odds-1)*STAKE,-STAKE)
p1=ret[(sub["D"]<PHASE).values]; p2=ret[(sub["D"]>=PHASE).values]
print(f"\nAway strategy phases: P1(2019-22) total=${p1.sum():+,.0f} Sharpe={p1.mean()/p1.std(ddof=1):+.3f} n={len(p1)}")
print(f"                      P2(2022-26) total=${p2.sum():+,.0f} Sharpe={p2.mean()/p2.std(ddof=1):+.3f} n={len(p2)}")
print("\nDONE.")
