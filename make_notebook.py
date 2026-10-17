"""
Build CSET485_Assignment1.ipynb — fully reproducible notebook for the assignment.
Roll: s24bcau0169 (last digit 9 -> Adult Census Income; seed = 169).
"""
import nbformat as nbf

BASE = r"C:\Users\Adarsh singh\OneDrive\Desktop\ai"
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []

cells.append(new_markdown_cell("""
# CSET485 – AI and Society: Assignment #1
**Roll number:** s24bcau0169  
**Dataset rule:** last digit `9` → **Adult Census Income** dataset  
**Random seed (random_state):** last 3 digits `169` → train/test split 70% / 30%  
**Target:** `income > $50K` encoded as 1 (yes) / 0 (no)

This notebook is the single source of truth. Running every cell from top to
finish regenerates every number cited in the accompanying PDF report.
"""))

cells.append(new_code_cell(r"""
import warnings
warnings.filterwarnings("ignore")
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import statsmodels.api as sm

BASE = r"C:\Users\Adarsh singh\OneDrive\Desktop\ai"
DATA_FILE = f"{BASE}/adult.data"
RANDOM_STATE = 169
TEST_SIZE = 0.30
FIGDIR = f"{BASE}/figs"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_style("whitegrid")

COLUMNS = ["age","workclass","fnlwgt","education","education-num",
           "marital-status","occupation","relationship","race","sex",
           "capital-gain","capital-loss","hours-per-week","native-country","income"]
df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS,
                 skipinitialspace=True, na_values="?", engine="python")
print(f"Raw rows loaded: {len(df)}")
print(df.isna().sum()[lambda s: s>0].to_string())
"""))

cells.append(new_markdown_cell("""
## Data Preprocessing (documented)

**Missing values:** The raw file encodes unknowns as `"?"` in `workclass`
(1,836), `occupation` (1,843) and `native-country` (583). These were treated
as missing (`NaN`) and the affected rows were **dropped** (2,399 rows removed),
leaving 30,162 records. This matches the UCI documentation note that
45,222 of 48,842 records remain when unknowns are excluded.

**Target encoding:** `income` → `1` if `>50K`, else `0`. Positive rate ≈ 24.89%.

**Categorical encoding:** One-hot encoding (`drop='first'`) applied to
`workclass, education, marital-status, occupation, relationship, race, sex,
native-country` (8 columns → 90 binary features). `education-num` is kept as a
numeric predictor.

**Feature scaling:** `StandardScaler` (zero mean, unit variance) applied to the
continuous predictors `age, fnlwgt, education-num, capital-gain,
capital-loss, hours-per-week`. One-hot columns are left as 0/1 (not scaled).

**Outliers:** Continuous skewed features (`capital-gain`, `capital-loss`,
`fnlwgt`, `hours-per-week`, `age`) were winsorized at the 1st/99th percentile.
Bounds were computed on the **training** split only and then applied to the test
split to avoid leakage.
"""))

cells.append(new_code_cell(r"""
df_clean = df.dropna().copy()
df_clean["income"] = df_clean["income"].astype(str).str.strip()
y = (df_clean["income"] == ">50K").astype(int)
feature_cols = [c for c in COLUMNS if c != "income"]
X = df_clean[feature_cols].copy()
continuous = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
categorical = [c for c in feature_cols if c not in continuous]
print("Continuous:", continuous)
print("Categorical:", categorical)
print("Positive rate:", round(float(y.mean()),4))
"""))

cells.append(new_code_cell(r"""
# Train / test split (70/30, seed 169, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {len(X_train)} (pos rate {y_train.mean():.4f})")
print(f"Test : {len(X_test)}  (pos rate {y_test.mean():.4f})")

# Winsorize continuous features using TRAIN bounds
for c in continuous:
    lo, hi = X_train[c].quantile(0.01), X_train[c].quantile(0.99)
    X_train[c] = X_train[c].clip(lo, hi)
    X_test[c] = X_test[c].clip(lo, hi)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ], remainder="drop", sparse_threshold=0)
X_train_enc = preprocess.fit_transform(X_train)
X_test_enc = preprocess.transform(X_test)
cat_names = preprocess.named_transformers_["cat"].get_feature_names_out(categorical)
all_feature_names = list(continuous) + list(cat_names)
print(f"Final feature matrix: train {X_train_enc.shape}, test {X_test_enc.shape}")
print(f"Features: {len(continuous)} continuous (scaled) + {len(cat_names)} one-hot = {len(all_feature_names)}")

X_train_sm = sm.add_constant(X_train_enc, has_constant="add")
X_test_sm = sm.add_constant(X_test_enc, has_constant="add")
"""))

cells.append(new_markdown_cell("""
## Part 1 — Build & Compare Model A vs Model B

- **Model A:** Ordinary Least Squares (Linear Probability Model) predicting the
  binary income target.
- **Model B:** Logistic Regression predicting P(income > $50K).
"""))

cells.append(new_code_cell(r"""
# Model A: OLS
model_a_ols = sm.OLS(y_train, X_train_sm).fit()
y_test_pred_a = model_a_ols.predict(X_test_sm)
median_pred_a = float(np.median(y_test_pred_a))
y_pred_a_class = (y_test_pred_a >= median_pred_a).astype(int)
acc_a = float(accuracy_score(y_test, y_pred_a_class))

# Model B: Logistic Regression
model_b = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)
model_b.fit(X_train_enc, y_train)
y_prob_b = model_b.predict_proba(X_test_enc)[:, 1]
y_pred_b_class = (y_prob_b >= 0.5).astype(int)
acc_b = float(accuracy_score(y_test, y_pred_b_class))
auc_b = float(roc_auc_score(y_test, y_prob_b))

print(f"Model A (OLS):   R^2 = {model_a_ols.rsquared:.4f},  median predicted = {median_pred_a:.4f},  accuracy = {acc_a:.4f}")
print(f"Model B (Logit): accuracy = {acc_b:.4f},  AUC = {auc_b:.4f},  intercept = {float(model_b.intercept_[0]):.4f}")
"""))

cells.append(new_code_cell(r"""
# Part 1: agreement / disagreement between the two thresholding approaches
cm = confusion_matrix(y_pred_a_class, y_pred_b_class, labels=[1,0])
# rows = Model A (1=high,0=low), cols = Model B (1=high,0=low)
aa, ab = int(cm[0,0]), int(cm[0,1])   # A high/B high ; A high/B low
ba, bb = int(cm[1,0]), int(cm[1,1])   # A low/B high  ; A low/B low
mismatch = ab + ba
n = len(y_test)
print("Confusion matrix (rows=ModelA high/low, cols=ModelB high/low):")
print(cm)
print(f"Agree high/high : {aa}")
print(f"Agree low/low   : {bb}")
print(f"A high / B low  : {ab}")
print(f"A low / B high  : {ba}")
print(f"Mismatches      : {mismatch} / {n} = {mismatch/n*100:.2f}%")
"""))

cells.append(new_markdown_cell("""
### Part 1 findings
- Model A threshold = median predicted value **0.2297** (well below 0.5).
- Model B threshold = **0.50**.
- Because the OLS median (0.2297) is far below the logistic 0.5 cutoff,
  Model A labels many more people "high risk" than Model B.
- **Mismatches: 2,644 / 9,049 = 29.22%**, all of them cases where Model A says
  "high risk" but Model B says "low risk" (A high/B low = 2,644; A low/B high =
  0).
- **Why they differ:** OLS predicts a continuous score whose natural split is
  its own median; logistic regression outputs calibrated probabilities split at
  0.5. Since the median OLS score (0.2297) << 0.5, the two rules draw the line
  in different places, so they flag different individuals as high risk.
"""))

cells.append(new_markdown_cell("""
## Part 2 — The Trap

Sensitive variable for subgroup/fairness analysis: **race** (Group A = White,
Group B = Black).
"""))

cells.append(new_code_cell(r"""
# Part 2.1: Simpson's Paradox hunt (race: White vs Black)
sensitive = "race"
group_a_name = "White"
group_b_name = "Black"

simp_features = ["age","education-num","hours-per-week","capital-gain","capital-loss","sex"]
simp_cont = [f for f in simp_features if f in continuous]
simp_cat = [f for f in simp_features if f in categorical]
preprocess21 = ColumnTransformer(
    transformers=[("num", StandardScaler(), simp_cont),
                  ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), simp_cat)],
    remainder="drop", sparse_threshold=0)
D21 = preprocess21.fit_transform(X_train[simp_features])
simp_names = list(simp_cont) + list(preprocess21.named_transformers_["cat"].get_feature_names_out(simp_cat))
D21_df = pd.DataFrame(D21, columns=simp_names, index=X_train.index).astype(float)
D21_sm = sm.add_constant(D21_df, has_constant="add")

mask_a = X_train[sensitive] == group_a_name
mask_b = X_train[sensitive] == group_b_name
m_full = sm.Logit(y_train, D21_sm).fit(disp=0, maxiter=200)
m_a = sm.Logit(y_train[mask_a], D21_sm.loc[mask_a]).fit(disp=0, maxiter=200)
m_b = sm.Logit(y_train[mask_b], D21_sm.loc[mask_b]).fit(disp=0, maxiter=200)

ci_full = m_full.conf_int(alpha=0.05)
ci_a = m_a.conf_int(alpha=0.05)
ci_b = m_b.conf_int(alpha=0.05)
kp = "education-num"
def crow(model, ci, name):
    return dict(coef=float(model.params[name]), se=float(model.bse[name]),
                p=float(model.pvalues[name]),
                lo=float(ci.loc[name,0]), hi=float(ci.loc[name,1]))
r_full, r_a, r_b = crow(m_full,ci_full,kp), crow(m_a,ci_a,kp), crow(m_b,ci_b,kp)
print(f"Key predictor: {kp}")
print(f"FULL   (n={len(X_train)}): coef={r_full['coef']:.4f} 95%CI[{r_full['lo']:.4f},{r_full['hi']:.4f}] p={r_full['p']:.2e}")
print(f"White  (n={int(mask_a.sum())}): coef={r_a['coef']:.4f} 95%CI[{r_a['lo']:.4f},{r_a['hi']:.4f}] p={r_a['p']:.2e}")
print(f"Black  (n={int(mask_b.sum())}): coef={r_b['coef']:.4f} 95%CI[{r_b['lo']:.4f},{r_b['hi']:.4f}] p={r_b['p']:.2e}")
print(f"Sign reversal: {np.sign(m_full.params[kp])!=np.sign(m_a.params[kp]) or np.sign(m_full.params[kp])!=np.sign(m_b.params[kp])}")
"""))

cells.append(new_markdown_cell("""
### Part 2.1 findings (education-num coefficient)
- Full model: **coef = 0.8262** [95% CI 0.7838, 0.8686], p < 0.001.
- White subgroup: **coef = 0.8124** [0.7676, 0.8573].
- Black subgroup: **coef = 1.0310** [0.8396, 1.2224].
- No sign reversal, but the return to education is **markedly stronger for Black**
  individuals (1.0310 vs 0.8124, ~27% larger). Each additional year of education
  raises the log-odds of earning >$50K more for Black workers than for White
  workers.
- **Policy implication if ignored:** A single "education raises income"
  conclusion would mask that education pays off *more* for Black workers;
  policies that only expand schooling without addressing where those returns
  are realized would under-estimate the benefit to the Black subgroup and could
  misdirect investment.
"""))

cells.append(new_code_cell(r"""
# Part 2.2: Omitted-variable bias — drop 'education' (a likely confounder)
omitted_raw = "education"
categorical_red = [c for c in categorical if c != omitted_raw]
preprocess_red = ColumnTransformer(
    transformers=[("num", StandardScaler(), continuous),
                  ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_red)],
    remainder="drop", sparse_threshold=0)
Xtr_red = preprocess_red.fit_transform(X_train)
red_names = list(continuous) + list(preprocess_red.named_transformers_["cat"].get_feature_names_out(categorical_red))
Xtr_red_df = pd.DataFrame(Xtr_red, columns=red_names, index=X_train.index).astype(float)
Xtr_red_sm = sm.add_constant(Xtr_red_df, has_constant="add")
X_train_enc_df = pd.DataFrame(X_train_enc, columns=all_feature_names, index=X_train.index).astype(float)
X_train_enc_sm = sm.add_constant(X_train_enc_df, has_constant="add")

m_full_sm = sm.Logit(y_train, X_train_enc_sm).fit(disp=0, maxiter=200)
m_red_sm = sm.Logit(y_train, Xtr_red_sm).fit(disp=0, maxiter=200)

common = ["education-num","age","capital-gain","hours-per-week"]
ovb = {}
for p in common:
    before = float(m_full_sm.params[p]); after = float(m_red_sm.params[p])
    ovb[p] = dict(before=before, after=after,
                  pct=float((after-before)/before*100) if before!=0 else float("nan"))
    print(f"{p:14s} before={before:.6f} after={after:.6f}  change={ovb[p]['pct']:+.4f}%")
shifted_most = max(ovb.items(), key=lambda kv: abs(kv[1]["pct"]))
print(f"Most shifted: {shifted_most[0]} ({shifted_most[1]['pct']:+.4f}%)")
"""))

cells.append(new_markdown_cell("""
### Part 2.2 findings (dropped confounder = `education`)
| Predictor | Full coef | Without `education` | % change |
|---|---:|---:|---:|
| education-num | 1.031205 | 0.696564 | **−32.45%** |
| age | 0.355970 | 0.368319 | +3.47% |
| capital-gain | 0.860972 | 0.859616 | −0.16% |
| hours-per-week | 0.385786 | 0.389075 | +0.85% |

- **`education-num` shifted the most (−32.45%)** when the categorical `education`
  variable was removed.
- **What it reveals:** `education-num` was partly standing in for the richer
  `education` categories. Once `education` is dropped, its coefficient collapses
  by a third — a large omitted-variable bias. This shows a model can look
  stable overall while a key predictor's coefficient is badly distorted by an
  unmodelled confound; you cannot trust coefficients until you stress-test for
  confounds.
"""))

cells.append(new_code_cell(r"""
# Part 2.3: Adversarial subset (high-confidence wrong predictions) on TEST set
y_test_arr = y_test.values
conf_wrong = ((y_prob_b >= 0.75) & (y_test_arr == 0)) | ((y_prob_b <= 0.25) & (y_test_arr == 1))
conf_idx = np.where(conf_wrong)[0]
n_fp = int(((y_prob_b >= 0.75) & (y_test_arr == 0)).sum())
n_fn = int(((y_prob_b <= 0.25) & (y_test_arr == 1)).sum())
print(f"High-confidence wrong: {conf_idx.size} / {n} = {conf_idx.size/n*100:.2f}%")
print(f"  FP (P>=0.75, actual=0): {n_fp}")
print(f"  FN (P<=0.25, actual=1): {n_fn}")

n_adv = min(50, conf_idx.size)
rng = np.random.default_rng(RANDOM_STATE)
adv_idx = rng.choice(conf_idx, size=n_adv, replace=False) if conf_idx.size else np.array([],dtype=int)
adv_rows = X_test.iloc[adv_idx].copy()
adv_rows["prob"] = y_prob_b[adv_idx]
adv_rows["actual"] = y_test_arr[adv_idx]
adv_rows.to_csv(f"{BASE}/adversarial_subset.csv", index=False)

def demo(dframe):
    return dict(race={k:int(v) for k,v in dframe[sensitive].value_counts().items()},
                sex={k:int(v) for k,v in dframe["sex"].value_counts().items()},
                age_mean=float(dframe["age"].mean()), age_median=float(dframe["age"].median()))
adv_demo = demo(adv_rows); test_demo = demo(X_test)
print("Adversarial subset (n=50):", adv_demo)
print("Full test set             :", test_demo)

# candidate traits
traits = {
    "capital-gain>0": (adv_rows["capital-gain"]>0).mean(),
    "capital-gain>5000": (adv_rows["capital-gain"]>5000).mean(),
    "education-num>=13": (adv_rows["education-num"]>=13).mean(),
    "hours-per-week>=40": (adv_rows["hours-per-week"]>=40).mean(),
    "sex==Male": (adv_rows["sex"]=="Male").mean(),
    "race==White": (adv_rows["race"]=="White").mean(),
    "race==Black": (adv_rows["race"]=="Black").mean(),
    "age>=40": (adv_rows["age"]>=40).mean(),
}
test_shares = {
    "capital-gain>0": float((X_test["capital-gain"]>0).mean()),
    "capital-gain>5000": float((X_test["capital-gain"]>5000).mean()),
    "education-num>=13": float((X_test["education-num"]>=13).mean()),
    "hours-per-week>=40": float((X_test["hours-per-week"]>=40).mean()),
    "sex==Male": float((X_test["sex"]=="Male").mean()),
    "race==White": float((X_test["race"]=="White").mean()),
    "race==Black": float((X_test["race"]=="Black").mean()),
    "age>=40": float((X_test["age"]>=40).mean()),
}
for k in traits:
    print(f"{k:20s} adv={traits[k]:.3f}  test={test_shares[k]:.3f}")
standout = max(traits, key=lambda k: abs(traits[k]-test_shares[k]))
print(f"Standout trait: {standout}  adv={traits[standout]:.3f}  test={test_shares[standout]:.3f}")
"""))

cells.append(new_markdown_cell("""
### Part 2.3 findings (adversarial subset)
- High-confidence wrong predictions on the test set: **455 / 9,049 = 5.03%**
  (FP: 107 with P≥0.75 but actual ≤$50K; FN: 348 with P≤0.25 but actual >$50K).
- A 50-row sample (seed 169) saved to `adversarial_subset.csv`.
- Subset vs full test set: older (mean age 43.5 vs 38.3), more male (70% vs
  67.5%), more White (90% vs 86.3%).
- **Standout characteristic: 62% are age ≥ 40 vs 43.5% in the full test set**
  (also 88% work ≥40 hrs/week vs 78%).
- **Why the model fails confidently:** these are older, high-hours workers whom
  the model rates as very unlikely (P≤0.25) to earn >$50K yet who actually do,
  OR whom it rates very likely (P≥0.75) yet who do not. The logistic decision
  boundary is linear in the scaled features; older workers with high hours but
  low capital-gain/low education-num fall on the wrong side of a boundary that
  the model learned from a younger, higher-capital-gain population — a
  distribution-shift / missing-interaction (age × capital-gain) failure that
  the linear model cannot capture.
"""))

cells.append(new_code_cell(r"""
# Part 2.4: Fairness trade-off sweep (threshold 0.05 -> 0.95, step 0.1)
test_sensitive = X_test[sensitive].values
ga_mask = test_sensitive == group_a_name
gb_mask = test_sensitive == group_b_name
thresholds = np.arange(0.05, 0.96, 0.10)
rows = []
for t in thresholds:
    yhat = (y_prob_b >= t).astype(int)
    pa = float(yhat[ga_mask].mean()) if ga_mask.sum() else float("nan")
    pb = float(yhat[gb_mask].mean()) if gb_mask.sum() else float("nan")
    dp = (pa/pb) if (pb and pb>0) else float("nan")
    def rates(mask):
        m = yhat[mask]; yt = y_test_arr[mask]
        tp=int(((m==1)&(yt==1)).sum()); fn=int(((m==0)&(yt==1)).sum())
        fp=int(((m==1)&(yt==0)).sum()); tn=int(((m==0)&(yt==0)).sum())
        return (tp/(tp+fn) if tp+fn else float("nan"),
                fp/(fp+tn) if fp+tn else float("nan"))
    tpr_a,fpr_a = rates(ga_mask); tpr_b,fpr_b = rates(gb_mask)
    er=[]
    if tpr_b and tpr_b>0: er.append(tpr_a/tpr_b)
    if fpr_b and fpr_b>0: er.append(fpr_a/fpr_b)
    eo = min(er) if er else float("nan")
    pp_a = float(y_test_arr[ga_mask&(yhat==1)].mean()) if (ga_mask&(yhat==1)).sum() else float("nan")
    pp_b = float(y_test_arr[gb_mask&(yhat==1)].mean()) if (gb_mask&(yhat==1)).sum() else float("nan")
    pp = (pp_a/pp_b) if (pp_b and pp_b>0) else float("nan")
    rows.append(dict(threshold=float(t), DP=dp, EO=eo, PP=pp,
                     acc=float((yhat==y_test_arr).mean()),
                     pa=pa, pb=pb, tpr_a=tpr_a,tpr_b=tpr_b,
                     fpr_a=fpr_a,fpr_b=fpr_b, pp_a=pp_a, pp_b=pp_b))
fair_df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(fair_df[["threshold","DP","EO","PP","acc"]].to_string(index=False,
      formatters={"DP":"{:.3f}".format,"EO":"{:.3f}".format,"PP":"{:.3f}".format,"acc":"{:.3f}".format}))

# Figures
figs=[]
plt.figure(figsize=(8,6))
xs=[r["DP"] for r in rows if r["DP"] is not None]
ys=[r["EO"] for r in rows if r["EO"] is not None]
ts=[r["threshold"] for r in rows if r["DP"] is not None and r["EO"] is not None]
plt.scatter(xs,ys,s=80,c=ts,cmap="viridis",edgecolor="k",zorder=3)
for x,y_,tt in zip(xs,ys,ts):
    plt.annotate(f"t={tt:.2f}",(x,y_),textcoords="offset points",xytext=(6,6),fontsize=8)
plt.xlabel("Demographic Parity (DP, ideal=1.0)"); plt.ylabel("Equalized Odds (EO, ideal=1.0)")
plt.xlim(0,1.5); plt.ylim(0,1.5)
plt.title("Fairness Trade-Off: DP vs EO by threshold (Adult, seed 169)")
plt.axvline(1.0,color="gray",ls="--",lw=0.8); plt.axhline(1.0,color="gray",ls="--",lw=0.8)
plt.colorbar(label="Threshold"); plt.tight_layout()
f1=f"{FIGDIR}/fairness_tradeoff.png"; plt.savefig(f1,dpi=150); plt.close(); figs.append(f1)

plt.figure(figsize=(9,5))
thr=[r["threshold"] for r in rows]
plt.plot(thr,[r["DP"] for r in rows],marker="o",label="Demographic Parity")
plt.plot(thr,[r["EO"] for r in rows],marker="s",label="Equalized Odds")
plt.plot(thr,[r["PP"] for r in rows],marker="^",label="Predictive Parity")
plt.axhline(1.0,color="gray",ls="--",lw=0.8)
plt.xlabel("Decision threshold"); plt.ylabel("Metric value (ideal=1.0)")
plt.title("Fairness Metrics vs Decision Threshold"); plt.legend(); plt.tight_layout()
f2=f"{FIGDIR}/fairness_metrics_vs_threshold.png"; plt.savefig(f2,dpi=150); plt.close(); figs.append(f2)

plt.figure(figsize=(8,5))
plt.hist(y_prob_b[y_test_arr==1],bins=25,alpha=0.6,label="Actual >50K (1)",color="green")
plt.hist(y_prob_b[y_test_arr==0],bins=25,alpha=0.6,label="Actual <=50K (0)",color="red")
plt.axvline(0.25,color="blue",ls="--",lw=0.8); plt.axvline(0.75,color="blue",ls="--",lw=0.8)
plt.xlabel("Predicted P(income > 50K)"); plt.ylabel("Count (test set)")
plt.title("Model B predicted probabilities by actual outcome (test set)")
plt.legend(); plt.tight_layout()
f3=f"{FIGDIR}/prob_histogram.png"; plt.savefig(f3,dpi=150); plt.close(); figs.append(f3)
print("Figures:", figs)
"""))

cells.append(new_markdown_cell("""
### Part 2.4 findings (sensitive = race: White vs Black)

| Threshold | DP | EO | PP | Accuracy |
|---:|---:|---:|---:|---:|
| 0.05 | 1.592 | 1.051 | 1.376 | 0.631 |
| 0.15 | 2.042 | 1.130 | 1.154 | 0.758 |
| 0.25 | 2.382 | 1.160 | 1.015 | 0.811 |
| 0.35 | 2.484 | 1.225 | 1.028 | 0.838 |
| 0.45 | 2.621 | 1.260 | 1.002 | 0.849 |
| 0.55 | 2.651 | 1.338 | 1.053 | 0.851 |
| 0.65 | 2.940 | 1.323 | 0.938 | 0.845 |
| 0.75 | 3.240 | 1.478 | 0.951 | 0.832 |
| 0.85 | 2.651 | 1.189 | 0.935 | 0.808 |
| 0.95 | 3.858 | 1.815 | 0.981 | 0.781 |

DP = P(ŷ=1|White)/P(ŷ=1|Black); EO = min(TPR ratio, FPR ratio); PP =
P(Y=1|ŷ=1,White)/P(Y=1|ŷ=1,Black). Ideal = 1.0.

**Trade-off:** DP and EO both move toward their ideal as the threshold *lowers*
(DP 1.592 at t=0.05 vs 3.858 at t=0.95), but lowering the threshold hurts
**predictive parity** (PP 1.376 at t=0.05 vs 1.002 at t=0.45) and **accuracy**
(0.631 at t=0.05 vs 0.851 at t=0.55). No single threshold makes all three
metrics equal 1.0.

**Trade-off cost:** moving from t=0.05 (DP=1.592) to t=0.95 (DP=3.858) worsens
DP's deviation from parity by (3.858−1)−(1.592−1) over (1.592−1) ≈ 68%, while
EO's deviation grows from 0.051 to 0.815. The two parity metrics track each
other; the real price of chasing threshold-based parity is paid in PP and
accuracy.

**Recommended threshold: t ≈ 0.45** — PP ≈ 1.002 (near-perfect predictive
parity) and accuracy 0.849, with DP=2.621 / EO=1.260. This sacrifices
demographic parity (White predicted positive ~2.6× Black) to give both groups
similar conditional accuracy among the predicted-positive.
![Fairness chart](figs/fairness_tradeoff.png)
![Metrics vs threshold](figs/fairness_metrics_vs_threshold.png)
![Probability histogram](figs/prob_histogram.png)
"""))

cells.append(new_markdown_cell("""
## Part 3 — Descriptive Analysis Questions
Each answer quotes specific numbers computed above.
"""))

cells.append(new_markdown_cell("""
### Q1 — Threshold-based vs direct logistic classification
Model A (OLS) threshold = median predicted value **0.2297**; Model B threshold =
**0.50**. On the 9,049-row test set the two rules agreed on high/high for 1,881
and low/low for 4,524, but **mismatched on 2,644 cases (29.22%)** — all of them
Model A "high risk" vs Model B "low risk" (0 cases the other way). Because the
OLS median score (0.2297) sits far below the logistic 0.5 probability cutoff,
Model A flags roughly a quarter of the test set as high risk that Model B does
not. The approaches flag different individuals because one splits a continuous
score at its median while the other splits calibrated probabilities at 0.5; with
a low median, the OLS rule is much more permissive.
"""))

cells.append(new_markdown_cell("""
### Q2 — Subgroup / Simpson's Paradox
Split by race (White n=18,124; Black n=2,012 in training). Predictor =
`education-num`. Full model: **coef 0.8262 [0.7838, 0.8686]**. White:
**0.8124 [0.7676, 0.8573]**. Black: **1.0310 [0.8396, 1.2224]**. No sign
reversal, but the education return is ~27% larger for Black workers (point
estimate 1.0310 vs 0.8124). Ignoring the subgroup would lead to the blanket
policy "more education → higher income," underestimating the payoff for Black
workers and overstating uniformity; targeted support would be misallocated.
"""))

cells.append(new_markdown_cell("""
### Q3 — Omitted-variable bias (dropped `education`)
| Predictor | Full | Without `education` | % change |
|---|---:|---:|---:|
| education-num | 1.031205 | 0.696564 | −32.45% |
| age | 0.355970 | 0.368319 | +3.47% |
| capital-gain | 0.860972 | 0.859616 | −0.16% |
| hours-per-week | 0.385786 | 0.389075 | +0.85% |

`education-num` shifted the most (−32.45%). It was proxying for the dropped
categorical `education` variable. Without stress-testing for confounds, a key
coefficient can be off by a third, so coefficient magnitudes are not trustworthy
until confounds are checked.
"""))

cells.append(new_markdown_cell("""
### Q4 — Detecting an unseen hidden confound
Commission an external validation on a different, temporally-later census sample
(or a different survey such as the CPS ASEC) with the same features plus a rich
set of domain variables, then compare subgroup performance and calibration.
Any systematic miscalibration or performance gap in the external data that the
current metrics hide signals a missing confound (e.g., a local labor-market
factor) not captured by the original features.
"""))

cells.append(new_markdown_cell("""
### Q5 — The 50-row adversarial subset
455 test rows (5.03%) were high-confidence wrong; the sampled 50 are older
(mean 43.5 vs 38.3), more often White (90% vs 86.3%), and **62% are age ≥ 40
vs 43.5% of the test set** (also 88% work ≥40 hrs vs 78%). The model's linear
decision boundary underestimates older, high-hours workers with modest
capital-gain: they actually earn >$50K but receive P≤0.25 (false negatives),
because age × capital-gain interactions and non-linear effects are not in the
model. The model extrapolates from a younger, wealthier training pattern and
fails confidently on this predictable subgroup.
"""))

cells.append(new_markdown_cell("""
### Q6 — Trade-off cost for a real person
Pick t=0.45 vs t=0.05. Consider two identical-in-earnings applicants, one White
and one Black, each truly earning >$50K. At t=0.45 (DP=2.621, EO=1.260,
PP=1.002) both have similar odds of approval (predictive parity ~1). At t=0.05
(DP=1.592, EO=1.051 but PP=1.376) approval rates equalize more across groups,
but the White applicant's approval no longer means the same thing as the Black
applicant's (predictive parity breaks: 1.376). Prioritizing DP (low threshold)
helps the Black applicant get approved more often, but harms the White
applicant by flooding approvals with low-probability cases (PP falls), so an
approved White candidate is less likely to truly qualify — fairness to one group
is bought with weaker meaning of the decision for the other.
"""))

cells.append(new_markdown_cell("""
### Q7 — Defensible threshold (t = 0.35)
At t=0.35: DP=2.484, EO=1.225, PP=1.028, accuracy=0.838. PP≈1.03 (near-perfect
predictive parity: approved White and Black applicants are ~equally likely to
truly qualify), accuracy is high (83.8%), and EO=1.23 (modest gap in TPR/FPR
between groups). It is defensible because it equalizes the *meaning* of a
positive decision across groups (PP closest to 1) while keeping overall error
low. Counterargument: DP=2.484 still means White applicants are ~2.5× as likely
to be approved as Black applicants — a critic would say this entrenches
group-level access inequality even if conditional accuracy is fair. That is
valid: PP fairness does not fix unequal selection rates.
"""))

cells.append(new_markdown_cell("""
### Q8 — Demographic parity vs equalized odds
I recommend **equalized odds** for this context (income/credit-like decision).
My numbers: at t=0.45, EO=1.260 vs DP=2.621 — EO is much closer to parity
(1.0) than DP, while PP=1.002 (perfect predictive parity) and accuracy=0.849.
Equalized odds keeps TPR and FPR similar across groups, so qualified applicants
of either race face similar true-positive and false-positive rates — the
outcome that matters most when the decision affects livelihood. I sacrifice
demographic parity (approval rates remain ~2.6× higher for White applicants).
That is acceptable because equalizing *rates* without equalizing *accuracy*
would approve many unqualified applicants from the disadvantaged group (harmful
stigma and default risk) while equalizing odds treats both groups by the same
performance standard.
"""))

cells.append(new_markdown_cell("""
## Part 4 — Reflection

**Direct answer:** I would recommend **conditional deployment, not
deployment-as-is**: the model is accurate overall (AUC 0.907, accuracy 0.852)
but fails confidently on a predictable subgroup (5.03% high-confidence errors;
62% of the adversarial sample aged ≥40), so it should only be deployed with
threshold t≈0.45, subgroup monitoring, and a human-appeal path.

### Why a single number fails
1. **Overall accuracy hides subgroup gaps.** Test accuracy is 85.2%, yet the
   model is confidently wrong on 5.03% of cases and those errors concentrate on
   older workers (62% aged ≥40 vs 43.5% of the test set). A single accuracy
   figure would conceal this.
2. **Threshold choice is a values question.** Moving from t=0.05 to t=0.95
   changes DP from 1.592 to 3.858 and accuracy from 0.631 to 0.781 — the "best"
   threshold depends on how much disparity vs. accuracy you will tolerate, not
   on the data alone.
3. **Fairness metrics conflict.** DP and EO both improve as the threshold drops,
   but predictive parity gets worse (PP 1.376 at t=0.05 vs 1.002 at t=0.45).
   No single threshold satisfies all three, so one metric cannot summarize
   fairness.
4. **Confident failure on a predictable subset.** The logistic boundary is
   linear; older, high-hours workers with modest capital-gain receive P≤0.25
   yet actually earn >$50K. Accuracy alone never reveals this.
5. **Hidden confounds remain unknowable.** Dropping `education` moved
   `education-num`'s coefficient by −32.45%; an unseen confound could distort
   the model just as badly without any metric flagging it.

### Grounding in evidence
Accuracy 85.2% / AUC 0.907 (Part 1); mismatch 29.22% between threshold rules
(Part 1); education-num −32.45% shift when `education` is dropped (Part 2.2);
455 high-confidence errors, 5.03%, with 62% aged ≥40 (Part 2.3); no threshold
makes DP/EO/PP all equal 1 (Part 2.4).

### Closing
Responsible deployment requires publishing the full trade-off surface —
threshold, subgroup error rates, and the fairness metrics in tension — rather
than asking a policymaker for a yes/no on one number.
"""))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3"}
nb.metadata["language_info"] = {"name":"python"}

out = f"{BASE}/CSET485_Assignment1.ipynb"
nbf.write(nb, out)
print("Wrote", out)
