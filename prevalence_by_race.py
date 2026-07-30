"""Race-stratified prevalence of RNPH with marginal standardization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11; plt.rcParams['figure.dpi'] = 300; plt.rcParams['axes.grid'] = False

RACE_COLORS = {"White": "#D3D3D3", "Black or African American": "#377EB8"}
INCLUDED_RACES = ["White", "Black or African American"]
INCLUDED_COHORTS = ["BLSA", "NACC", "ADNI"]

EVANS_THRESH = 0.30
CALLOSAL_ANGLE_THRESH = 90
SPPB_THRESH = 10
GAIT_ABN_THRESH_NACC = 1
ADNI_GAIT_ABN = 2
MIN_CASES_LOGIT = 10        # events-per-arm floor before a multivariable model is trusted

map_blsa = {6: "White", 3: "Black or African American"}
map_nacc = {1: "White", 2: "Black or African American"}
map_adni = {'5': "White", '4': "Black or African American"}

# 1. PREPROCESSING  (radiologic classifiability fixed; clinical sentinels excluded)
def standardize_cols(df):
    for source in ['Sex', 'SEX', 'PTGENDER', 'Gender', 'sex_code']:
        if source in df.columns:
            df = df.rename(columns={source: 'sex'}); break
    if 'sex' in df.columns:
        df['sex_code'] = df['sex'].replace({'Male': 1, 'Female': 0, 'M': 1, 'F': 0,
                                            1: 1, 2: 0, 0: 0, '1': 1, '2': 0, 1.0: 1, 2.0: 0, 0.0: 0})
    else:
        df['sex_code'] = np.nan
    found_icv = next((c for c in ['ICV', 'NACCICV', 'icv', 'intracranial_vol'] if c in df.columns), None)
    df['icv_scaled'] = (pd.to_numeric(df[found_icv], errors='coerce') / 100.0) if found_icv else np.nan
    found_educ = next((c for c in ['maxeducation', 'EDUC', 'PTEDUCAT', 'educ'] if c in df.columns), None)
    df['educ_val'] = pd.to_numeric(df[found_educ], errors='coerce') if found_educ else np.nan
    return df

def apply_row_flags(df, cohort, ei_col, ca_col, gait_col, cog_col):
    df = df.copy()
    for c in [ei_col, ca_col, gait_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    ei, ca = df[ei_col], df[ca_col]
    # --- RADIOLOGIC: classifiable = EI present AND (EI<=0.30 OR CA present); CA is ~only measured if EI>0.30 ---
    df['rad_v'] = ei.notna() & ((ei <= EVANS_THRESH) | ca.notna())
    df['rad_a'] = ((ei > EVANS_THRESH) & (ca < CALLOSAL_ANGLE_THRESH)).fillna(False)

    raw = pd.to_numeric(df[cog_col], errors='coerce') if cog_col in df.columns else pd.Series(np.nan, index=df.index)
    g = df[gait_col] if gait_col in df.columns else pd.Series(np.nan, index=df.index)
    df['g_v'] = False; df['g_a'] = False; df['c_v'] = False; df['c_a'] = False

    if cohort == 'BLSA':
        gv = g.between(0, 12); df['g_v'] = gv; df.loc[gv, 'g_a'] = g[gv] <= SPPB_THRESH
        cog = raw.replace({3: 0})                                   # 3 -> normal (validated recode)
        cv = cog.notna(); df['c_v'] = cv; df.loc[cv, 'c_a'] = cog[cv].isin([1, 2])
    elif cohort == 'BIOCARD':
        gv = g.between(0, 12); df['g_v'] = gv; df.loc[gv, 'g_a'] = g[gv] <= SPPB_THRESH
        s = df[cog_col].astype(str).str.upper(); cv = df[cog_col].notna()
        df['c_v'] = cv; df.loc[cv, 'c_a'] = s[cv].isin(['MCI', 'DEMENTIA'])
    elif cohort == 'NACC':
        gv = g.isin([0, 1, 2, 3, 4]); df['g_v'] = gv                # exclude 8 (unknown) and negatives
        df.loc[gv, 'g_a'] = g[gv] >= GAIT_ABN_THRESH_NACC
        cv = raw.notna() & (raw != 99); df['c_v'] = cv              # exclude CDRGLOB==99 sentinel
        df.loc[cv, 'c_a'] = raw[cv] >= 0.5
    elif cohort == 'ADNI':
        gv = g.isin([1, 2]); df['g_v'] = gv; df.loc[gv, 'g_a'] = g[gv] == ADNI_GAIT_ABN
        cv = raw.notna() & (raw != 99); df['c_v'] = cv
        df.loc[cv, 'c_a'] = raw[cv] >= 0.5
    return df

def preprocess_blsa(df):
    df = standardize_cols(df.copy())
    df['age'] = (pd.to_datetime(df.get('visitdate'), errors='coerce') -
                 pd.to_datetime(df.get('birthdate'), errors='coerce')).dt.days / 365.25
    df['race_std'] = pd.to_numeric(df['race'], errors='coerce').map(map_blsa) if 'race' in df.columns else np.nan
    return apply_row_flags(df, 'BLSA', 'EI', 'CA', 'sppb', 'cogn_status').assign(cohort='BLSA', pid=df['pid'])

def preprocess_nacc(df):
    df = standardize_cols(df.copy())
    df['age'] = pd.to_numeric(df['MRIYR'], errors='coerce') - pd.to_numeric(df['BIRTHYR'], errors='coerce')
    df['race_std'] = pd.to_numeric(df['RACE'], errors='coerce').map(map_nacc) if 'RACE' in df.columns else np.nan
    return apply_row_flags(df, 'NACC', 'EI', 'CA', 'GAIT', 'CDRGLOB').assign(cohort='NACC', pid=df['NACCID'])

def preprocess_adni(df):
    df = standardize_cols(df.copy())
    df['age'] = pd.to_numeric(df['Age'], errors='coerce')
    if 'PTRACCAT' in df.columns:
        df['race_std'] = df['PTRACCAT'].astype(str).str.split('|').str[0].str.strip().map(map_adni)
    else:
        df['race_std'] = np.nan
    return apply_row_flags(df, 'ADNI', 'EI', 'CA', 'NXGAIT', 'CDGLOBAL').assign(cohort='ADNI', pid=df['Subject'])

# 2. PATIENT-LEVEL AGGREGATION  (broad "ever" phenotype, + n_visits for differential follow-up)
def create_patient_level_data(df):
    if df.empty:
        return pd.DataFrame()
    df = df[df['age'] >= 60].copy()
    bools = df.groupby('pid').agg(rad_v=('rad_v', 'any'), rad_a=('rad_a', 'any'),
                                  g_v=('g_v', 'any'), g_a=('g_a', 'any'),
                                  c_v=('c_v', 'any'), c_a=('c_a', 'any'))
    covs = df.sort_values('age').groupby('pid').agg(
        age=('age', 'first'),                       # baseline age (not mean)
        sex_code=('sex_code', 'first'), race_std=('race_std', 'first'),
        educ_val=('educ_val', 'first'), icv_scaled=('icv_scaled', 'first'), cohort=('cohort', 'first'))
    nvis = df.groupby('pid').size().rename('n_visits')
    nrad = df.groupby('pid')['rad_v'].sum().rename('n_rad_visits')
    pt = bools.join(covs).join(nvis).join(nrad).reset_index()
    pt['n_visits_z'] = (pt['n_visits'] - pt['n_visits'].mean()) / pt['n_visits'].std(ddof=0)

    pt['RNPH-All'] = pt['rad_a'];                       pt['denom_all'] = pt['rad_v']
    pt['RNPH-Gait'] = pt['rad_a'] & pt['g_a'];          pt['denom_gait'] = pt['rad_v'] & pt['g_v']
    pt['RNPH-Cog'] = pt['rad_a'] & pt['c_a'];           pt['denom_cog'] = pt['rad_v'] & pt['c_v']
    pt['RNPH-GC'] = pt['rad_a'] & pt['g_a'] & pt['c_a']; pt['denom_gc'] = pt['rad_v'] & pt['g_v'] & pt['c_v']
    return pt[pt['race_std'].isin(INCLUDED_RACES)]

#    - covariates kept only if <30% missing in the modelling set (avoids silent complete-case wipeout)
#    - g-computation (marginal standardization) for adjusted prevalence
#    - VALID bootstrap: rows resampled and the model REFIT each iteration
#    - sparse cells (min arm cases < 10) -> Fisher's exact, unadjusted
def evaluate_cohort(pt_df, phenotype_col, denom_col, n_boot=300, seed=0):
    valid = pt_df[pt_df[denom_col] == True].dropna(subset=['age', 'sex_code', 'race_std', phenotype_col]).copy()
    if valid.empty:
        return None
    valid[phenotype_col] = valid[phenotype_col].astype(int)
    counts = {r: {'N': int((valid['race_std'] == r).sum()),
                  'Cases': int(valid.loc[valid['race_std'] == r, phenotype_col].sum())} for r in INCLUDED_RACES}
    min_cases = min(counts[r]['Cases'] for r in INCLUDED_RACES)

    def fisher_block():
        wc, wn = counts['White']['Cases'], counts['White']['N'] - counts['White']['Cases']
        bc, bn = counts['Black or African American']['Cases'], counts['Black or African American']['N'] - counts['Black or African American']['Cases']
        _, p = stats.fisher_exact([[wc, wn], [bc, bn]])
        out = []
        for r in INCLUDED_RACES:
            n, k = counts[r]['N'], counts[r]['Cases']; prev = k / n if n else 0
            lo, hi = stats.binom.interval(0.95, n, prev) if n else (0, 0)
            out.append({'Race': r, 'Crude_Prev': prev, 'Adj_Prev': prev,
                        'CI_Low': lo / n if n else 0, 'CI_High': hi / n if n else 0, 'N': n, 'Cases': k})
        return out, p, "Fisher's exact (unadjusted)"

    if min_cases < MIN_CASES_LOGIT:
        return fisher_block()

    # candidate covariates kept only when sufficiently observed in this modelling set
    base = ['age', 'sex_code']
    optional = [c for c in ['educ_val', 'icv_scaled'] if valid[c].notna().mean() >= 0.70]
    cov = base + optional + ['n_visits_z']
    reg = valid.dropna(subset=cov).copy()
    # recount events on the actual modelling sample so crude % and adjusted % refer to the same people
    counts = {r: {'N': int((reg['race_std'] == r).sum()),
                  'Cases': int(reg.loc[reg['race_std'] == r, phenotype_col].sum())} for r in INCLUDED_RACES}
    if min(counts[r]['Cases'] for r in INCLUDED_RACES) < MIN_CASES_LOGIT:
        return fisher_block()

    rhs = "C(race_std, Treatment(reference='White')) + " + " + ".join(cov)
    formula = f"Q('{phenotype_col}') ~ {rhs}"
    try:
        model = smf.logit(formula, data=reg).fit(disp=0)
    except Exception:
        return fisher_block()
    p = model.pvalues.get("C(race_std, Treatment(reference='White'))[T.Black or African American]", np.nan)
    cov_lbl = "LogReg (Age, Sex" + (", Educ" if 'educ_val' in cov else "") + (", ICV" if 'icv_scaled' in cov else "") + ", #Visits)"

    rng = np.random.default_rng(seed)
    boot = {r: [] for r in INCLUDED_RACES}
    for _ in range(n_boot):
        bs = reg.sample(len(reg), replace=True, random_state=int(rng.integers(1e9)))
        try:
            bm = smf.logit(formula, data=bs).fit(disp=0)
        except Exception:
            continue
        for r in INCLUDED_RACES:
            sim = reg.copy(); sim['race_std'] = r
            boot[r].append(bm.predict(sim).mean())

    out = []
    for r in INCLUDED_RACES:
        n, k = counts[r]['N'], counts[r]['Cases']
        sim = reg.copy(); sim['race_std'] = r
        adj = model.predict(sim).mean()
        lo, hi = (np.percentile(boot[r], [2.5, 97.5]) if len(boot[r]) > 20 else (np.nan, np.nan))
        out.append({'Race': r, 'Crude_Prev': k / n if n else 0, 'Adj_Prev': adj,
                    'CI_Low': lo, 'CI_High': hi, 'N': n, 'Cases': k})
    return out, p, cov_lbl

try:
    ALL_PT = pd.concat([create_patient_level_data(preprocess_blsa(BLSA)),
                        create_patient_level_data(preprocess_nacc(NACC)),
                        create_patient_level_data(preprocess_adni(ADNI))], ignore_index=True)
except NameError:
    BLSA = pd.read_csv("BLSA_epidemiology.csv", low_memory=False).loc[:, lambda d: ~d.columns.duplicated()]
    NACC = pd.read_csv("NACC_epidemiology.csv", low_memory=False)
    ADNI = pd.read_csv("ADNI_epidemiology.csv", low_memory=False)
    ALL_PT = pd.concat([create_patient_level_data(preprocess_blsa(BLSA)),
                        create_patient_level_data(preprocess_nacc(NACC)),
                        create_patient_level_data(preprocess_adni(ADNI))], ignore_index=True)

scenarios = [('RNPH-All', 'RNPH-All', 'denom_all'), ('RNPH-Gait', 'RNPH-Gait', 'denom_gait'),
             ('RNPH-Cog', 'RNPH-Cog', 'denom_cog'), ('RNPH-GC (Possible NPH)', 'RNPH-GC', 'denom_gc')]

table_rows, plot_data = [], []
for cohort in INCLUDED_COHORTS:
    c_df = ALL_PT[ALL_PT['cohort'] == cohort]
    if c_df.empty:
        continue
    for pheno_name, col, denom in scenarios:
        res = evaluate_cohort(c_df, col, denom)
        if not res:
            continue
        r_list, p, method = res
        for r in r_list:
            r.update({'Cohort': cohort, 'Phenotype': pheno_name, 'P_Value': p, 'Method': method})
            plot_data.append(r)
        w = next(x for x in r_list if x['Race'] == 'White'); b = next(x for x in r_list if x['Race'] != 'White')
        p_str = "<.001" if p < 0.001 else f"{p:.3f}"
        adj = lambda d: (f"{d['Adj_Prev']*100:.1f} ({d['CI_Low']*100:.1f}-{d['CI_High']*100:.1f})"
                         if pd.notna(d['CI_Low']) else f"{d['Adj_Prev']*100:.1f}")
        table_rows.append({'Cohort': cohort, 'Phenotype': pheno_name,
                           'White n/N (Crude %)': f"{w['Cases']}/{w['N']} ({w['Crude_Prev']*100:.1f}%)",
                           'White Adj % (95% CI)': adj(w),
                           'Black n/N (Crude %)': f"{b['Cases']}/{b['N']} ({b['Crude_Prev']*100:.1f}%)",
                           'Black Adj % (95% CI)': adj(b), 'P-Value': p_str, 'Test Used': method})

TABLE_DF = pd.DataFrame(table_rows); PLOT_DF = pd.DataFrame(plot_data)


print("LIFETIME (EVER-ABNORMAL) PREVALENCE OF RADIOLOGIC NPH PHENOTYPES BY COHORT AND RACE")
print("Denominator = radiologically classifiable (EI present AND (EI<=0.30 OR CA present)). Crude % and marginal-standardized Adj % (95% CI, refit bootstrap).")

print(TABLE_DF)
TABLE_DF.to_csv("NPH_Race_Adjusted_Prevalence_corrected.csv", index=False)

if not PLOT_DF.empty:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    phenos = [s[0] for s in scenarios]; x = np.arange(len(INCLUDED_COHORTS)); width = 0.25
    for i, ax in enumerate(axes):
        sub = PLOT_DF[PLOT_DF['Phenotype'] == phenos[i]]
        p_txt = []
        for c in INCLUDED_COHORTS:
            row = sub[sub['Cohort'] == c]
            if row.empty:
                p_txt.append(f"{c}: -"); continue
            pv = row.iloc[0]['P_Value']
            s = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
            if "Fisher" in row.iloc[0]['Method']:
                s += "\u2020"
            p_txt.append(f"{c}: {s}")
        ax.set_title(f"{phenos[i]}\n({', '.join(p_txt)})", fontweight='bold', fontsize=12, pad=15)
        for j, race in enumerate(INCLUDED_RACES):
            rd = sub[sub['Race'] == race].set_index('Cohort').reindex(INCLUDED_COHORTS)
            adjv = rd['Adj_Prev'].fillna(0); crude = rd['Crude_Prev'].fillna(0)
            yerr = [(rd['Adj_Prev'] - rd['CI_Low']).fillna(0).clip(lower=0),
                    (rd['CI_High'] - rd['Adj_Prev']).fillna(0).clip(lower=0)]
            off = (j - 0.5) * width * 2.2
            ax.bar(x + off, adjv * 100, width=width * 2, color=RACE_COLORS[race],
                   yerr=[y * 100 for y in yerr], capsize=3, edgecolor='white', alpha=0.9,
                   label=f"{race} (Adj)" if i == 0 else "")
            ax.scatter(x + off, crude * 100, color='white', edgecolor='black', s=50, zorder=10,
                       label="Crude" if (i == 0 and j == 0) else "")
        ax.set_xticks(x); ax.set_xticklabels(INCLUDED_COHORTS, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.set_ylabel("Lifetime Prevalence (%)", fontweight='bold', fontsize=13)
            ax.legend(frameon=False, fontsize=9, loc='upper left')
            ax.text(0.02, 0.80, "Bar = Adjusted\nDot = Crude\n\u2020 = Fisher's exact",
                    transform=ax.transAxes, fontsize=9, va='top', fontstyle='italic')
    plt.tight_layout()
    plt.savefig('NPH_Race_Stratified_Adjusted_corrected.tiff', dpi=300, bbox_inches='tight')
    plt.show()
