"""Figure 1: stacked-bar prevalence of RNPH phenotypes by age band and cohort."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = False

COHORT_COLORS = {'BLSA': '#4E79A7', 'BIOCARD': '#59A14F', 'NACC': '#E15759', 'ADNI': '#B07AA1'}
EVANS_THRESH = 0.30
CA_ACUTE_THRESH = 90
SPPB_ABN_THRESH = 10
AGE_BINS = [60, 70, 80, np.inf]
AGE_LABELS = ['60-69', '70-79', '≥80']
OVERALL_LABEL = '≥60 (All)'
AGE_LABELS_PLUS = AGE_LABELS + [OVERALL_LABEL]   # decade bands + total 60-and-over column

LABEL_MAP = ['Not NPH', 'RNPH-All', 'RNPH-A', 'RNPH-G', 'RNPH-C', 'RNPH-GC (Possible NPH)']
COHORT_ORDER = ['BLSA', 'BIOCARD', 'NACC', 'ADNI']

NACC = pd.read_csv("NACC_epidemiology.csv", low_memory=False).groupby(['NACCID', 'VISITYR'], as_index=False).first()
biocard = pd.read_csv("BIOCARD_epidemiology.csv", low_memory=False).groupby(['JHUANONID', 'VISITYEAR'], as_index=False).first()
ADNI = pd.read_csv("ADNI_epidemiology.csv", low_memory=False)
ADNI = ADNI[ADNI['Sex'] != 'X'].groupby(['Subject', 'Visit'], as_index=False).first()
BLSA = pd.read_csv("BLSA_epidemiology.csv", low_memory=False)
BLSA = BLSA.loc[:, ~BLSA.columns.duplicated()].groupby(['pid', 'visitdate'], as_index=False).first()
RAW = {'BLSA': BLSA, 'BIOCARD': biocard, 'NACC': NACC, 'ADNI': ADNI}

def standardize(df, cohort):
    d = df.copy(); out = pd.DataFrame()
    if cohort == 'BLSA':
        out['subject'] = d['pid'].values
        out['age'] = ((pd.to_datetime(d['visitdate'], errors='coerce') - pd.to_datetime(d['birthdate'], errors='coerce')).dt.days / 365.25).values
        gait = pd.to_numeric(d['sppb'], errors='coerce').where(lambda s: (s >= 0) & (s <= 12))
        cog = pd.to_numeric(d['cogn_status'], errors='coerce').replace({3: 0})
        gait_abn = gait <= SPPB_ABN_THRESH; cog_abn = cog.isin([1, 2])
    elif cohort == 'BIOCARD':
        out['subject'] = d['JHUANONID'].values
        out['age'] = (pd.to_numeric(d['VISITYEAR'], errors='coerce') - pd.to_numeric(d['BIRTHYEAR'], errors='coerce')).values
        gait = pd.to_numeric(d['SPPB'], errors='coerce').where(lambda s: (s >= 0) & (s <= 12))
        cog = d['DIAGNOSIS'].replace({'IMPAIRED NOT MCI': 'NORMAL'}).astype(str).str.upper()
        gait_abn = gait <= SPPB_ABN_THRESH; cog_abn = cog.isin(['MCI', 'DEMENTIA'])
    elif cohort == 'NACC':
        out['subject'] = d['NACCID'].values
        out['age'] = (pd.to_numeric(d['MRIYR'], errors='coerce') - pd.to_numeric(d['BIRTHYR'], errors='coerce')).values
        gait = pd.to_numeric(d['GAIT'], errors='coerce').where(lambda s: (s >= 0) & (s != 8))
        cog = pd.to_numeric(d['CDRGLOB'], errors='coerce').where(lambda s: s != 99)
        gait_abn = gait >= 1; cog_abn = cog >= 0.5
    elif cohort == 'ADNI':
        out['subject'] = d['Subject'].values
        out['age'] = pd.to_numeric(d['Age'], errors='coerce').values
        gait = pd.to_numeric(d['NXGAIT'], errors='coerce'); cog = pd.to_numeric(d['CDGLOBAL'], errors='coerce')
        gait_abn = gait == 2; cog_abn = cog >= 0.5
    out['ei'] = pd.to_numeric(d['EI'], errors='coerce').values
    out['ca'] = pd.to_numeric(d['CA'], errors='coerce').values
    out['gait_avail'] = gait.notna().values; out['cog_avail'] = cog.notna().values
    out['gait_abn'] = (gait_abn.fillna(False) & gait.notna()).values
    out['cog_abn'] = (cog_abn.fillna(False) & cog.notna()).values
    out['cohort'] = cohort
    return out

# 3. RADIOLOGIC CLASSIFICATION (per visit)
def classify_radiologic(d):
    ei, ca = d['ei'], d['ca']
    status = np.where(ei.isna(), 'no_ei',
             np.where(ei <= EVANS_THRESH, 'negative',
             np.where(ca.isna(), 'indeterminate',
             np.where(ca < CA_ACUTE_THRESH, 'positive', 'negative'))))
    d = d.copy()
    d['classifiable'] = pd.Series(status, index=d.index).isin(['positive', 'negative'])
    d['rad_pos'] = pd.Series(status, index=d.index) == 'positive'
    return d

#    Decade level (one record per subject per band) AND overall >=60
#    (one record per subject across all >=60 visits). A feature is flagged
#    if it occurs at ANY qualifying visit within the window.
AGG_SPEC = dict(rad_valid=('classifiable', 'any'), rad_pos=('rad_pos', 'any'),
                gait_avail=('gait_avail', 'any'), gait_abn=('gait_abn', 'any'),
                cog_avail=('cog_avail', 'any'), cog_abn=('cog_abn', 'any'))

frames = []
for cohort, df in RAW.items():
    d = standardize(df, cohort)
    d = d[d['age'] >= 60].copy()
    d = classify_radiologic(d)
    d['age_band'] = pd.cut(d['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    d = d.dropna(subset=['age_band'])
    # decade bands
    band = d.groupby(['subject', 'age_band'], observed=True).agg(**AGG_SPEC).reset_index()
    band['age_band'] = band['age_band'].astype(str)
    # overall >=60 (subject-level, across all bands)
    over = d.groupby('subject').agg(**AGG_SPEC).reset_index()
    over['age_band'] = OVERALL_LABEL
    both = pd.concat([band, over], ignore_index=True)
    both['cohort'] = cohort
    frames.append(both)

SEL = pd.concat(frames, ignore_index=True)
SEL = SEL[SEL['rad_valid']].copy()
for _col in ['rad_pos', 'gait_avail', 'cog_avail', 'gait_abn', 'cog_abn', 'rad_valid']:
    SEL[_col] = SEL[_col].astype(bool)

# 5. PHENOTYPE PREVALENCE + WILSON 95% CI
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)

results = []
for cohort in COHORT_ORDER:
    cdf = SEL[SEL['cohort'] == cohort]
    for ag in AGE_LABELS_PLUS:
        b = cdf[cdf['age_band'] == ag]
        rad_denom = len(b)
        comp = b[b['gait_avail'] & b['cog_avail']]
        clin_denom = len(comp)
        counts = {
            'Not NPH': int(rad_denom - int(b['rad_pos'].sum())),
            'RNPH-All': int(b['rad_pos'].sum()),
            'RNPH-A': int((comp['rad_pos'] & ~comp['gait_abn'] & ~comp['cog_abn']).sum()),
            'RNPH-G': int((comp['rad_pos'] & comp['gait_abn'] & ~comp['cog_abn']).sum()),
            'RNPH-C': int((comp['rad_pos'] & ~comp['gait_abn'] & comp['cog_abn']).sum()),
            'RNPH-GC (Possible NPH)': int((comp['rad_pos'] & comp['gait_abn'] & comp['cog_abn']).sum()),
        }
        for label in LABEL_MAP:
            denom = rad_denom if label in ['Not NPH', 'RNPH-All'] else clin_denom
            k = counts[label]; p, lo, hi = wilson_ci(k, denom)
            results.append({'cohort': cohort, 'type': label, 'age_group': ag,
                            'eligible_n': denom, 'cases_n': k, 'prevalence': p, 'ci_low': lo, 'ci_high': hi})

ALL = pd.DataFrame(results)
ALL['error_low'] = (ALL['prevalence'] - ALL['ci_low']).clip(lower=0)
ALL['error_high'] = (ALL['ci_high'] - ALL['prevalence']).clip(lower=0)

# 6. POINT-ESTIMATE PLOT (decade bands + >=60 overall)
def plot_point_estimates(df):
    fig, axes = plt.subplots(1, 5, figsize=(28, 7), sharey=True)
    pheno_order = LABEL_MAP[1:]
    titles = ['A. RNPH-All', 'B. RNPH-A', 'C. RNPH-G', 'D. RNPH-C', 'E. RNPH-GC\n(Possible NPH)']
    x_pos = np.arange(len(AGE_LABELS_PLUS)); dodge = 0.15
    for idx, ax in enumerate(axes):
        sub_df = df[df['type'] == pheno_order[idx]]
        for c_idx, cohort in enumerate(COHORT_ORDER):
            cd = sub_df[sub_df['cohort'] == cohort].set_index('age_group').reindex(AGE_LABELS_PLUS)
            off = (c_idx - 1.5) * dodge
            ax.errorbar(x_pos + off, cd['prevalence'] * 100,
                        yerr=[cd['error_low'].fillna(0) * 100, cd['error_high'].fillna(0) * 100],
                        fmt='o', label=cohort if idx == 0 else "", color=COHORT_COLORS[cohort],
                        linewidth=4, markersize=13, capsize=4, capthick=4, elinewidth=4,
                        markeredgecolor='white', markeredgewidth=1)
        ax.axvline(len(AGE_LABELS) - 0.5, color='#999', ls=':', lw=1.5)   # separate bands from overall
        ax.set_title(titles[idx], fontweight='bold', pad=15, fontsize=14)
        ax.set_xticks(x_pos); ax.set_xticklabels(AGE_LABELS_PLUS, fontsize=10, rotation=20)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray'); ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel('Period Prevalence (%)', fontweight='bold', fontsize=13)
            ax.legend(loc='upper left', frameon=False, fontsize=11)
        else:
            ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig('RNPH_Prevalence_PointEstimates.tiff', dpi=300, format='tiff')
    plt.show()

plot_point_estimates(ALL)

# 7. PREVALENCE TABLE  (decade bands + >=60 total column)
ALL['Formatted'] = ((ALL['prevalence'] * 100).round(1).astype(str) + '% [' +
                    (ALL['ci_low'] * 100).round(1).astype(str) + '-' +
                    (ALL['ci_high'] * 100).round(1).astype(str) + ']').replace('nan% [nan-nan]', 'N/A')
pivot_table = ALL.pivot_table(index=['cohort', 'type'], columns='age_group', values='Formatted', aggfunc='first')[AGE_LABELS_PLUS]
pivot_table = pivot_table.reindex([(c, t) for c in COHORT_ORDER for t in LABEL_MAP])
pivot_table.index.names = ['Cohort', 'Phenotype']; pivot_table.columns.name = 'Age Range'

print("TABLE 2. PERIOD PREVALENCE (%) [95% CI] BY AGE BAND AND OVERALL (≥60)")

print(pivot_table)
pivot_table.to_csv("RNPH_Prevalence_Final_Analysis_Aggregated.csv")

# 8. COUNTS TABLE  (cases / eligible, incl. >=60 total)
ALL['Counts'] = ALL['cases_n'].astype(int).astype(str) + ' / ' + ALL['eligible_n'].astype(int).astype(str)
counts_table = ALL.pivot_table(index=['cohort', 'type'], columns='age_group', values='Counts', aggfunc='first')[AGE_LABELS_PLUS]
counts_table = counts_table.reindex([(c, t) for c in COHORT_ORDER for t in LABEL_MAP])
counts_table.index.names = ['Cohort', 'Phenotype']; counts_table.columns.name = 'Age Range'

print("TABLE 3. CASES / ELIGIBLE  (numerator / denominator), incl. ≥60 total")
print("Not NPH & RNPH-All: denominator = classifiable Evans index in window; RNPH-A/G/C/GC: gait AND cognition measured in window")

print(counts_table)
counts_table.to_csv("RNPH_Prevalence_Counts.csv")
