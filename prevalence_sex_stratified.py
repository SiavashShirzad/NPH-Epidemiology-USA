"""Age- and sex-stratified prevalence of RNPH phenotypes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = False

COHORT_COLORS = {'BLSA': '#648FFF', 'BIOCARD': '#785EF0', 'NACC': '#DC267F', 'ADNI': '#FE6100'}
EVANS_THRESH = 0.30
CA_ACUTE_THRESH = 90
SPPB_ABN_THRESH = 10
AGE_BINS, AGE_LABELS = [60, 70, 80, np.inf], ['60-69', '70-79', '≥80']
OVERALL_LABEL = '≥60 (All)'
AGE_LABELS_PLUS = AGE_LABELS + [OVERALL_LABEL]

LABEL_MAP = ['Not NPH', 'RNPH-All', 'RNPH-A', 'RNPH-G', 'RNPH-C', 'RNPH-GC (Possible NPH)']
COHORT_ORDER = ['BLSA', 'BIOCARD', 'NACC', 'ADNI']

def standardize_sex(val):
    s = str(val).strip().upper()
    if s in ['1', '1.0', 'M', 'MALE']: return 'M'
    if s in ['2', '2.0', 'F', 'FEMALE']: return 'F'
    return np.nan

NACC = pd.read_csv("NACC_epidemiology.csv", low_memory=False).groupby(['NACCID', 'VISITYR'], as_index=False).first()
biocard = pd.read_csv("BIOCARD_epidemiology.csv", low_memory=False).groupby(['JHUANONID', 'VISITYEAR'], as_index=False).first()
ADNI = pd.read_csv("ADNI_epidemiology.csv", low_memory=False)
ADNI = ADNI[ADNI['Sex'] != 'X'].groupby(['Subject', 'Visit'], as_index=False).first()
BLSA = pd.read_csv("BLSA_epidemiology.csv", low_memory=False)
BLSA = BLSA.loc[:, ~BLSA.columns.duplicated()].groupby(['pid', 'visitdate'], as_index=False).first()
RAW = {'BLSA': BLSA, 'BIOCARD': biocard, 'NACC': NACC, 'ADNI': ADNI}

# 2. STANDARDIZE TO A COMMON VISIT-LEVEL TABLE (+ sex)
def standardize(df, cohort):
    d = df.copy(); out = pd.DataFrame()
    if cohort == 'BLSA':
        out['subject'] = d['pid'].values
        out['age'] = ((pd.to_datetime(d['visitdate'], errors='coerce') - pd.to_datetime(d['birthdate'], errors='coerce')).dt.days / 365.25).values
        out['sex'] = d['sex'].apply(standardize_sex).values
        gait = pd.to_numeric(d['sppb'], errors='coerce').where(lambda s: (s >= 0) & (s <= 12))
        cog = pd.to_numeric(d['cogn_status'], errors='coerce').replace({3: 0})
        gait_abn = gait <= SPPB_ABN_THRESH; cog_abn = cog.isin([1, 2])
    elif cohort == 'BIOCARD':
        out['subject'] = d['JHUANONID'].values
        out['age'] = (pd.to_numeric(d['VISITYEAR'], errors='coerce') - pd.to_numeric(d['BIRTHYEAR'], errors='coerce')).values
        out['sex'] = d['SEX'].apply(standardize_sex).values
        gait = pd.to_numeric(d['SPPB'], errors='coerce').where(lambda s: (s >= 0) & (s <= 12))
        cog = d['DIAGNOSIS'].replace({'IMPAIRED NOT MCI': 'NORMAL'}).astype(str).str.upper()
        gait_abn = gait <= SPPB_ABN_THRESH; cog_abn = cog.isin(['MCI', 'DEMENTIA'])
    elif cohort == 'NACC':
        out['subject'] = d['NACCID'].values
        out['age'] = (pd.to_numeric(d['MRIYR'], errors='coerce') - pd.to_numeric(d['BIRTHYR'], errors='coerce')).values
        out['sex'] = d['SEX'].apply(standardize_sex).values
        gait = pd.to_numeric(d['GAIT'], errors='coerce').where(lambda s: (s >= 0) & (s != 8))
        cog = pd.to_numeric(d['CDRGLOB'], errors='coerce').where(lambda s: s != 99)
        gait_abn = gait >= 1; cog_abn = cog >= 0.5
    elif cohort == 'ADNI':
        out['subject'] = d['Subject'].values
        out['age'] = pd.to_numeric(d['Age'], errors='coerce').values
        sx = d['Sex'].apply(standardize_sex)
        if 'PTGENDER' in d.columns:
            sx = sx.fillna(d['PTGENDER'].apply(standardize_sex))
        out['sex'] = sx.values
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

# 4. CO-OCCURRENCE: decade bands + overall >=60 (subject-level), sex per subject
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
    sexmap = (d.dropna(subset=['sex']).groupby('subject')['sex']
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan))
    band = d.groupby(['subject', 'age_band'], observed=True).agg(**AGG_SPEC).reset_index()
    band['age_band'] = band['age_band'].astype(str)
    over = d.groupby('subject').agg(**AGG_SPEC).reset_index()
    over['age_band'] = OVERALL_LABEL
    both = pd.concat([band, over], ignore_index=True)
    both['sex'] = both['subject'].map(sexmap)
    both['cohort'] = cohort
    frames.append(both)

SEL = pd.concat(frames, ignore_index=True)
SEL = SEL[SEL['rad_valid']].copy()
for _col in ['rad_pos', 'gait_avail', 'cog_avail', 'gait_abn', 'cog_abn', 'rad_valid']:
    SEL[_col] = SEL[_col].astype(bool)

print("Sex resolved per subject; subjects with unknown sex are excluded from strata.")
for cohort in COHORT_ORDER:
    c = SEL[(SEL['cohort'] == cohort) & (SEL['age_band'] == OVERALL_LABEL)]
    print("  %-8s (>=60)  M=%d  F=%d  unknown=%d" % (cohort, (c['sex'] == 'M').sum(), (c['sex'] == 'F').sum(), c['sex'].isna().sum()))

# 5. AGE- AND SEX-STRATIFIED PREVALENCE + WILSON 95% CI (incl. >=60 total)
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)

def calculate_stratified_prevalence(sex_code):
    results = []
    for cohort in COHORT_ORDER:
        cdf = SEL[(SEL['cohort'] == cohort) & (SEL['sex'] == sex_code)]
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
                                'eligible_n': denom, 'cases_n': k, 'prev': p, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(results)

M_RES = calculate_stratified_prevalence('M')
F_RES = calculate_stratified_prevalence('F')

# 6. PLOTTING (decade bands + >=60 overall)
def plot_sex_data(df, sex_label):
    fig, axes = plt.subplots(1, 5, figsize=(28, 6.5), sharey=True)
    phenos = LABEL_MAP[1:]; dodge = 0.16; x_base = np.arange(len(AGE_LABELS_PLUS))
    for i, p in enumerate(phenos):
        ax = axes[i]; sub = df[df['type'] == p]
        for c_idx, coh in enumerate(COHORT_ORDER):
            data = sub[sub['cohort'] == coh].set_index('age_group').reindex(AGE_LABELS_PLUS)
            x = x_base + (c_idx - 1.5) * dodge
            y = data['prev'] * 100
            err_low = (data['prev'] - data['ci_low']).clip(0) * 100
            err_high = (data['ci_high'] - data['prev']).clip(0) * 100
            ax.errorbar(x, y, yerr=[err_low, err_high], fmt='o', color=COHORT_COLORS[coh],
                        markersize=13, capsize=5, elinewidth=3.5, markeredgecolor='white',
                        markeredgewidth=1.5, label=coh if i == 0 else "")
        ax.axvline(len(AGE_LABELS) - 0.5, color='#999', ls=':', lw=1.5)
        ax.set_title(f"{p}\n({sex_label})", fontweight='bold', pad=18, fontsize=14)
        ax.set_xticks(x_base); ax.set_xticklabels(AGE_LABELS_PLUS, fontsize=10, rotation=20)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=':', alpha=0.6, color='gray')
        if i == 0:
            ax.set_ylabel('Period Prevalence (%)', fontweight='bold', fontsize=13)
            ax.legend(loc='upper left', frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(f'RNPH_Prevalence_{sex_label}.tiff', dpi=300, format='tiff')
    plt.show()

plot_sex_data(M_RES, "Males")
plot_sex_data(F_RES, "Females")

# 7. FINAL COMBINED TABLE (incl. >=60 total)
M_RES['Sex'], F_RES['Sex'] = 'Male', 'Female'
COMBINED = pd.concat([M_RES, F_RES], ignore_index=True)
COMBINED['Formatted'] = (
    COMBINED['cases_n'].astype(int).astype(str) + " / " + COMBINED['eligible_n'].astype(int).astype(str) + " (" +
    (COMBINED['prev'] * 100).round(1).astype(str) + "% [" + (COMBINED['ci_low'] * 100).round(1).astype(str) + "-" +
    (COMBINED['ci_high'] * 100).round(1).astype(str) + "])").replace(r'^0 / 0.*', 'N/A', regex=True)
COMBINED['age_group'] = pd.Categorical(COMBINED['age_group'], categories=AGE_LABELS_PLUS, ordered=True)
final_table = COMBINED.pivot_table(index=['cohort', 'age_group', 'Sex'], columns='type', values='Formatted', aggfunc='first', observed=True)
final_table = final_table.reindex(COHORT_ORDER, level=0)[LABEL_MAP]
final_table.index.names = ['Cohort', 'Age Decade', 'Sex']; final_table.columns.name = 'Phenotype'

print("TABLE 2. AGE- AND SEX-STRATIFIED PERIOD PREVALENCE (incl. ≥60 total): CASES / TOTAL (PREVALENCE % [95% CI])")

print(final_table)
final_table.to_csv("RNPH_Sex_Stratified_Prevalence_Final.csv")
