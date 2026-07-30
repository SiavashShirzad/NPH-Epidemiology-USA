"""Mixed-effects models of cognitive and gait trajectories (supplementary analysis)."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

map_blsa = {6: "White", 3: "Black or African American", 9: "Asian"}
map_nacc = {1: "White", 2: "Black or African American", 5: "Asian"}
map_biocard = {1: "White", 2: "Black or African American", 5: "Asian"}
map_adni = {'5': "White", '4': "Black or African American", '2': "Asian"}
EVANS_THRESH = 0.30; CA_THRESH = 90; SPPB_ABN = 10

def clean_mmse(s):
    v = pd.to_numeric(s, errors='coerce'); return v.where((v >= 0) & (v <= 30), np.nan)

def std_sex(df):
    for src in ['Sex', 'SEX', 'PTGENDER', 'Gender']:
        if src in df.columns:
            df = df.rename(columns={src: 'sex'}); break
    if 'sex' in df.columns:
        df['sex_code'] = df['sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0, 1: 1, 2: 0, 0: 0, '1': 1, '2': 0, 1.0: 1, 2.0: 0})
    else:
        df['sex_code'] = np.nan
    return df

def std_race(df, m, raw):
    s = df[raw]
    if df[raw].dtype == object:
        s = df[raw].astype(str).str.split('|').str[0].str.strip()
    else:
        s = pd.to_numeric(df[raw], errors='coerce')
    df['race_std'] = s.map(m)
    valid = ['White', 'Black or African American', 'Asian']
    df['race_std'] = df['race_std'].apply(lambda x: x if (pd.isna(x) or x in valid) else 'Other')
    return df

def calc_lincom(model, terms):
    p = model.params; cov = model.cov_params()
    t = [x for x in terms if x in p.index]
    if not t: return np.nan, np.nan, np.nan
    est = p[t].sum(); se = np.sqrt(sum(cov.loc[a, b] for a in t for b in t))
    return est, est - 1.96 * se, est + 1.96 * se

# 2. COHORT PREP -> unified schema (pid, age, ei, ca, sex_code, race_std, mmse_clean, gait_metric, gait_type)
def prep_blsa(df):
    d = std_sex(df.copy())
    d['age'] = (pd.to_datetime(d.get('visitdate'), errors='coerce') - pd.to_datetime(d.get('birthdate'), errors='coerce')).dt.days / 365.25
    d['ei'] = pd.to_numeric(d['EI'], errors='coerce'); d['ca'] = pd.to_numeric(d['CA'], errors='coerce')
    d['mmse_clean'] = clean_mmse(d.get('mmse_total'))
    s = pd.to_numeric(d['sppb'], errors='coerce'); d['gait_metric'] = s.where((s >= 0) & (s <= 12)); d['gait_type'] = 'continuous'
    d = std_race(d, map_blsa, 'race')
    return d.assign(cohort='BLSA', pid=d['pid'])

def prep_biocard(df):
    d = std_sex(df.copy().rename(columns={'JHUANONID': 'pid'}))
    d['age'] = pd.to_numeric(d['VISITYEAR'], errors='coerce') - pd.to_numeric(d['BIRTHYEAR'], errors='coerce')
    d['ei'] = pd.to_numeric(d['EI'], errors='coerce'); d['ca'] = pd.to_numeric(d['CA'], errors='coerce')
    d['mmse_clean'] = clean_mmse(d.get('MMSE'))
    s = pd.to_numeric(d['SPPB'], errors='coerce'); d['gait_metric'] = s.where((s >= 0) & (s <= 12)); d['gait_type'] = 'continuous'
    d = std_race(d, map_biocard, 'RACE')
    return d.assign(cohort='BIOCARD', pid=d['pid'])

def prep_nacc(df):
    d = std_sex(df.copy().rename(columns={'NACCID': 'pid'}))
    d['age'] = pd.to_numeric(d['MRIYR'], errors='coerce') - pd.to_numeric(d['BIRTHYR'], errors='coerce')
    d['ei'] = pd.to_numeric(d['EI'], errors='coerce'); d['ca'] = pd.to_numeric(d['CA'], errors='coerce')
    d['mmse_clean'] = clean_mmse(d.get('NACCMMSE'))
    g = pd.to_numeric(d['GAIT'], errors='coerce')                          # 0 normal, 1-4 abn, 8/neg excluded
    d['gait_metric'] = np.select([g == 0, g.between(1, 4)], [0, 1], default=np.nan); d['gait_type'] = 'binary'
    d = std_race(d, map_nacc, 'RACE')
    return d.assign(cohort='NACC', pid=d['pid'])

def prep_adni(df):
    d = std_sex(df.copy().rename(columns={'Subject': 'pid'}))
    d['age'] = pd.to_numeric(d['Age'], errors='coerce')
    d['ei'] = pd.to_numeric(d['EI'], errors='coerce'); d['ca'] = pd.to_numeric(d['CA'], errors='coerce')
    d['mmse_clean'] = clean_mmse(d.get('MMSCORE'))
    g = pd.to_numeric(d['NXGAIT'], errors='coerce')                        # 1 normal, 2 abnormal
    d['gait_metric'] = np.select([g == 1, g == 2], [0, 1], default=np.nan); d['gait_type'] = 'binary'
    d = std_race(d, map_adni, 'PTRACCAT')
    return d.assign(cohort='ADNI', pid=d['pid'])

# 3. GROUPS BY RADIOLOGY ONLY (baseline = first classifiable scan; no clinical conditioning)
def assign_groups(df):
    df = df[df['age'] >= 60].copy()
    df['rnph'] = (df['ei'] > EVANS_THRESH) & (df['ca'] < CA_THRESH)
    df['ei_normal'] = (df['ei'] < EVANS_THRESH) & df['ei'].notna()
    df['arm_elig'] = df['rnph'] | df['ei_normal']        # RNPH or no-hydrocephalus; middle group (EI>0.3 & CA>=90) excluded

    def grp(g):
        e = g.sort_values('age'); e = e[e['arm_elig']]
        if e.empty: return 'Excluded'
        return 'Radiologic' if bool(e.iloc[0]['rnph']) else 'Normal Aging'
    label = df.groupby('pid', group_keys=False).apply(grp)
    df['Group'] = df['pid'].map(label)
    df = df[df['Group'].isin(['Radiologic', 'Normal Aging'])].drop_duplicates(subset=['pid', 'age'])
    df['age_c'] = df['age'] - 75
    return df

# 4. MODELS (random intercept + random slope; GEE for binary gait)
FORMULA = " ~ age_c * C(Group, Treatment(reference='Normal Aging')) + sex_code + C(race_std, Treatment(reference='White'))"

def extract(model, data, mtype):
    out = {}
    for grp in ['Normal Aging', 'Radiologic']:
        n = data[data['Group'] == grp]['pid'].nunique()
        if n < 3: continue
        if grp == 'Normal Aging':
            bt, st, pI = ['Intercept'], ['age_c'], np.nan
        else:
            gt = f"C(Group, Treatment(reference='Normal Aging'))[T.{grp}]"
            it = f"age_c:{gt}"
            if it not in model.params.index: it = f"{gt}:age_c"
            bt, st, pI = ['Intercept', gt], ['age_c', it], model.pvalues.get(it, np.nan)
        be, blo, bhi = calc_lincom(model, bt); se, slo, shi = calc_lincom(model, st)
        if mtype == 'logistic':
            f = lambda z: np.exp(z) / (1 + np.exp(z))
            be, blo, bhi = f(be), f(blo), f(bhi); se, slo, shi = np.exp(se), np.exp(slo), np.exp(shi)
        out[grp] = {'N': n, 'Base_Val': be, 'Base_CI': (blo, bhi), 'Slope_Val': se, 'Slope_CI': (slo, shi), 'P_Inter': pI}
    return out

def fit_lme(formula, data):
    try:
        return smf.mixedlm(formula, data, groups=data['pid'], re_formula='~age_c').fit(disp=False)
    except Exception:
        return smf.mixedlm(formula, data, groups=data['pid']).fit(disp=False)   # fallback: random intercept

def run(df):
    res = {'mmse': {}, 'gait': {}}
    cog = df.dropna(subset=['mmse_clean', 'age_c', 'Group', 'sex_code', 'race_std'])
    if cog['Group'].nunique() == 2:
        try: res['mmse'] = extract(fit_lme("mmse_clean" + FORMULA, cog), cog, 'linear')
        except Exception as e: print("  MMSE:", e)
    gait = df.dropna(subset=['gait_metric', 'age_c', 'Group', 'sex_code', 'race_std'])
    if gait['Group'].nunique() == 2:
        try:
            if gait['gait_type'].iloc[0] == 'continuous':
                res['gait'] = extract(fit_lme("gait_metric" + FORMULA, gait), gait, 'linear'); res['gait']['metric_type'] = 'SPPB'
            else:
                m = smf.gee("gait_metric" + FORMULA, data=gait, groups=gait['pid'],
                            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                res['gait'] = extract(m, gait, 'logistic'); res['gait']['metric_type'] = 'binary'
        except Exception as e: print("  Gait:", e)
    return res

loaders = [('BLSA', 'BLSA_epidemiology.csv', prep_blsa), ('BIOCARD', 'BIOCARD_epidemiology.csv', prep_biocard),
           ('NACC', 'NACC_epidemiology.csv', prep_nacc), ('ADNI', 'ADNI_epidemiology.csv', prep_adni)]
cog_rows, gait_rows = [], []
for name, path, prep in loaders:
    try:
        raw = pd.read_csv(path, low_memory=False)
        if name == 'BLSA': raw = raw.loc[:, ~raw.columns.duplicated()]
        df = assign_groups(prep(raw))
    except Exception as e:
        print(f"{name}: load/prep failed -> {e}"); continue
    nR = df[df['Group'] == 'Radiologic']['pid'].nunique(); nN = df[df['Group'] == 'Normal Aging']['pid'].nunique()
    print(f"Processing {name}: Radiologic n={nR}, Normal Aging n={nN}")
    res = run(df)
    for grp, s in res.get('mmse', {}).items():
        pv = "<.001" if (not np.isnan(s['P_Inter']) and s['P_Inter'] < 0.001) else (f"{s['P_Inter']:.3f}" if not np.isnan(s['P_Inter']) else "-")
        cog_rows.append({'Cohort': name, 'Group': grp, 'N': s['N'],
                         'Baseline MMSE (age 75)': f"{s['Base_Val']:.1f} [{s['Base_CI'][0]:.1f}, {s['Base_CI'][1]:.1f}]",
                         'Annual change (95% CI)': f"{s['Slope_Val']:.2f} [{s['Slope_CI'][0]:.2f}, {s['Slope_CI'][1]:.2f}]",
                         'P(slope diff vs Normal)': pv})
    g = res.get('gait', {}); mt = g.get('metric_type', '')
    for grp, s in g.items():
        if grp == 'metric_type': continue
        pv = "<.001" if (not np.isnan(s['P_Inter']) and s['P_Inter'] < 0.001) else (f"{s['P_Inter']:.3f}" if not np.isnan(s['P_Inter']) else "-")
        if mt == 'SPPB':
            base = f"{s['Base_Val']:.1f} [{s['Base_CI'][0]:.1f}, {s['Base_CI'][1]:.1f}] pts"; slope = f"{s['Slope_Val']:.2f} [{s['Slope_CI'][0]:.2f}, {s['Slope_CI'][1]:.2f}] pts/yr"; mn = 'SPPB (0-12)'
        else:
            base = f"{s['Base_Val']*100:.1f}% [{s['Base_CI'][0]*100:.1f}-{s['Base_CI'][1]*100:.1f}%]"; slope = f"OR {s['Slope_Val']:.2f} [{s['Slope_CI'][0]:.2f}-{s['Slope_CI'][1]:.2f}]/yr"; mn = 'Gait abnormality'
        gait_rows.append({'Cohort': name, 'Group': grp, 'Metric': mn, 'N': s['N'],
                          'Baseline (age 75)': base, 'Trajectory (95% CI)': slope, 'P(slope diff vs Normal)': pv})


print("TABLE 3: LONGITUDINAL COGNITIVE (MMSE) TRAJECTORIES - RADIOLOGIC NPH vs NORMAL AGING (EI<0.30), per cohort")
print("Mixed-effects LME, random intercept + random slope; adjusted for baseline age, sex, race. Ref profile: 75-yo White female.")

if cog_rows: print(pd.DataFrame(cog_rows).to_string(index=False))
pd.DataFrame(cog_rows).to_csv("RateOfChange_Cognition.csv", index=False)

print("\n\n" + "=" * 120)
print("TABLE 4: LONGITUDINAL GAIT TRAJECTORIES - RADIOLOGIC NPH vs NORMAL AGING (EI<0.30), per cohort")
print("LME (BLSA/BIOCARD, SPPB) and GEE (NACC/ADNI, binary); adjusted for baseline age, sex, race.")

if gait_rows: print(pd.DataFrame(gait_rows).to_string(index=False))
pd.DataFrame(gait_rows).to_csv("RateOfChange_Gait.csv", index=False)
