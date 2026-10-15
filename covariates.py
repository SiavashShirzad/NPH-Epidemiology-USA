import re
from data import *


def strip_num(s):
    return num(s.astype(str).str.replace('<', '', regex=False).str.replace('>', '', regex=False))


nacc_amyloid = CLIN[CLIN['cohort'] == 'NACC'][['pid', 'age', 'amyl_pet', 'amyl_csf']]
nacc_amyloid = pd.concat([nacc_amyloid[['pid', 'age']].assign(amyloid=nacc_amyloid['amyl_pet'], source='PET'),
                          nacc_amyloid[['pid', 'age']].assign(amyloid=nacc_amyloid['amyl_csf'], source='CSF')]).dropna(subset=['amyloid'])

adni_pet = read('adni_amyloid_pet')
adni_pet = adni_pet[(num(adni_pet['qc_flag']) >= 1) & before_cutoff(adni_pet['SCANDATE'])]
adni_pet = pd.DataFrame({'pid': adni_pet['PTID'], 'age': decimal_year(adni_pet['SCANDATE']) - adni_pet['PTID'].map(adni_birth),
                         'amyloid': num(adni_pet['AMYLOID_STATUS'])})

adni_csf = read('adni_csf')
adni_csf = adni_csf[before_cutoff(adni_csf['EXAMDATE'])]
ratio = strip_num(adni_csf['PTAU']) / strip_num(adni_csf['ABETA42'])
adni_csf = pd.DataFrame({'pid': adni_csf['PTID'], 'age': decimal_year(adni_csf['EXAMDATE']) - adni_csf['PTID'].map(adni_birth),
                         'amyloid': (ratio > ELECSYS_RATIO_CUT).astype(float).where(ratio.notna())})

biocard_csf = read('biocard_csf')
ab_ratio = num(biocard_csf['AB42']) / num(biocard_csf['AB40'])
biocard_csf = pd.DataFrame({'pid': biocard_csf['JHUANONID'].astype(str),
                            'age': pd.to_datetime(biocard_csf['CSF_DATE'], errors='coerce').dt.year - num(biocard_csf['BIRTHYEAR']),
                            'amyloid': np.where(ab_ratio <= LUMIPULSE_RATIO_POSITIVE, 1.0, np.where(ab_ratio >= LUMIPULSE_RATIO_NEGATIVE, 0.0, np.nan))})

pib_raw = read('blsa_amyloid_pet')
blsa_birthdate = blsa_raw.groupby('pid')['birthdate'].first()
blsa_birthdate.index = blsa_birthdate.index.astype(str)
blsa_pib = pd.DataFrame({'pid': pib_raw['idno'].astype(str),
                         'age': decimal_year(pib_raw['obsdate_pibpet']) - decimal_year(pib_raw['idno'].astype(str).map(blsa_birthdate)),
                         'amyloid': pib_raw['pib_group'].map({'PiB+': 1.0, 'PiB-': 0.0})}).dropna()

biocard_pib = read('biocard_amyloid_pet')
biocard_pib = pd.DataFrame({'pid': biocard_pib['JHUANONID'].astype(str), 'age': num(biocard_pib['AGE_at_PETscandate']),
                            'amyloid': biocard_pib['Amyloid_status'].map({'Positive': 1.0, 'Negative': 0.0})}).dropna()

AMYLOID = pd.concat([nacc_amyloid.assign(cohort='NACC'), adni_pet.assign(cohort='ADNI', source='PET'), adni_csf.assign(cohort='ADNI', source='CSF'),
                     blsa_pib.assign(cohort='BLSA', source='PET'), biocard_pib.assign(cohort='BIOCARD', source='PET'),
                     biocard_csf.assign(cohort='BIOCARD', source='CSF')], ignore_index=True)[['cohort', 'pid', 'age', 'amyloid', 'source']].dropna()
AMYLOID_PET_BY_PERSON = {k: g for k, g in AMYLOID[AMYLOID['source'] == 'PET'].groupby(['cohort', 'pid'])}
AMYLOID_CSF_BY_PERSON = {k: g for k, g in AMYLOID[AMYLOID['source'] == 'CSF'].groupby(['cohort', 'pid'])}
BLSA_AMYLOID_ONSET = pib_raw.groupby('idno')['eaoa'].first().dropna()
BLSA_AMYLOID_ONSET.index = BLSA_AMYLOID_ONSET.index.astype(str)
BLSA_NEVER_POSITIVE = set(blsa_pib.groupby('pid')['amyloid'].max().loc[lambda s: s == 0].index)
BLSA_LAST_NEGATIVE_AGE = blsa_pib[blsa_pib['amyloid'] == 0].groupby('pid')['age'].max()


def status_at_baseline(by_person, column, cohort, pid, age0, window):
    x = by_person.get((cohort, pid))
    if x is None:
        return np.nan
    positive_before = x[(x[column] == 1) & (x['age'] <= age0 + window)]
    negative_after = x[(x[column] == 0) & (x['age'] >= age0 - window)]
    if len(positive_before) and not len(negative_after):
        return 1.0
    if len(negative_after) and not len(positive_before):
        return 0.0
    if len(positive_before) and len(negative_after):
        return x.iloc[(x['age'] - age0).abs().argsort().values].iloc[0][column]
    return np.nan


def amyloid_at_baseline(cohort, pid, age0, window=AMYLOID_WINDOW, pet_only=False):
    if cohort == 'BLSA':
        if pid in BLSA_AMYLOID_ONSET.index:
            return float(age0 >= BLSA_AMYLOID_ONSET[pid])
        if pid in BLSA_NEVER_POSITIVE:
            return 0.0 if age0 <= BLSA_LAST_NEGATIVE_AGE[pid] + window else np.nan
    status = status_at_baseline(AMYLOID_PET_BY_PERSON, 'amyloid', cohort, pid, age0, window)
    if pd.isna(status) and not pet_only:
        status = status_at_baseline(AMYLOID_CSF_BY_PERSON, 'amyloid', cohort, pid, age0, window)
    return status


adni_tau = read('adni_tau_pet')
adni_tau = adni_tau[(adni_tau['TRACER'] == 'FTP') & (num(adni_tau['qc_flag']) >= 1) & before_cutoff(adni_tau['SCANDATE'])]
adni_tau = pd.DataFrame({'pid': adni_tau['PTID'], 'age': decimal_year(adni_tau['SCANDATE']) - adni_tau['PTID'].map(adni_birth),
                         'suvr': num(adni_tau['META_TEMPORAL_SUVR'])}).dropna()
adni_tau['tau'] = (adni_tau['suvr'] > FTP_META_TEMPORAL_CUT).astype(float)

biocard_tau = read('biocard_tau_pet')
biocard_tau = pd.DataFrame({'pid': biocard_tau['JHUANONID'].astype(str), 'age': num(biocard_tau['AGE_at_PETscandate']),
                            'suvr': num(biocard_tau[['BRAAK_III_tau_SUVR', 'BRAAK_IV_tau_SUVR']].mean(axis=1))}).dropna()
bio_cog = CLIN[(CLIN['cohort'] == 'BIOCARD') & CLIN['cog_abn'].notna()]
nearest_cog = biocard_tau.merge(bio_cog[['pid', 'age', 'cog_abn']], on='pid', suffixes=('', '_cog'))
nearest_cog = nearest_cog[(nearest_cog['age'] - nearest_cog['age_cog']).abs() <= 1].assign(gap=lambda d: (d['age'] - d['age_cog']).abs())
nearest_cog = nearest_cog.sort_values('gap').drop_duplicates(['pid', 'age']).set_index(['pid', 'age'])['cog_abn']
biocard_tau['unimpaired'] = [nearest_cog.get((p, a), np.nan) == 0 for p, a in zip(biocard_tau['pid'], biocard_tau['age'])]
biocard_tau['amyloid'] = [amyloid_at_baseline('BIOCARD', p, a) for p, a in zip(biocard_tau['pid'], biocard_tau['age'])]
tau_reference = biocard_tau[biocard_tau['unimpaired'] & (biocard_tau['amyloid'] == 0)].drop_duplicates('pid')['suvr']
BIOCARD_TAU_CUT = tau_reference.mean() + TAU_REFERENCE_SD * tau_reference.std()
biocard_tau['tau'] = (biocard_tau['suvr'] > BIOCARD_TAU_CUT).astype(float)
TAU = pd.concat([adni_tau.assign(cohort='ADNI'), biocard_tau.assign(cohort='BIOCARD')], ignore_index=True)[['cohort', 'pid', 'age', 'suvr', 'tau']]
TAU_BY_PERSON = {k: g for k, g in TAU.groupby(['cohort', 'pid'])}


def tau_at_baseline(cohort, pid, age0, window=TAU_WINDOW):
    return status_at_baseline(TAU_BY_PERSON, 'tau', cohort, pid, age0, window)


nacc_park = CLIN[(CLIN['cohort'] == 'NACC') & CLIN['etpr'].isin([2, 3, 4, 5])]
adni_park = adni_long('DXPARK', frozen=False)
blsa_park_ids = SUBJ[(SUBJ['cohort'] == 'BLSA') & num(SUBJ['last_cogn_status_type1']).isin([8, 14])]['pid']
blsa_park_visits = CLIN[(CLIN['cohort'] == 'BLSA') & CLIN['pid'].isin(blsa_park_ids)]
blsa_park_onset = (blsa_park_visits[blsa_park_visits['cog_score'] == 2].groupby('pid')['age'].min()
                   .combine_first(blsa_park_visits[blsa_park_visits['cog_score'] >= 1].groupby('pid')['age'].min())
                   .combine_first(blsa_park_visits.groupby('pid')['age'].max()))
PARKINSONISM = pd.concat([nacc_park[['pid', 'age']].assign(cohort='NACC'), adni_park[adni_park['DXPARK'] == 1][['pid', 'age']].assign(cohort='ADNI'),
                          pd.DataFrame({'pid': blsa_park_onset.index, 'age': blsa_park_onset.values, 'cohort': 'BLSA'})], ignore_index=True)

bs = blsa_raw.groupby('pid').agg(stroke=('stroke_hx_ever', 'max'), tia=('tia_hx_ever', 'max')).apply(num)
av = tab.groupby('subject_id').agg(stroke=('HMSTROKE', 'max')).apply(num)
flag = lambda s: (s > 0).astype(float).where(s.notna())
VASCULAR = pd.concat([pd.DataFrame({'cohort': 'BLSA', 'pid': bs.index.astype(str),
                                    'stroke': flag(bs['stroke'].fillna(0) + bs['tia'].fillna(0)).where(bs['stroke'].notna()).values}),
                      pd.DataFrame({'cohort': 'ADNI', 'pid': av.index, 'stroke': flag(av['stroke']).values})], ignore_index=True)


def add_baseline_covariates(d):
    d = d.copy()
    d['amyloid'] = [amyloid_at_baseline(c, p, a) for c, p, a in zip(d['cohort'], d['pid'], d['age0'])]
    d['amyloid_pet'] = [amyloid_at_baseline(c, p, a, pet_only=True) for c, p, a in zip(d['cohort'], d['pid'], d['age0'])]
    d['tau'] = [tau_at_baseline(c, p, a) for c, p, a in zip(d['cohort'], d['pid'], d['age0'])]
    pk = PARKINSONISM.merge(d[['cohort', 'pid', 'age0']], on=['cohort', 'pid'])
    pk = pk[pk['age'] <= pk['age0'] + 0.5][['cohort', 'pid']].drop_duplicates().assign(parkinsonism=True)
    d = d.merge(pk, on=['cohort', 'pid'], how='left').merge(VASCULAR, on=['cohort', 'pid'], how='left')
    d['parkinsonism'] = d['parkinsonism'].fillna(False).astype(bool)
    return d


VOLUMES = read('brain_volumes')
VOLUMES['image_id'] = VOLUMES['image_id'].astype(str)
VOLUMES['tbv_pct'] = 100 * VOLUMES['tbv'] / VOLUMES['icv']
nacc_mrfi = nacc_raw.dropna(subset=['NACCMRFI']).drop_duplicates('id').set_index('id')['NACCMRFI'].astype(str).str.replace('.zip', 'ni', regex=False)
vol_by_cohort = {c: g.set_index('image_id') for c, g in VOLUMES.dropna(subset=['tbv_pct']).groupby('cohort')}


def match_volume(cohort, pid, scan_id, ei):
    v = vol_by_cohort.get(cohort)
    if v is None:
        return np.nan
    keys = [scan_id] + ([nacc_mrfi[scan_id]] if cohort == 'NACC' and scan_id in nacc_mrfi.index else [])
    cand = v[v.index.isin(keys)]
    if cand.empty and cohort == 'NACC':
        cand = v[v.index.str.contains(re.escape(pid))]
    if cand.empty:
        return np.nan
    if len(cand) > 1 and pd.notna(ei):
        cand = cand.iloc[[(cand['EI_image'] - ei).abs().argmin()]]
    return cand['tbv_pct'].iloc[0]


with_volumes = SCANS60['cohort'].isin(vol_by_cohort)
SCANS60.loc[with_volumes, 'tbv_pct'] = [match_volume(c, p, str(s), e) for c, p, s, e in
                                        SCANS60.loc[with_volumes, ['cohort', 'pid', 'scan_id', 'EI']].itertuples(index=False)]


def add_volumes(d):
    v = SCANS60[['cohort', 'pid', 'age', 'tbv_pct']].assign(age0_key=lambda x: x['age'].round(4)).drop(columns='age')
    d = d.assign(age0_key=d['age0'].round(4)).merge(v.drop_duplicates(['cohort', 'pid', 'age0_key']),
                                                   on=['cohort', 'pid', 'age0_key'], how='left').drop(columns='age0_key')
    d['tbv_z'] = d.groupby('cohort')['tbv_pct'].transform(lambda x: (x - x.mean()) / x.std())
    return d
