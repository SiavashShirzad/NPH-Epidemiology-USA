from config import *

blsa_raw = read('blsa')
blsa_raw = blsa_raw.loc[:, ~blsa_raw.columns.duplicated()]
blsa_raw = blsa_raw.groupby(['pid', 'visitdate'], as_index=False).first()
blsa_raw['age'] = decimal_year(blsa_raw['visitdate']) - decimal_year(blsa_raw['birthdate'])

blsa_subj = blsa_raw.groupby('pid').agg(sex=('sex', 'first'), race_code=('race', 'first'), educ=('maxeducation', 'first'),
                                        birth=('birthdate', 'first'), death=('dateofdeath', 'first'),
                                        last_cogn_status_type1=('last_cogn_status_type1', 'first'),
                                        last_cogn_status_type2=('last_cogn_status_type2', 'first')).reset_index()
blsa_subj['death_age'] = decimal_year(blsa_subj['death']) - decimal_year(blsa_subj['birth'])
blsa_subj['race'] = num(blsa_subj['race_code']).map(RACE_BLSA).fillna('Unknown')
blsa_subj = blsa_subj.drop(columns=['race_code', 'birth', 'death'])

blsa_scans = blsa_raw[num(blsa_raw['EI']).notna()].assign(scan_id=lambda d: d['id'].astype(str))
blsa_scans = blsa_scans[['pid', 'scan_id', 'age', 'EI', 'CA', 'DESI']]

sppb = num(blsa_raw['sppb']).where(lambda s: (s >= 0) & (s <= 12))
cogn = num(blsa_raw['cogn_status'])
blsa_clin = pd.DataFrame({'pid': blsa_raw['pid'], 'age': blsa_raw['age'],
                          'gait_score': sppb, 'gait_abn': (sppb <= SPPB_THRESH).where(sppb.notna()),
                          'cog_score': cogn, 'cog_abn': cogn.isin([1, 2]).where(cogn.notna())})

bio_raw = read('biocard').groupby(['JHUANONID', 'VISITYEAR'], as_index=False).first()
bio_raw['age'] = num(bio_raw['VISITYEAR']) - num(bio_raw['BIRTHYEAR'])

bio_subj = bio_raw.groupby('JHUANONID').agg(sex=('SEX', 'first'), race_code=('RACE', 'first'), educ=('EDUC', 'first'),
                                            birthyear=('BIRTHYEAR', 'first'), deathyear=('DEATHYEAR', 'first')).reset_index().rename(columns={'JHUANONID': 'pid'})
bio_subj['sex'] = num(bio_subj['sex']).map({1: 'M', 2: 'F'})
bio_subj['race'] = num(bio_subj['race_code']).map(RACE_NACC).fillna('Unknown')
bio_subj['death_age'] = num(bio_subj['deathyear']) - num(bio_subj['birthyear'])
bio_subj = bio_subj.drop(columns=['race_code', 'birthyear', 'deathyear'])

bio_scans = bio_raw[num(bio_raw['EI']).notna()].rename(columns={'JHUANONID': 'pid'})
bio_scans['scan_id'] = bio_scans['pid'].astype(str) + '_' + bio_scans['VISITYEAR'].astype(int).astype(str)
bio_scans = bio_scans[['pid', 'scan_id', 'age', 'EI', 'CA', 'DESI']]

bio_sppb = num(bio_raw['SPPB']).where(lambda s: (s >= 0) & (s <= 12))
bio_dx = bio_raw['DIAGNOSIS_x'].astype(str).str.upper().where(bio_raw['DIAGNOSIS_x'].notna())
bio_clin = pd.DataFrame({'pid': bio_raw['JHUANONID'], 'age': bio_raw['age'],
                         'gait_score': bio_sppb, 'gait_abn': (bio_sppb <= SPPB_THRESH).where(bio_sppb.notna()),
                         'cog_score': bio_dx, 'cog_abn': bio_dx.isin(['MCI', 'DEMENTIA']).where(bio_dx.notna())})

nacc_raw = read('nacc')
nacc_birth = num(nacc_raw['BIRTHYR']) + 0.5

nacc_subj = nacc_raw.groupby('NACCID').agg(sex=('SEX', 'first'), race_code=('RACE', 'first'), educ=('EDUC', 'first'),
                                           birthyr=('BIRTHYR', 'first'), yod=('NACCYOD', 'first'),
                                           braak=('NACCBRAA', 'max'), neuritic=('NACCNEUR', 'max')).reset_index().rename(columns={'NACCID': 'pid'})
nacc_subj['sex'] = num(nacc_subj['sex']).map({1: 'M', 2: 'F'})
nacc_subj['race'] = num(nacc_subj['race_code']).map(RACE_NACC).fillna('Unknown')
nacc_subj['educ'] = num(nacc_subj['educ']).where(lambda s: (s >= 0) & (s <= 36))
nacc_subj['death_age'] = num(nacc_subj['yod']).where(lambda s: (s > 1900) & (s < 2100)) - num(nacc_subj['birthyr'])
for c in ['braak', 'neuritic']:
    nacc_subj[c] = num(nacc_subj[c]).where(lambda s: (s >= 0) & (s <= 7))
nacc_subj = nacc_subj.drop(columns=['race_code', 'birthyr', 'yod'])

scan_date = pd.to_datetime(num(nacc_raw['date']).astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
nacc_raw['scan_age'] = decimal_year(scan_date).fillna(num(nacc_raw['MRIYR']) + 0.5) - nacc_birth
nacc_scans = nacc_raw[num(nacc_raw['EI']).notna()].drop_duplicates('id').rename(columns={'NACCID': 'pid'})
nacc_scans = nacc_scans.assign(scan_id=nacc_scans['id'].astype(str), age=nacc_scans['scan_age'])
nacc_scans = nacc_scans[['pid', 'scan_id', 'age', 'EI', 'CA', 'DESI']]

visit_date = pd.to_datetime(dict(year=num(nacc_raw['VISITYR']), month=num(nacc_raw['VISITMO']), day=num(nacc_raw['VISITDAY'])), errors='coerce')
nacc_raw['visit_age'] = decimal_year(visit_date) - nacc_birth
nacc_visits = nacc_raw[nacc_raw['visit_age'].notna()].drop_duplicates(['NACCID', 'VISITYR', 'VISITMO', 'VISITDAY'])
gait = num(nacc_visits['GAIT']).where(lambda s: s.isin([0, 1, 2, 3, 4]))
cdr = num(nacc_visits['CDRGLOB']).where(lambda s: s.isin([0, 0.5, 1, 2, 3]))
nacc_clin = pd.DataFrame({'pid': nacc_visits['NACCID'], 'age': nacc_visits['visit_age'],
                          'gait_score': gait, 'gait_abn': (gait >= 1).where(gait.notna()),
                          'cog_score': cdr, 'cog_abn': (cdr >= 0.5).where(cdr.notna()),
                          'etpr': num(nacc_visits['NACCETPR']),
                          'amyl_pet': num(nacc_visits['AMYLPET']).where(lambda s: s.isin([0, 1])),
                          'amyl_csf': num(nacc_visits['AMYLCSF']).where(lambda s: s.isin([0, 1]))})

adni_raw = read('adni')
adni_raw = adni_raw[adni_raw['Sex'] != 'X']
ADNI_FREEZE_KEYS = set(zip(adni_raw['Subject'], adni_raw['Visit']))
adni_sex_fallback = adni_raw.groupby('Subject')['Sex'].first()

tab = read('adni_visits')
keys = ['subject_id', 'visit']


def adni_long(var, frozen=True):
    v = tab[keys + [var, var + '_age']]
    v = v[num(v[var]).notna() & num(v[var + '_age']).notna()]
    if frozen:
        v = v[[k in ADNI_FREEZE_KEYS for k in zip(v['subject_id'], v['visit'])]]
    return pd.DataFrame({'pid': v['subject_id'], 'age': num(v[var + '_age']), var: num(v[var])})


birth_pairs = []
for var in ['CDGLOBAL', 'NXGAIT', 'MMSCORE', 'DXPARK']:
    d = tab[keys + [var + '_date', var + '_age']].dropna()
    birth_pairs.append(pd.DataFrame({'pid': d['subject_id'], 'birth': decimal_year(d[var + '_date']) - num(d[var + '_age'])}))
adni_birth = pd.concat(birth_pairs).groupby('pid')['birth'].median()

visit_dates = tab.melt(id_vars=keys, value_vars=[c for c in tab.columns if c.endswith('_date')], value_name='date')
visit_dates['date'] = pd.to_datetime(visit_dates['date'], errors='coerce')
ADNI_VISIT_DATE = visit_dates.dropna(subset=['date']).groupby(keys)['date'].median()

adni_scans = adni_raw[num(adni_raw['EI']).notna()].groupby('Image Data ID', as_index=False).first()
adni_scans = adni_scans.rename(columns={'Subject': 'pid', 'Image Data ID': 'scan_id'})
adni_scans['visit_date'] = [ADNI_VISIT_DATE.get((p, v)) for p, v in zip(adni_scans['pid'], adni_scans['Visit'])]
adni_scans['scan_date'] = pd.to_datetime(adni_scans['image_date'], errors='coerce').fillna(pd.to_datetime(adni_scans['visit_date']))
adni_scans['age'] = (decimal_year(adni_scans['scan_date']) - adni_scans['pid'].map(adni_birth)).fillna(num(adni_scans['Age']))
adni_scans['session_key'] = adni_scans['pid'].astype(str) + '_' + adni_scans['scan_date'].dt.strftime('%Y-%m-%d').fillna(adni_scans['Visit'].astype(str))
adni_scans = adni_scans[['pid', 'scan_id', 'age', 'EI', 'CA', 'DESI', 'session_key', 'scan_date']]
ADNI_DATA_CUTOFF = adni_scans['scan_date'].max()
before_cutoff = lambda dates: pd.to_datetime(dates, errors='coerce') <= ADNI_DATA_CUTOFF

cdr_long = adni_long('CDGLOBAL')
gait_long = adni_long('NXGAIT')
cdr_long = cdr_long[cdr_long['CDGLOBAL'].isin([0, 0.5, 1, 2, 3])]
gait_long = gait_long[gait_long['NXGAIT'].isin([1, 2])]
adni_clin = pd.concat([pd.DataFrame({'pid': cdr_long['pid'], 'age': cdr_long['age'], 'cog_score': cdr_long['CDGLOBAL'],
                                     'cog_abn': cdr_long['CDGLOBAL'] >= 0.5}),
                       pd.DataFrame({'pid': gait_long['pid'], 'age': gait_long['age'], 'gait_score': gait_long['NXGAIT'],
                                     'gait_abn': gait_long['NXGAIT'] == 2})], ignore_index=True)
adni_clin['gait_abn'] = adni_clin['gait_abn'].astype('object').where(adni_clin['gait_score'].notna())
adni_clin['cog_abn'] = adni_clin['cog_abn'].astype('object').where(adni_clin['cog_score'].notna())

demo = tab.groupby('subject_id').agg(sex=('PTGENDER', 'first'), educ=('PTEDUCAT', 'first'),
                                     race_code=('PTRACCAT', 'first'), dod=('NPDODYR', 'max')).reset_index()
demo['sex'] = num(demo['sex']).map({1: 'M', 2: 'F'})
race_code = demo['race_code'].astype(str)
demo['race'] = np.where(race_code.str.contains(r'\|'), 'More than one race', num(race_code).map(RACE_ADNI).fillna('Unknown'))
demo = demo.rename(columns={'subject_id': 'pid'})
adni_deaths = tab[['subject_id', 'AEHDTHDT']].dropna()
adni_ae_death_year = pd.to_datetime(adni_deaths['AEHDTHDT'], errors='coerce').dt.year.groupby(adni_deaths['subject_id']).min()
demo['dod'] = pd.concat([num(demo['dod']), demo['pid'].map(adni_ae_death_year)], axis=1).min(axis=1)
demo['dod'] = demo['dod'].where(demo['dod'] + 0.5 <= decimal_year(pd.Series([ADNI_DATA_CUTOFF])).iloc[0])
demo['death_age'] = demo['dod'] + 0.5 - demo['pid'].map(adni_birth)
adni_subj = demo[['pid', 'sex', 'race', 'educ', 'death_age']].copy()
missing_ids = sorted(set(adni_scans['pid']) - set(adni_subj['pid']))
adni_subj = pd.concat([adni_subj, pd.DataFrame({'pid': missing_ids, 'race': 'Unknown'})], ignore_index=True)
adni_subj['sex'] = adni_subj['sex'].fillna(adni_subj['pid'].map(adni_sex_fallback))

parts = {'BLSA': (blsa_subj, blsa_scans, blsa_clin), 'BIOCARD': (bio_subj, bio_scans, bio_clin),
         'NACC': (nacc_subj, nacc_scans, nacc_clin), 'ADNI': (adni_subj, adni_scans, adni_clin)}
SUBJ = pd.concat([p[0].assign(cohort=c) for c, p in parts.items()], ignore_index=True)
SCANS = pd.concat([p[1].assign(cohort=c) for c, p in parts.items()], ignore_index=True)
CLIN = pd.concat([p[2].assign(cohort=c) for c, p in parts.items()], ignore_index=True)
for d in (SUBJ, SCANS, CLIN):
    d['pid'] = d['pid'].astype(str)

imaging_cols = ['EI', 'CA', 'DESI']
SCANS[imaging_cols] = SCANS[imaging_cols].apply(num)
SCANS = SCANS.dropna(subset=['age'])
SCANS['session'] = SCANS['session_key'].where(SCANS['session_key'].notna(), SCANS['age'].round(2).astype(str))
SCANS = (SCANS.groupby(['cohort', 'pid', 'session'], as_index=False)
              .agg(scan_id=('scan_id', 'first'), n_acquisitions=('scan_id', 'size'), age=('age', 'mean'),
                   **{c: (c, 'mean') for c in imaging_cols})
              .drop(columns='session'))
CLIN = CLIN.dropna(subset=['age'])

SCANS['undeterminable'] = False
SCANS['review_desh'] = np.nan


def classify(scans):
    ei, ca, desi = scans['EI'], scans['CA'], scans['DESI']
    usable = ei.notna() & ~scans['undeterminable']
    reviewed_desh = scans['review_desh'].notna()
    desh_features = np.where(reviewed_desh, scans['review_desh'] == 'Positive', desi > DESI_THRESH)
    scans['classifiable_ca'] = usable & ((ei <= EVANS_THRESH) | ca.notna())
    scans['rad_ca'] = usable & (ei > EVANS_THRESH) & (ca < CA_THRESH)
    scans['classifiable_desh'] = usable & ((ei <= EVANS_THRESH) | desi.notna() | reviewed_desh)
    scans['rad_desh'] = usable & (ei > EVANS_THRESH) & desh_features
    scans['classifiable_ei'] = usable
    scans['rad_ei'] = usable & (ei > EVANS_THRESH)
    scans['no_ventriculomegaly'] = usable & (ei < EVANS_THRESH)
    return scans


def link(scans, clin, value_cols, avail_col):
    out = []
    for cohort, window in LINK_WINDOW.items():
        s = scans[scans['cohort'] == cohort][['cohort', 'pid', 'scan_id', 'age']].sort_values('age')
        c = clin[(clin['cohort'] == cohort) & clin[avail_col].notna()][['pid', 'age'] + value_cols]
        c = c.rename(columns={'age': 'clin_age'}).sort_values('clin_age')
        out.append(pd.merge_asof(s, c, left_on='age', right_on='clin_age', by='pid', direction='nearest', tolerance=window))
    return pd.concat(out, ignore_index=True).drop(columns=['age'])


SCANS = classify(SCANS)
AUTOMATED = SCANS[['cohort', 'scan_id', 'rad_ca', 'rad_desh']].copy()

review = read('scan_review', dtype=str)
review_ei, review_ca = num(review['Review_EI']), num(review['Review_CA'])
ca_typed_as_ei = review_ei > 1
ei_typed_as_ca = review_ca < 10
review_ca = review_ca.where(~ei_typed_as_ca, np.nan).fillna(num(review['Review_EI']).where(ca_typed_as_ei))
review = review.assign(review_ca=review_ca, decision=review['Review_decision'].replace({'Needs revision': 'Revised'}),
                       review_desh_call=review['Review_DESH'].where(review['Review_DESH'].isin(['Positive', 'Negative'])))

keyed = review.drop_duplicates(['Cohort', 'Scan_ID']).set_index(['Cohort', 'Scan_ID'])
key = pd.MultiIndex.from_arrays([SCANS['cohort'], SCANS['scan_id'].astype(str)])
decision = pd.Series(keyed['decision'].reindex(key).values, index=SCANS.index)
revised = decision == 'Revised'
SCANS.loc[revised, 'CA'] = pd.Series(keyed['review_ca'].reindex(key).values, index=SCANS.index)[revised].fillna(SCANS.loc[revised, 'CA'])
SCANS['review_desh'] = pd.Series(keyed['review_desh_call'].reindex(key).values, index=SCANS.index).where(revised)
SCANS['undeterminable'] = decision == 'Undeterminable'
SCANS = classify(SCANS)

compare = AUTOMATED.set_index(['cohort', 'scan_id']).join(SCANS.set_index(['cohort', 'scan_id'])[['rad_ca', 'rad_desh', 'undeterminable']], rsuffix='_after')
audit = []
for c in COHORTS:
    r = review[review['Cohort'] == c]
    x = compare.loc[c]
    audit.append({'Cohort': c, 'Scans reviewed': len(r),
                  'Correct': int((r['decision'] == 'Correct').sum()), 'Revised': int((r['decision'] == 'Revised').sum()),
                  'Undeterminable': int((r['decision'] == 'Undeterminable').sum()),
                  'CA classification: negative to positive': int((~x['rad_ca'] & x['rad_ca_after']).sum()),
                  'CA classification: positive to negative': int((x['rad_ca'] & ~x['rad_ca_after'] & ~x['undeterminable']).sum()),
                  'DESH classification: negative to positive': int((~x['rad_desh'] & x['rad_desh_after']).sum()),
                  'DESH classification: positive to negative': int((x['rad_desh'] & ~x['rad_desh_after'] & ~x['undeterminable']).sum())})

g = link(SCANS, CLIN, ['gait_score', 'gait_abn'], 'gait_abn').rename(columns={'clin_age': 'gait_age'})
c = link(SCANS, CLIN, ['cog_score', 'cog_abn'], 'cog_abn').rename(columns={'clin_age': 'cog_age'})
SCANS = (SCANS.merge(g, on=['cohort', 'pid', 'scan_id'], how='left')
              .merge(c, on=['cohort', 'pid', 'scan_id'], how='left')
              .merge(SUBJ[['cohort', 'pid', 'sex', 'race', 'educ', 'death_age']], on=['cohort', 'pid'], how='left'))
SCANS60 = SCANS[SCANS['age'] >= MIN_AGE].copy()

nacc_gait_rows = CLIN[(CLIN['cohort'] == 'NACC') & CLIN['gait_abn'].notna()]
nacc_any_rows = CLIN[(CLIN['cohort'] == 'NACC') & (CLIN['gait_abn'].notna() | CLIN['cog_abn'].notna())]
nacc_last_gait = nacc_gait_rows.groupby('pid')['age'].max()
nacc_last_visit = nacc_any_rows.groupby('pid')['age'].max()
NACC_GAIT_END = nacc_last_gait[nacc_last_visit.reindex(nacc_last_gait.index) > nacc_last_gait + 0.25]

if __name__ == '__main__':
    save_table(pd.DataFrame(audit).set_index('Cohort').T.assign(Total=lambda d: d.sum(axis=1)), 'scan_review_summary')
