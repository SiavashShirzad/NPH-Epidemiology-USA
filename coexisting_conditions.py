from progression_competing_risks import *

nacc_all = nacc_raw[nacc_raw['visit_age'].notna()]
nacc_ids = lambda mask: set(nacc_all.loc[mask, 'NACCID'].astype(str))
nacc_code = num(nacc_all['NACCETPR'])
nacc_flag = lambda col: nacc_ids(num(nacc_all[col]) == 1)
nacc_path = SUBJ[SUBJ['cohort'] == 'NACC'].set_index('pid')[['braak', 'neuritic']].dropna()
dx_all = read('adni_diagnosis')
dx_all = dx_all[before_cutoff(dx_all['EXAMDATE'])]
dx_ids = lambda mask: set(dx_all.loc[mask, 'PTID'])
blsa_codes = SUBJ[SUBJ['cohort'] == 'BLSA'].set_index('pid')[['last_cogn_status_type1', 'last_cogn_status_type2']].apply(num)
blsa_has = lambda codes: set(blsa_codes.index[blsa_codes.isin(codes).any(axis=1)])
bio_flags = bio_raw.groupby('JHUANONID')[['PROBAD', 'POSSAD', 'VASC', 'PARK', 'DLB', 'FTD', 'ALCDEM', 'MEDS']].max().apply(num)
bio_flags.index = bio_flags.index.astype(str)
bio_has = lambda mask: set(bio_flags.index[mask])
stroke_ids = lambda c: set(VASCULAR[(VASCULAR['cohort'] == c) & (VASCULAR['stroke'] == 1)]['pid'])

EVIDENCE = {
    'NACC': {'ad_pathology': set(nacc_path.index[(nacc_path['braak'] >= 3) & (nacc_path['neuritic'] >= 1)]),
             'ad_probable': nacc_flag('PROBAD'), 'ad_possible': nacc_flag('POSSAD'),
             'pd': nacc_flag('PARK') | nacc_flag('MSA') | nacc_flag('CORT') | nacc_ids(nacc_code.isin([2, 3, 4, 5])),
             'vd_clinical': nacc_flag('VASC') | nacc_flag('VASCPS') | nacc_ids(nacc_code == 8), 'stroke_history': set(),
             'other': nacc_flag('FTD') | nacc_flag('HUNT') | nacc_flag('PRION') | nacc_ids(nacc_code.isin(list(range(9, 14)) + list(range(15, 31)))),
             'nph_clinical': nacc_flag('HYCEPH') | nacc_ids(nacc_code == 14)},
    'ADNI': {'ad_pathology': set(), 'ad_probable': dx_ids(num(dx_all['DXAPP']) == 1),
             'ad_possible': dx_ids((num(dx_all['DXAPP']) == 2) | (num(dx_all['DXMDUE']) == 1)),
             'pd': dx_ids((num(dx_all['DXPARK']) == 1) | (num(dx_all['DXAPROB']) == 5) | (num(dx_all['DXAPOSS']) == 5)),
             'vd_clinical': dx_ids((num(dx_all['DXAPROB']) == 2) | (num(dx_all['DXAPOSS']) == 2)), 'stroke_history': stroke_ids('ADNI'),
             'other': dx_ids((num(dx_all['DXMDUE']) == 2) | (num(dx_all['DXDDUE']) == 2) | (num(dx_all['DXOTHDEM']) == 1)),
             'nph_clinical': set()},
    'BLSA': {'ad_pathology': blsa_has([1]), 'ad_probable': blsa_has([2]), 'ad_possible': blsa_has([3, 5]), 'pd': blsa_has([8, 14]),
             'vd_clinical': blsa_has([4, 17]), 'stroke_history': stroke_ids('BLSA'),
             'other': blsa_has([6, 7, 10, 11, 15, 18]), 'nph_clinical': blsa_has([12])},
    'BIOCARD': {'ad_pathology': set(), 'ad_probable': bio_has(bio_flags['PROBAD'] == 1), 'ad_possible': bio_has(bio_flags['POSSAD'] == 1),
                'pd': bio_has((bio_flags['PARK'] == 1) | (bio_flags['DLB'] == 1)), 'vd_clinical': bio_has(bio_flags['VASC'] == 1),
                'stroke_history': set(), 'other': bio_has((bio_flags['FTD'] == 1) | (bio_flags['ALCDEM'] == 1) | (bio_flags['MEDS'] == 1)),
                'nph_clinical': set()}}
AMYLOID_EVER = (AMYLOID[AMYLOID['source'] == 'PET'].groupby(['cohort', 'pid'])['amyloid'].max()
                .combine_first(AMYLOID[AMYLOID['source'] == 'CSF'].groupby(['cohort', 'pid'])['amyloid'].max()))


def classify_evidence(cohort, pid):
    e = EVIDENCE[cohort]
    amyloid = AMYLOID_EVER.get((cohort, pid), np.nan)
    clinical_ad = pid in e['ad_probable'] or pid in e['ad_possible']
    if pid in e['ad_pathology']:
        ad = 'Pathology'
    elif amyloid == 1:
        ad = 'Amyloid biomarker'
    elif clinical_ad and amyloid == 0:
        ad = 'Clinical, amyloid-negative'
    elif clinical_ad:
        ad = 'Alzheimer clinical syndrome'
    else:
        ad = None
    has_ad = ad in ('Pathology', 'Amyloid biomarker')
    has_clinical_ad = ad == 'Alzheimer clinical syndrome'
    has_pd = pid in e['pd']
    vascular = pid in e['vd_clinical'] or pid in e['stroke_history']
    n_causes = has_ad + has_clinical_ad + has_pd + vascular
    primary = ('Mixed' if n_causes >= 2 else 'AD (biomarker)' if has_ad else 'Alzheimer clinical syndrome' if has_clinical_ad
               else 'PD/parkinsonism' if has_pd else 'Vascular' if vascular else 'Other diagnosis' if pid in e['other'] else 'None identified')
    return {'has_ad': has_ad, 'has_clinical_ad': has_clinical_ad, 'has_pd': has_pd, 'vascular': vascular, 'primary': primary,
            'nph_clinical': pid in e['nph_clinical']}


symptomatic = {}
for c in COHORTS:
    s = SCANS60[(SCANS60['cohort'] == c) & SCANS60['classifiable_ca']]
    for arm, mask in [('NPH imaging features', s['rad_ca']), ('No ventriculomegaly', s['no_ventriculomegaly'].astype(bool))]:
        for pid, g in s[mask].groupby('pid'):
            gait, cog = (g['gait_abn'] == True).any(), (g['cog_abn'] == True).any()
            if gait or cog:
                prior = symptomatic.get((c, arm, pid), (False, False))
                symptomatic[(c, arm, pid)] = (prior[0] or gait, prior[1] or cog)
for sc, (gait, cog) in [('Gait impairment', (True, False)), ('Cognitive impairment', (False, True)), ('Gait and cognitive impairment', (True, True))]:
    d = ARMS_CA[sc]
    for _, r in d[d['event'] == 1].iterrows():
        key = (r['cohort'], 'NPH imaging features' if r['arm'] == 'Target' else 'No ventriculomegaly', r['pid'])
        prior = symptomatic.get(key, (False, False))
        symptomatic[key] = (prior[0] or gait, prior[1] or cog)
SYMPTOMATIC = pd.DataFrame([{'cohort': c, 'arm': a, 'pid': p, 'symptoms': 'Gait and cognition' if g and k else 'Gait only' if g else 'Cognition only',
                             **classify_evidence(c, p)} for (c, a, p), (g, k) in symptomatic.items()])

outcome_rows = []
for cohort in COHORTS + ['Pooled']:
    s = SCANS60[SCANS60['classifiable_ca'] & (SCANS60['cohort'] == cohort if cohort != 'Pooled' else True)]
    n = len(set(zip(s.loc[s['rad_ca'], 'cohort'], s.loc[s['rad_ca'], 'pid'])))
    sym = SYMPTOMATIC[(SYMPTOMATIC['arm'] == 'NPH imaging features') & (SYMPTOMATIC['cohort'] == cohort if cohort != 'Pooled' else True)]
    n_none = int((sym['primary'] == 'None identified').sum())
    outcome_rows.append({'Cohort': cohort, 'Participants with NPH imaging features, No.': n,
                         'No gait or cognitive impairment, No. (%)': n_pct(n - len(sym), n),
                         'Impairment with a co-existing condition, No. (%)': n_pct(len(sym) - n_none, n),
                         'Impairment with no co-existing condition identified, No. (%)': n_pct(n_none, n)})

condition_rows = []
for cohort in COHORTS + ['Pooled']:
    for arm in ['NPH imaging features', 'No ventriculomegaly']:
        g = SYMPTOMATIC[(SYMPTOMATIC['arm'] == arm) & (SYMPTOMATIC['cohort'] == cohort if cohort != 'Pooled' else True)]
        n = len(g)
        if n == 0:
            continue
        pct = lambda k: f'{int(k)} ({100 * k / n:.0f})'
        condition_rows.append({'Cohort': cohort, 'Group': arm, 'Symptomatic participants, No.': n,
                               'Any co-existing condition': pct((g['primary'] != 'None identified').sum()),
                               'AD, biomarker or pathology confirmed': pct(g['has_ad'].sum()),
                               'Alzheimer clinical syndrome, no biomarker data': pct(g['has_clinical_ad'].sum()),
                               'Parkinsonism': pct(g['has_pd'].sum()), 'Cerebrovascular disease': pct(g['vascular'].sum()),
                               'Other diagnosis only': pct((g['primary'] == 'Other diagnosis').sum()),
                               'None identified': pct((g['primary'] == 'None identified').sum()),
                               'Clinically diagnosed NPH': int(g['nph_clinical'].sum())})

imaging_sym = SYMPTOMATIC[SYMPTOMATIC['arm'] == 'NPH imaging features']
PRIMARY_ORDER = ['AD (biomarker)', 'Alzheimer clinical syndrome', 'PD/parkinsonism', 'Vascular', 'Mixed', 'Other diagnosis', 'None identified']
PRIMARY_COLORS = {'AD (biomarker)': '#3B6FA0', 'Alzheimer clinical syndrome': '#9DBBD6', 'PD/parkinsonism': '#D6952F', 'Vascular': '#6A9A58',
                  'Mixed': '#8C6BB1', 'Other diagnosis': '#ADB8BE', 'None identified': '#A6323B'}
by_symptom = pd.crosstab(imaging_sym['symptoms'], imaging_sym['primary'], normalize='index').mul(100).reindex(columns=PRIMARY_ORDER, fill_value=0)
by_symptom = by_symptom.reindex(['Gait only', 'Cognition only', 'Gait and cognition'])
counts = imaging_sym['symptoms'].value_counts()

fig, ax = plt.subplots(figsize=(6.5, 2.4))
left = np.zeros(len(by_symptom))
for cat in PRIMARY_ORDER:
    vals = by_symptom[cat].values
    ax.barh(range(len(by_symptom)), vals, left=left, color=PRIMARY_COLORS[cat], edgecolor='white', lw=0.5, label=cat)
    for y, (v, l0) in enumerate(zip(vals, left)):
        if v >= 7:
            ax.text(l0 + v / 2, y, f'{v:.0f}%', ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
    left += vals
ax.set_yticks(range(len(by_symptom)))
ax.set_yticklabels([f'{s} (n = {counts.get(s, 0)})' for s in by_symptom.index])
ax.set_xlim(0, 100)
ax.invert_yaxis()
ax.set_xlabel('Participants with NPH imaging features, %')
ax.legend(ncol=4, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.3))
fig.tight_layout()

save_table(pd.DataFrame(outcome_rows).set_index('Cohort'), 'eTable9A_outcomes')
save_table(pd.DataFrame(condition_rows).set_index(['Cohort', 'Group']).T, 'eTable9B_coexisting_conditions')
save_table(by_symptom.round(0).assign(n=counts), 'coexisting_conditions_by_presentation')
save_figure(fig, 'eFigure9_coexisting_conditions')
