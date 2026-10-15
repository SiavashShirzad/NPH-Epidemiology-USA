from data import *

table1, schedule, missing, rep = {}, {}, {}, []
RACE_GROUPS = ['Asian', 'Black or African American', 'White', 'Other race']
US_60_2024['Other race'] = round(100 - US_60_2024['Asian'] - US_60_2024['Black or African American'] - US_60_2024['White'], 1)

for c in COHORTS:
    s = SCANS60[(SCANS60['cohort'] == c) & SCANS60['classifiable_ca']].sort_values('age')
    base = s.drop_duplicates('pid')
    n = len(base)
    baseline_age = base.set_index('pid')['age']
    clin = CLIN[(CLIN['cohort'] == c) & CLIN['pid'].isin(base['pid'])]
    followup = clin[clin['age'] > clin['pid'].map(baseline_age) + 0.25].groupby('pid')['age'].max() - baseline_age
    sessions = s.groupby('pid').size()
    gait_known, cog_known = int(base['gait_abn'].notna().sum()), int(base['cog_abn'].notna().sum())
    gait_k, cog_k = int((base['gait_abn'] == True).sum()), int((base['cog_abn'] == True).sum())
    cdr = num(base['cog_score'])
    table1[c] = {
        'Participants, No.': str(n),
        'Age at baseline, mean (SD), y': mean_sd(base['age']),
        'Women, No. (%)': n_pct((base['sex'] == 'F').sum(), n),
        'Men, No. (%)': n_pct((base['sex'] == 'M').sum(), n),
        'Asian, No. (%)': n_pct((base['race'] == 'Asian').sum(), n),
        'Black or African American, No. (%)': n_pct((base['race'] == 'Black or African American').sum(), n),
        'White, No. (%)': n_pct((base['race'] == 'White').sum(), n),
        'Other or unknown race, No. (%)': n_pct((~base['race'].isin(['White', 'Black or African American', 'Asian'])).sum(), n),
        'Education, mean (SD), y': mean_sd(base['educ']),
        'Gait impairment at baseline, No./total (%)': f'{gait_k}/{gait_known} ({100 * gait_k / max(gait_known, 1):.1f})',
        'Cognitive impairment at baseline, No./total (%)': f'{cog_k}/{cog_known} ({100 * cog_k / max(cog_known, 1):.1f})',
        'SPPB score at baseline, mean (SD)': mean_sd(base['gait_score']) if c in ('BLSA', 'BIOCARD') else '',
        'Global CDR at baseline, 0 / 0.5 / ≥1, No.': (f'{int((cdr == 0).sum())} / {int((cdr == 0.5).sum())} / {int((cdr >= 1).sum())}'
                                                       if c in ('NACC', 'ADNI') else ''),
        'Evans Index at baseline, mean (SD)': mean_sd(base['EI'], 3),
        'MRI sessions, No.': str(len(s)),
        'MRI sessions per participant, median (IQR)': median_iqr(sessions, 0),
        'Clinical follow-up, median (IQR), y': median_iqr(followup)}

    visits = clin.drop_duplicates(['pid', 'age']).sort_values(['pid', 'age'])
    cls = s.sort_values(['pid', 'age'])
    schedule[c] = {'Sessions per participant, median (IQR)': median_iqr(sessions, 0),
                   'Participants with ≥2 sessions, No. (%)': n_pct((sessions >= 2).sum(), len(sessions)),
                   'Interval between MRIs, median (IQR), y': median_iqr(cls.groupby('pid')['age'].diff()),
                   'Interval between clinical visits, median (IQR), y': median_iqr(visits.groupby('pid')['age'].diff())}
    pct = lambda k, n: f'{int(k)}/{n} ({100 * k / n:.1f})'
    missing[c] = {'Sessions without linked gait assessment': pct(cls['gait_abn'].isna().sum(), len(cls)),
                  'Sessions without linked cognitive assessment': pct(cls['cog_abn'].isna().sum(), len(cls)),
                  'Participants with death recorded': int(SUBJ[(SUBJ['cohort'] == c) & SUBJ['pid'].isin(cls['pid'])]['death_age'].notna().sum())}

    known = base[base['race'] != 'Unknown']
    race = known['race'].where(known['race'].isin(RACE_GROUPS[:3]), 'Other race')
    values = {'Women': 100 * (base['sex'] == 'F').mean(),
              **{g: 100 * (race == g).mean() for g in RACE_GROUPS},
              "Bachelor's degree or higher": 100 * (num(base['educ']).dropna() >= 16).mean()}
    dissimilarity = 0.5 * sum(abs(values[g] - US_60_2024[g]) for g in RACE_GROUPS)
    rep.append({'Cohort': c, 'Participants aged ≥60 y, No.': str(n),
                **{k: f'{v:.1f} ({v / US_60_2024[k]:.2f})' for k, v in values.items()},
                'Index of dissimilarity for race, %': f'{dissimilarity:.1f}'})

rep.append({'Cohort': 'US population, age ≥60 y', 'Participants aged ≥60 y, No.': 'NA',
            **{k: f'{v:.1f}' for k, v in US_60_2024.items()}, 'Index of dissimilarity for race, %': 'Reference'})

save_table(pd.DataFrame(table1), 'Table1_characteristics')
save_table(pd.concat([pd.DataFrame(schedule), pd.DataFrame(missing)]), 'eTable2_schedule_missing_data')
save_table(pd.DataFrame(rep).set_index('Cohort').T, 'eTable3_representativeness')
