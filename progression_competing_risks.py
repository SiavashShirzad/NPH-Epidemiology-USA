from covariates import *
from scipy.stats import chi2
from lifelines import AalenJohansenFitter, CoxPHFitter
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

SCENARIOS = {'Gait impairment': ('gait_abn', None),
             'Cognitive impairment': ('cog_abn', None),
             'Gait and cognitive impairment': ('comb_abn', None),
             'Gait only to gait and cognitive impairment': ('cog_abn', lambda s: (s['base_gait'] == 1) & (s['base_cog'] == 0))}
ADMIN_END = {('NACC', 'gait_abn'): NACC_GAIT_END, ('NACC', 'comb_abn'): NACC_GAIT_END}

BASE_COVARIATES = ['age0', 'male']
ADJ_COVARIATES = ['age0', 'male', 'amyloid_positive', 'amyloid_unknown', 'parkinsonism', 'stroke_yes', 'stroke_unknown']
TAU_COVARIATES = ['tau_positive', 'tau_unknown']
VOLUME_COVARIATES = ['tbv_z']
MODELS = {'Model 1': (BASE_COVARIATES, False), 'Model 2': (ADJ_COVARIATES, False), 'Model 3': (BASE_COVARIATES + VOLUME_COVARIATES, True)}


def model_covariates(d):
    d = add_baseline_covariates(d)
    d['parkinsonism'] = d['parkinsonism'].astype(int)
    d['amyloid_positive'] = (d['amyloid'] == 1).astype(int)
    d['amyloid_unknown'] = d['amyloid'].isna().astype(int)
    d['amyloid_pet_positive'] = (d['amyloid_pet'] == 1).astype(int)
    d['amyloid_pet_unknown'] = d['amyloid_pet'].isna().astype(int)
    d['tau_positive'] = (d['tau'] == 1).astype(int)
    d['tau_unknown'] = d['tau'].isna().astype(int)
    d['stroke_yes'] = (d['stroke'] == 1).astype(int)
    d['stroke_unknown'] = d['stroke'].isna().astype(int)
    return d


gait_rows = CLIN[CLIN['gait_abn'].notna()][['cohort', 'pid', 'age', 'gait_abn']].sort_values('age')
cog_rows = CLIN[CLIN['cog_abn'].notna()][['cohort', 'pid', 'age', 'cog_abn']].sort_values('age')
CLIN_PAIRED = pd.concat([pd.merge_asof(gait_rows[gait_rows['cohort'] == k], cog_rows[cog_rows['cohort'] == k][['pid', 'age', 'cog_abn']].rename(columns={'age': 'cog_age'}),
                                       left_on='age', right_on='cog_age', by='pid', direction='nearest', tolerance=LINK_WINDOW[k]).dropna(subset=['cog_abn'])
                         for k in PROG_COHORTS], ignore_index=True)
CLIN_PAIRED['comb_abn'] = CLIN_PAIRED['gait_abn'].astype(bool) & CLIN_PAIRED['cog_abn'].astype(bool)
FOLLOW = {}


def outcome_table(cohort, outcome):
    if (cohort, outcome) not in FOLLOW:
        d = (CLIN_PAIRED[CLIN_PAIRED['cohort'] == cohort][['pid', 'age', 'comb_abn']] if outcome == 'comb_abn'
             else CLIN[(CLIN['cohort'] == cohort) & CLIN[outcome].notna()][['pid', 'age', outcome]])
        FOLLOW[(cohort, outcome)] = {p: (x['age'].values, x[outcome].astype(bool).values) for p, x in d.groupby('pid')}
    return FOLLOW[(cohort, outcome)]


def build_arms(cohort, rad, outcome, base_filter=None, control='no_ventriculomegaly', death_rule='observed', exclude_prior_abnormal=False):
    s = SCANS60[(SCANS60['cohort'] == cohort) & SCANS60[rad.replace('rad_', 'classifiable_')]].copy().sort_values(['pid', 'age'])
    s['target'] = s[rad].astype(bool)
    s['base_gait'], s['base_cog'] = s['gait_abn'].astype(float), s['cog_abn'].astype(float)
    s['base_comb'] = ((s['base_gait'] == 1) & (s['base_cog'] == 1)).astype(float)
    base_col = {'gait_abn': 'base_gait', 'cog_abn': 'base_cog', 'comb_abn': 'base_comb'}[outcome]
    conversion = s[s['target']].groupby('pid')['age'].min()
    if control == 'not_nph':
        ctl = s[~s['target']]
        ctl = ctl[ctl['age'] < ctl['pid'].map(conversion).fillna(np.inf)]
        tgt = s[s['target']]
        if base_filter is not None:
            ctl, tgt = ctl[base_filter(ctl)], tgt[base_filter(tgt)]
        pools = [('Control', ctl.drop_duplicates('pid'), conversion), ('Target', tgt.drop_duplicates('pid'), None)]
    else:
        s = s[s['target'] | s['no_ventriculomegaly'].astype(bool)]
        if base_filter is not None:
            s = s[base_filter(s)]
        first = s.drop_duplicates('pid', keep='first')
        pools = [('Control', first[~first['target']], conversion), ('Target', first[first['target']], None)]
    follow, admin_end = outcome_table(cohort, outcome), ADMIN_END.get((cohort, outcome))
    death = SUBJ[SUBJ['cohort'] == cohort].set_index('pid')['death_age']
    rows = []
    for arm, pool, conversions in pools:
        pool = pool.set_index('pid')
        pool = pool[pool[base_col] != 1]
        for pid, b in pool.iterrows():
            base_age = b['age']
            conversion_age = conversions.get(pid, np.inf) if conversions is not None else np.inf
            stop = min(conversion_age, admin_end.get(pid, np.inf) if admin_end is not None else np.inf)
            if stop <= base_age:
                continue
            ages, abn = follow.get(pid, (np.array([]), np.array([], bool)))
            prior = ages <= base_age + 0.5
            if exclude_prior_abnormal and prior.any() and abn[prior][np.argmin(ages[prior])]:
                continue
            keep = (ages > base_age) & (ages <= stop)
            ages, abn = ages[keep], abn[keep]
            if death_rule == 'observed' and len(ages) == 0:
                continue
            death_age = death.get(pid, np.nan)
            last_seen = ages.max() if len(ages) else base_age
            death_counts = (pd.notna(death_age) and base_age < death_age <= stop and
                            (death_rule == 'any' or death_age <= last_seen + DEATH_WINDOW))
            if abn.any():
                dur, status = ages[abn].min() - base_age, 1
            elif death_counts:
                dur, status = death_age - base_age, 2
            elif np.isfinite(conversion_age):
                dur, status = stop - base_age, 0
            elif len(ages):
                dur, status = last_seen - base_age, 0
            else:
                continue
            if dur > FOLLOWUP_CAP:
                dur, status = FOLLOWUP_CAP, 0
            if dur > 0 and b['sex'] in ('M', 'F'):
                rows.append({'cohort': cohort, 'pid': pid, 'arm': arm, 'duration': dur, 'event': status, 'age0': base_age,
                             'male': int(b['sex'] == 'M'), 'educ': num(pd.Series([b['educ']])).iloc[0]})
    return pd.DataFrame(rows)


def estimable(d, c):
    v = d[c].dropna()
    if c in BASE_COVARIATES or not set(v.unique()) <= {0, 1}:
        return v.nunique() > 1
    return v.nunique() > 1 and min((v == 1).sum(), (v == 0).sum()) >= 10 and ((d[c] == 1) & (d['event'] == 1)).sum() >= 3


def model_frame(d, covariates, strata=None):
    covs = [c for c in covariates if estimable(d, c) and (not strata or c in BASE_COVARIATES or (d.groupby('cohort')[c].nunique() > 1).any())]
    key = d['cohort'] + '_' + d['pid'].astype(str)
    m = d.assign(imaging=(d['arm'] == 'Target').astype(int), cs=(d['event'] == 1).astype(int), key=key)
    m = m[['duration', 'cs', 'imaging'] + covs + ([strata] if strata else []) + ['key']].dropna()
    repeated = m['key'].duplicated().any()
    return (m if repeated else m.drop(columns='key')), repeated


def cox(d, covariates=BASE_COVARIATES, strata=None, min_events=5):
    tgt, ctl = d[d['arm'] == 'Target'], d[d['arm'] == 'Control']
    if len(tgt) < MIN_IMAGING_N or (tgt['event'] == 1).sum() < min_events or (ctl['event'] == 1).sum() < min_events:
        return (np.nan,) * 4
    m, repeated = model_frame(d, covariates, strata)
    for penalizer in (0.0, 0.01):
        try:
            model = CoxPHFitter(penalizer=penalizer).fit(m, 'duration', 'cs', strata=[strata] if strata else None,
                                                          cluster_col='key' if repeated else None, robust=repeated)
            r = model.summary.loc['imaging']
            return np.exp(r['coef']), np.exp(r['coef lower 95%']), np.exp(r['coef upper 95%']), r['p']
        except Exception:
            continue
    return (np.nan,) * 4


def heterogeneity(d, covariates):
    logs, variances = [], []
    for c in PROG_COHORTS:
        hr, lo, hi, _ = cox(d[d['cohort'] == c], covariates)
        if pd.notna(hr):
            logs.append(np.log(hr))
            variances.append(((np.log(hi) - np.log(lo)) / 3.92) ** 2)
    if len(logs) < 2:
        return np.nan
    b, w = np.array(logs), 1 / np.array(variances)
    return chi2.sf(np.sum(w * (b - np.sum(w * b) / w.sum()) ** 2), len(logs) - 1)


def progression_arms(rad, control='no_ventriculomegaly', death_rule='observed', exclude_prior_abnormal=False):
    return {sc: add_volumes(model_covariates(pd.concat([build_arms(c, rad, o, bf, control, death_rule, exclude_prior_abnormal)
                                                        for c in PROG_COHORTS], ignore_index=True)))
            for sc, (o, bf) in SCENARIOS.items()}


def progression_table(arms):
    rows, hr_rows = [], []
    for sc, d in arms.items():
        for c in PROG_COHORTS + ['Pooled']:
            x = d if c == 'Pooled' else d[d['cohort'] == c]
            t, k = x[x['arm'] == 'Target'], x[x['arm'] == 'Control']
            if t.empty:
                continue
            strata = 'cohort' if c == 'Pooled' else None
            row = {'Outcome': sc, 'Cohort': c, 'NPH imaging features, No.': len(t), 'Controls, No.': len(k),
                   'Events, NPH imaging features, No.': int((t['event'] == 1).sum()), 'Events, controls, No.': int((k['event'] == 1).sum())}
            for model, (covs, needs_volumes) in MODELS.items():
                h = cox(x.dropna(subset=VOLUME_COVARIATES) if needs_volumes else x, covs, strata=strata)
                row[f'{model}, HR (95% CI)'] = fmt_hr(h)
                row[f'{model}, P'] = fmt_p(h[3])
                hr_rows.append({'outcome': sc, 'cohort': c, 'model': model, 'hr': h[0], 'lo': h[1], 'hi': h[2], 'p': h[3]})
            row['Heterogeneity P'] = fmt_p(heterogeneity(d, ADJ_COVARIATES)) if c == 'Pooled' else ''
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Outcome', 'Cohort']), pd.DataFrame(hr_rows)


def fit_cif(g):
    if len(g) < MIN_AT_RISK or (g['event'] == 1).sum() == 0:
        return None
    return AalenJohansenFitter(calculate_variance=True).fit(g['duration'], g['event'], event_of_interest=1)


ARMS_CA = progression_arms('rad_ca')
ARMS_DESH = progression_arms('rad_desh')
TABLE_CA, HR_CA = progression_table(ARMS_CA)
TABLE_DESH, HR_DESH = progression_table(ARMS_DESH)
HRS = {'Callosal angle': HR_CA, 'DESH': HR_DESH}
COG_ARMS = {label: add_volumes(model_covariates(pd.concat([build_arms(c, rad, 'cog_abn') for c in COGNITION_COHORTS], ignore_index=True)))
            for label, rad in [('Callosal angle', 'rad_ca'), ('DESH', 'rad_desh')]}


def cognition_three_cohorts():
    rows = []
    for label, d in COG_ARMS.items():
        for c in COGNITION_COHORTS + ['Pooled, BLSA and ADNI', 'Pooled, BLSA, ADNI, and NACC']:
            x = d[d['cohort'].isin(PROG_COHORTS)] if c == 'Pooled, BLSA and ADNI' else d if c.startswith('Pooled') else d[d['cohort'] == c]
            strata = 'cohort' if c.startswith('Pooled') else None
            t, k = x[x['arm'] == 'Target'], x[x['arm'] == 'Control']
            row = {'Definition': label, 'Cohort': c, 'NPH imaging features, No.': len(t), 'Controls, No.': len(k),
                   'Events, NPH imaging features, No.': int((t['event'] == 1).sum()), 'Events, controls, No.': int((k['event'] == 1).sum())}
            for model in ['Model 1', 'Model 2']:
                hr = cox(x, MODELS[model][0], strata=strata)
                row[f'{model}, HR (95% CI)'] = fmt_hr(hr)
                row[f'{model}, P'] = fmt_p(hr[3])
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Definition', 'Cohort'])


def sensitivity_analyses():
    alternative = {'Control: all without NPH imaging features': progression_arms('rad_ca', control='not_nph'),
                   'Earliest recorded assessment abnormal excluded': progression_arms('rad_ca', exclude_prior_abnormal=True),
                   'Deaths at any time; no follow-up assessment required': progression_arms('rad_ca', death_rule='any')}
    rows = []
    for sc in SCENARIOS:
        base = ARMS_CA[sc]
        comorbid = (base['parkinsonism'] == 1) | (base['stroke'] == 1) | (base['amyloid'] == 1)
        specs = [('Primary (model 2)', base, ADJ_COVARIATES),
                 ('Amyloid-negative participants only', base[base['amyloid'] == 0], [c for c in ADJ_COVARIATES if not c.startswith('amyloid')]),
                 ('Amyloid biomarker status known (complete case)', base[base['amyloid'].notna()], [c for c in ADJ_COVARIATES if c != 'amyloid_unknown']),
                 ('Amyloid biomarker status from PET only', base, [c.replace('amyloid_', 'amyloid_pet_') for c in ADJ_COVARIATES]),
                 ('Model 2 plus tau PET status', base, ADJ_COVARIATES + TAU_COVARIATES),
                 ('Parkinsonism, stroke, and amyloid positivity excluded', base[~comorbid], BASE_COVARIATES),
                 ('Model 2 plus education', base.dropna(subset=['educ']), ADJ_COVARIATES + ['educ'])]
        specs += [(name, arms[sc], ADJ_COVARIATES) for name, arms in alternative.items()]
        for name, d, covs in specs:
            row = {'Outcome': sc, 'Analysis': name, 'NPH imaging features/controls, No.': f"{(d['arm'] == 'Target').sum()}/{(d['arm'] == 'Control').sum()}"}
            for c in PROG_COHORTS:
                row[f'{c}, HR (95% CI)'] = fmt_hr(cox(d[d['cohort'] == c], covs))
            pooled = cox(d, covs, strata='cohort')
            row['Pooled, HR (95% CI)'] = fmt_hr(pooled)
            row['P'] = fmt_p(pooled[3])
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Outcome', 'Analysis'])


def cumulative_incidence_table():
    rows = []
    for label, arms in [('Callosal angle', ARMS_CA), ('DESH', ARMS_DESH)]:
        for sc, d in arms.items():
            for c in PROG_COHORTS:
                for arm, group in [('Target', 'NPH imaging features'), ('Control', 'No ventriculomegaly')]:
                    x = d[(d['cohort'] == c) & (d['arm'] == arm)]
                    if len(x) < MIN_IMAGING_N:
                        continue
                    ajf = AalenJohansenFitter(calculate_variance=True, seed=1).fit(x['duration'], x['event'], event_of_interest=1)
                    idx = ajf.cumulative_density_.index[ajf.cumulative_density_.index <= 5].max()
                    est = 100 * ajf.cumulative_density_.loc[idx].iloc[0]
                    lo, hi = (100 * ajf.confidence_interval_cumulative_density_.loc[idx]).tolist()
                    rows.append({'Definition': label, 'Outcome': sc, 'Cohort': c, 'Group': group, 'n': len(x),
                                 '5-y cumulative incidence, % (95% CI)': f'{est:.1f} ({lo:.1f}-{hi:.1f})',
                                 'At risk at 5 y': int((x['duration'] >= 5).sum())})
    return pd.DataFrame(rows).set_index(['Definition', 'Outcome', 'Cohort', 'Group'])


PROG_COLORS = {'Gait impairment': '#3B6FA0', 'Cognitive impairment': '#C9822B', 'Gait and cognitive impairment': '#9E2B35',
               'Gait only to gait and cognitive impairment': '#9E2B35', 'control': '#6E6E6E'}
PROG_TITLES = {'Gait impairment': 'Gait impairment', 'Cognitive impairment': 'Cognitive impairment',
               'Gait and cognitive impairment': 'Gait and cognitive\nimpairment', 'Gait only to gait and cognitive impairment': 'Gait only to gait and\ncognitive impairment'}
RISK_TIMES = [0, 2, 4, 6, 8, 10, 12]


def p_label(p):
    t = fmt_p(p)
    return f'P < {t[1:]}' if t.startswith('<') else f'P = {t}'


def draw_curve(ax, g, color, ls):
    a = fit_cif(g)
    if a is None:
        return
    curve, band = a.cumulative_density_.iloc[:, 0] * 100, a.confidence_interval_ * 100
    ok = [t for t in np.arange(0, FOLLOWUP_CAP + 0.01, 0.05) if (g['duration'] >= t).sum() >= MIN_AT_RISK]
    main = curve.index <= (max(ok) if ok else 0)
    ax.step(curve.index[main], curve.values[main], where='post', color=color, ls=ls, lw=1.5, solid_capstyle='butt')
    ax.fill_between(band.index[main], band.iloc[main, 0], band.iloc[main, 1], step='post', color=color, alpha=0.13, lw=0)
    tail = curve.index >= curve.index[main].max()
    if tail.sum() > 1:
        ax.step(curve.index[tail], curve.values[tail], where='post', color=color, ls=ls, lw=1.0, alpha=0.35)


def plot_curves(arms, hrs, cohorts, name, control_label='No ventriculomegaly', model='Model 2'):
    scenarios = list(SCENARIOS)
    n_rows, n_cols = len(cohorts), len(scenarios)
    height = 2.35 * n_rows + 0.4
    fig = plt.figure(figsize=(7.2, height))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.26, left=0.17, right=0.985, top=1 - 0.32 / height, bottom=0.3 / height)
    letters = iter('ABCDEFGHIJKLMNOP')
    for i, cohort in enumerate(cohorts):
        row_axes = []
        for j, sc in enumerate(scenarios):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[4.4, 0.9], hspace=0.5)
            ax, risk_ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            risk_ax.axis('off')
            d = arms[sc] if cohort == 'Pooled' else arms[sc][arms[sc]['cohort'] == cohort]
            tgt, ctl = d[d['arm'] == 'Target'], d[d['arm'] == 'Control']
            letter = next(letters)
            ax.set_title(f'{letter}  {PROG_TITLES[sc]}' if i == 0 else letter, loc='left', fontsize=7.0, fontweight='bold', pad=3, linespacing=1.15)
            if len(tgt) < MIN_IMAGING_N or (tgt['event'] == 1).sum() == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[['left', 'bottom']].set_visible(False)
                if j == 0 and n_rows > 1:
                    ax.set_ylabel(f'{cohort}\nCumulative incidence, %', fontsize=6.8)
                ax.text(0.5, 0.5, f'Insufficient participants\n(n = {len(tgt)} with NPH imaging features)', ha='center', va='center',
                        fontsize=6.3, color='#8A8A8A', transform=ax.transAxes)
                continue
            draw_curve(ax, ctl, PROG_COLORS['control'], (0, (4, 2)))
            draw_curve(ax, tgt, PROG_COLORS[sc], '-')
            h = hrs[(hrs['outcome'] == sc) & (hrs['cohort'] == cohort) & (hrs['model'] == model)]
            if len(h) and pd.notna(h['hr'].iloc[0]):
                h = h.iloc[0]
                ax.text(0.04, 0.96, f"HR, {h['hr']:.2f} ({h['lo']:.2f}-{h['hi']:.2f})\n{p_label(h['p'])}", transform=ax.transAxes,
                        ha='left', va='top', fontsize=6.0, linespacing=1.3)
            ax.set_xlim(0, FOLLOWUP_CAP)
            ax.set_xticks(RISK_TIMES)
            ax.tick_params(labelsize=6.3, length=2.5, width=0.6, pad=1.5)
            ax.set_xlabel('Years since index MRI', fontsize=6.6, labelpad=1.5)
            ax.grid(axis='y', color='#EDEDED', lw=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(f'{cohort}\nCumulative incidence, %' if n_rows > 1 else 'Cumulative incidence, %', fontsize=6.8)
            elif not row_axes:
                ax.set_ylabel('Cumulative incidence, %', fontsize=6.8)
            risk_ax.set_xlim(ax.get_xlim())
            risk_ax.set_ylim(-0.6, 2.2)
            for r, (g, col) in enumerate([(tgt, PROG_COLORS[sc]), (ctl, PROG_COLORS['control'])]):
                for t in RISK_TIMES:
                    risk_ax.text(t, 1 - r, str(int((g['duration'] >= t).sum())), ha='center', va='center', fontsize=5.6, color=col)
            if not row_axes:
                risk_ax.text(-1.1, 1.95, 'No. at risk', ha='right', va='center', fontsize=5.6, fontweight='bold')
                risk_ax.text(-1.1, 1, 'NPH imaging features', ha='right', va='center', fontsize=5.6)
                risk_ax.text(-1.1, 0, control_label, ha='right', va='center', fontsize=5.6)
            row_axes.append(ax)
        if row_axes:
            top = 10 * np.ceil(max(a.get_ylim()[1] for a in row_axes) / 10)
            for a in row_axes:
                a.set_ylim(0, top)
    handles = [Line2D([], [], color='#3A3A3A', lw=1.5, label='NPH imaging features'),
               Line2D([], [], color=PROG_COLORS['control'], lw=1.5, ls=(0, (4, 2)), label=control_label)]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=6.6, bbox_to_anchor=(0.58, -0.01))
    save_figure(fig, name)


def plot_forest(name, model='Model 2'):
    labels = PROG_COHORTS + ['Pooled']
    outcomes = list(SCENARIOS)
    fig, axes = plt.subplots(1, len(outcomes), figsize=(1.8 * len(outcomes) + 0.9, 2.2), sharey=True)
    for letter, ax, sc in zip('ABCD', axes, outcomes):
        for k, (definition, hrs) in enumerate(HRS.items()):
            h = hrs[(hrs['outcome'] == sc) & (hrs['model'] == model)].set_index('cohort').reindex(labels)
            for y, (cohort, r) in enumerate(h.iterrows()):
                if pd.notna(r['hr']):
                    pooled = cohort == 'Pooled'
                    ax.errorbar(r['hr'], y + (k - 0.5) * 0.32, xerr=[[r['hr'] - r['lo']], [r['hi'] - r['hr']]],
                                fmt='D' if pooled else 's', color=DEFINITION_COLORS[definition], ms=4.6 if pooled else 3.4,
                                capsize=0, lw=1.0, mfc=DEFINITION_COLORS[definition] if pooled else 'white', mew=1.0)
        ax.axvline(1, color='#8A8A8A', lw=0.7, ls=(0, (3, 2)))
        ax.axhspan(len(labels) - 1.5, len(labels) - 0.5, color='#F2F2F2', zorder=0)
        ax.set_xscale('log')
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.set_xticklabels(['0.5', '1', '2', '4', '8'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.35, 12)
        ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
        ax.set_xlabel('Hazard ratio (95% CI)', fontsize=6.8)
        ax.set_title(f'{letter}  {PROG_TITLES[sc]}', fontsize=7.2, fontweight='bold', loc='left', linespacing=1.15)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    handles = [Line2D([], [], color=DEFINITION_COLORS[d], marker='s', mfc='white', ls='', ms=4, label=f'{d} definition') for d in HRS]
    handles.append(Line2D([], [], color='#3A3A3A', marker='D', ls='', ms=4.5, label='Pooled estimate (stratified by cohort)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=1.2)
    save_figure(fig, name)


FIG4_DEFINITIONS = [('Callosal angle', TABLE_CA, 'CA definition'), ('DESH', TABLE_DESH, 'DESH definition')]
FIG4_OUTCOMES = {'Gait impairment': 'Gait impairment', 'Cognitive impairment': 'Cognitive impairment',
                 'Gait and cognitive impairment': 'Gait and cognitive impairment',
                 'Gait only to gait and cognitive impairment': 'Gait only to gait and cognition'}
FIG4_WIDTH, FIG4_ROW = 7.0, 0.165
FIG4_X = {'label': 0.08, 'model': 0.32, 'n_target': 2.56, 'n_control': 3.62, 'fp_lo': 4.12, 'fp_hi': 5.82, 'hr': 5.98}
FIG4_HR_RANGE = (0.4, 11.0)
RULE_DARK, RULE_LIGHT, BAND, MUTED = '#3d3d3a', '#b9b8b1', '#f1f0eb', '#5f5e59'


def hr_to_x(v):
    lo, hi = np.log(FIG4_HR_RANGE[0]), np.log(FIG4_HR_RANGE[1])
    return FIG4_X['fp_lo'] + (np.log(v) - lo) / (hi - lo) * (FIG4_X['fp_hi'] - FIG4_X['fp_lo'])


def fig4_rows(cohort):
    rows, y = [], 0.0
    for definition, table, header in FIG4_DEFINITIONS:
        rows.append({'kind': 'definition', 'y': y, 'label': header, 'definition': definition})
        y += 1.25
        outcomes = [sc for sc in FIG4_OUTCOMES if (sc, cohort) in table.index]
        for k, sc in enumerate(outcomes):
            t = table.loc[(sc, cohort)]
            rows.append({'kind': 'outcome', 'y': y, 'label': FIG4_OUTCOMES[sc], 'definition': definition,
                         'n': (f"{int(t['NPH imaging features, No.'])} ({int(t['Events, NPH imaging features, No.'])})",
                               f"{int(t['Controls, No.'])} ({int(t['Events, controls, No.'])})")})
            y += 1
            for m in MODELS:
                h = HRS[definition]
                h = h[(h['outcome'] == sc) & (h['cohort'] == cohort) & (h['model'] == m)]
                rows.append({'kind': 'model', 'y': y, 'label': m, 'definition': definition, 'hr': h.iloc[0] if len(h) else None})
                y += 1
            if k < len(outcomes) - 1:
                rows.append({'kind': 'separator', 'y': y - 0.5 + 0.2})
                y += 0.4
        y += 0.35
    return rows, y - 0.35


def plot_hr_models(name, cohort='Pooled'):
    rows, last = fig4_rows(cohort)
    header_top, header_bottom = -3.0, -0.85
    body_bottom = last - 0.4
    axis_bottom = body_bottom + 2.5
    fig = plt.figure(figsize=(FIG4_WIDTH, (axis_bottom - header_top) * FIG4_ROW))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG4_WIDTH)
    ax.set_ylim(axis_bottom, header_top)
    ax.axis('off')
    hline = lambda yy, **k: ax.plot([0.04, FIG4_WIDTH - 0.04], [yy, yy], solid_capstyle='butt', **k)
    hline(header_top + 0.25, color=RULE_DARK, lw=1.0)
    hline(header_bottom, color=RULE_DARK, lw=0.6)
    hline(body_bottom, color=RULE_DARK, lw=1.0)
    head_y = (header_top + 0.25 + header_bottom) / 2
    head = dict(fontsize=7, fontweight='bold', color=INK, va='center', linespacing=1.15)
    ax.text(FIG4_X['label'], head_y, 'Outcome', ha='left', **head)
    ax.text(FIG4_X['n_target'], head_y, 'NPH imaging features,\nNo. (events)', ha='center', **head)
    ax.text(FIG4_X['n_control'], head_y, 'Controls,\nNo. (events)', ha='center', **head)
    ax.text((FIG4_X['fp_lo'] + FIG4_X['fp_hi']) / 2, head_y, 'Hazard ratio\n(95% CI)', ha='center', **head)
    ax.text(FIG4_X['hr'], head_y, 'HR (95% CI)', ha='left', **head)
    ref = hr_to_x(1)
    ax.plot([ref, ref], [header_bottom, body_bottom], color='#8a8a8a', lw=0.7, ls=(0, (3, 2)), zorder=1)
    for r in rows:
        if r['kind'] == 'separator':
            hline(r['y'], color=RULE_LIGHT, lw=0.5, ls=(0, (2.5, 2)))
            continue
        y, color = r['y'], DEFINITION_COLORS.get(r.get('definition'), INK)
        if r['kind'] == 'definition':
            ax.add_patch(plt.Rectangle((0.04, y - 0.55), FIG4_WIDTH - 0.08, 1.1, color=BAND, lw=0, zorder=0))
            ax.text(FIG4_X['label'], y, r['label'], fontsize=7.6, fontweight='bold', color=color, va='center')
        elif r['kind'] == 'outcome':
            ax.text(FIG4_X['label'], y, r['label'], fontsize=7, fontweight='bold', color=INK, va='center')
            ax.text(FIG4_X['n_target'], y, r['n'][0], fontsize=6.9, color=INK, va='center', ha='center')
            ax.text(FIG4_X['n_control'], y, r['n'][1], fontsize=6.9, color=INK, va='center', ha='center')
        else:
            primary = r['label'] == 'Model 2'
            style = dict(fontsize=6.8, va='center', color=INK if primary else MUTED, fontweight='bold' if primary else 'normal')
            ax.text(FIG4_X['model'], y, r['label'], **style)
            h = r['hr']
            if h is None or pd.isna(h['hr']):
                ax.text(FIG4_X['hr'], y, 'NE', **style)
                continue
            lo, hi = max(h['lo'], FIG4_HR_RANGE[0]), min(h['hi'], FIG4_HR_RANGE[1])
            ax.plot([hr_to_x(lo), hr_to_x(hi)], [y, y], color=color, lw=1.5 if primary else 0.9, solid_capstyle='butt', zorder=2)
            ax.plot(hr_to_x(h['hr']), y, 's' if primary else 'o', ms=4.6 if primary else 3.6, color=color,
                    mfc=color if primary else 'white', mec=color, mew=0.9, zorder=3)
            ax.text(FIG4_X['hr'], y, fmt_hr((h['hr'], h['lo'], h['hi'])), **style)
    tick_top = body_bottom + 0.35
    for v in [0.5, 1, 2, 4, 8]:
        ax.plot([hr_to_x(v)] * 2, [body_bottom, tick_top], color=RULE_DARK, lw=0.6)
        ax.text(hr_to_x(v), tick_top + 0.25, f'{v:g}', fontsize=6.8, color=INK, ha='center', va='top')
    ax.text((FIG4_X['fp_lo'] + FIG4_X['fp_hi']) / 2, tick_top + 1.3, 'Hazard ratio (log scale)', fontsize=7, color=INK, ha='center', va='top')
    save_figure(fig, name)


if __name__ == '__main__':
    save_table(pd.concat({'Callosal angle': TABLE_CA, 'DESH': TABLE_DESH}, names=['Definition']), 'eTable6_hazard_ratios')
    save_table(sensitivity_analyses(), 'eTable7_sensitivity_analyses')
    save_table(cognition_three_cohorts(), 'eTable8_cognitive_impairment_NACC')
    save_table(cumulative_incidence_table(), 'cumulative_incidence_5y')
    plot_curves(ARMS_CA, HR_CA, ['BLSA'], 'Figure3_progression_BLSA')
    plot_hr_models('Figure4_hazard_ratios_models')
    plot_curves(ARMS_CA, HR_CA, ['ADNI', 'Pooled'], 'eFigure5A_progression_ADNI_pooled_CA')
    plot_curves(ARMS_DESH, HR_DESH, ['BLSA', 'ADNI', 'Pooled'], 'eFigure5B_progression_DESH')
    plot_forest('eFigure6_hazard_ratios_by_cohort')
