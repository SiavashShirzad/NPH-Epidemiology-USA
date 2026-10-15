from progression_competing_risks import *
import statsmodels.formula.api as smf

COG_COHORTS = ['BLSA', 'BIOCARD', 'ADNI']
COG_LONGITUDINAL = ['BLSA', 'ADNI']
LINK_WINDOW_COG = {'BLSA': 1.0, 'BIOCARD': 1.0, 'ADNI': 0.5}
COVARIATES = ['age0', 'male', 'educ']
TRAILS_MAX = 300


def speed(seconds):
    return -np.log(num(seconds).clip(lower=1, upper=TRAILS_MAX))


blsa_dates = blsa_raw[['idno', 'visit', 'pid', 'age']].dropna(subset=['idno', 'visit']).drop_duplicates(['idno', 'visit'])
blsa_dates[['idno', 'visit']] = blsa_dates[['idno', 'visit']].apply(num)
tests = read('blsa_cognitive_tests')
tests[['idno', 'visit']] = tests[['idno', 'visit']].apply(num)
tests = tests.drop_duplicates(['idno', 'visit']).sort_values(['idno', 'visit'])
blsa_cog = pd.DataFrame({'idno': tests['idno'], 'visit': tests['visit'],
                         'trails_a': speed(num(tests['TRATS']).fillna(num(tests['trails_a_test_time']))),
                         'trails_b': speed(num(tests['TRBTS']).fillna(num(tests['trails_b_test_time']))),
                         'dsst': num(tests['DSSTot']), 'fluency_letter': num(tests['FLULet']),
                         'cvlt_total': num(tests['CVLtca']), 'cvlt_delay': num(tests['CVLfrl'])})
blsa_cog = blsa_cog.merge(blsa_dates, on=['idno', 'visit'], how='inner')

uw = read('adni_cognition')
uw = uw[before_cutoff(uw['EXAMDATE'])]
uw['age'] = decimal_year(uw['EXAMDATE']) - uw['PTID'].map(adni_birth)
adni_cog = uw.dropna(subset=['PTID', 'age']).rename(columns={'PTID': 'pid', 'ADNI_EF': 'adni_ef', 'ADNI_MEM': 'adni_mem'})[['pid', 'age', 'adni_ef', 'adni_mem']]

bio = read('biocard_cognition')
bio_birth = bio_raw.groupby('JHUANONID')['BIRTHYEAR'].first().apply(num)
bio['pid'] = bio['JHUANONID'].astype(str)
bio['age'] = pd.to_datetime(bio['VISITDATE'], errors='coerce').dt.year - bio['JHUANONID'].map(bio_birth)
biocard_cog = bio.rename(columns={'Composite_executive': 'biocard_ef', 'Composite_episodic_memory': 'biocard_mem'})[['pid', 'age', 'biocard_ef', 'biocard_mem']].dropna(subset=['age'])

TESTS = {'BLSA': {'executive': ['trails_a', 'trails_b', 'dsst', 'fluency_letter'], 'memory': ['cvlt_total', 'cvlt_delay']},
         'BIOCARD': {'executive': ['biocard_ef'], 'memory': ['biocard_mem']},
         'ADNI': {'executive': ['adni_ef'], 'memory': ['adni_mem']}}
COG_RAW = {'BLSA': blsa_cog, 'BIOCARD': biocard_cog, 'ADNI': adni_cog}
for c, d in COG_RAW.items():
    d['pid'] = d['pid'].astype(str)
    d.drop(d[d['age'] < MIN_AGE].index, inplace=True)


def index_scans(rad):
    s = SCANS60[SCANS60['cohort'].isin(COG_COHORTS) & SCANS60[rad.replace('rad_', 'classifiable_')]].sort_values(['cohort', 'pid', 'age'])
    s = s[s[rad].astype(bool) | s['no_ventriculomegaly'].astype(bool)]
    first = s.drop_duplicates(['cohort', 'pid'], keep='first').copy()
    first['imaging'] = first[rad].astype(int)
    first = first.rename(columns={'age': 'age0'})
    first['male'] = (first['sex'] == 'M').astype(int)
    return first[['cohort', 'pid', 'age0', 'male', 'educ', 'imaging', 'cog_abn']]


def standardize(cohort, idx):
    d = COG_RAW[cohort].merge(idx[['pid', 'age0', 'male', 'educ', 'imaging', 'cog_abn']], on='pid', how='inner')
    reference = d[(d['imaging'] == 0) & (d['cog_abn'] == 0) & ((d['age'] - d['age0']).abs() <= 2)]
    for domain, tests in TESTS[cohort].items():
        zs = []
        for t in tests:
            ref = reference.dropna(subset=[t, 'age', 'male', 'educ'])
            if len(ref) < 30:
                continue
            fit = smf.ols(f'{t} ~ age + I(age ** 2) + male + educ', data=ref).fit()
            pred = fit.predict(d.assign(educ=d['educ'].fillna(ref['educ'].median())))
            zs.append((d[t] - pred) / np.sqrt(fit.scale))
        d[domain] = pd.concat(zs, axis=1).mean(axis=1, skipna=True) if zs else np.nan
        ref_sd = d.loc[reference.index, domain].std() if len(reference) else 1
        d[domain] = d[domain] / ref_sd
    d['years'] = d['age'] - d['age0']
    return d.assign(cohort=cohort)


def usable(d, covs, extra=''):
    covs = [c for c in covs if d[c].notna().any() and d[c].nunique() > 1]
    while covs:
        x = smf.ols(f"{covs[0]} ~ {' + '.join(['imaging'] + covs)}{extra}", data=d.dropna(subset=covs + ['imaging'])).exog
        if np.linalg.matrix_rank(x) == x.shape[1]:
            break
        covs = covs[:-1]
    return covs


def ols_effect(d, outcome, cohort_fe):
    covs = usable(d, COVARIATES, ' + C(cohort)' if cohort_fe else '')
    d = d.dropna(subset=[outcome, 'imaging'] + covs)
    if d['imaging'].sum() < 5 or len(d) < 30:
        return (np.nan,) * 5
    rhs = ' + '.join(['imaging'] + covs + (['C(cohort)'] if cohort_fe else []))
    fit = smf.ols(f'{outcome} ~ {rhs}', data=d).fit(cov_type='HC3')
    lo, hi = fit.conf_int().loc['imaging']
    return fit.params['imaging'], lo, hi, fit.pvalues['imaging'], int(d['imaging'].sum())


def mixed_slope(d, outcome, cohort_fe):
    covs = usable(d, COVARIATES, ' + C(cohort)' if cohort_fe else '')
    d = d.dropna(subset=[outcome, 'years', 'imaging'] + covs)
    d = d[(d['years'] >= -0.5) & (d['years'] <= FOLLOWUP_CAP)]
    counts = d.groupby('pid').size()
    d = d[d['pid'].isin(counts[counts >= 2].index)]
    if d.loc[d['imaging'] == 1, 'pid'].nunique() < 5:
        return (np.nan,) * 5
    rhs = ' + '.join(['years * imaging'] + covs + (['C(cohort) * years'] if cohort_fe else []))
    fit = None
    for re_formula in ('~years', None):
        try:
            fit = smf.mixedlm(f'{outcome} ~ {rhs}', d, groups=d['cohort'] + '_' + d['pid'], re_formula=re_formula).fit(method='lbfgs', reml=True)
            break
        except Exception:
            continue
    if fit is None:
        return (np.nan,) * 5
    lo, hi = fit.conf_int().loc['years:imaging']
    return fit.params['years:imaging'], lo, hi, fit.pvalues['years:imaging'], int(d.loc[d['imaging'] == 1, 'pid'].nunique())


fmt_beta = lambda r: '' if pd.isna(r[0]) else f'{r[0]:.2f} ({r[1]:.2f} to {r[2]:.2f})'

cross_rows, slope_rows = [], []
for label, rad in [('Callosal angle', 'rad_ca'), ('DESH', 'rad_desh')]:
    idx = index_scans(rad)
    cog = pd.concat([standardize(c, idx[idx['cohort'] == c]) for c in COG_COHORTS], ignore_index=True)
    at_index = cog[cog['years'].abs() <= cog['cohort'].map(LINK_WINDOW_COG)].assign(dist=lambda d: d['years'].abs())
    base = at_index.sort_values('dist').drop_duplicates(['cohort', 'pid'])
    for outcome in ['executive', 'memory']:
        for c in COG_COHORTS + ['Pooled']:
            d = base if c == 'Pooled' else base[base['cohort'] == c]
            r = ols_effect(d, outcome, c == 'Pooled')
            cross_rows.append({'Definition': label, 'Outcome': outcome, 'Cohort': c, 'n imaging': r[4],
                               'Difference, SD (95% CI)': fmt_beta(r), 'P': fmt_p(r[3]), 'beta': r[0], 'lo': r[1], 'hi': r[2]})
        for c in COG_LONGITUDINAL + ['Pooled']:
            d = cog[cog['cohort'].isin(COG_LONGITUDINAL)] if c == 'Pooled' else cog[cog['cohort'] == c]
            r = mixed_slope(d, outcome, c == 'Pooled')
            slope_rows.append({'Definition': label, 'Outcome': outcome, 'Cohort': c, 'n imaging': r[4],
                               'Difference in annual change, SD/y (95% CI)': fmt_beta(r), 'P': fmt_p(r[3]), 'beta': r[0], 'lo': r[1], 'hi': r[2]})
CROSS, SLOPES = pd.DataFrame(cross_rows), pd.DataFrame(slope_rows)

COG_PANELS = {'executive': 'Executive function and processing speed', 'memory': 'Episodic memory'}


def plot_cognition(name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    specs = [(CROSS, COG_COHORTS + ['Pooled'], 'A  At the index MRI', 'Difference vs controls, SD'),
             (SLOPES, COG_LONGITUDINAL + ['Pooled'], 'B  Change over time', 'Difference in annual change, SD per year')]
    shown = [c for c in COG_COHORTS + ['Pooled'] if CROSS.loc[CROSS['Cohort'] == c, 'beta'].notna().any()]
    rows, labels, y = [], [], 0
    for domain in COG_PANELS:
        for c in shown:
            rows.append((domain, c, y))
            labels.append(c)
            y += 1
        y += 0.9
    for ax, (table, cohorts, title, xlabel) in zip(axes, specs):
        for k, definition in enumerate(['Callosal angle', 'DESH']):
            t = table[table['Definition'] == definition]
            for domain, c, yy in rows:
                r = t[(t['Outcome'] == domain) & (t['Cohort'] == c)]
                if c in cohorts and len(r) and pd.notna(r['beta'].iloc[0]):
                    r = r.iloc[0]
                    pooled = c == 'Pooled'
                    ax.errorbar(r['beta'], yy + (k - 0.5) * 0.3, xerr=[[r['beta'] - r['lo']], [r['hi'] - r['beta']]],
                                fmt='D' if pooled else 's', ms=4.4 if pooled else 3.4, lw=1.0, capsize=0, color=DEFINITION_COLORS[definition],
                                mfc=DEFINITION_COLORS[definition] if pooled else 'white', mew=1.0)
        ax.axvline(0, color='#8A8A8A', lw=0.7, ls=(0, (3, 2)))
        for domain, c, yy in rows:
            if c == 'Pooled':
                ax.axhspan(yy - 0.5, yy + 0.5, color='#F2F2F2', zorder=0)
        ax.set_title(title, loc='left', fontsize=7.4, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=6.8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
    axes[0].set_yticks([r[2] for r in rows])
    axes[0].set_yticklabels(labels, fontsize=6.6)
    axes[0].set_ylim(y - 0.4, -1.2)
    for domain, dname in COG_PANELS.items():
        y0 = min(r[2] for r in rows if r[0] == domain)
        axes[0].text(-0.02, y0 - 0.8, dname, transform=axes[0].get_yaxis_transform(), ha='right', va='center', fontsize=6.6, fontweight='bold')
    handles = [Line2D([], [], color=DEFINITION_COLORS[d], marker='s', mfc='white', ls='', ms=4, label=f'{d} definition') for d in ['Callosal angle', 'DESH']]
    handles.append(Line2D([], [], color='#3A3A3A', marker='D', ls='', ms=4.4, label='Pooled estimate (cohort fixed effects)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.5)
    save_figure(fig, name)


if __name__ == '__main__':
    save_table(CROSS.drop(columns=['beta', 'lo', 'hi']).set_index(['Definition', 'Outcome', 'Cohort']), 'cognitive_profile_index_mri')
    save_table(SLOPES.drop(columns=['beta', 'lo', 'hi']).set_index(['Definition', 'Outcome', 'Cohort']), 'cognitive_profile_change')
    plot_cognition('eFigure8_cognitive_profile')
