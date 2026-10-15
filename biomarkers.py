from covariates import *
import statsmodels.formula.api as smf

GROUPS = {'NPH imaging features': 0, 'Ventriculomegaly only': 1, 'No ventriculomegaly': 2}
GROUP_COLORS = {'NPH imaging features': COLORS['RNPH-GC'], 'Ventriculomegaly only': COLORS['RNPH-C'], 'No ventriculomegaly': COLORS['control']}


def positivity(marker, values, cohorts, window, min_positive=0):
    rows, plot_rows = [], []
    for c in cohorts:
        s = SCANS60[(SCANS60['cohort'] == c) & SCANS60['classifiable_ca']].copy()
        s['group'] = np.where(s['rad_ca'], 'NPH imaging features', np.where(s['no_ventriculomegaly'].astype(bool), 'No ventriculomegaly', 'Ventriculomegaly only'))
        m = s[['pid', 'age', 'group', 'sex']].merge(values[values['cohort'] == c][['pid', 'age', marker]], on='pid', suffixes=('', '_marker'))
        m = m[(m['age'] - m['age_marker']).abs() <= window]
        m = m.assign(order=m['group'].map(GROUPS), gap=(m['age'] - m['age_marker']).abs()).sort_values(['order', 'gap']).drop_duplicates('pid')
        for grp in GROUPS:
            x = m[m['group'] == grp]
            p, lo, hi = wilson(int(x[marker].sum()), len(x))
            rows.append({'Cohort': c, 'Group': grp, 'n': len(x), 'Positive, No. (%)': n_pct(x[marker].sum(), len(x))})
            plot_rows.append({'cohort': c, 'group': grp, 'p': 100 * p, 'lo': 100 * lo, 'hi': 100 * hi})
        md = m[m['group'].isin(['NPH imaging features', 'No ventriculomegaly'])].assign(
            imaging=lambda d: (d['group'] == 'NPH imaging features').astype(int), male=lambda d: (d['sex'] == 'M').astype(int))
        if md['imaging'].sum() >= 5 and md[marker].sum() >= min_positive:
            fit = smf.logit(f'{marker} ~ imaging + age + male', md).fit(disp=0)
            ci = np.exp(fit.conf_int().loc['imaging'])
            rows.append({'Cohort': c, 'Group': 'OR, NPH imaging features vs no ventriculomegaly (age, sex)', 'n': len(md),
                         'Positive, No. (%)': f"{np.exp(fit.params['imaging']):.2f} ({ci[0]:.2f}-{ci[1]:.2f}); P = {fmt_p(fit.pvalues['imaging'])}"})
    return pd.DataFrame(rows).set_index(['Cohort', 'Group']), pd.DataFrame(plot_rows)


AMY_TABLE, AMY_PLOT = positivity('amyloid', AMYLOID, COHORTS, AMYLOID_WINDOW)
TAU_TABLE, TAU_PLOT = positivity('tau', TAU, ['BIOCARD', 'ADNI'], TAU_WINDOW, min_positive=5)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [2, 1.1]})
for ax, data, cohorts, ylabel, title in [(a1, AMY_PLOT, COHORTS, 'Amyloid positive, %', 'A  Amyloid (PET or CSF)'),
                                         (a2, TAU_PLOT, ['BIOCARD', 'ADNI'], 'Tau positive, %', 'B  Tau (PET)')]:
    for k, grp in enumerate(GROUPS):
        x = data[data['group'] == grp].set_index('cohort').reindex(cohorts)
        ax.bar(np.arange(len(cohorts)) + (k - 1) * 0.25, x['p'], 0.24, yerr=[x['p'] - x['lo'], x['hi'] - x['p']],
               color=GROUP_COLORS[grp], capsize=2, label=grp, error_kw={'lw': 0.8})
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color='#EDEDED', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=8.5)
handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.07, 1, 1])

save_table(AMY_TABLE, 'amyloid_by_imaging_group')
save_table(TAU_TABLE, 'tau_by_imaging_group')
save_figure(fig, 'eFigure7_amyloid_tau')
