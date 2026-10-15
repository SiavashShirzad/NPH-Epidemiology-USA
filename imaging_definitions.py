from prevalence_phenotypes import prevalence
from progression_competing_risks import *

DEFINITIONS = {'Evans Index >0.30 alone': 'rad_ei', 'Evans Index + callosal angle <90°': 'rad_ca', 'Evans Index + DESH': 'rad_desh'}
DEFINITION_TICKS = {'Evans Index >0.30 alone': 'Evans Index\nalone', 'Evans Index + callosal angle <90°': 'Evans Index +\ncallosal angle',
                    'Evans Index + DESH': 'Evans Index +\nDESH'}
COHORT_MARKERS = {'BLSA': 'o', 'BIOCARD': 's', 'NACC': '^', 'ADNI': 'D'}
OUTCOMES = ['Gait impairment', 'Cognitive impairment', 'Gait and cognitive impairment']

rows, prevalence_points, hr_points = [], [], []
for label, rad in DEFINITIONS.items():
    prev = prevalence(SCANS60, rad, rad.replace('rad_', 'classifiable_')).set_index(['cohort', 'band', 'category'])
    row = {'Definition': label}
    for c in COHORTS:
        for band in ['≥60', '≥80']:
            x = prev.loc[(c, band, 'RNPH-All')]
            row[f'Prevalence, {c}, {band}, %'] = f"{100 * x['prev']:.1f} ({100 * x['lo']:.1f}-{100 * x['hi']:.1f})"
        prevalence_points.append({'definition': label, 'cohort': c, 'prev': 100 * prev.loc[(c, '≥60', 'RNPH-All'), 'prev']})
    arms = {'rad_ca': ARMS_CA, 'rad_desh': ARMS_DESH}.get(rad) or progression_arms(rad)
    for sc in OUTCOMES:
        hr = cox(arms[sc], ADJ_COVARIATES, strata='cohort')
        row[f'{sc}: pooled HR (95% CI)'] = fmt_hr(hr)
        row[f'{sc}: P'] = fmt_p(hr[3])
        hr_points.append({'definition': label, 'outcome': sc, 'hr': hr[0], 'lo': hr[1], 'hi': hr[2]})
    rows.append(row)

pp, hp = pd.DataFrame(prevalence_points), pd.DataFrame(hr_points)
labels = list(DEFINITIONS)
positions = np.arange(len(labels))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1, 1.25]})
means = pp.groupby('definition')['prev'].mean().reindex(labels)
a1.bar(positions, means.values, 0.58, color=COLORS['RNPH-All'], alpha=0.22, edgecolor=COLORS['RNPH-All'], lw=0.8)
for c in COHORTS:
    v = pp[pp['cohort'] == c].set_index('definition')['prev'].reindex(labels)
    a1.scatter(positions + (COHORTS.index(c) - 1.5) * 0.1, v.values, marker=COHORT_MARKERS[c], s=16, color=COLORS['RNPH-All'],
               edgecolor='white', lw=0.4, zorder=3, label=c)
highest = pp.groupby('definition')['prev'].max().reindex(labels)
for x, m, top in zip(positions, means.values, highest.values):
    a1.text(x, top + 2.0, f'{m:.1f}%', ha='center', va='bottom', fontsize=6.6, fontweight='bold', color=COLORS['RNPH-All'])
a1.set_xticks(positions)
a1.set_xticklabels([DEFINITION_TICKS[l] for l in labels], fontsize=6.6)
a1.set_ylabel('Prevalence at age ≥60, %', fontsize=7)
a1.set_ylim(0, 10 * np.ceil((pp['prev'].max() + 8) / 10))
a1.tick_params(labelsize=6.5, length=2.5, width=0.6)
a1.grid(axis='y', color='#EDEDED', lw=0.5)
a1.set_axisbelow(True)
a1.legend(frameon=False, fontsize=6.2, ncol=2, loc='upper right', handletextpad=0.2, columnspacing=0.8)
a1.set_title('A  Prevalence (bars, mean across cohorts)', loc='left', fontsize=7.2, fontweight='bold')
for k, sc in enumerate(OUTCOMES):
    x = hp[hp['outcome'] == sc].set_index('definition').reindex(labels)
    a2.errorbar(positions + (k - 1) * 0.18, x['hr'], yerr=[x['hr'] - x['lo'], x['hi'] - x['hr']], fmt='o', ms=4, lw=1.0, capsize=0,
                color=PROG_COLORS[sc], label=PROG_TITLES[sc].replace('\n', ' '))
a2.axhline(1, color='#8A8A8A', lw=0.7, ls=(0, (3, 2)))
a2.set_yscale('log')
a2.set_yticks([0.75, 1, 1.5, 2, 3, 4, 6])
a2.set_yticklabels(['0.75', '1', '1.5', '2', '3', '4', '6'])
a2.yaxis.set_minor_locator(NullLocator())
a2.set_ylim(0.7, 6.5)
a2.set_xticks(positions)
a2.set_xticklabels([DEFINITION_TICKS[l] for l in labels], fontsize=6.6)
a2.set_xlim(-0.5, len(labels) - 0.5)
a2.set_ylabel('Pooled hazard ratio (95% CI)', fontsize=7)
a2.tick_params(labelsize=6.5, length=2.5, width=0.6)
a2.grid(axis='y', color='#EDEDED', lw=0.5)
a2.set_axisbelow(True)
a2.legend(frameon=False, fontsize=6.2, loc='upper left')
a2.set_title('B  Progression to symptoms (BLSA and ADNI)', loc='left', fontsize=7.2, fontweight='bold')
fig.tight_layout(w_pad=2.0)

save_table(pd.DataFrame(rows).set_index('Definition').T, 'imaging_definitions')
save_figure(fig, 'eFigure3_imaging_definitions')
