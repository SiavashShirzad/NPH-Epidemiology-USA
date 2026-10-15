from data import *
from scipy.stats import chi2

STRATA = list(US_POP_STRATA)
STRATUM_BANDS = [(60, 70, '60-69'), (70, 80, '70-79'), (80, 200, '80+')]
OUTCOMES = {'RNPH': (False, False), 'RNPH-G': (True, False), 'RNPH-C': (False, True), 'RNPH-GC': (True, True)}
ADMIN_END = {('NACC', 'RNPH-G'): NACC_GAIT_END, ('NACC', 'RNPH-GC'): NACC_GAIT_END}
OUTCOME_LABELS = {'RNPH': 'NPH imaging features', 'RNPH-G': 'With gait impairment', 'RNPH-C': 'With cognitive impairment',
                  'RNPH-GC': 'With gait and cognitive impairment'}


def participant_record(scan_ages, scan_rad, clin_ages, clin_g, clin_c, needs_g, needs_c, baseline_free):
    entry = scan_ages[0]
    if baseline_free:
        base = clin_ages <= entry + max(LINK_WINDOW.values())
        for needed, vals in [(needs_g, clin_g), (needs_c, clin_c)]:
            if needed:
                known = base & ~np.isnan(vals)
                if not known.any() or np.nanmax(vals[known]) == 1:
                    return None
    t = np.concatenate([scan_ages, clin_ages])
    rad = np.concatenate([scan_rad.astype(float), np.full(len(clin_ages), np.nan)])
    g = np.concatenate([np.full(len(scan_ages), np.nan), clin_g])
    c = np.concatenate([np.full(len(scan_ages), np.nan), clin_c])
    order = np.argsort(t, kind='stable')
    t, rad, g, c = t[order], rad[order], g[order], c[order]
    met = np.maximum.accumulate(np.nan_to_num(rad)) > 0
    if needs_g:
        met &= np.maximum.accumulate(np.nan_to_num(g)) > 0
    if needs_c:
        met &= np.maximum.accumulate(np.nan_to_num(c)) > 0
    hit = np.where(met & (t > entry))[0]
    if len(hit):
        return entry, t[hit[0]], 1
    return (entry, t[-1], 0) if t[-1] > entry else None


def split_person_time(entry, exit_age, event, sex):
    out = {k: [0.0, 0] for k in STRATA}
    for lo, hi, lab in STRATUM_BANDS:
        a, b = max(entry, lo), min(exit_age, hi)
        if a < b:
            out[(sex, lab)][0] += b - a
    if event:
        out[(sex, next(l for lo, hi, l in STRATUM_BANDS if lo <= exit_age < hi))][1] += 1
    return out


def exact_poisson(k, py, per=1000):
    if py <= 0:
        return np.nan, np.nan, np.nan
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    return k / py * per, lo / py * per, chi2.ppf(0.975, 2 * (k + 1)) / 2 / py * per


def fay_feuer(d, n, w, per=1000):
    d, n, w = (np.array(x, float) for x in (d, n, w))
    m = n > 0
    y = np.sum(w[m] * d[m] / n[m])
    v = np.sum(w[m] ** 2 * d[m] / n[m] ** 2)
    wm = np.max(w[m] / n[m])
    lo = 0.0 if y == 0 else (v / (2 * y)) * chi2.ppf(0.025, 2 * y ** 2 / v)
    hi = ((v + wm ** 2) / (2 * (y + wm))) * chi2.ppf(0.975, 2 * (y + wm) ** 2 / (v + wm ** 2))
    return y * per, lo * per, hi * per


def incidence(rad, classifiable, baseline_free=False):
    weights = [US_POP_STRATA[k] / sum(US_POP_STRATA.values()) for k in STRATA]
    empty = (np.array([]), np.array([]), np.array([]))
    rows = []
    for cohort in COHORTS:
        s = SCANS60[(SCANS60['cohort'] == cohort) & SCANS60[classifiable]].sort_values(['pid', 'age'])
        first = s.drop_duplicates('pid', keep='first').set_index('pid')
        eligible = first.index[(s.groupby('pid').size().reindex(first.index) >= 2) & ~first[rad].astype(bool)]
        s = s[s['pid'].isin(eligible)]
        cl = CLIN[(CLIN['cohort'] == cohort) & CLIN['pid'].isin(eligible)]
        scans_by = {p: (d['age'].values, d[rad].values) for p, d in s.groupby('pid')}
        clin_by = {p: (d['age'].values, d['gait_abn'].astype(float).values, d['cog_abn'].astype(float).values) for p, d in cl.groupby('pid')}
        for label, (ng, nc) in OUTCOMES.items():
            if cohort not in INCIDENCE_OUTCOME_COHORTS[label]:
                continue
            strata = {k: [0.0, 0] for k in STRATA}
            n_risk = 0
            end = ADMIN_END.get((cohort, label))
            for p in eligible:
                ca_, cg, cc = clin_by.get(p, empty)
                sa, sr = scans_by[p]
                if end is not None and p in end.index:
                    stop = end[p]
                    if sa[0] >= stop:
                        continue
                    kc, ks = ca_ <= stop, sa <= stop
                    ca_, cg, cc, sa, sr = ca_[kc], cg[kc], cc[kc], sa[ks], sr[ks]
                if (ng and np.all(np.isnan(cg))) or (nc and np.all(np.isnan(cc))):
                    continue
                rec = participant_record(sa, sr, ca_, cg, cc, ng, nc, baseline_free)
                if rec is None or first.loc[p, 'sex'] not in ('M', 'F'):
                    continue
                n_risk += 1
                for k, (py, ev) in split_person_time(*rec, first.loc[p, 'sex']).items():
                    strata[k][0] += py
                    strata[k][1] += ev
            events, pyears = [strata[k][1] for k in STRATA], [strata[k][0] for k in STRATA]
            crude, adj = exact_poisson(int(sum(events)), sum(pyears)), fay_feuer(events, pyears, weights)
            rows.append({'cohort': cohort, 'outcome': label, 'n_at_risk': n_risk, 'events': int(sum(events)),
                         'person_years': round(sum(pyears)), 'crude': crude[0], 'crude_lo': crude[1], 'crude_hi': crude[2],
                         'adj': adj[0], 'adj_lo': adj[1], 'adj_hi': adj[2]})
    return pd.DataFrame(rows)


def incidence_table(results):
    out = []
    for definition, res in results.items():
        for _, r in res.iterrows():
            out.append({'Definition': definition, 'Outcome': OUTCOME_LABELS[r['outcome']], 'Cohort': r['cohort'],
                        'Events, No.': r['events'], 'At risk, No.': r['n_at_risk'], 'Person-years': r['person_years'],
                        'Crude rate (95% CI)': f"{r['crude']:.1f} ({r['crude_lo']:.1f}-{r['crude_hi']:.1f})",
                        'Standardized rate (95% CI)': f"{r['adj']:.1f} ({r['adj_lo']:.1f}-{r['adj_hi']:.1f})"})
    return pd.DataFrame(out).set_index(['Definition', 'Outcome', 'Cohort'])


INC_COLORS = {'Callosal angle': '#1F3A5F', 'DESH': '#A6323B'}
SYMPTOM_LABELS = {'RNPH-G': 'Gait impairment', 'RNPH-C': 'Cognitive impairment', 'RNPH-GC': 'Gait and cognitive impairment'}


def plot_incidence(results, name):
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    x = np.arange(len(COHORTS))
    for k, (label, res) in enumerate(results.items()):
        r = res.set_index(['cohort', 'outcome'])
        vals = np.array([r.loc[(c, 'RNPH'), 'adj'] for c in COHORTS])
        lo = vals - np.array([r.loc[(c, 'RNPH'), 'adj_lo'] for c in COHORTS])
        hi = np.array([r.loc[(c, 'RNPH'), 'adj_hi'] for c in COHORTS]) - vals
        pos = x + (k - 0.5) * 0.34
        ax_a.bar(pos, vals, 0.32, color=INC_COLORS[label], alpha=0.9, label=f'{label} definition')
        ax_a.errorbar(pos, vals, yerr=[lo, hi], fmt='none', ecolor='#333333', elinewidth=0.8, capsize=2.5)
        for p, v, h in zip(pos, vals, hi):
            ax_a.text(p, v + h + 0.4, f'{v:.1f}', ha='center', va='bottom', fontsize=6.4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(COHORTS, fontweight='bold')
    ax_a.set_ylabel('Incidence per 1000\nperson-years')
    ax_a.set_ylim(0, np.ceil(max(res['adj_hi'].max() for res in results.values()) / 2) * 2 + 2)
    ax_a.grid(axis='y', color='#EDEDED', lw=0.5)
    ax_a.set_axisbelow(True)
    ax_a.legend(frameon=False, fontsize=7, loc='upper left')
    ax_a.set_title('A  NPH imaging features, with or without symptoms', loc='left', fontweight='bold', fontsize=8)
    cohorts_b = [c for c in COHORTS if any(c in INCIDENCE_OUTCOME_COHORTS[o] for o in SYMPTOM_LABELS)]
    top = max(res[res['outcome'].isin(list(SYMPTOM_LABELS))]['adj_hi'].max() for res in results.values())
    for j, (label, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, j])
        r = res.set_index(['cohort', 'outcome'])
        xb = np.arange(len(cohorts_b))
        for k, (outcome, olabel) in enumerate(SYMPTOM_LABELS.items()):
            for p, c in zip(xb + (k - 1) * 0.26, cohorts_b):
                if (c, outcome) not in r.index:
                    ax.text(p, 0.15, 'NA', ha='center', va='bottom', fontsize=5.6, color='#8A8A8A')
                    continue
                v, l, h = r.loc[(c, outcome), ['adj', 'adj_lo', 'adj_hi']]
                ax.bar(p, v, 0.24, color=COLORS[outcome], label=olabel if c == cohorts_b[0] else None)
                ax.errorbar(p, v, yerr=[[v - l], [h - v]], fmt='none', ecolor='#333333', elinewidth=0.7, capsize=2)
        ax.set_xticks(xb)
        ax.set_xticklabels(cohorts_b, fontweight='bold')
        ax.set_ylim(0, np.ceil(top / 2) * 2 + 1)
        ax.grid(axis='y', color='#EDEDED', lw=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel('Incidence per 1000\nperson-years')
        ax.set_title(f"{'BC'[j]}  With symptoms, {'callosal angle' if label == 'Callosal angle' else 'DESH'} definition",
                     loc='left', fontweight='bold', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[o]) for o in SYMPTOM_LABELS]
    fig.legend(handles, list(SYMPTOM_LABELS.values()), loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.045, 'Rates standardized to the 2024 US population by age and sex; error bars show 95% CIs. NA, outcome not assessed consistently in that cohort.',
             ha='center', fontsize=6.2, color='#555555')
    save_figure(fig, name)


if __name__ == '__main__':
    INC_CA = incidence('rad_ca', 'classifiable_ca')
    INC_DESH = incidence('rad_desh', 'classifiable_desh')
    INC_SYMPTOM_FREE = incidence('rad_ca', 'classifiable_ca', baseline_free=True)
    save_table(incidence_table({'Callosal angle': INC_CA, 'DESH': INC_DESH}), 'eTable5_incidence')
    save_table(incidence_table({'Callosal angle': INC_SYMPTOM_FREE}), 'incidence_no_baseline_impairment')
    plot_incidence({'Callosal angle': INC_CA, 'DESH': INC_DESH}, 'eFigure4_incidence')
