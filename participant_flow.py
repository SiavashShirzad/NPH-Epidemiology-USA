from progression_competing_risks import *
from matplotlib.patches import FancyBboxPatch

flow = {}
for c in COHORTS:
    all_scans = SCANS[SCANS['cohort'] == c]
    s60 = all_scans[all_scans['age'] >= MIN_AGE]
    cls = s60[s60['classifiable_ca']]
    complete = cls[cls['gait_abn'].notna() & cls['cog_abn'].notna()]
    first = cls.sort_values('age').drop_duplicates('pid').set_index('pid')
    n_sessions = cls.groupby('pid').size()
    col = {'Participants with ≥1 MRI': all_scans['pid'].nunique(), 'With ≥1 MRI at age ≥60': s60['pid'].nunique(),
           'Excluded: no classifiable MRI': s60['pid'].nunique() - cls['pid'].nunique(),
           'Prevalence sample, imaging': cls['pid'].nunique(), 'Prevalence sample, clinical presentations': complete['pid'].nunique(),
           'Incidence sample': int(((n_sessions.reindex(first.index) >= 2) & ~first['rad_ca'].astype(bool)).sum())}
    for sc, d in ARMS_CA.items():
        d = d[d['cohort'] == c]
        if len(d):
            col[f'Progression, {sc.lower()} (imaging / control)'] = f"{(d['arm'] == 'Target').sum()} / {(d['arm'] == 'Control').sum()}"
    if c in COGNITION_COHORTS and c not in PROG_COHORTS:
        d = COG_ARMS['Callosal angle']
        d = d[d['cohort'] == c]
        col['Progression, cognitive impairment (imaging / control)'] = f"{(d['arm'] == 'Target').sum()} / {(d['arm'] == 'Control').sum()}"
    flow[c] = col
FLOW = pd.DataFrame(flow).fillna('')


def flow_box(ax, x, y, h, text, w=0.9, dashed=False, bold_first=False, size=6.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.01,rounding_size=0.04', fc='white',
                                ec='#A0A0A0' if dashed else '#6E6E6E', lw=0.8, ls='--' if dashed else '-'))
    lines = text.split('\n')
    if bold_first:
        ax.text(x, y + h / 2 - 0.2, lines[0], ha='center', va='top', fontsize=size, fontweight='bold')
        ax.text(x, y + h / 2 - 0.45, '\n'.join(lines[1:]), ha='center', va='top', fontsize=size - 0.4, linespacing=1.45)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=size, linespacing=1.35, color='#8A8A8A' if dashed else 'black')


def flow_arrow(ax, x, y0, y1, note=None):
    ax.annotate('', xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#6E6E6E', mutation_scale=7))
    if note:
        ax.text(x + 0.04, (y0 + y1) / 2, note, ha='left', va='center', fontsize=5.5, color='#6E6E6E', linespacing=1.2)


fig, ax = plt.subplots(figsize=(7.2, 6.6))
ax.set_xlim(0, 4)
ax.set_ylim(0, 10)
ax.axis('off')
for i, c in enumerate(COHORTS):
    f, x = FLOW[c], i + 0.5
    n = lambda key: int(f[key])
    ax.text(x, 9.75, c, ha='center', va='center', fontsize=8.5, fontweight='bold')
    flow_box(ax, x, 9.05, 0.6, f"With ≥1 MRI\nn = {n('Participants with ≥1 MRI'):,}")
    flow_arrow(ax, x, 8.72, 7.98, f"Excluded: no MRI\nat age ≥60 y,\nn = {n('Participants with ≥1 MRI') - n('With ≥1 MRI at age ≥60'):,}")
    flow_box(ax, x, 7.65, 0.6, f"≥1 MRI at age ≥60 y\nn = {n('With ≥1 MRI at age ≥60'):,}")
    flow_arrow(ax, x, 7.32, 6.62, f"Excluded: no\nclassifiable MRI,\nn = {n('Excluded: no classifiable MRI'):,}")
    flow_box(ax, x, 5.95, 1.25, f"Prevalence\nClassifiable MRI: n = {n('Prevalence sample, imaging'):,}\nWith gait and cognitive\n"
                                f"assessments: n = {n('Prevalence sample, clinical presentations'):,}", bold_first=True)
    flow_arrow(ax, x, 5.3, 4.95)
    flow_box(ax, x, 4.55, 0.7, f"Incidence\nn = {n('Incidence sample'):,}", bold_first=True)
    flow_arrow(ax, x, 4.17, 3.65)
    progression = [(k.replace('Progression, ', '').replace(' (imaging / control)', ''), v) for k, v in f.items() if k.startswith('Progression') and v]
    if progression:
        text = 'Progression\nNPH imaging features / controls\n' + '\n'.join(
            f"{k.replace('gait only to gait and cognitive impairment', 'gait only to both').capitalize()}:\n{v}" for k, v in progression)
        flow_box(ax, x, 3.65 - (0.55 + 0.5 * len(progression)) / 2, 0.55 + 0.5 * len(progression), text, bold_first=True, size=6.1)
    else:
        flow_box(ax, x, 3.3, 0.7, 'Progression not\nanalyzed', dashed=True, size=6.4)
ax.plot([0.1, 3.9], [0.55, 0.55], color='#D0D0D0', lw=0.6)
ax.text(2.0, 0.3, 'Progression counts are participants free of the outcome at the index MRI (NPH imaging features / no ventriculomegaly), callosal angle definition.',
        ha='center', va='center', fontsize=5.8, color='#6E6E6E')

save_table(FLOW, 'eFigure1_flow_counts')
save_figure(fig, 'eFigure1_participant_flow')
