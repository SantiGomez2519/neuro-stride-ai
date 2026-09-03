from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_raw_vs_clean(raw, clean, time_col, value_col, title, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw[time_col], raw[value_col], label='Raw', linewidth=1)
    ax.plot(clean[time_col], clean[value_col], label='Clean', linewidth=1.5, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig


def plot_normalized_variable(normalized, variable, title=None, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    subset = normalized[normalized.variable == variable]
    for (_, cycle), grp in subset.groupby(['side','cycle_id']):
        ax.plot(grp.phase_pct, grp.value, linewidth=0.8, alpha=0.35)
    for side, grp in subset.groupby('side'):
        mean = grp.groupby('phase_pct').value.mean()
        ax.plot(mean.index, mean.values, linewidth=2, label=f'{side} mean')
    ax.set_title(title or variable)
    ax.set_xlabel('Gait cycle [%]')
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig


def plot_with_normative(normalized, normative, variable, side, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    obs = normalized[(normalized.variable == variable) & (normalized.side == side)]
    ref = normative[(normative.variable == variable) & (normative.side == side)]
    for _, grp in obs.groupby('cycle_id'):
        ax.plot(grp.phase_pct, grp.value, linewidth=0.8, alpha=0.25)
    if len(ref):
        ax.fill_between(ref.phase_pct, ref['mean'] - ref['std'], ref['mean'] + ref['std'], alpha=0.2, label='Synthetic TD ±1 SD')
        ax.plot(ref.phase_pct, ref['mean'], linewidth=2, linestyle='--', label='Synthetic TD mean')
    if len(obs):
        mean = obs.groupby('phase_pct').value.mean()
        ax.plot(mean.index, mean.values, linewidth=2.2, label=f'{side} observed mean')
    ax.set_title(f'{variable} - side {side}')
    ax.set_xlabel('Gait cycle [%]')
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig


def write_markdown_summary(subject, trial_meta, features, symmetry, gps, output_path):
    lines = [
        '# Reporte demostrativo de marcha - datos sintéticos',
        '',
        '> Documento de validación de software. No constituye diagnóstico ni recomendación clínica.',
        '',
        f'**Sujeto sintético:** {subject.subject_id}',
        f'**Cohorte sintética:** {subject.cohort}',
        f'**Lado afectado simulado:** {subject.affected_side}',
        f'**Ensayo:** {trial_meta.trial_id}',
        '',
        '## Parámetros temporoespaciales por lado',
        '',
        features.groupby('side')[['stance_pct','swing_pct','double_support_pct','cycle_duration_s','cadence_steps_min','step_length_m','stride_length_m','step_width_m','mean_speed_m_s']].mean().round(3).to_markdown(),
        '',
        '## Asimetría',
        '',
        symmetry.round(2).to_markdown(index=False) if len(symmetry) else 'No disponible.',
        '',
        '## Gait Profile Score sintético',
        '',
        gps.groupby('side')['gps_deg'].agg(['mean','std']).round(2).to_markdown() if len(gps) else 'No disponible.',
        '',
        '## Interpretación técnica',
        '',
        '- Revisar parámetros con mayor asimetría absoluta y curvas fuera de la envolvente sintética TD.',
        '- Usar las características por fase como entradas explicables de modelos de detección y de control adaptativo.',
        '- Toda interpretación clínica requiere revisión del equipo de rehabilitación y datos reales aprobados éticamente.',
    ]
    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
