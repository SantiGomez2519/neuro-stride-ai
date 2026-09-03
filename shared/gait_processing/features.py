import numpy as np
import pandas as pd


def symmetry_index(right, left, absolute=False):
    denom = 0.5 * (np.abs(right) + np.abs(left))
    value = 100.0 * (right - left) / denom if denom > 1e-12 else np.nan
    return abs(value) if absolute else value


def _interp_at(df, time_s, col):
    x = df['time_s'].to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.interp(time_s, x[valid], y[valid])) if valid.sum() >= 2 else np.nan


def compute_cycle_features(kin, analog, contact_df, cycles, mass_kg):
    rows = []
    # Interpolate contact flags onto kinematic timestamps for double-support calculation.
    ctime = contact_df['time_s'].to_numpy()
    cL = contact_df['L'].to_numpy()
    cR = contact_df['R'].to_numpy()
    pelvis_x = 0.25 * (kin['LASI_X_mm'] + kin['RASI_X_mm'] + kin['LPSI_X_mm'] + kin['RPSI_X_mm']) / 1000.0
    kin_local = kin.copy()
    kin_local['pelvis_x_m'] = pelvis_x
    for cyc in cycles.itertuples(index=False):
        side = cyc.side
        other = 'L' if side == 'R' else 'R'
        duration = cyc.cycle_duration_s
        stance = cyc.stance_duration_s
        start = cyc.start_s
        end = cyc.end_s
        seg = kin_local[(kin_local.time_s >= start) & (kin_local.time_s <= end)]
        if len(seg) < 10:
            continue
        tseg = seg.time_s.to_numpy()
        Lc = np.interp(tseg, ctime, cL) > 0.5
        Rc = np.interp(tseg, ctime, cR) > 0.5
        ds_pct = 100.0 * np.mean(Lc & Rc)
        heel_x = f'{side}HEE_X_mm'
        heel_y = f'{side}HEE_Y_mm'
        other_heel_x = f'{other}HEE_X_mm'
        other_heel_y = f'{other}HEE_Y_mm'
        stride_length = abs(_interp_at(kin_local, end, heel_x) - _interp_at(kin_local, start, heel_x)) / 1000.0
        step_length = abs(_interp_at(kin_local, start, heel_x) - _interp_at(kin_local, start, other_heel_x)) / 1000.0
        step_width = abs(_interp_at(kin_local, start, heel_y) - _interp_at(kin_local, start, other_heel_y)) / 1000.0
        speed = abs(_interp_at(kin_local, end, 'pelvis_x_m') - _interp_at(kin_local, start, 'pelvis_x_m')) / duration
        row = {
            'side': side,
            'cycle_id': cyc.cycle_id,
            'start_s': start,
            'cycle_duration_s': duration,
            'stance_pct': 100.0 * stance / duration,
            'swing_pct': 100.0 * (duration - stance) / duration,
            'double_support_pct': ds_pct,
            'cadence_steps_min': 120.0 / duration,
            'stride_length_m': stride_length,
            'step_length_m': step_length,
            'step_width_m': step_width,
            'mean_speed_m_s': speed,
        }
        # Curve-derived clinically interpretable features.
        for joint in ['HipFlex','HipAbd','KneeFlex','KneeValgus','AnkleDorsi','FootProgression']:
            col = f'{side}_{joint}_deg'
            if col in seg:
                vals = seg[col].to_numpy(dtype=float)
                row[f'{joint}_min_deg'] = float(np.nanmin(vals))
                row[f'{joint}_max_deg'] = float(np.nanmax(vals))
                row[f'{joint}_rom_deg'] = float(np.nanmax(vals) - np.nanmin(vals))
        toe_col = f'{side}TOE_Z_mm'
        if toe_col in seg:
            swing_seg = seg[seg.time_s >= cyc.toe_off_s]
            if len(swing_seg):
                row['peak_toe_clearance_mm'] = float(np.nanmax(swing_seg[toe_col]))
                mid = swing_seg.iloc[max(1, int(0.15*len(swing_seg))):max(2, int(0.85*len(swing_seg)))]
                row['minimum_mid_swing_toe_clearance_mm'] = float(np.nanmin(mid[toe_col])) if len(mid) else np.nan
        for name in ['HipMomentFlex_Nm_kg','KneeMomentFlex_Nm_kg','AnkleMomentDorsi_Nm_kg','HipPower_W_kg','KneePower_W_kg','AnklePower_W_kg']:
            col = f'{side}_{name}'
            if col in seg:
                vals = seg[col].to_numpy(dtype=float)
                row[f'{name}_min'] = float(np.nanmin(vals))
                row[f'{name}_max'] = float(np.nanmax(vals))
        # Vertical GRF peaks normalized to body weight.
        aseg = analog[(analog.time_s >= start) & (analog.time_s <= cyc.toe_off_s)]
        if len(aseg):
            fz = aseg[f'{side}_GRF_V_N'].to_numpy(dtype=float) / (mass_kg * 9.81)
            row['vertical_grf_peak_bw'] = float(np.nanmax(fz))
            row['vertical_grf_impulse_bw_s'] = float(np.trapz(fz, aseg.time_s.to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_features(cycle_features: pd.DataFrame, group_cols=('subject_id','trial_id','side')):
    numeric = cycle_features.select_dtypes(include=np.number).columns.tolist()
    keep = [c for c in numeric if c not in ('start_s',)]
    agg = cycle_features.groupby(list(group_cols))[keep].agg(['mean','std','median'])
    agg.columns = ['__'.join(c) for c in agg.columns]
    return agg.reset_index()


def pairwise_trial_symmetry(cycle_features):
    metrics = ['stance_pct','swing_pct','cycle_duration_s','step_length_m','stride_length_m','mean_speed_m_s','KneeFlex_rom_deg','AnkleDorsi_rom_deg','peak_toe_clearance_mm']
    rows = []
    for (subject_id, trial_id), grp in cycle_features.groupby(['subject_id','trial_id']):
        left = grp[grp.side == 'L']
        right = grp[grp.side == 'R']
        row = {'subject_id': subject_id, 'trial_id': trial_id}
        for metric in metrics:
            if metric in grp:
                lv = float(left[metric].mean())
                rv = float(right[metric].mean())
                row[f'{metric}__L_mean'] = lv
                row[f'{metric}__R_mean'] = rv
                row[f'{metric}__symmetry_index_pct'] = symmetry_index(rv, lv)
                row[f'{metric}__absolute_asymmetry_pct'] = symmetry_index(rv, lv, absolute=True)
        rows.append(row)
    return pd.DataFrame(rows)
