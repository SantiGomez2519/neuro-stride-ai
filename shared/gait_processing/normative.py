import numpy as np
import pandas as pd


def build_normative_reference(normalized_long: pd.DataFrame, cycle_metadata: pd.DataFrame, cohort='TD'):
    keys = cycle_metadata.query('cohort == @cohort')[['trial_id','side','cycle_id']].drop_duplicates()
    ref = normalized_long.merge(keys, on=['trial_id','side','cycle_id'], how='inner')
    summary = ref.groupby(['side','variable','phase_pct'])['value'].agg(['mean','std','count']).reset_index()
    return summary


def add_gps_scores(normalized_long: pd.DataFrame, normative: pd.DataFrame, variables):
    ref = normative[normative.variable.isin(variables)][['side','variable','phase_pct','mean']]
    merged = normalized_long[normalized_long.variable.isin(variables)].merge(ref, on=['side','variable','phase_pct'], how='inner')
    merged['sq_error'] = (merged['value'] - merged['mean']) ** 2
    gvs = merged.groupby(['subject_id','trial_id','side','cycle_id','variable'])['sq_error'].mean().pow(0.5).rename('gvs_deg').reset_index()
    gps = gvs.groupby(['subject_id','trial_id','side','cycle_id'])['gvs_deg'].apply(lambda x: float(np.sqrt(np.mean(np.square(x))))).rename('gps_deg').reset_index()
    return gps, gvs
