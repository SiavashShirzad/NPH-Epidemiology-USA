"""Table 1: baseline demographics and follow-up duration by cohort."""

import pandas as pd
import numpy as np

# Race Maps
map_blsa = {6: "White", 3: "Black", 9: "Asian", 2: "Am. Indian", 10: "Hawaiian", 7: "Other", 8: "Other", 11: "Other"}
map_nacc = {1: "White", 2: "Black", 5: "Asian", 3: "Am. Indian", 4: "Hawaiian", 50: "Other"}
map_biocard = {1: "White", 2: "Black", 5: "Asian", 3: "Am. Indian", 4: "Hawaiian"}
map_adni = {'5': "White", '4': "Black", '2': "Asian", '1': "Am. Indian", '3': "Hawaiian"}

def standardize_sex_col(df):
    """Standardizes sex column to 'Male'/'Female' strings"""
    col_map = {'Sex': 'sex', 'SEX': 'sex', 'PTGENDER': 'sex', 'Gender': 'sex', 'sex_code': 'sex'}
    for source, target in col_map.items():
        if source in df.columns:
            df = df.rename(columns={source: target})
            break
            
    if 'sex' in df.columns:
        # standardize to string for display
        df['sex_label'] = df['sex'].replace({
            1: 'Male', 0: 'Female', 2: 'Female',
            'M': 'Male', 'F': 'Female', 
            'Male': 'Male', 'Female': 'Female'
        })
    else:
        df['sex_label'] = 'Unknown'
    return df

def get_stats_for_cohort(df, cohort_name, pid_col, age_col, ei_col, race_col, race_map, time_col):
    """
    Generates a dictionary of stats for a single cohort, formatted for rows.
    """
    # Rename columns to standard internal names
    df = df.copy().rename(columns={pid_col: 'pid', age_col: 'age', ei_col: 'EI'})
    
    # CRITICAL FIX: If time_col was the same as age_col (e.g. 'Age'), update pointer to 'age'
    if time_col == age_col:
        time_col = 'age'
    elif time_col == pid_col:
        time_col = 'pid'
        
    df = standardize_sex_col(df)
    
    # Race Mapping
    if race_col and race_col in df.columns:
        if cohort_name == 'ADNI':
            # Handle ADNI '5|4' strings
            df['race_val'] = df[race_col].astype(str).str.split('|').str[0].str.strip()
            df['Race_Raw'] = df['race_val'].map(race_map).fillna('Other')
        else:
            df['Race_Raw'] = df[race_col].map(race_map).fillna('Other')
    else:
        df['Race_Raw'] = 'Other'

    # Bin everything into White, Black, Asian, or Other
    df['Race'] = df['Race_Raw'].apply(lambda x: x if x in ['White', 'Black', 'Asian'] else 'Other')

    # Ensure Numeric Age/EI
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['EI'] = pd.to_numeric(df['EI'], errors='coerce')

    # --- INCLUSION FILTERING ---
    # Criteria: Age >= 60 AND Valid MRI (EI not null)
    mask_included = (df['age'] >= 60) & (df['EI'].notna())
    df_inc = df[mask_included].copy()

    n_inc_pts = df_inc['pid'].nunique()
    n_inc_scans = len(df_inc)

    if n_inc_pts == 0:
        return {
            'Cohort': cohort_name, 'N participants': 0, 'N scans': 0
        }

    # --- DEMOGRAPHICS (Unique Participant Level) ---
    # Take the BASELINE visit (earliest age) for each included participant
    baseline = df_inc.sort_values('age').groupby('pid').first().reset_index()

    # Sex (Male)
    n_male = (baseline['sex_label'] == 'Male').sum()
    perc_male = (n_male / n_inc_pts) * 100 if n_inc_pts > 0 else 0

    # Race Distribution (n and % in same cell)
    race_counts = baseline['Race'].value_counts()
    race_dist = {}
    for r in ['White', 'Black', 'Asian', 'Other']:
        cnt = race_counts.get(r, 0)
        pct = (cnt / n_inc_pts) * 100 if n_inc_pts > 0 else 0
        race_dist[r] = f"{cnt} ({pct:.1f}%)"

    # --- LONGITUDINAL STATS ---
    # Follow-up Duration (Max Age - Min Age)
    if time_col not in df_inc.columns:
        fu_mean, fu_sd = 0, 0
    else:
        if pd.api.types.is_datetime64_any_dtype(df_inc[time_col]):
            fu_calc = df_inc.groupby('pid')[time_col].agg(lambda x: (x.max() - x.min()).days / 365.25)
        else:
            df_inc[time_col] = pd.to_numeric(df_inc[time_col], errors='coerce')
            fu_calc = df_inc.groupby('pid')[time_col].agg(lambda x: x.max() - x.min())
        
        fu_mean = fu_calc.mean()
        fu_sd = fu_calc.std()

    # --- COMPILE RESULTS ---
    # Arranged specifically to match the desired horizontal table structure
    return {
        'Cohort': cohort_name,
        'N participants': f"{n_inc_pts:,}",
        'N scans': f"{n_inc_scans:,}",
        'Sex (Male)': f"{n_male} ({perc_male:.1f}%)",
        'Race (White)': race_dist['White'],
        'Race (Black)': race_dist['Black'],
        'Race (Asian)': race_dist['Asian'],
        'Race (Other)': race_dist['Other'],
        'Follow-up years': f"{fu_mean:.2f} ± {fu_sd:.2f}"
    }

stats_list = []

# BLSA Setup
if 'BLSA' in locals():
    BLSA_temp = BLSA.copy()
    BLSA_temp['visitdate'] = pd.to_datetime(BLSA_temp.get('visitdate'), errors='coerce')
    BLSA_temp['birthdate'] = pd.to_datetime(BLSA_temp.get('birthdate'), errors='coerce')
    BLSA_temp['calc_age'] = (BLSA_temp['visitdate'] - BLSA_temp['birthdate']).dt.days / 365.25
    
    stats_list.append(get_stats_for_cohort(
        BLSA_temp, "BLSA", 
        'pid', 'calc_age', 'EI', 'race', map_blsa, 'visitdate'
    ))

# BIOCARD Setup
if 'biocard' in locals():
    BIO_temp = biocard.copy()
    BIO_temp['calc_age'] = pd.to_numeric(BIO_temp['VISITYEAR']) - pd.to_numeric(BIO_temp['BIRTHYEAR'])
    
    stats_list.append(get_stats_for_cohort(
        BIO_temp, "BIOCARD", 
        'JHUANONID', 'calc_age', 'EI', 'RACE', map_biocard, 'VISITYEAR'
    ))

# NACC Setup
if 'NACC' in locals():
    NACC_temp = NACC.copy()
    NACC_temp['calc_age'] = pd.to_numeric(NACC_temp['MRIYR']) - pd.to_numeric(NACC_temp['BIRTHYR'])
    
    stats_list.append(get_stats_for_cohort(
        NACC_temp, "NACC", 
        'NACCID', 'calc_age', 'EI', 'RACE', map_nacc, 'MRIYR'
    ))

# ADNI Setup
if 'ADNI' in locals():
    # ADNI uses 'Age' directly as both the age column and the time column
    stats_list.append(get_stats_for_cohort(
        ADNI, "ADNI", 
        'Subject', 'Age', 'EI', 'PTRACCAT', map_adni, 'Age'
    ))

if stats_list:
    # Convert to DataFrame and set Cohort as the Index (Rows)
    df_stats = pd.DataFrame(stats_list)
    df_final = df_stats.set_index('Cohort')
    
    
    print(" TABLE 1: STUDY DEMOGRAPHICS")
    
    
    try:
        print(df_final)
    except NameError:
        print(df_final.to_string())
    
    # Export to CSV
    df_final.to_csv("Table1_Demographics_Rows.csv")

else:
    print("Error: No cohort dataframes found (BLSA, biocard, NACC, ADNI).")
