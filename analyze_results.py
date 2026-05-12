import pandas as pd
import glob
import argparse
import numpy as np

def analyze(csv_pattern):
    files = glob.glob(csv_pattern)
    if not files:
        print(f"No files found matching {csv_pattern}")
        return

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean boolean parsing just in case
    if df['is_correct'].dtype == object:
        df['is_correct'] = df['is_correct'].astype(str).str.strip().str.upper()
        df['is_correct'] = df['is_correct'].map({'TRUE': True, 'FALSE': False})
        
    # Clean is_over_updating parsing
    if 'is_over_updating' in df.columns:
        if df['is_over_updating'].dtype == object:
            df['is_over_updating'] = df['is_over_updating'].astype(str).str.strip().str.upper()
            df['is_over_updating'] = df['is_over_updating'].map({'TRUE': True, 'FALSE': False, 'NONE': False})
            df['is_over_updating'] = df['is_over_updating'].fillna(False)
        else:
            df['is_over_updating'] = df['is_over_updating'].astype(bool)
    else:
        df['is_over_updating'] = False
    
    print("="*80)
    print("OVERALL SYSTEM PERFORMANCE (Separated by is_over_updating)")
    print("="*80)
    system_perf = df.groupby(['system', 'is_over_updating'])['is_correct'].agg(['mean', 'sum', 'count']).rename(columns={'mean': 'Accuracy', 'sum': 'Correct', 'count': 'Total'})
    system_perf['Accuracy'] = (system_perf['Accuracy'] * 100).round(2).astype(str) + '%'
    print(system_perf.to_string())
    print("\n")
    
    print("="*80)
    print("PERFORMANCE BY PROBE LOGIC (Separated by is_over_updating)")
    print("="*80)
    probe_perf = df.groupby(['system', 'is_over_updating', 'probe_logic'])['is_correct'].agg(['mean', 'sum', 'count']).rename(columns={'mean': 'Accuracy', 'sum': 'Correct', 'count': 'Total'})
    probe_perf['Accuracy'] = (probe_perf['Accuracy'] * 100).round(2).astype(str) + '%'
    print(probe_perf.to_string())
    print("\n")
    
    print("="*80)
    print("CHAIN CORRECTION VS RIPPLE EFFECT (Separated by is_over_updating)")
    print("="*80)
    # Pivot to have one row per trial_id + system + is_over_updating
    pivot = df.pivot_table(index=['system', 'is_over_updating', 'trial_id'], columns='probe_logic', values='is_correct', aggfunc='first')
    
    if 'Chain_Correction_Success' in pivot.columns and 'Ripple_Effect_Locality' in pivot.columns:
        pivot.dropna(subset=['Chain_Correction_Success', 'Ripple_Effect_Locality'], inplace=True)
        
        # Calculate boolean conditions
        pivot['Both_Correct'] = pivot['Chain_Correction_Success'] & pivot['Ripple_Effect_Locality']
        pivot['Only_Chain_Correct'] = pivot['Chain_Correction_Success'] & ~pivot['Ripple_Effect_Locality']
        pivot['Only_Ripple_Correct'] = ~pivot['Chain_Correction_Success'] & pivot['Ripple_Effect_Locality']
        pivot['Both_Wrong'] = ~pivot['Chain_Correction_Success'] & ~pivot['Ripple_Effect_Locality']
        
        stats = pivot.groupby(['system', 'is_over_updating'])[['Both_Correct', 'Only_Chain_Correct', 'Only_Ripple_Correct', 'Both_Wrong']].sum()
        stats['Total_Trials'] = pivot.groupby(['system', 'is_over_updating']).size()
        
        # RDR (Ripple Degradation Rate) = Only_Chain_Correct / (Both_Correct + Only_Chain_Correct)
        chain_correct_total = stats['Both_Correct'] + stats['Only_Chain_Correct']
        stats['RDR_%'] = (stats['Only_Chain_Correct'] / chain_correct_total * 100).round(2)
        stats['RDR_%'] = stats['RDR_%'].fillna(0).astype(str) + '%'
        
        # Add general percentages
        stats['Both_Correct_%'] = (stats['Both_Correct'] / stats['Total_Trials'] * 100).round(2).astype(str) + '%'
        stats['Ripple_Dropoff_Total_%'] = (stats['Only_Chain_Correct'] / stats['Total_Trials'] * 100).round(2).astype(str) + '%'
        
        print("Trial-Level Pairwise Analysis (Counts & Percentages):")
        print("- Both_Correct: System got both the fact and the implication right.")
        print("- RDR_% (Ripple Degradation Rate): Proportion of trials where implication was forgotten given the fact was remembered.")
        print("- Ripple_Dropoff_Total_%: Only_Chain_Correct / Total_Trials.")
        print("-" * 80)
        print(stats.to_string())
    else:
        print("Missing required probe logic columns to compare Chain vs Ripple.")
        
    print("\n" + "="*80)
    print("OVERALL DOMAIN DIFFICULTY (Mean across all systems & conditions)")
    print("="*80)
    domain_overall = (df.groupby('domain')['is_correct'].mean() * 100).round(2).sort_values()
    print(domain_overall.to_string())
    dead_topics = domain_overall[domain_overall < 50].index.tolist()
    print(f"\nIdentified 'Dead' Topics (overall accuracy < 50%): {dead_topics}")

    print("\n" + "="*80)
    print("PERFORMANCE BY DOMAIN (Separated by is_over_updating)")
    print("="*80)
    domain_perf = df.groupby(['system', 'is_over_updating', 'domain'])['is_correct'].mean().unstack().round(3) * 100
    print(domain_perf.to_string(na_rep='-'))

    print("\n" + "="*80)
    print("IMPACT OF OVER-UPDATING BY DOMAIN (Accuracy without - Accuracy with)")
    print("="*80)
    if True in domain_perf.index.get_level_values('is_over_updating') and False in domain_perf.index.get_level_values('is_over_updating'):
        delta_df = domain_perf.xs(False, level='is_over_updating') - domain_perf.xs(True, level='is_over_updating')
        print("Positive values indicate performance dropped when over-updating (noise) was introduced.")
        print("Negative values indicate performance improved with over-updating.")
        print("-" * 80)
        print(delta_df.round(2).to_string(na_rep='-'))
        
        print("\n" + "="*80)
        print("VARIANCE ACROSS DOMAINS (Separated by is_over_updating)")
        print("="*80)
        print("Higher variance means the system's performance is highly sensitive to the specific domain.")
        print("-" * 80)
        var_df = pd.DataFrame({
            'Variance': domain_perf.var(axis=1).round(2),
            'Std_Dev': domain_perf.std(axis=1).round(2),
            'Mean_Accuracy': domain_perf.mean(axis=1).round(2)
        })
        print(var_df.to_string())

        print("\n" + "="*80)
        print("REFINED VARIANCE ACROSS DOMAINS (Excluding 'Dead' Topics)")
        print("="*80)
        print(f"Excluding domains that the LLM fundamentally struggles with: {dead_topics}")
        print("This isolates the memory system's volatility from baseline LLM logic failures.")
        print("-" * 80)
        valid_domains = [d for d in domain_perf.columns if d not in dead_topics]
        refined_domain_perf = domain_perf[valid_domains]
        refined_var_df = pd.DataFrame({
            'Refined_Variance': refined_domain_perf.var(axis=1).round(2),
            'Refined_Std_Dev': refined_domain_perf.std(axis=1).round(2),
            'Refined_Mean_Accuracy': refined_domain_perf.mean(axis=1).round(2)
        })
        print(refined_var_df.to_string())

        print("\n" + "="*80)
        print("WITHIN-DOMAIN VARIANCE (Separated by is_over_updating)")
        print("="*80)
        print("This shows how erratic the performance is within each specific domain.")
        print("Variance of 0 means the system was consistently right (or consistently wrong) on all trials in that domain.")
        print("Higher variance (up to ~0.25 for binary data) means the system was inconsistent (random guessing / noise).")
        print("-" * 80)
        domain_var = df.groupby(['system', 'is_over_updating', 'domain'])['is_correct'].var().unstack().round(3)
        print(domain_var.to_string(na_rep='-'))
        
        print("\n" + "="*80)
        print("AVERAGE WITHIN-DOMAIN NOISE (Mean variance across all domains)")
        print("="*80)
        print("This single number summarizes the average 'noisiness' across all domains.")
        print("Higher means more random guessing overall.")
        print("-" * 80)
        mean_noise = domain_var.mean(axis=1).round(3).to_frame("Mean_Variance_Noise")
        print(mean_noise.to_string())

    else:
        print("Could not calculate over-updating deltas (missing True/False data).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze memory evaluation results.")
    parser.add_argument("--pattern", default="revised_extreme/*.csv", help="Glob pattern for CSV files (default: revised_extreme/*.csv)")
    args = parser.parse_args()
    analyze(args.pattern)
