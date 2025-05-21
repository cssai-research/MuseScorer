import numpy as np
import pandas as pd
import os
import re
from scipy import stats
from copy import deepcopy
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import krippendorff
from scipy.stats import linregress
from typing import List, Tuple
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, v_measure_score
from sklearn.metrics import homogeneity_score, completeness_score
from tqdm.notebook import tqdm
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, v_measure_score
from sklearn.utils import resample
import numpy as np
import pandas as pd
import powerlaw
import platform

# for CQ
# from symspellpy.symspellpy import SymSpell, Verbosity
# import copy
# import string
# import nltk
# from nltk import download
# from nltk.corpus import stopwords, wordnet as wn
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
# download('stopwords')
# download('punkt')  
# stop_words = stopwords.words('english')
# from collections import defaultdict
# import math
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import minimum_spanning_tree


def load_csv_data(folder_name, file_name):
    csv_path = os.path.join(folder_name, file_name+".csv")
    return pd.read_csv(csv_path, encoding='utf-8-sig')#, encoding='utf-8-sig' latin-1


if platform.system() == "Windows":
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.0"


def compute_icc(df, print_=False, full_output=False):
    """
    Compute ICC(3,1) and ICC(3,k) from a pandas DataFrame using R's `psych::ICC` function.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame of ratings (e.g., rows = subjects, columns = raters)
        print_ (bool): Whether to print the full ICC table
        full_output (bool): Whether to return the full ICC results table

    Returns:
        dict: A dictionary with ICC(3,1) and ICC(3,k) values, CIs, p-values, and F-statistics.
              If full_output=True, also includes the full table as 'full_table' (pandas.DataFrame)
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    psych = importr('psych')

    # Convert pandas DataFrame to R
    r_df = pandas2ri.py2rpy(df)

    # Run the ICC analysis in R
    icc_result = psych.ICC(r_df)
    icc_table = icc_result.rx2('results')

    # Convert to pandas
    icc_df = pandas2ri.rpy2py(icc_table)

    if print_:
        print("\nFull ICC Table:\n", icc_df)

    # Extract ICC(3,1) and ICC(3,k)
    icc_3_1_row = icc_df[icc_df['type'] == 'ICC3'].iloc[0]
    icc_3_k_row = icc_df[icc_df['type'] == 'ICC3k'].iloc[0]

    result = {
        'ICC(3,1)': icc_3_1_row['ICC'],
        'CI(3,1)': (icc_3_1_row['lower bound'], icc_3_1_row['upper bound']),
        'F(3,1)': icc_3_1_row['F'],
        'df(3,1)': (icc_3_1_row['df1'], icc_3_1_row['df2']),
        'p(3,1)': icc_3_1_row['p'],

        'ICC(3,k)': icc_3_k_row['ICC'],
        'CI(3,k)': (icc_3_k_row['lower bound'], icc_3_k_row['upper bound']),
        'F(3,k)': icc_3_k_row['F'],
        'df(3,k)': (icc_3_k_row['df1'], icc_3_k_row['df2']),
        'p(3,k)': icc_3_k_row['p']
    }

    if full_output:
        result['full_table'] = icc_df

    return result

def format_icc3k_result(stats: dict) -> str:
    icc = stats['ICC(3,k)']
    ci_low, ci_high = stats['CI(3,k)']
    f_val = stats['F(3,k)']
    df1, df2 = stats['df(3,k)']
    p_val = stats['p(3,k)']
    
    p_formatted = "p < .001" if p_val < 0.001 else f"p = {p_val:.3f}"

    return (
        f"ICC(3,k) = {icc:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
        f"F({df1:.0f}, {df2:.0f}) = {f_val:.2f}, {p_formatted}"
    )

def print_ICC(df):
    result = compute_icc(df, print_=False, full_output=False)
    print(format_icc3k_result(result))

def score_participants(df, annotator, threshold=1):    
    annotations_dict = {}
    fluency_dict = {}
    nonredun_dict = {}
    shap_dict = {}
    rarity_dict = {}
    orig_dict = {}
    norm_nonredun_dict = {}
    norm_shap_dict = {}
    norm_rarity_dict = {}
    norm_orig_dict = {}
    df2 = df[['for_user_id','round_id',annotator]]
    round_ids = sorted(df['round_id'].unique())
    
    # construct annotations dict
    for index, row in df2.iterrows():
        user_id = row["for_user_id"]
        round_id = row["round_id"]
        annotation = row[annotator]

        if user_id not in annotations_dict: 
            annotations_dict[user_id]={k:set() for k in round_ids} 
            fluency_dict[user_id]={k:0 for k in round_ids} 
        annotations_dict[user_id][round_id].add(annotation)
        fluency_dict[user_id][round_id]+=1
    
    # construct tallies
    tally_dict = {k:{} for k in round_ids}  # separate dicts for rounds
    for user_id, user_dict in annotations_dict.items():
        for round_id, round_set in user_dict.items():
            for idea_id in list(round_set):
                if idea_id not in tally_dict[round_id]:
                    tally_dict[round_id][idea_id] = 0
                tally_dict[round_id][idea_id] += 1

    num_participants = len(annotations_dict)

    # shap_score_dict and rarity_score_dict
    shap_score_dict = {r:{} for r in round_ids}
    rarity_score_dict = {r:{} for r in round_ids}
    orig_score_dict = {r:{} for r in round_ids}

    for round_id, round_dict in tally_dict.items():
        for idea_id, count in round_dict.items():
            # Shap-style 1/n
            shap_score_dict[round_id][idea_id] = 1 / count

            # Rarity: 1 - (count / N)
            rarity_score_dict[round_id][idea_id] = 1 - (count / num_participants)

            # Originality points
            freq = count / num_participants
            if freq > 0.10:
                orig_score = 0
            elif freq > 0.03:
                orig_score = 1
            elif freq > 0.01:
                orig_score = 2
            else:
                orig_score = 3
            orig_score_dict[round_id][idea_id] = orig_score

    # scoring participants
    for user_id in annotations_dict.keys():
        nonredun_dict[user_id] = {r:0 for r in round_ids}
        shap_dict[user_id] = {r:0 for r in round_ids}
        rarity_dict[user_id] = {r:0 for r in round_ids}
        orig_dict[user_id] = {r:0 for r in round_ids}

        # fluency_dict[user_id]={r:len(annotations_dict[user_id][r]) for r in round_ids}

        for round_id in round_ids:
            for idea_id in annotations_dict[user_id][round_id]:
                count = tally_dict[round_id][idea_id]
                if count <= threshold:
                    nonredun_dict[user_id][round_id] += 1
                shap_dict[user_id][round_id] += shap_score_dict[round_id][idea_id]
                rarity_dict[user_id][round_id] += rarity_score_dict[round_id][idea_id]
                orig_dict[user_id][round_id] += orig_score_dict[round_id][idea_id]

        # get normalized scores
        norm_nonredun_dict[user_id] = {r:nonredun_dict[user_id][r]/fluency_dict[user_id][r] for r in round_ids}
        norm_shap_dict[user_id] = {r:shap_dict[user_id][r]/fluency_dict[user_id][r] for r in round_ids}
        norm_rarity_dict[user_id] = {r:rarity_dict[user_id][r]/fluency_dict[user_id][r] for r in round_ids}
        norm_orig_dict[user_id] = {r:orig_dict[user_id][r]/fluency_dict[user_id][r] for r in round_ids}
        
        # total score
        nonredun_dict[user_id]['total'] = sum(nonredun_dict[user_id][r] for r in round_ids)
        shap_dict[user_id]['total'] = sum(shap_dict[user_id][r] for r in round_ids)
        rarity_dict[user_id]['total'] = sum(rarity_dict[user_id][r] for r in round_ids)
        orig_dict[user_id]['total'] = sum(orig_dict[user_id][r] for r in round_ids)
        fluency_dict[user_id]['total'] = sum(fluency_dict[user_id][r] for r in round_ids)
        norm_nonredun_dict[user_id]['total'] = sum(norm_nonredun_dict[user_id][r] for r in round_ids)
        norm_shap_dict[user_id]['total'] = sum(norm_shap_dict[user_id][r] for r in round_ids)
        norm_rarity_dict[user_id]['total'] = sum(norm_rarity_dict[user_id][r] for r in round_ids)
        norm_orig_dict[user_id]['total'] = sum(norm_orig_dict[user_id][r] for r in round_ids)

    return nonredun_dict, shap_dict, rarity_dict, orig_dict, fluency_dict, norm_nonredun_dict, norm_shap_dict, norm_rarity_dict,norm_orig_dict

def generate_lists(dict1,dict2):
    list1 = []
    list2 = []
    for k,v in dict1.items():
        list1.append(v['total'])
        list2.append(dict2[k]['total'])
    return list1,list2

def pearson_corr_with_ci(list1, list2, alpha=0.05):
    """
    Compute Pearson correlation with 95% CI and p-value using scipy.stats.

    Parameters:
        list1, list2 (list or np.array): Input vectors
        alpha (float): Significance level (default 0.05 for 95% CI)

    Returns:
        dict: r, p-value, CI lower/upper, n
    """
    result = stats.pearsonr(list1, list2)
    ci = result.confidence_interval(confidence_level=1 - alpha)
    return {
        "r": result.statistic,
        "p": result.pvalue,
        "CI_lower": ci.low,
        "CI_upper": ci.high,
        "n": len(list1)
    }


def print_pearson_result(result):
    print(f"Pearson's r = {result['r']:.3f}, "
          f"95% CI = [{result['CI_lower']:.3f}, {result['CI_upper']:.3f}], "
          f"p = {result['p']:.5f}, n = {result['n']}")

def spearman_corr_with_ci(list1, list2, alpha=0.05):
    """
    Compute Spearman correlation with 95% CI (via Fisher z-transform approximation) and p-value.

    Parameters:
        list1, list2 (list or np.array): Input vectors
        alpha (float): Significance level (default 0.05 for 95% CI)

    Returns:
        dict: rho, p-value, CI lower/upper, n
    """
    # Calculate Spearman correlation and p-value
    res = stats.spearmanr(list1, list2)
    rho = res.statistic
    pval = res.pvalue
    n = len(list1)

    # Fisher z-transformation for CI
    if abs(rho) == 1 or n <= 3:
        ci_lower, ci_upper = np.nan, np.nan
    else:
        stderr = 1.0 / np.sqrt(n - 3)
        z = np.arctanh(rho)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = z - z_crit * stderr, z + z_crit * stderr
        ci_lower, ci_upper = np.tanh([lo_z, hi_z])

    return {
        "rho": rho,
        "p": pval,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "n": n
    }

def print_spearman_result(result):
    print(f"Spearman's rho = {result['rho']:.3f}, "
          f"95% CI = [{result['CI_lower']:.3f}, {result['CI_upper']:.3f}], "
          f"p = {result['p']:.5f}, n = {result['n']}")


def kendall_corr_with_ci(list1, list2, alpha=0.05):
    """
    Compute Kendall's tau with 95% CI (via Fisher z approximation) and p-value.

    Parameters:
        list1, list2 (list or np.array): Input vectors
        alpha (float): Significance level (default 0.05 for 95% CI)

    Returns:
        dict: tau, p-value, CI lower/upper, n
    """
    res = stats.kendalltau(list1, list2)
    tau = res.statistic
    pval = res.pvalue
    n = len(list1)

    # Approximate CI using Fisher z transformation (only valid for large n)
    if abs(tau) == 1 or n <= 10:
        ci_lower, ci_upper = np.nan, np.nan
    else:
        stderr = np.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
        z = np.arctanh(tau)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = z - z_crit * stderr, z + z_crit * stderr
        ci_lower, ci_upper = np.tanh([lo_z, hi_z])

    return {
        "tau": tau,
        "p": pval,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "n": n
    }


def print_kendall_result(result):
    print(f"Kendall's tau = {result['tau']:.3f}, "
          f"95% CI = [{result['CI_lower']:.3f}, {result['CI_upper']:.3f}], "
          f"p = {result['p']:.5f}, n = {result['n']}")


def print_correlations2(list1,list2,corr='pearson'):
    if corr=='pearson':
        result = pearson_corr_with_ci(list1, list2)
        print_pearson_result(result)
 
    elif corr=='spearman':
        result = spearman_corr_with_ci(list1, list2)
        print_spearman_result(result)

    elif corr=='kendall':
        result = kendall_corr_with_ci(list1, list2)
        print_kendall_result(result)

def print_correlations(dict1,dict2,corr='pearson'):
    list1,list2 = generate_lists(dict1,dict2)
    if corr=='pearson':
        result = pearson_corr_with_ci(list1, list2)
        print_pearson_result(result)
 
    elif corr=='spearman':
        result = spearman_corr_with_ci(list1, list2)
        print_spearman_result(result)

    elif corr=='kendall':
        result = kendall_corr_with_ci(list1, list2)
        print_kendall_result(result)

def get_correlations(dict1,dict2,corr='pearson'):
    list1,list2 = generate_lists(dict1,dict2)
    if corr=='pearson':
        return pearson_corr_with_ci(list1,list2)
    elif corr=='spearman':
        return spearman_corr_with_ci(list1,list2)
    elif corr=='kendall':
        return kendall_corr_with_ci(list1,list2)


def get_ann_score_dicts(df,annotators,threshold=1):
    ann_score_dicts = {ann:{} for ann in annotators}
    for ann in annotators:
        nonredun_dict, shap_dict, rarity_dict, orig_dict, fluency_dict, norm_nonredun_dict, norm_shap_dict, norm_rarity_dict, norm_orig_dict = score_participants(df,ann,threshold)
        ann_score_dicts[ann]['nonredun'] = deepcopy(nonredun_dict)
        ann_score_dicts[ann]['shap'] = deepcopy(shap_dict)
        ann_score_dicts[ann]['rarity'] = deepcopy(rarity_dict)
        ann_score_dicts[ann]['orig'] = deepcopy(orig_dict)
        ann_score_dicts[ann]['fluency'] = deepcopy(fluency_dict)
        ann_score_dicts[ann]['norm_nonredun'] = deepcopy(norm_nonredun_dict)
        ann_score_dicts[ann]['norm_shap'] = deepcopy(norm_shap_dict)
        ann_score_dicts[ann]['norm_rarity'] = deepcopy(norm_rarity_dict)
        ann_score_dicts[ann]['norm_orig'] = deepcopy(norm_orig_dict)
    return ann_score_dicts


def generate_all_lists(ann_score_dicts,scoring_method,round_key='total'):
    ann_list_dict = {ann:[] for ann in ann_score_dicts.keys()}
    participant_ids = set(ann_score_dicts[list(ann_score_dicts.keys())[0]][scoring_method].keys())
    
    for p in participant_ids: # make sure the same participant is in the same index for every annotator
        for ann,ann_dict in ann_score_dicts.items(): # loop through each annotator
            ann_list_dict[ann].append(ann_dict[scoring_method][p][round_key])
    return ann_list_dict



def calculate_correlations(df,annotators,threshold=1):
    ann_score_dicts = get_ann_score_dicts(df,annotators,threshold)

    for scoring_method in ['nonredun','shap','rarity','orig','fluency','norm_nonredun','norm_shap','norm_rarity','norm_orig']:
        print(f"Generating correlations for the scoring method {scoring_method}")
        pairs = list(itertools.combinations(annotators, 2))
        for ann1,ann2 in pairs:
            for corr_method in ['pearson','spearman','kendall']:
                print(f"{corr_method} correlation between {ann1} and {ann2}:") 
                print_correlations(ann_score_dicts[ann1][scoring_method],ann_score_dicts[ann2][scoring_method],corr_method)
                print("")
        print("\n")



def generate_icc_dataframe_allrounds(df,annotators):
    ann_score_dicts = get_ann_score_dicts(df,annotators,threshold=1)
    scoring_methods = ['nonredun','shap','rarity','orig','fluency','norm_nonredun','norm_shap','norm_rarity','norm_orig']
    temp_corr_dataframes = {s:[] for s in scoring_methods}
    corr_dataframes = {s:[] for s in scoring_methods}
    participant_ids = list(set(ann_score_dicts[list(ann_score_dicts.keys())[0]][scoring_methods[0]].keys()))
    round_ids = sorted(df['round_id'].unique())
    for scoring_method in scoring_methods:
        for round_key in round_ids:
            ann_list_dict = generate_all_lists_allrounds(ann_score_dicts,scoring_method,round_key,participant_ids)
            temp_corr_dataframes[scoring_method].append(pd.DataFrame(ann_list_dict))

    for scoring_method in scoring_methods:
        dfs_with_round = [df.assign(round_id=i+1, for_user_id=participant_ids) for i, df in enumerate(temp_corr_dataframes[scoring_method])]
        corr_dataframes[scoring_method] = pd.concat(dfs_with_round, ignore_index=True)    

    return corr_dataframes


def generate_icc_dataframe(df,annotators):
    corr_dataframes_allrounds = generate_icc_dataframe_allrounds(df,annotators)
    corr_dataframes = {}

    for k,v in corr_dataframes_allrounds.items():
        corr_dataframes[k] = v.groupby('for_user_id')[annotators].sum().reset_index()[annotators]

    return corr_dataframes


def generate_all_lists_allrounds(ann_score_dicts,scoring_method,round_key,participant_ids):
    ann_list_dict = {ann:[] for ann in ann_score_dicts.keys()}

    for p in participant_ids: # make sure the same participant is in the same index for every annotator
        for ann,ann_dict in ann_score_dicts.items(): # loop through each annotator
            ann_list_dict[ann].append(ann_dict[scoring_method][p][round_key])
    return ann_list_dict


def summarize_mean_clustering_agreement(df, annotator1='ali', annotator2='krish', alpha=0.05):
    """
    Compute the mean and CI across the 5 rounds for AMI, NMI, V-measure, Homogeneity, and Completeness.

    This version uses the 5 rounds as the population (no bootstrapping).
    """
    metrics = {
        'AMI': [],
        'NMI': [],
        'V-measure': [],
        'Homogeneity': [],
        'Completeness': [],
    }

    for round_id in sorted(df['round_id'].unique()):
        df_round = df[df['round_id'] == round_id]
        labels1 = df_round[annotator1].astype(str).values
        labels2 = df_round[annotator2].astype(str).values

        metrics['AMI'].append(adjusted_mutual_info_score(labels1, labels2))
        metrics['NMI'].append(normalized_mutual_info_score(labels1, labels2))
        metrics['V-measure'].append(v_measure_score(labels1, labels2))
        metrics['Homogeneity'].append(homogeneity_score(labels1, labels2))
        metrics['Completeness'].append(completeness_score(labels1, labels2))

    # Compute mean and 95% CI using t-distribution
    summary = []
    n_rounds = len(df['round_id'].unique())
    for metric_name, values in metrics.items():
        values = np.array(values)
        mean = values.mean()
        std_err = values.std(ddof=1) / np.sqrt(n_rounds)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n_rounds - 1)
        ci_lower = mean - t_crit * std_err
        ci_upper = mean + t_crit * std_err
        summary.append({
            'Metric': metric_name,
            'Mean': mean,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'n_rounds': n_rounds
        })

    return pd.DataFrame(summary)



def analyze_power_law_fit_per_round(df, annotators=['ali', 'krish']):
    """
    Analyze power-law fit for each annotator separately, round by round.
    Each round is treated as an independent sample of idea bucket frequencies.
    Returns a nested dictionary with results per annotator and per round.
    """
    results = {}

    for annotator in annotators:
        annotator_results = {}
        for round_id in sorted(df['round_id'].unique()):
            df_round = df[df['round_id'] == round_id]
            bucket_counts = df_round[annotator].value_counts().values

            # Fit power-law model to bucket frequencies
            try:
                fit = powerlaw.Fit(bucket_counts, discrete=True, verbose=False)
                R, p = fit.distribution_compare('power_law', 'lognormal')

                annotator_results[round_id] = {
                    'alpha': fit.alpha,
                    'xmin': fit.xmin,
                    'vs_lognormal_R': R,
                    'vs_lognormal_p': p,
                    'powerlaw_significantly_better': (R > 0 and p < 0.05),
                    'n_buckets': len(bucket_counts)
                }
            except Exception as e:
                annotator_results[round_id] = {'error': str(e)}

        results[annotator] = annotator_results

    return results


def summarize_bucket_counts_with_powerlaw(results_dict):
    """
    Extend power-law summary with average number of distinct buckets and its 95% CI per annotator.
    """
    summary = []

    for annotator, round_results in results_dict.items():
        valid_rounds = [r for r in round_results.values() if 'error' not in r]

        alphas = np.array([r['alpha'] for r in valid_rounds])
        n_buckets = np.array([r['n_buckets'] for r in valid_rounds])
        powerlaw_success = [r['powerlaw_significantly_better'] for r in valid_rounds]
        n = len(n_buckets)

        # Mean and CI for alpha
        mean_alpha = np.mean(alphas)
        se_alpha = np.std(alphas, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_alpha = (mean_alpha - t_crit * se_alpha, mean_alpha + t_crit * se_alpha)

        # Mean and CI for bucket counts
        mean_buckets = np.mean(n_buckets)
        se_buckets = np.std(n_buckets, ddof=1) / np.sqrt(n)
        ci_buckets = (mean_buckets - t_crit * se_buckets, mean_buckets + t_crit * se_buckets)

        summary.append({
            'annotator': annotator,
            'mean_alpha': mean_alpha,
            'alpha_CI_lower': ci_alpha[0],
            'alpha_CI_upper': ci_alpha[1],
            'prop_significant_powerlaw': np.mean(powerlaw_success),
            'mean_n_buckets': mean_buckets,
            'n_buckets_CI_lower': ci_buckets[0],
            'n_buckets_CI_upper': ci_buckets[1],
            'n_rounds': n
        })

    return pd.DataFrame(summary)

def get_llm_colname(model_, embedding_, prompting_, k_):
    outstr = ""
    if model_=="llama":
        outstr+="llama33"
    elif model_=="qwen":
        outstr+="qwen3"
    elif model_=="phi":
        outstr+="phi4"

    outstr+= f"k{k_}"

    if prompting_=="baseline":
        outstr += "BSL"
    elif prompting_ == "cot":
        outstr += "CoT"

    if embedding_ == "mpnet":
        outstr += "r1"
    elif embedding_ == "bge":
        outstr += "r1001"
    elif embedding_ == "mxbai":
        outstr += "r2001"
    elif embedding_ == "e5":
        outstr += "r3001"
    
    return outstr    


def get_clustering_colname(model_, embedding_):
    outstr = model_+"_"
    if embedding_ == "mpnet":
        outstr += "r1"
    elif embedding_ == "bge":
        outstr += "r1001"
    elif embedding_ == "mxbai":
        outstr += "r2001"
    elif embedding_ == "e5":
        outstr += "r3001"
    return outstr



def plot_grouped_bar_with_ci(data_dict, baseline, ylabel='Score', title='Grouped Bar Plot with CI',
                              figsize=(12, 4), ymin=0.4, ymax=None, show_values=True):
    """
    Plots a grouped barplot with asymmetric confidence intervals and a baseline.

    Parameters:
    - data_dict: nested dict: {embedding: {llama: {"mean", "ci_lower", "ci_upper"}}}
    - baseline: dict with keys: "mean", "ci_lower", "ci_upper"
    - ylabel: y-axis label
    - title: plot title
    - figsize: figure size
    - ymin, ymax: optional y-axis limits
    - show_values: whether to print mean values above bars
    """
    groups = list(data_dict.keys())
    conditions = list(next(iter(data_dict.values())).keys())
    n_groups = len(groups)
    n_conditions = len(conditions)

    means = [[data_dict[group][cond]["mean"] for cond in conditions] for group in groups]
    lower_CIs = [[data_dict[group][cond]["ci_lower"] for cond in conditions] for group in groups]
    upper_CIs = [[data_dict[group][cond]["ci_upper"] for cond in conditions] for group in groups]

    x_groups = np.arange(n_groups)
    width = 0.8 / n_conditions
    offset = 1.0  # leave space for baseline

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline
    baseline_x = 0
    baseline_mean = baseline['mean']
    baseline_err = [[baseline_mean - baseline['ci_lower']], [baseline['ci_upper'] - baseline_mean]]
    bar = ax.bar(baseline_x, baseline_mean, width=0.6, yerr=baseline_err, capsize=5, label="Baseline", color='gray')
    if show_values:
        ax.text(baseline_x, baseline_mean + 0.04, f"{baseline_mean:.3f}", ha='center', va='bottom', fontsize=9)

    # Grouped bars
    for i, cond in enumerate(conditions):
        mean_vals = [m[i] for m in means]
        err_low = [m[i] - l[i] for m, l in zip(means, lower_CIs)]
        err_up = [u[i] - m[i] for m, u in zip(means, upper_CIs)]
        bar_x = x_groups + offset + i * width
        bars = ax.bar(bar_x, mean_vals, width, yerr=[err_low, err_up], capsize=5, label=cond)
        
        if show_values:
            for x, y in zip(bar_x, mean_vals):
                ax.text(x, y + 0.02, f"{y:.3f}", ha='center', va='bottom', fontsize=9)

    # Axis and legend
    all_xticks = [baseline_x] + list(x_groups + offset + width * (n_conditions - 1) / 2)
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(["Baseline"] + groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


def wrapper_icc_3_1(df,ann1,ann2,scoring_method):
    corr_dataframes_llm = generate_icc_dataframe(df,[ann1,ann2])
    rarity_dataframe_llm = corr_dataframes_llm[scoring_method]
    icc_output = compute_icc(rarity_dataframe_llm, print_=False)
    return {"mean": icc_output['ICC(3,1)'], "ci_lower": icc_output['CI(3,1)'][0], "ci_upper": icc_output['CI(3,1)'][1]}

def wrapper_icc_3_k(df,ann1,ann2,scoring_method):
    corr_dataframes_llm = generate_icc_dataframe(df,[ann1,ann2])
    rarity_dataframe_llm = corr_dataframes_llm[scoring_method]
    icc_output = compute_icc(rarity_dataframe_llm, print_=False)
    return {"mean": icc_output['ICC(3,k)'], "ci_lower": icc_output['CI(3,k)'][0], "ci_upper": icc_output['CI(3,k)'][1]}

def wrapper_pearson(df,ann1,ann2,scoring_method):
    corr_dataframes_llm = generate_icc_dataframe(df,[ann1,ann2])
    rarity_dataframe_llm = corr_dataframes_llm[scoring_method]
    result_llm = pearson_corr_with_ci(rarity_dataframe_llm[ann1], rarity_dataframe_llm[ann2])
    return {"mean": result_llm['r'], "ci_lower": result_llm['CI_lower'], "ci_upper": result_llm['CI_upper']}

def wrapper_spearman(df,ann1,ann2,scoring_method):
    corr_dataframes_llm = generate_icc_dataframe(df,[ann1,ann2])
    rarity_dataframe_llm = corr_dataframes_llm[scoring_method]
    result_llm = spearman_corr_with_ci(rarity_dataframe_llm[ann1], rarity_dataframe_llm[ann2])
    return {"mean": result_llm['rho'], "ci_lower": result_llm['CI_lower'], "ci_upper": result_llm['CI_upper']}



def wrapper_clustering(df,ann1,ann2,eval_):
    summary_df = summarize_mean_clustering_agreement(df, annotator1=ann1, annotator2=ann2)
    row = summary_df[summary_df['Metric'] == eval_]
    row = row.iloc[0]
    return {
        'mean': row['Mean'],
        'ci_lower': row['CI_lower'],
        'ci_upper': row['CI_upper']
    }


def wrapper_scorer(df,ann1,ann2,eval_='pearson',scoring_method='norm_orig'):
    if eval_=="ICC(3,1)":
        return wrapper_icc_3_1(df,ann1,ann2,scoring_method)
    elif eval_=="ICC(3,k)":
        return wrapper_icc_3_k(df,ann1,ann2,scoring_method)
    elif eval_=="pearson":
        return wrapper_pearson(df,ann1,ann2,scoring_method)
    elif eval_=="spearman":
        return wrapper_spearman(df,ann1,ann2,scoring_method)
    else: #it has to be clustering then
        return wrapper_clustering(df,ann1,ann2,eval_)


def get_grouped_plots_sweep_embedding(df,primary_h1, secondary_h2, eval_, prompting_,comparison_group='llm',scoring_method='norm_orig'):
    # generate baseline
    baseline = wrapper_scorer(df,primary_h1,secondary_h2,eval_,scoring_method)
    # generate datadict
    data_dict = {}
    if comparison_group == 'llm':
        models = ['llama','qwen','phi']
    elif comparison_group == 'clustering':
        models = ['kmm_sil','kmm_sem','agg_sil','agg_sem']
    for embedding_ in ['mpnet','bge','mxbai','e5']:
        data_dict[embedding_]={}
        for model_ in models:
            if prompting_=='baseline' and model_=='qwen':
                continue
            current_llm_colname = get_llm_colname(model_, embedding_, prompting_, 10) if comparison_group == 'llm' else get_clustering_colname(model_, embedding_)
            data_dict[embedding_][model_] = wrapper_scorer(df,primary_h1,current_llm_colname,eval_,scoring_method)
    plot_grouped_bar_with_ci(data_dict, baseline, ylabel=eval_, title=f"{eval_} performance against annotator H1, {prompting_} prompting")


def get_grouped_plots_sweep_k(df,primary_h1, secondary_h2, eval_, prompting_,comparison_group='llm',scoring_method='norm_rarity'):
    # generate baseline
    baseline = wrapper_scorer(df,primary_h1,secondary_h2,eval_,scoring_method)
    embedding_ = 'mxbai'
    # generate datadict
    data_dict = {}
    if comparison_group == 'llm':
        models = ['phi'] #'llama','qwen','phi'
    elif comparison_group == 'clustering':
        models = ['kmm_sil','kmm_sem','agg_sil','agg_sem']
    
    for k in [5,10,15,20,30,40,50,75,100]: #'mpnet','bge','mxbai','e5'
        data_dict[k]={}
        for model_ in models:
            if prompting_=='baseline' and model_=='qwen':
                continue
            current_llm_colname = get_llm_colname(model_, embedding_, prompting_, k) if comparison_group == 'llm' else get_clustering_colname(model_, embedding_)
            data_dict[k][model_] = wrapper_scorer(df,primary_h1,current_llm_colname,eval_,scoring_method)
    plot_grouped_bar_with_ci(data_dict, baseline, ylabel=eval_, title=f"{eval_} performance against annotator H1, {prompting_} prompting")


def plot_grouped_bar_with_ci2(data_dict, ylabel,group_gap=1,title='Grouped Bar Plot with CI',
                              figsize=(12, 4), ymin=None, ymax=None, show_values=True,save_fig=False):
    """
    Plots a grouped barplot with mean and confidence intervals, with separation between groups.

    Args:
        data_dict (dict): Nested dictionary in the form:
            {
                'group1': {'label1': {'mean': float, 'ci_lower': float, 'ci_upper': float}, ...},
                'group2': ...
            }
        group_gap (float): Gap (in bar units) between groups.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Assign consistent colors to shared keys
    all_keys = {k for group in data_dict.values() for k in group}
    color_map = {}
    color_palette = plt.get_cmap('tab10')
    for i, key in enumerate(sorted(all_keys)):
        color_map[key] = color_palette(i)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = 0
    x_ticks = []
    x_tick_labels = []
    added_labels = set()

    for group, items in data_dict.items():
        group_keys = list(items.keys())
        for i, key in enumerate(group_keys):
            stats = items[key]
            mean = stats['mean']
            ci_lower = stats['ci_lower']
            ci_upper = stats['ci_upper']
            ci = [[mean - ci_lower], [ci_upper - mean]]

            label = key if key not in added_labels else ""
            ax.bar(x, mean, yerr=ci, capsize=5, color=color_map[key], label=label)
            ax.text(x, ci_upper + 0.02, f"{mean:.3f}", ha='center', va='bottom', fontsize=9)

            x_ticks.append(x)
            x_tick_labels.append(f"{group}\n{key}")
            x += 1

        x += group_gap  # Add gap between groups
        added_labels.update(group_keys)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    ax.set_ylabel("Mean Value")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Model")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'Figures/grouped_plots_bucket_level_{title}.pdf', format='pdf', dpi=300)
    
    plt.show()



def get_grouped_plots_bucket_level(df,primary_h1, eval_='AMI',embedding_='e5',save_fig=True):
    # generate baseline

    data_dict = {}
    # First, do the llms
    models_dict = {'llm':['llama','qwen','phi'], 'clustering':['kmm_sil','kmm_sem','agg_sil','agg_sem']}
    
#     data_dict['Human ann'] = {}
#     data_dict['Human ann']['H2'] = wrapper_scorer(df,primary_h1,'krish',eval_)
    
    for prompting_ in ['cot','baseline']:
        data_dict[prompting_] = {}
        for model_ in models_dict['llm']:
            if prompting_=='baseline' and model_=='qwen':
                continue
            current_llm_colname = get_llm_colname(model_, embedding_, prompting_, 10) 
            data_dict[prompting_][model_] = wrapper_scorer(df,primary_h1,current_llm_colname,eval_)
            
    # now do clustering baselines
    data_dict['clustering'] = {}
    for model_ in models_dict['clustering']:
        current_llm_colname = get_clustering_colname(model_, embedding_)
        data_dict['clustering'][model_] = wrapper_scorer(df,primary_h1,current_llm_colname,eval_)
   
    plot_grouped_bar_with_ci2(data_dict, ylabel=eval_, title=f"{eval_} performance against annotator H1, {embedding_} embedding",save_fig=save_fig)
    return data_dict



def bland_altman_plot_analysis(data1, data2, judge1_name='Judge1', judge2_name='Judge2',save_fig=False):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)  # mean difference (bias)
    sd = np.std(diff, axis=0)  # standard deviation of differences
    
    # Calculate limits of agreement
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd
    
    # 1. Is bias close to zero? (threshold: |bias| < 0.1 * SD considered close)
    bias_threshold =  1.96 * sd #0.1 * sd
    bias_close_to_zero = np.abs(md) < bias_threshold
    
    # 2. Are most points within limits of agreement? (threshold: >95% inside limits)
    n_points = len(diff)
    n_within_limits = np.sum((diff > loa_lower) & (diff < loa_upper))
    perc_within_limits = n_within_limits / n_points * 100
    points_within_limits_good = perc_within_limits >= 95
    
    # 3. Proportional bias: Regression of difference vs mean
    slope, intercept, r_value, p_value, std_err = linregress(mean, diff)
    proportional_bias = p_value < 0.05  # Significant slope implies proportional bias
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.scatter(mean, diff, s=40,color='purple')
    plt.axhline(md, color='red', linestyle='--', label=f'Mean diff (bias) = {md:.2f}')
    plt.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
    plt.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')
    
#     Regression line (for proportional bias visualization)
    # plt.plot(mean, intercept + slope*mean, color='green', linestyle=':', label=f'Slope = {slope:.2f} (p={p_value:.3f})')

#     plt.title(f'Bland-Altman Plot: {judge1_name} vs {judge2_name}')
    plt.xlabel('Mean of two judges')
    plt.ylabel('Difference between two judges')
    plt.legend()
    plt.grid(True)
    
    # Text box summary
    summary_text = (
        f"Bias close to zero? {'✅ Yes' if bias_close_to_zero else '❌ No'} (Bias={md:.2f})\n"
        f">95% within LoA? {'✅ Yes' if points_within_limits_good else '❌ No'} ({perc_within_limits:.1f}%)\n"
        f"Proportional bias? {'❌ No' if not proportional_bias else '✅ Yes'} (p={p_value:.3f})"
    )
    plt.gcf().text(0.65, 0.25, summary_text, fontsize=11, bbox=dict(facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'Figures/bland_altman.pdf', format='pdf', dpi=300)
    plt.show()



def bland_altman_plot_analysis_wrapper(dataframe,annotators, save_fig=False):
    pairs = list(itertools.combinations(annotators, 2))
    for ann1,ann2 in pairs:
        bland_altman_plot_analysis(dataframe[ann1],dataframe[ann2],ann1,ann2,save_fig)



def judge_pairplots(df_judges):
    sns.pairplot(df_judges, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.suptitle('Pairwise Judge Score Comparisons', y=1.02)
    plt.show()


def plot_loglog_scatter(df, annotator, save_fig=False):

    round_ids = sorted(df['round_id'].unique())
    size_distributions = []

    for r in round_ids:
        round_df = df[df['round_id'] == r]
        bucket_sizes = round_df[annotator].value_counts().values
        size_distributions.append(bucket_sizes)

    # Concatenate all size arrays to form a combined distribution
    bucket_counts = np.concatenate(size_distributions)
    
    # Get raw frequency distribution
    unique, counts = np.unique(bucket_counts, return_counts=True)
    prob = counts / counts.sum()

    # Plot raw data as scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(unique, prob, color='purple', s=15, alpha=0.90)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Bucket size $m_k$')
    plt.ylabel(r'$P(m_k)$')

    plt.title(f'Log-Log Bucket Distribution ({annotator})')
    plt.tight_layout()

    if save_fig:
        plt.savefig(f'Figures/loglogdistr4_{annotator}.pdf', format='pdf', dpi=300)

    plt.show()


# class CQ_Calculator(object):
#     def __init__(self,
#                 df_annotated,
#                 process = True):
        
        
#         self.subjectDict_collection = {}
#         self.df_idea_annotations = df_annotated
#         self.all_ids = self.df_idea_annotations['for_user_id'].unique()
#         self.fact_dict = [1,1,2,6]
        
#         self.stop = set(stopwords.words('english'))
#         self.stop.add('use')
#         self.stop.add('make')
#         self.stop.add('something')
#         self.stop.add('put')
        
#         #This sets up the symspell dictionary for autocorrect to work
#         max_edit_distance_dictionary = 4;
#         prefix_length = 7;
#         # create object
#         self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length);
#         # load dictionary
#         dictionary_path = "data/frequency_dictionary_en_82_765.txt";
#         term_index = 0  # column of the term in the dictionary text file
#         count_index = 1  # column of the term frequency in the dictionary text file
#         if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index):
#             print("Dictionary file not found")
#         self.hyponymCountDict = {}
#         self.icDict = {}
        
#         self.df_cq = None

#         if process:
#             self.init_subjectDict()
#             self.extract_cq()
#             self.generate_df_cq()
            
#     def wordnet_pos(self,treebank):
#         return wn.ADJ if treebank.startswith('J') else \
#                 (wn.VERB if treebank.startswith('V') else \
#                  (wn.NOUN if treebank.startswith('N') else \
#                   (wn.ADV if treebank.startswith('R') else '')))

#     def preprocess(self,s):
#         result = s.lower();
#         result = nltk.pos_tag(nltk.word_tokenize(result));
#         result = [(wnl.lemmatize(word.translate(str.maketrans(string.punctuation,len(string.punctuation)*' '))).strip(),self.wordnet_pos(pos)) for word,pos in result if (word not in self.stop and self.wordnet_pos(pos) is not '')];
#         result = [(word,pos) for word,pos in result if not(len(word.split(' ')) == 2 and len(word.split(' ')[0]) == 1 and len(word.split(' ')[1]) == 1) and len(word) > 2];

#         return result;
    
#     def autocorrect(self,s,edit_dist=2):
#         return self.sym_spell.lookup_compound(s, edit_dist)[0].term if len(self.sym_spell.lookup_compound(s, edit_dist)) > 0 else s;
    
#     def toNoun(self,synset):
#         if synset.pos() == 'n':
#             return synset;

#         return max([(derivation.synset(),derivation.count()) for lemma in synset.lemmas() for derivation in lemma.derivationally_related_forms() + lemma.pertainyms() if derivation.synset().pos() == 'n']+[(None,-1)], key=lambda t: t[1])[0];

#     #Hyponyms are children
#     def countHyponyms(self,synset,result=None):
#         initialCall = result == None;
        
#         if initialCall and str(synset) in self.hyponymCountDict:
#             return self.hyponymCountDict[str(synset)];

#         result = result if not initialCall else set();

#         result.update(synset.hyponyms()+synset.instance_hyponyms());
#         for s in synset.hyponyms() + synset.instance_hyponyms():
#             self.countHyponyms(s,result);

#         if(initialCall):
#             self.hyponymCountDict[str(synset)] = len(result)
#         return len(result)
    
#     # NOUNS ONLY
#     def ic(self,synset):
        
#         self.icDict[str(synset)] = self.icDict[str(synset)] if self.icDict.get(str(synset)) else 1 - math.log2(self.countHyponyms(synset)+1)/math.log2(82115);
#         return self.icDict[str(synset)];
    
#     def sim(self,s1,s2):
#         return 1 - .5*(self.ic(s1)+self.ic(s2)-2*max([self.ic(s) for s in s1.common_hypernyms(s2)]))
    
#     def init_subjectDict(self):
        
#         subjectDict_template = {k:[] for k in self.all_ids}

#         self.subjectDict_collection = {1:{1:copy.deepcopy(subjectDict_template),2:copy.deepcopy(subjectDict_template),'both':copy.deepcopy(subjectDict_template)},
#                                  2:{1:copy.deepcopy(subjectDict_template),2:copy.deepcopy(subjectDict_template),'both':copy.deepcopy(subjectDict_template)},
#                                  3:{1:copy.deepcopy(subjectDict_template),2:copy.deepcopy(subjectDict_template),'both':copy.deepcopy(subjectDict_template)},
#                                  4:{1:copy.deepcopy(subjectDict_template),2:copy.deepcopy(subjectDict_template),'both':copy.deepcopy(subjectDict_template)},
#                                  5:{1:copy.deepcopy(subjectDict_template),2:copy.deepcopy(subjectDict_template),'both':copy.deepcopy(subjectDict_template)}}
        
#         # deal with the old annotations file
#         for index, row in self.df_idea_annotations.iterrows():

#             converted_id = row["for_user_id"]
#             self.subjectDict_collection[row['round_id']][row['turn_id']][converted_id].append((row["idea_content"],1))
#             self.subjectDict_collection[row['round_id']]['both'][converted_id].append((row["idea_content"],1))

    
#     def roundwise_extractor_cq(self,round_id,turn='both',group_=None):
        
#         subjectDict = self.subjectDict_collection[round_id][turn]
        
#         #This just makes a new dict where all the ideas have been preprocessed
#         processedSubjectDict = dict([(uid, [(self.preprocess(datum[0]), datum[1]) for datum in lst]) for uid,lst in subjectDict.items()]);
        
#         #This edits the processedSubjectDict to replace pairs of words with one new concept
#         for key in processedSubjectDict:
#             for idea in processedSubjectDict[key]:
#                 for index in range(1,len(idea[0])):
#                     if(len(wn.synsets(idea[0][index-1][0] + '_' + idea[0][index][0])) > 0):
#                         syn = wn.synsets(idea[0][index-1][0] + '_' + idea[0][index][0])[0];
#                         #print(syn);                
#                         idea[0][index-1] = (idea[0][index-1][0] + '_' + idea[0][index][0],syn.pos());
#                         idea[0][index] = idea[0][index-1];
        
#         subjectSynsets = defaultdict(set)
        
#         #This code makes a set of synsets for each person
#         for person in processedSubjectDict:
#             for idea in processedSubjectDict[person]:
#                 for word in idea[0]:
#                     needsToBeChecked = True;
#                     while(needsToBeChecked):
#                         needsToBeChecked = False;
#                         if(len(wn.synsets(word[0],word[1])) > 0):
#                             subjectSynsets[person].add(wn.synsets(word[0],word[1])[0]);
#                         elif(len(wn.synsets(word[0])) > 0):
#                             subjectSynsets[person].add(wn.synsets(word[0])[0]);
#                         else:
#                             if(len(wn.synsets(word[0].replace(' ', '-'))) > 0):
#                                 subjectSynsets[person].add(wn.synsets(word[0].replace(' ', '-'))[0]);
#                             elif(len(word[0].split(' ')) > 1 and len(wn.synsets(word[0].split(' ')[1])) > 0):
#                                 subjectSynsets[person].add(wn.synsets(word[0].split(' ')[1])[0]);
#                             elif(word[0] != self.autocorrect(word[0])):
#                                 word = (self.autocorrect(word[0]), word[1]);
#                                 needsToBeChecked = True; 
       
#         subjectSynsetsN = defaultdict(set)
        
#         #Code to make a noun-only version of the synset sets
#         for subject in subjectSynsets:
#             for synset in subjectSynsets[subject]:
#                 if synset.pos() == 'n':
#                     subjectSynsetsN[subject].add(synset);
#                 elif self.toNoun(synset):
#                     subjectSynsetsN[subject].add(self.toNoun(synset));
                    
#         subjectTree = {};

#         for subject in subjectSynsetsN:
#             synsetCount = len(subjectSynsetsN[subject]);
#             synsetList = list(subjectSynsetsN[subject]);
#             subjectTree[subject] = np.zeros((synsetCount,synsetCount));
#             for i in range(synsetCount):
#                 for j in range(i+1,synsetCount):
#                     subjectTree[subject][(i,j)] = self.sim(synsetList[i],synsetList[j]);
        
#         I = {};

#         for subject in subjectTree: #{k:subjectTree[k] for k in subjectTree if k == '30'}:
#             result = minimum_spanning_tree(subjectTree[subject]*-1);
#             I[subject] = result.sum()*-1;
        
#         Q = {};

#         for subject in subjectTree:
#             Q[subject] = len(subjectTree[subject]) - I[subject];
        
#         return Q
    

#     def extract_cq(self):
#         print("==Loading CQ data==")
#         self.cq_dict = {}
#         for id_ in self.all_ids: # do it for all alters, control, usf, and pred egos (converted IDs)
#             self.cq_dict[id_]={1:{1:0.0,2:0.0,'both':0.0},
#                                 2:{1:0.0,2:0.0,'both':0.0},
#                                 3:{1:0.0,2:0.0,'both':0.0},
#                                 4:{1:0.0,2:0.0,'both':0.0},
#                                 5:{1:0.0,2:0.0,'both':0.0},
#                                'all':{1:0.0,2:0.0,'both':0.0} # 'all' is linear sum of the 5 rounds
#                               }
        
#         for round_ in range(1,6): # compute cq in rounds 1-5
#             print("Computing CQ of round "+str(round_)+"/5")
#             for turn_ in ['both']:
#                 subjectDict = self.roundwise_extractor_cq(round_id=round_,turn=turn_)
#                 for key,val in subjectDict.items():
#                     self.cq_dict[key][round_][turn_] = val
        
#         for id_ in self.all_ids:
#             for turn_ in ['both']:
#                 self.cq_dict[id_]['all'][turn_] = np.sum([self.cq_dict[id_][k][turn_] for k in range(1,6)])  
                
#     def generate_df_cq(self):
 
#         round_list = []
#         cq_list = []

#         id_list = []

#         for round_ in range(1,6):
#             for turn_ in ['both']:
#                 for id_ in self.all_ids:
 
#                     round_list.append(round_)
#                     cq_list.append(self.cq_dict[id_][round_][turn_]) 
#                     id_list.append(id_)


#         self.df_cq = pd.DataFrame(list(zip(round_list,cq_list,id_list)), columns =['round_id','cq','for_user_id']) 

def check_variance(df):
    return df.std(axis=0)  # std of each column (rater)

def wrapper_icc_3_1_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference='ali'):
    df2 = generate_icc_dataframe_allrounds(df,[current_llm_colname])[scoring_method]
    df3 = df2.groupby('for_user_id')[[current_llm_colname]].sum().reset_index()
    current_llm_dict = pd.Series(df3[current_llm_colname].values, index = df3['for_user_id']).to_dict()
    df_person_level[current_llm_colname] = df_person_level['for_user_id'].map(current_llm_dict)
    icc_df = df_person_level[[reference,current_llm_colname]]
    stds = check_variance(icc_df)
    if (stds == 0).any():
        print("Warning: Some raters have zero variance, which may cause singular fits.")

    icc_output = compute_icc(icc_df, print_=False)
    return {"mean": icc_output['ICC(3,1)'], "ci_lower": icc_output['CI(3,1)'][0], "ci_upper": icc_output['CI(3,1)'][1]}

def wrapper_icc_3_k_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference='ali'):
    df2 = generate_icc_dataframe_allrounds(df,[current_llm_colname])[scoring_method]
    df3 = df2.groupby('for_user_id')[[current_llm_colname]].sum().reset_index()
    current_llm_dict = pd.Series(df3[current_llm_colname].values, index = df3['for_user_id']).to_dict()
    df_person_level[current_llm_colname] = df_person_level['for_user_id'].map(current_llm_dict)
    icc_df = df_person_level[[reference,current_llm_colname]]
    stds = check_variance(icc_df)
    if (stds == 0).any():
        print("Warning: Some raters have zero variance, which may cause singular fits.")

    icc_output = compute_icc(icc_df, print_=False)
    return {"mean": icc_output['ICC(3,k)'], "ci_lower": icc_output['CI(3,k)'][0], "ci_upper": icc_output['CI(3,k)'][1]}

def wrapper_pearson_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference='ali'):
    df2 = generate_icc_dataframe_allrounds(df,[current_llm_colname])[scoring_method]
    df3 = df2.groupby('for_user_id')[[current_llm_colname]].sum().reset_index()
    current_llm_dict = pd.Series(df3[current_llm_colname].values, index = df3['for_user_id']).to_dict()
    df_person_level[current_llm_colname] = df_person_level['for_user_id'].map(current_llm_dict)
    icc_df = df_person_level[[reference,current_llm_colname]]
    
    result_llm = pearson_corr_with_ci(icc_df[reference], icc_df[current_llm_colname])
    return {"mean": result_llm['r'], "ci_lower": result_llm['CI_lower'], "ci_upper": result_llm['CI_upper']}

def wrapper_spearman_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference='ali'):
    df2 = generate_icc_dataframe_allrounds(df,[current_llm_colname])[scoring_method]
    df3 = df2.groupby('for_user_id')[[current_llm_colname]].sum().reset_index()
    current_llm_dict = pd.Series(df3[current_llm_colname].values, index = df3['for_user_id']).to_dict()
    df_person_level[current_llm_colname] = df_person_level['for_user_id'].map(current_llm_dict)
    icc_df = df_person_level[[reference,current_llm_colname]]
    
    result_llm = spearman_corr_with_ci(icc_df[reference], icc_df[current_llm_colname])
    return {"mean": result_llm['rho'], "ci_lower": result_llm['CI_lower'], "ci_upper": result_llm['CI_upper']}

def wrapper_scorer_personlevel(df,df_person_level,current_llm_colname,eval_='pearson',scoring_method='norm_orig',reference='ali'):
    if eval_=="ICC(3,1)":
        return wrapper_icc_3_1_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference)
    elif eval_=="ICC(3,k)":
        return wrapper_icc_3_k_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference)
    elif eval_=="pearson":
        return wrapper_pearson_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference)
    elif eval_=="spearman":
        return wrapper_spearman_personlevel(df,df_person_level,current_llm_colname,scoring_method,reference)

    
def get_grouped_plots_person_level(df, eval_='pearson',scoring_method ='norm_orig', embedding_='e5',reference='ali',save_fig=False):
    # first prepare the reference value
    df2 = generate_icc_dataframe_allrounds(df,['ali','krish'])[scoring_method]
    df_person_level = df2.groupby('for_user_id')[['ali', 'krish']].sum().reset_index()
    df_person_level['mean_score'] = df_person_level[['ali', 'krish']].mean(axis=1)
    
    data_dict = {}
#     data_dict['human_ann'] = {}
#     data_dict['human_ann']['H2'] = wrapper_scorer_personlevel(df,df_person_level,'krish',eval_,scoring_method,reference)
    # First, do the llms
    models_dict = {'llm':['llama','qwen','phi'], 'clustering':['kmm_sil','kmm_sem','agg_sil','agg_sem']}
    
    for prompting_ in ['cot','baseline']:
        data_dict[prompting_] = {}
        for model_ in models_dict['llm']:
            if prompting_=='baseline' and model_=='qwen':
                continue
            current_llm_colname = get_llm_colname(model_, embedding_, prompting_, 10) 
            data_dict[prompting_][model_] = wrapper_scorer_personlevel(df,df_person_level,current_llm_colname,eval_,scoring_method,reference)
            
    # now do clustering baselines
    data_dict['clustering'] = {}
    for model_ in models_dict['clustering']:
        current_llm_colname = get_clustering_colname(model_, embedding_)
        data_dict['clustering'][model_] = wrapper_scorer_personlevel(df,df_person_level,current_llm_colname,eval_,scoring_method,reference)
   
    plot_grouped_bar_with_ci2(data_dict, ylabel=eval_, title=f"{eval_} performance against annotator H1, {embedding_} embedding",save_fig=save_fig)
    return data_dict



def get_mean_avg_rating_per_user(df,rater_cols):
    # Step 1: Compute average rating per idea
    df['avg_rating'] = df[rater_cols].mean(axis=1)
    
    # Step 2 & 3: Group by user and get max avg_rating
    max_avg_dict = df.groupby('for_user_id')['avg_rating'].mean().to_dict()
    
    return max_avg_dict


def get_total_ideas_per_user(df):
    # Group by for_user_id and count the number of rows (ideas) per user
    idea_count_dict = df['for_user_id'].value_counts().to_dict()
    
    return idea_count_dict



def get_personality_dicts(df):
    traits = ['n_ffi', 'e_ffi', 'o_ffi', 'a_ffi', 'c_ffi']
    personality_dicts = {}

    for trait in traits:
        # Create dictionary only for non-null values
        personality_dicts[trait] = df.loc[df[trait].notna(), ['for_user_id', trait]] \
                                      .set_index('for_user_id')[trait].to_dict()

    return personality_dicts['n_ffi'], personality_dicts['e_ffi'], personality_dicts['o_ffi'], \
           personality_dicts['a_ffi'], personality_dicts['c_ffi']


def get_mean_metaphor_rating(df, meta_rater_list):
    # Compute mean for meta1 and meta2 per user
    df['meta1_avg'] = df[meta_rater_list].mean(axis=1)
    df['meta2_avg'] = df[meta_rater_list].mean(axis=1)

    # Compute mean of the two metaphor scores
    df['meta_overall_avg'] = df[['meta1_avg', 'meta2_avg']].mean(axis=1)

    # Create dictionary {for_user_id: meta_overall_avg}, excluding any NaNs
    meta_dict = df.loc[df['meta_overall_avg'].notna(), ['for_user_id', 'meta_overall_avg']] \
                  .set_index('for_user_id')['meta_overall_avg'].to_dict()

    return meta_dict


def get_gf_score_dict(df,gf_col):
    #'gf_cfiq', 'gf_letters', 'gf_numbers'

    # Drop rows with any NaNs in the gf columns
    valid_df = df[['for_user_id'] + gf_col].dropna()

    # Compute the row-wise average
    valid_df['gf_avg'] = valid_df[gf_col].mean(axis=1)

    # Create the dictionary {for_user_id: gf_avg}
    gf_dict = valid_df.set_index('for_user_id')['gf_avg'].to_dict()

    return gf_dict


def get_ssci_identity_score_dict(df):
    ssci_cols = ['ssci_creative_identity']# 'ssci_creative_efficacy'

    # Drop rows with any NaNs in the SSCI columns
    valid_df = df[['for_user_id'] + ssci_cols].dropna()

    # Compute the row-wise average
    valid_df['ssci_avg'] = valid_df[ssci_cols].mean(axis=1)

    # Create the dictionary {for_user_id: ssci_avg}
    ssci_dict = valid_df.set_index('for_user_id')['ssci_avg'].to_dict()

    return ssci_dict

def get_ssci_efficacy_score_dict(df):
    ssci_cols = ['ssci_creative_efficacy']# 'ssci_creative_efficacy'

    # Drop rows with any NaNs in the SSCI columns
    valid_df = df[['for_user_id'] + ssci_cols].dropna()

    # Compute the row-wise average
    valid_df['ssci_avg'] = valid_df[ssci_cols].mean(axis=1)

    # Create the dictionary {for_user_id: ssci_avg}
    ssci_dict = valid_df.set_index('for_user_id')['ssci_avg'].to_dict()

    return ssci_dict


def get_clean_sorted_lists(df, col1, col2):
    # Drop rows where either column has NaN
    filtered_df = df[[col1, col2, 'for_user_id']].dropna()

    # Sort by for_user_id
    sorted_df = filtered_df.sort_values(by='for_user_id')

    # Extract the two columns as lists
    list1 = sorted_df[col1].tolist()
    list2 = sorted_df[col2].tolist()

    return list1, list2

def get_o_score_dict(df):
    o_cols = ['o_aestheticappreciation', 'o_inquisitiveness', 'o_creativity', 'o_unconventionality'] 

    # Drop rows with any NaNs in the gf columns
    valid_df = df[['for_user_id'] + o_cols].dropna()

    # Compute the row-wise average
    valid_df['o_avg'] = valid_df[o_cols].mean(axis=1)

    # Create the dictionary {for_user_id: gf_avg}
    o_dict = valid_df.set_index('for_user_id')['o_avg'].to_dict()

    return o_dict


def get_flexibility_rating(df):
    flexibility_cols = ['flexibility']# 'ssci_creative_efficacy'

    # Drop rows with any NaNs in the SSCI columns
    valid_df = df[['for_user_id'] + flexibility_cols].dropna()

    # Compute the row-wise average
    valid_df['flexibility_avg'] = valid_df[flexibility_cols].mean(axis=1)

    # Create the dictionary {for_user_id: ssci_avg}
    flexibility_dict = valid_df.set_index('for_user_id')['flexibility_avg'].to_dict()

    return flexibility_dict

























































