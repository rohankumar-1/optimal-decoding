import pandas as pd
import gzip
import os
import ssl
import urllib.request
import re
import numpy as np


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def MC_calcs(scores_true, scores_false, ref_true, ref_best) -> dict:
    """Given model scores for true / false reference answers, calculates MC scores"""
    scores = {}
    scores['max'] = max(scores_true)
    scores['diff'] = max(scores_true) - max(scores_false)
    scores['scores-true'] = scores_true
    scores['scores-false'] = scores_false

    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[ref_true.index(ref_best)] > max_false:
        scores['MC1'] = 1.0
    else:
        scores['MC1'] = 0.0

    # compute MC3: 1vFalse -- each correct answer vs all false answers
    max_false = max(scores_false)
    onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
    scores['MC3'] = onevall

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    while sum(probs_true) == 0:
        print("WARNING: all zero scores_true")
        scores_true = [x/2.0 for x in scores_true]
        probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)
    while sum(probs_false) == 0:
        print("WARNING: all zero scores_false")
        scores_false = [x/2.0 for x in scores_false]
        probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    
    # check nan
    if np.isnan(sum(probs_true)):
        scores['MC2'] = 0.0
        print(f"WARNING: nan in probs_true: sum(probs_true)={sum(probs_true)}, sum(probs_false)={sum(probs_false)}")
    else:
        scores['MC2'] = sum(probs_true)

    return scores

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def split_multi_answer(ans, sep=';', close=True):
    """Splits string of all reference answers into a list of formatted answers"""
    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


def format_best(best_ans, close=True):
    """Formats best answer to match format of reference answers"""
    best = best_ans.strip()
    if close:
        if best[-1] != '.':
            best = best + '.'
    return best

def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def load_csv(file_path, is_gzip=False):
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        list_data = list(df['Question'])
    return list_data

def load_csv_dict(file_path, is_gzip=False):
    open_func = open if not is_gzip else gzip.open
    list_data = []
    with open_func(file_path, 'r') as f:
        df = pd.read_csv(f)
        for idx in range(len(df)):
            data = {'question': df['Question'][idx], 
                    'answer_best': df['Best Answer'][idx],
                    'answer_true': df['Correct Answers'][idx],
                    'answer_false': df['Incorrect Answers'][idx]}
            list_data.append(data)

    return list_data

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path