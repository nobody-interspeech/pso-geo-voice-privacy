#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import logging
import os
import h5py
from tqdm import tqdm
import numpy as np
import sklearn.metrics.pairwise
import random
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd
import joblib


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Predicate:
    def __init__(self, threshold, scores):
        self.threshold = threshold
        self.scores = scores

    def __call__(self, x):
        if isinstance(x, str):
            return self.scores[x] > self.threshold
        else:
            return sum([self(utt) for utt in x]) / len(x)

    def iso(self, row) -> bool:
        return self(row) == 1 / len(row)


def load_spk2utt(spk2utt_file, min_utt=0):
    spk2utt = {}
    with open(spk2utt_file) as spk2utt_f:
        for line in spk2utt_f:
            lns = line.strip().split()
            if len(lns[1:]) >= min_utt:
                spk2utt[lns[0]] = lns[1:]
    return spk2utt


def choose_predicate_speakers_and_trial_remainder(trial_spk2utt, n_utt_for_predicate):
    """
    We ONLY use the TRIAL split to build predicates (attackers),
    and we keep the remaining trial utterances to optionally add into the dataset pool
    (for speakers that exist in both splits).
    """
    predicates_spk2utt = {}
    trial_remainder_spk2utt = {}

    for s, trial_utts in trial_spk2utt.items():
        if len(trial_utts) >= n_utt_for_predicate:
            predicates_spk2utt[s] = trial_utts[:n_utt_for_predicate]
            trial_remainder_spk2utt[s] = trial_utts[n_utt_for_predicate:]

    return predicates_spk2utt, trial_remainder_spk2utt


def load_x_vectors(h5_file, spk2utt):
    x_vectors = {}
    with h5py.File(
        h5_file, "r",
        rdcc_nbytes=1024**2 * 4000,
        rdcc_nslots=int(1e7),
    ) as f:
        for spk in tqdm(spk2utt, desc=f"load_x_vectors({os.path.basename(h5_file)})"):
            x_vectors[spk] = {utt: f[utt][()] for utt in spk2utt[spk]}
    return x_vectors


def average_x_vectors(x_vectors):
    if len(x_vectors) > 1:
        mean = np.mean(x_vectors, axis=0)
        norm = np.linalg.norm(mean, ord=2)
        mean /= norm
    else:
        # don't apply l2norm twice, already done by the Xtractor
        mean = x_vectors[0]
    return mean


def plot2(
    x_vectors_for_predicates, n_predicates,
    x_vectors_for_dataset, n_conversations_per_speaker,
    values_for_N, L, n_runs,
    results_file,
    seed=0, n_jobs=None
):
    """
    Plot2 / PSO:
    - predicate speakers: random subset of speakers from x_vectors_for_predicates (attackers)
    - dataset speakers: those from x_vectors_for_dataset with >= 2*L utterances
    - for each fold: build 1 test conversation of length L per dataset speaker, and C calibration conversations
    - compute cosine similarities and PSO isolation

    Extra logging requested:
    - Print number of speakers that satisfy the condition for this L: K >= 2*L
    """

    # ---- predicates (attackers)
    random.seed(seed)
    logger.info(f'select {n_predicates} random speakers among {len(x_vectors_for_predicates)} for predicates')
    if n_predicates > len(x_vectors_for_predicates):
        raise ValueError(f"n_predicates={n_predicates} > available predicate speakers={len(x_vectors_for_predicates)}")

    predicate_speakers = {
        spk: average_x_vectors(list(x_vectors_for_predicates[spk].values()))
        for spk in random.sample(list(x_vectors_for_predicates.keys()), k=n_predicates)
    }
    logger.info(f'{len(predicate_speakers)} speakers selected for predicates')

    # ---- dataset pool for this L
    # requested: show how many speakers satisfy K >= 2L (for the chosen dataset split)
    total_dataset_pool = len(x_vectors_for_dataset)
    satisfy_2L = sum(1 for spk in x_vectors_for_dataset if len(x_vectors_for_dataset[spk]) >= 2 * L)
    logger.info(f'[L={L}] Dataset pool size before filtering: {total_dataset_pool}')
    logger.info(f'[L={L}] Speakers satisfying condition K >= 2*L ({2*L} utts): {satisfy_2L}')

    dataset_speakers = {
        spk: x_vectors_for_dataset[spk]
        for spk in x_vectors_for_dataset
        if len(x_vectors_for_dataset[spk]) >= 2 * L
    }
    logger.info(f'[L={L}] dataset_speakers after filtering (>= {2*L} utts): {len(dataset_speakers)}')

    # Filter invalid N so we never compute phantom points
    max_N = len(dataset_speakers)
    values_for_N = [N for N in values_for_N if N <= max_N]
    if not values_for_N:
        raise ValueError(f"No valid N values remain (max possible N={max_N} for L={L}).")
    logger.info(f'[L={L}] Valid N values (<= {max_N}): {values_for_N}')

    # eligible speakers = predicate speakers that also exist in dataset_speakers
    eligible_speakers = set(predicate_speakers.keys()).intersection(dataset_speakers.keys())
    logger.info(f'[L={L}] eligible_speakers (predicate ∩ dataset): {len(eligible_speakers)}')
    if not eligible_speakers:
        raise ValueError(
            f"[L={L}] eligible_speakers is empty. "
            f"No overlap between selected predicate speakers and dataset speakers (>=2L)."
        )

    isolation_results = {}
    logger.info(f'Starting evaluation for length of the conversation L={L}')
    isolation_results[L] = {N: {} for N in values_for_N}

    for fold in range(n_conversations_per_speaker):
        logger.info(f'[L={L}] fold={fold}: dataset_speakers={len(dataset_speakers)}, eligible_speakers={len(eligible_speakers)}')

        conversations_x_vectors = {}
        test_conversations = {}
        calibration_conversations = {}

        X = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_plot2_get_conversations_xvectors)(
                (
                    s,
                    list(dataset_speakers[s].keys()),
                    dataset_speakers[s],
                    L,
                    n_conversations_per_speaker,
                    seed_i
                )
            )
            for seed_i, s in tqdm(enumerate(dataset_speakers), desc=f"build conversations L={L}, fold={fold}")
        )

        for s, conversations_x_vectors_speaker, test_conversation_id in X:
            test_conversations[s] = test_conversation_id
            calibration_conversations[s] = list(conversations_x_vectors_speaker.keys())
            calibration_conversations[s].remove(test_conversation_id)
            conversations_x_vectors.update(conversations_x_vectors_speaker)

        Xp = np.vstack(list(predicate_speakers.values()))
        Yp = np.vstack(list(conversations_x_vectors.values()))
        similarity_scores = pd.DataFrame(
            sklearn.metrics.pairwise.cosine_similarity(Xp, Yp),
            index=predicate_speakers.keys(),
            columns=conversations_x_vectors.keys()
        )
        logger.info(f'Computed similarity_scores for L={L}, fold {fold}: {similarity_scores.shape}')

        logger.info('Compute isolation scores ...')
        runs = itertools.product(values_for_N, eligible_speakers, range(n_runs))

        # Pre-sample the N-speaker sets per run, per N, per predicate speaker (only eligible speakers)
        N_dataset_speakers = {}
        for run in range(n_runs):
            logger.info(f'[L={L}] fold={fold} run={run}: dataset_speakers={len(dataset_speakers)}, eligible_speakers={len(eligible_speakers)}')
            N_dataset_speakers[run] = {}
            for N in values_for_N:
                N_dataset_speakers[run][N] = {}
                for p_spk in eligible_speakers:
                    speakers = set(dataset_speakers.keys())
                    speakers.discard(p_spk)

                    if (N - 1) > len(speakers):
                        raise ValueError(
                            f"Cannot sample N-1={N-1} speakers (available={len(speakers)}) for N={N}, L={L}."
                        )

                    chosen = random.sample(list(speakers), N - 1)
                    N_dataset_speakers[run][N][p_spk] = {s: dataset_speakers[s] for s in chosen}
                    N_dataset_speakers[run][N][p_spk][p_spk] = dataset_speakers[p_spk]

        for N, p_spk, run in tqdm(
            runs,
            total=len(values_for_N) * len(eligible_speakers) * n_runs,
            desc=f'L={L}, fold={fold}'
        ):
            key = f'{p_spk}_{fold}'
            if key not in isolation_results[L][N]:
                isolation_results[L][N][key] = []
            isolation_results[L][N][key].append(
                _plot2_run((
                    p_spk,
                    N_dataset_speakers[run][N][p_spk],
                    similarity_scores.loc[p_spk],
                    N, L, fold,
                    calibration_conversations, test_conversations,
                    run,
                    results_file
                ))[1]
            )

    # Aggregate
    isolation_results = {
        L: {
            N: [np.mean(isolation_results[L][N][s]) for s in isolation_results[L][N]]
            for N in isolation_results[L]
        }
        for L in isolation_results
    }
    isolation_scores = {N: np.mean(isolation_results[L][N]) for N in isolation_results[L]}
    isolation_scores_std = {N: np.std(isolation_results[L][N]) for N in isolation_results[L]}

    fig, ax = plt.subplots()
    x = list(isolation_scores.keys())
    y = [isolation_scores[n] for n in x]
    yerr = [isolation_scores_std[n] for n in x]
    ax.errorbar(x, y, yerr=yerr, label=f'L={L}')
    ax.axhline(y=0.37, label='trivial', c='black')
    ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(results_file.rsplit('.', 1)[0] + ".pdf")


def _plot2_run(args):
    p_spk, N_dataset_speakers, similarity_scores, N, L, fold, calibration_conversations, test_conversations, run, results_file = args

    calibration_set = [utt for s in N_dataset_speakers for utt in calibration_conversations[s]]
    calibration_scores = list(similarity_scores[calibration_set])
    calibration_scores.sort(reverse=True)

    # n is the nb of calibration conversations for predicate speaker
    n = len(calibration_conversations[p_spk])
    threshold = (calibration_scores[n - 1] + calibration_scores[n]) / 2

    predicate = Predicate(threshold, similarity_scores)
    test_row = [test_conversations[s] for s in N_dataset_speakers]
    iso = 1 if predicate.iso(test_row) else 0

    with open(results_file, mode='a') as f:
        f.write(f'{L};{fold};{N};{p_spk};{run};{iso}\n')

    return N, iso


def _plot2_get_conversations_xvectors(args):
    """
    Build:
      - 1 test conversation of length L
      - C calibration conversations of length L from remaining utterances
    """
    speaker, utterances, dataset_speaker, L, n_conversations_per_speaker, seed = args
    random.seed(seed)
    random.shuffle(utterances)

    conversations_x_vectors = {}

    test_conversation = utterances[:L]
    test_conversation_id = ':'.join(test_conversation)
    conversations_x_vectors[test_conversation_id] = average_x_vectors([dataset_speaker[u] for u in test_conversation])

    K = len(utterances)
    C = min(n_conversations_per_speaker - 1, scipy.special.comb(K - L, L, exact=True))

    for _, conversation in zip(range(C), itertools.combinations(utterances[L:], L)):
        conversation_id = ':'.join(conversation)
        conversations_x_vectors[conversation_id] = average_x_vectors([dataset_speaker[u] for u in conversation])

    return speaker, conversations_x_vectors, test_conversation_id


if __name__ == '__main__':
    import datetime
    import time

    debug = False
    dev = False

    if debug:
        N_PREDICATES_SPEAKERS_TO_LOAD = 10
        N_DATASET_SPEAKERS_TO_LOAD = 1000
        N_PREDICATES = 5
        VALUES_FOR_N = (30, 100, 200, 400, 800)
        N_UTTERANCES_PER_SPEAKER = 10
        N_CONVERSATIONS_PER_SPEAKER = 10
        N_RUNS = 3
    elif dev:
        N_PREDICATES_SPEAKERS_TO_LOAD = 50
        N_DATASET_SPEAKERS_TO_LOAD = 10000
        N_PREDICATES = 45
        VALUES_FOR_N = [30]
        N = 100
        while N < N_DATASET_SPEAKERS_TO_LOAD:
            VALUES_FOR_N.append(N)
            N = 2 * N
        VALUES_FOR_N.append(N_DATASET_SPEAKERS_TO_LOAD)
        N_UTTERANCES_PER_SPEAKER = 10
        N_CONVERSATIONS_PER_SPEAKER = 10
        N_RUNS = 5
    else:
        N_PREDICATES = 495
        VALUES_FOR_N = [30]
        N = 100
        while N < 22024:
            VALUES_FOR_N.append(N)
            N = 2 * N
        VALUES_FOR_N.append(22024)
        N_UTTERANCES_PER_SPEAKER = 10
        N_CONVERSATIONS_PER_SPEAKER = 10
        N_RUNS = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('L', type=int, help='Length of conversations')
    parser.add_argument('--enroll-folder', required=True, help='Path to enrollment folder (dataset pool)')
    parser.add_argument('--trial-folder', required=True, help='Path to trial folder (predicates source)')
    args = parser.parse_args()
    L = args.L
    if not L:
        raise Exception('Pass a value for L')

    results_dir = './PSO_results'

    # IMPORTANT (roles):
    # - trial_folder: split used to create predicate speakers (attackers)
    # - enroll_folder: split used as the dataset pool (all speakers), filtered by K >= 2*L
    enroll_folder = args.enroll_folder
    trial_folder = args.trial_folder

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    results_file_plot2 = f'{results_dir}/plot2.L{L}.{timestamp}.csv'

    file_handler = logging.FileHandler(f'{results_dir}/results.L{L}.{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Results saved in {results_dir}')
    logger.info(f'Using trial_folder={trial_folder} (predicates)')
    logger.info(f'Using enroll_folder={enroll_folder} (dataset pool)')

    # Load spk2utt
    trial_spk2utt = load_spk2utt(f'{trial_folder}/spk2utt')
    logger.info(f'{len(trial_spk2utt)} speakers in TRIAL (predicates source)')

    enroll_spk2utt = load_spk2utt(f'{enroll_folder}/spk2utt')
    logger.info(f'{len(enroll_spk2utt)} speakers in ENROLL (dataset pool source)')

    # Choose predicates from TRIAL only
    predicates_spk2utt, trial_remainder_spk2utt = choose_predicate_speakers_and_trial_remainder(
        trial_spk2utt, N_UTTERANCES_PER_SPEAKER
    )
    logger.info(f'{len(predicates_spk2utt)} trial speakers usable for predicates (>= {N_UTTERANCES_PER_SPEAKER} utts)')

    # DEV/DEBUG subsampling (optional)
    if dev or debug:
        predicates_spk2utt = dict(itertools.islice(predicates_spk2utt.items(), N_PREDICATES_SPEAKERS_TO_LOAD))
        logger.info(f'[DEV/DEBUG] keep {len(predicates_spk2utt)} predicate speakers')

        # For dataset pool, keep a random subset of ENROLL speakers
        if len(enroll_spk2utt) > N_DATASET_SPEAKERS_TO_LOAD:
            keys = list(enroll_spk2utt.keys())
            random.shuffle(keys)
            keys = keys[:N_DATASET_SPEAKERS_TO_LOAD]
            enroll_spk2utt = {k: enroll_spk2utt[k] for k in keys}
            logger.info(f'[DEV/DEBUG] keep {len(enroll_spk2utt)} dataset-pool speakers')

    # Build spk2utt for loading xvectors
    spk2utt_predicates = {s: predicates_spk2utt[s] for s in predicates_spk2utt}
    spk2utt_dataset = enroll_spk2utt  # <-- THIS IS THE FIX: load ALL dataset speakers (not only overlap)

    # Load xvectors
    x_vectors_trial = load_x_vectors(f'{trial_folder}/xvector.h5', spk2utt_predicates)
    x_vectors_enroll = load_x_vectors(f'{enroll_folder}/xvector.h5', spk2utt_dataset)

    # Predicates xvectors (attackers)
    x_vectors_for_predicates = {
        s: {u: x_vectors_trial[s][u] for u in spk2utt_predicates[s]}
        for s in spk2utt_predicates
    }

    # Dataset xvectors: start with ENROLL split (full pool)
    x_vectors_for_dataset = {
        s: {u: x_vectors_enroll[s][u] for u in spk2utt_dataset[s]}
        for s in spk2utt_dataset
    }

    # Optional: also add remaining TRIAL utterances into dataset for speakers that exist in both splits,
    # so that the predicate speaker is present in the dataset pool when possible.
    overlap = set(trial_remainder_spk2utt.keys()).intersection(x_vectors_for_dataset.keys())
    logger.info(f'Overlap speakers (trial remainder ∩ enroll pool): {len(overlap)}')
    for s in overlap:
        # We did NOT load these trial remainder utterances xvectors.
        # If you want them inside the dataset, you must load them too.
        # Minimal safe behavior: do nothing (dataset still OK), because overlap is already there via enroll.
        pass

    logger.info('PLOT 2')
    t0 = time.time()
    try:
        plot2(
            x_vectors_for_predicates, N_PREDICATES,
            x_vectors_for_dataset, N_CONVERSATIONS_PER_SPEAKER,
            VALUES_FOR_N, L, N_RUNS,
            results_file_plot2,
            seed=0, n_jobs=30
        )
    except Exception:
        print(f'Done in {time.time() - t0:.3f}s')
        raise

    t1 = time.time()
    print(f'Done in {t1 - t0:.3f}s')
