#!/usr/bin/env python3
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


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# -----------------------------
# Metadata loading / indexing
# -----------------------------
def load_geolocation(path: str) -> pd.DataFrame:
    """
    Reads geolocation.csv with columns:
      spk_id, gender, household_id, hh_size, OA, LSOA, MSOA
    Separator may be comma or tab; we auto-detect.
    """
    try:
        df = pd.read_csv(path, sep=",", dtype=str)
        if "spk_id" not in df.columns:
            raise ValueError("spk_id not found with comma sep")
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str)

    required = {"spk_id", "household_id", "OA", "LSOA", "MSOA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in geolocation file: {missing}")

    df = df.set_index("spk_id", drop=False)
    return df


def extract_base_spk_id(full_spk_id: str) -> str:
    """
    Extract base speaker ID from full ID.
    E.g., 'spk-f-65389-spk-f-65389-common-voice-en' -> 'spk-f-65389'
    """
    parts = full_spk_id.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return full_spk_id


def build_index(df: pd.DataFrame, col: str) -> dict:
    """Mapping: value -> set(spk_id)."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not in geolocation dataframe.")
    idx = {}
    for spk_id, v in zip(df["spk_id"], df[col]):
        if pd.isna(v):
            continue
        idx.setdefault(v, set()).add(spk_id)
    return idx


# -----------------------------
# PSO predicate
# -----------------------------
class Predicate:
    def __init__(self, threshold, scores):
        self.threshold = threshold
        self.scores = scores  # pandas Series indexed by conversation_id

    def __call__(self, x):
        if isinstance(x, str):
            return self.scores[x] > self.threshold
        else:
            return sum(self(utt) for utt in x) / len(x)

    def iso(self, row) -> bool:
        # legacy; we use robust integer check in _plot2_run
        return self(row) == 1 / len(row)


# -----------------------------
# IO helpers
# -----------------------------
def load_spk2utt(spk2utt_file, min_utt=0):
    spk2utt = {}
    with open(spk2utt_file) as f:
        for line in f:
            lns = line.strip().split()
            if len(lns) < 2:
                continue
            spk, utts = lns[0], lns[1:]
            if len(utts) >= min_utt:
                spk2utt[spk] = utts
    return spk2utt


def choose_predicate_speakers_and_trial_remainder(trial_spk2utt, n_utt_for_predicate):
    """
    Predicate speakers (attackers) built ONLY from TRIAL:
    - keep first n_utt_for_predicate utterances as predicate material
    - keep the remainder as potential dataset fallback
    """
    predicates_spk2utt = {}
    trial_remainder_spk2utt = {}
    for s, trial_utts in trial_spk2utt.items():
        if len(trial_utts) >= n_utt_for_predicate:
            predicates_spk2utt[s] = trial_utts[:n_utt_for_predicate]
            trial_remainder_spk2utt[s] = trial_utts[n_utt_for_predicate:]
    return predicates_spk2utt, trial_remainder_spk2utt


def build_dataset_from_enroll_with_trial_fallback(
    enroll_spk2utt: dict,
    trial_remainder_spk2utt: dict,
    L: int
):
    """
    1) Dataset pool = TOUS les speakers ENROLL avec leurs utterances enroll
    2) Si K_enroll < 2L, on complète avec des utterances TRIAL du même speaker
       (MAIS uniquement le remainder, donc on exclut les 10 utts utilisées pour le prédicat)
    3) Si après complément K_total < 2L => speaker drop (impossible de construire test+calib)
    """
    need = 2 * L
    dataset = {}

    only_enroll_ok = 0
    topped_up_ok = 0
    dropped = 0

    for s, enroll_utts in enroll_spk2utt.items():
        utts = list(enroll_utts)

        if len(utts) >= need:
            dataset[s] = utts
            only_enroll_ok += 1
            continue

        extra = trial_remainder_spk2utt.get(s, [])
        if extra:
            missing = need - len(utts)
            utts.extend(extra[:missing])

        if len(utts) >= need:
            dataset[s] = utts
            topped_up_ok += 1
        else:
            dropped += 1

    stats = {
        "enroll_total": len(enroll_spk2utt),
        "need_2L": need,
        "only_enroll_ok": only_enroll_ok,
        "topped_up_ok": topped_up_ok,
        "kept_total": len(dataset),
        "dropped": dropped,
    }
    return dataset, stats


def load_x_vectors(h5_file, spk2utt):
    """
    spk2utt: dict spk -> list[utt_id]
    Loads only requested utterances from a H5 where keys are utterance IDs.
    """
    x_vectors = {}
    with h5py.File(
        h5_file, "r",
        rdcc_nbytes=1024**2 * 4000,
        rdcc_nslots=int(1e7),
    ) as f:
        for spk in tqdm(spk2utt, desc=f"load_x_vectors({os.path.basename(h5_file)})"):
            d = {}
            for utt in spk2utt[spk]:
                d[utt] = f[utt][()]  # fail fast if missing
            x_vectors[spk] = d
    return x_vectors


def average_x_vectors(x_vectors):
    if len(x_vectors) > 1:
        mean = np.mean(x_vectors, axis=0)
        norm = np.linalg.norm(mean, ord=2)
        if norm > 0:
            mean /= norm
    else:
        mean = x_vectors[0]  # already normalized by extractor
    return mean


# -----------------------------
# Plot2 internals
# -----------------------------
def _plot2_get_conversations_xvectors(args):
    """
    Build:
      - 1 test conversation of length L
      - C calibration conversations (length L) from remaining utterances
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


def _plot2_run(args):
    """
    One evaluation for (geo_level, known_frac, run, N, predicate speaker, fold).
    Returns (N, iso).
    """
    (
        p_spk,
        N_dataset_speakers,       # dict: spk -> {utt: xvec}
        similarity_scores,        # pandas Series indexed by conversation_id
        N, L, fold,
        calibration_conversations, test_conversations,
        run,
        results_file,
        geo_level,
        geo_df,
        geo_index,
        speaker_to_convs,
        known_frac,
        known_speakers_set,       # set of speakers whose geo is known
    ) = args

    sim_scores = similarity_scores.copy()

    # Geo restriction (PARTIAL KNOWLEDGE):
    # - attacker knows geo only for known_speakers_set
    # - attacker can eliminate ONLY known speakers that are outside the target geo bucket
    if geo_level != "none":
        if geo_df is None or geo_index is None:
            raise ValueError("geo_df/geo_index must be provided when geo_level != 'none'.")

        p_spk_base = extract_base_spk_id(p_spk)
        if p_spk_base not in geo_df.index:
            raise ValueError(f"Predicate speaker {p_spk} (base: {p_spk_base}) not found in geolocation file.")

        region_value = geo_df.loc[p_spk_base, geo_level]
        if pd.isna(region_value):
            raise ValueError(f"Predicate speaker {p_spk} has no value for {geo_level}.")

        allowed_base_ids = set(geo_index.get(region_value, set()))

        disallowed_speakers = set()
        for s in known_speakers_set:
            if extract_base_spk_id(s) not in allowed_base_ids:
                disallowed_speakers.add(s)

        if disallowed_speakers:
            disallowed_convs = set()
            for s in disallowed_speakers:
                disallowed_convs.update(speaker_to_convs[s])
            disallowed_convs = disallowed_convs.intersection(sim_scores.index)
            if disallowed_convs:
                sim_scores.loc[list(disallowed_convs)] = -np.inf

    # Calibration threshold
    calibration_set = [utt for s in N_dataset_speakers for utt in calibration_conversations[s]]
    calibration_scores = list(sim_scores[calibration_set])
    calibration_scores.sort(reverse=True)

    n = len(calibration_conversations[p_spk])
    threshold = (calibration_scores[n - 1] + calibration_scores[n]) / 2

    predicate = Predicate(threshold, sim_scores)

    # Test row (one item per speaker)
    test_row = [test_conversations[s] for s in N_dataset_speakers]

    # Robust iso check (avoid float equality)
    iso = 1 if (sum(predicate(utt) for utt in test_row) == 1) else 0

    # CSV columns:
    # L;fold;N;predicate;run;known_frac;iso
    with open(results_file, mode='a') as f:
        f.write(f'{L};{fold};{N};{p_spk};{run};{known_frac:.3f};{iso}\n')

    return N, iso


def plot2(
    x_vectors_for_predicates, n_predicates,
    x_vectors_for_dataset, n_conversations_per_speaker,
    values_for_N, L, n_runs,
    results_file,
    geo_level="none",
    geo_df=None,
    geo_index=None,
    known_fracs=(1.0,),
    seed=0,
    n_jobs=None
):
    """
    Writes CSV rows:
      L;fold;N;predicate;run;known_frac;isolation

    geo_level: "none" or one of {"MSOA","LSOA","OA","household_id"}
    known_fracs: fractions of dataset speakers whose geo is known by the attacker.
    """
    random.seed(seed)
    known_fracs = sorted(set(float(f) for f in known_fracs))
    known_fracs = [f for f in known_fracs if 0.0 < f <= 1.0]
    if not known_fracs:
        raise ValueError("known_fracs is empty after filtering. Provide values in (0,1].")

    # ---- predicates (attackers)
    logger.info(f'[geo={geo_level}] select {n_predicates} random speakers among {len(x_vectors_for_predicates)}')
    if n_predicates > len(x_vectors_for_predicates):
        raise ValueError(f"n_predicates={n_predicates} > available predicate speakers={len(x_vectors_for_predicates)}")

    predicate_speakers = {
        spk: average_x_vectors(list(x_vectors_for_predicates[spk].values()))
        for spk in random.sample(list(x_vectors_for_predicates.keys()), k=n_predicates)
    }
    logger.info(f'[geo={geo_level}] {len(predicate_speakers)} speakers selected for predicates')

    # ---- dataset pool for this L
    logger.info(f'[geo={geo_level}] Dataset pool size before filtering: {len(x_vectors_for_dataset)}')
    dataset_speakers = {spk: x_vectors_for_dataset[spk] for spk in x_vectors_for_dataset if len(x_vectors_for_dataset[spk]) >= 2 * L}
    logger.info(f'[geo={geo_level}] dataset_speakers after filtering (>= {2*L} utts): {len(dataset_speakers)}')

    max_N = len(dataset_speakers)
    values_for_N = [N for N in values_for_N if N <= max_N]
    if not values_for_N:
        raise ValueError(f"No valid N values remain (max possible N={max_N} for L={L}).")
    logger.info(f'[geo={geo_level}] Valid N values (<= {max_N}): {values_for_N}')

    eligible_speakers = set(predicate_speakers.keys()).intersection(dataset_speakers.keys())
    if not eligible_speakers:
        raise ValueError("eligible_speakers is empty: no overlap between predicate_speakers and dataset_speakers.")
    logger.info(f'[geo={geo_level}] Global counts for L={L}: dataset_speakers={len(dataset_speakers)}, eligible_speakers={len(eligible_speakers)}')
    logger.info(f'[geo={geo_level}] Known fractions: {known_fracs}')

    # isolation_results[known_frac][L][N][key] -> list[iso]
    isolation_results = {frac: {L: {N: {} for N in values_for_N}} for frac in known_fracs}
    logger.info(f'[geo={geo_level}] Starting evaluation for conversation length L={L}')

    # Ensure CSV has header (optional). We won't overwrite if exists.
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        with open(results_file, "a") as f:
            f.write("L;fold;N;predicate;run;known_frac;iso\n")

    for fold in range(n_conversations_per_speaker):
        logger.info(f'[geo={geo_level}][L={L}] fold={fold}: dataset_speakers={len(dataset_speakers)}, eligible_speakers={len(eligible_speakers)}')

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
                    seed_i,
                )
            )
            for seed_i, s in tqdm(enumerate(dataset_speakers), desc=f"build conversations L={L}, fold={fold}")
        )

        for s, conversations_x_vectors_speaker, test_conversation_id in X:
            test_conversations[s] = test_conversation_id
            calibration_conversations[s] = list(conversations_x_vectors_speaker.keys())
            calibration_conversations[s].remove(test_conversation_id)
            conversations_x_vectors.update(conversations_x_vectors_speaker)

        # speaker -> all conv ids (test + calibration), needed for masking
        speaker_to_convs = {}
        for s in dataset_speakers:
            convs = set(calibration_conversations[s])
            convs.add(test_conversations[s])
            speaker_to_convs[s] = convs

        # cosine similarities (computed ONCE; reused for all fracs)
        Xp = np.vstack(list(predicate_speakers.values()))
        Yp = np.vstack(list(conversations_x_vectors.values()))
        similarity_scores = pd.DataFrame(
            sklearn.metrics.pairwise.cosine_similarity(Xp, Yp),
            index=predicate_speakers.keys(),
            columns=conversations_x_vectors.keys(),
        )
        logger.info(f'[geo={geo_level}] Computed similarity_scores for L={L}, fold={fold}: {similarity_scores.shape}')

        # pre-sample datasets per run/N/predicate + pre-sample geo-known sets per frac
        N_dataset_speakers = {}
        known_sets = {}  # known_sets[run][N][p_spk][frac] = set(speakers)

        for run in range(n_runs):
            logger.info(f'[geo={geo_level}][L={L}] fold={fold} run={run}: dataset_speakers={len(dataset_speakers)}, eligible_speakers={len(eligible_speakers)}')
            N_dataset_speakers[run] = {}
            known_sets[run] = {}

            for N in values_for_N:
                N_dataset_speakers[run][N] = {}
                known_sets[run][N] = {}

                for p_spk in eligible_speakers:
                    speakers = set(dataset_speakers.keys())
                    speakers.discard(p_spk)
                    if (N - 1) > len(speakers):
                        raise ValueError(f"Cannot sample N-1={N-1} speakers (available={len(speakers)}) for N={N}, L={L}.")
                    chosen = random.sample(list(speakers), N - 1)

                    # dataset sample (size N)
                    sample_dict = {s: dataset_speakers[s] for s in chosen}
                    sample_dict[p_spk] = dataset_speakers[p_spk]
                    N_dataset_speakers[run][N][p_spk] = sample_dict

                    # pre-sample known speakers sets for each frac (ensure p_spk included)
                    speaker_list = list(sample_dict.keys())  # size N
                    others = [s for s in speaker_list if s != p_spk]

                    known_sets[run][N].setdefault(p_spk, {})
                    for frac in known_fracs:
                        k = max(1, int(round(frac * len(speaker_list))))
                        k_others = max(0, k - 1)
                        known = set(random.sample(others, k_others)) if k_others > 0 else set()
                        known.add(p_spk)
                        known_sets[run][N][p_spk][frac] = known

        runs_iter = itertools.product(values_for_N, eligible_speakers, range(n_runs), known_fracs)
        total = len(values_for_N) * len(eligible_speakers) * n_runs * len(known_fracs)

        for N, p_spk, run, frac in tqdm(
            runs_iter,
            total=total,
            desc=f'geo={geo_level}, L={L}, fold={fold}',
        ):
            key = f'{p_spk}_{fold}'

            if key not in isolation_results[frac][L][N]:
                isolation_results[frac][L][N][key] = []

            iso = _plot2_run((
                p_spk,
                N_dataset_speakers[run][N][p_spk],
                similarity_scores.loc[p_spk],
                N, L, fold,
                calibration_conversations, test_conversations,
                run,
                results_file,
                geo_level, geo_df, geo_index, speaker_to_convs,
                frac,
                known_sets[run][N][p_spk][frac],
            ))[1]

            isolation_results[frac][L][N][key].append(iso)

    # -----------------------------
    # Aggregate for plotting (one curve per known_frac)
    # -----------------------------
    fig, ax = plt.subplots()

    for frac in known_fracs:
        # For each N: mean over speakers(keys), where each key has mean over (runs * folds)
        isolation_results_agg = {
            N: [np.mean(isolation_results[frac][L][N][s]) for s in isolation_results[frac][L][N]]
            for N in isolation_results[frac][L]
        }
        isolation_scores = {N: np.mean(isolation_results_agg[N]) for N in isolation_results_agg}
        isolation_scores_std = {N: np.std(isolation_results_agg[N]) for N in isolation_results_agg}

        x = list(isolation_scores.keys())
        y = [isolation_scores[n] for n in x]
        yerr = [isolation_scores_std[n] for n in x]

        ax.errorbar(x, y, yerr=yerr, label=f'known={frac:.1f}')

    ax.axhline(y=0.37, label='trivial', c='black')
    ax.set_ylim(0, 1)
    ax.set_xlabel('N (dataset speakers)')
    ax.set_ylabel('Isolation probability')
    ax.set_title(f'PSO Plot2 — L={L} — geo={geo_level}')
    plt.legend(ncols=3)

    out_pdf = results_file.rsplit('.', 1)[0] + ".pdf"
    plt.savefig(out_pdf)
    logger.info(f'[geo={geo_level}] Saved plot: {out_pdf}')


# -----------------------------
# Main
# -----------------------------
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
    parser.add_argument('L', type=int)
    parser.add_argument('--enroll-folder', required=True, help='Path to enrollment folder (dataset pool)')
    parser.add_argument('--trial-folder', required=True, help='Path to trial folder (predicates source)')
    parser.add_argument('--geo-file', required=True, help='Path to geolocation CSV file')
    parser.add_argument('--geo-level', type=str, default='all',
                        choices=['all', 'none', 'MSOA', 'LSOA', 'OA', 'household_id'])
    parser.add_argument('--geo-known-fracs', type=str, default='1.0',
                        help="Comma-separated fractions of dataset speakers whose geo is known by attacker. "
                             "Example: 0.1,0.2,...,0.9")
    parser.add_argument('--n-jobs', type=int, default=30)
    args = parser.parse_args()

    L = args.L
    if not L:
        raise Exception('Pass a value for L')

    # Parse known fractions
    known_fracs = []
    for x in args.geo_known_fracs.split(','):
        x = x.strip()
        if not x:
            continue
        try:
            known_fracs.append(float(x))
        except ValueError:
            raise ValueError(f"Cannot parse fraction '{x}' in --geo-known-fracs")
    known_fracs = [f for f in known_fracs if 0.0 < f <= 1.0]
    known_fracs = sorted(set(known_fracs))
    if not known_fracs:
        raise ValueError("No valid fractions in --geo-known-fracs (must be in (0,1]).")
    logger.info(f'Known geo fractions: {known_fracs}')

    results_dir = './PSO_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    # IMPORTANT:
    # - ENROLL = 22024 speakers => base dataset
    # - TRIAL  = 4949 speakers  => predicates + fallback utterances
    enroll_folder = args.enroll_folder
    trial_folder = args.trial_folder

    file_handler = logging.FileHandler(f'{results_dir}/results.L{L}.{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'Results saved in {results_dir}')

    # Load geolocation once
    geo_df = load_geolocation(args.geo_file)

    # Determine which geo levels to run
    if args.geo_level == "all":
        geo_levels = ["none", "MSOA", "LSOA", "OA", "household_id"]
    else:
        geo_levels = [args.geo_level]

    # Load spk2utt
    enroll_spk2utt = load_spk2utt(f'{enroll_spk2utt_folder}/spk2utt')
    logger.info(f'{len(enroll_spk2utt)} speakers in ENROLL')

    trial_spk2utt = load_spk2utt(f'{trial_spk2utt_folder}/spk2utt')
    logger.info(f'{len(trial_spk2utt)} speakers in TRIAL')

    # Predicates from TRIAL; keep remainder for fallback
    predicates_spk2utt, trial_remainder_spk2utt = choose_predicate_speakers_and_trial_remainder(
        trial_spk2utt, N_UTTERANCES_PER_SPEAKER
    )
    logger.info(f'{len(predicates_spk2utt)} trial speakers usable for predicates (>= {N_UTTERANCES_PER_SPEAKER} utts)')

    # Filter predicates to only those present in geolocation file (using base ID)
    geo_spk_ids = set(geo_df.index)
    predicates_spk2utt_filtered = {s: utts for s, utts in predicates_spk2utt.items() if extract_base_spk_id(s) in geo_spk_ids}
    logger.info(f'{len(predicates_spk2utt_filtered)} predicate speakers after geo filtering (base ID present in geolocation.csv)')
    predicates_spk2utt = predicates_spk2utt_filtered

    # Build dataset: ENROLL-first + TRIAL remainder fallback to satisfy K >= 2L
    dataset_spk2utt_merged, stats = build_dataset_from_enroll_with_trial_fallback(
        enroll_spk2utt=enroll_spk2utt,
        trial_remainder_spk2utt=trial_remainder_spk2utt,
        L=L
    )
    logger.info(
        f"[L={L}] Dataset build stats: enroll_total={stats['enroll_total']}, need={stats['need_2L']}, "
        f"only_enroll_ok={stats['only_enroll_ok']}, topped_up_ok={stats['topped_up_ok']}, "
        f"kept_total={stats['kept_total']}, dropped={stats['dropped']}"
    )

    # Filter dataset speakers to only those in geolocation file (using base ID)
    dataset_spk2utt_merged = {s: utts for s, utts in dataset_spk2utt_merged.items() if extract_base_spk_id(s) in geo_spk_ids}
    logger.info(f'[L={L}] {len(dataset_spk2utt_merged)} dataset speakers after geo filtering')

    # DEV/DEBUG subsampling
    if dev or debug:
        predicates_spk2utt = dict(itertools.islice(predicates_spk2utt.items(), N_PREDICATES_SPEAKERS_TO_LOAD))
        logger.info(f'[DEV/DEBUG] keep {len(predicates_spk2utt)} predicate speakers')
        if len(dataset_spk2utt_merged) > N_DATASET_SPEAKERS_TO_LOAD:
            keys = list(dataset_spk2utt_merged.keys())
            random.shuffle(keys)
            keys = keys[:N_DATASET_SPEAKERS_TO_LOAD]
            dataset_spk2utt_merged = {k: dataset_spk2utt_merged[k] for k in keys}
            logger.info(f'[DEV/DEBUG] keep {len(dataset_spk2utt_merged)} dataset speakers')

    # Split dataset utterances by which H5 they belong to:
    enroll_set_by_spk = {s: set(utts) for s, utts in enroll_spk2utt.items()}

    spk2utt_dataset_enroll = {}
    spk2utt_dataset_trial = {}

    for s, utts in dataset_spk2utt_merged.items():
        enroll_set = enroll_set_by_spk.get(s, set())
        e_utts = [u for u in utts if u in enroll_set]
        t_utts = [u for u in utts if u not in enroll_set]  # fallback from trial remainder
        if e_utts:
            spk2utt_dataset_enroll[s] = e_utts
        if t_utts:
            spk2utt_dataset_trial[s] = t_utts

    # Predicates xvectors: from TRIAL H5 only
    spk2utt_predicates = predicates_spk2utt

    # Load xvectors
    logger.info("Loading xvectors for predicates (TRIAL H5)...")
    x_vectors_trial_pred = load_x_vectors(f'{trial_folder}/xvectors.h5', spk2utt_predicates)

    logger.info("Loading xvectors for dataset ENROLL part (ENROLL H5)...")
    x_vectors_enroll_ds = load_x_vectors(f'{enroll_folder}/xvectors.h5', spk2utt_dataset_enroll) if spk2utt_dataset_enroll else {}

    logger.info("Loading xvectors for dataset TRIAL fallback part (TRIAL H5)...")
    x_vectors_trial_ds = load_x_vectors(f'{trial_folder}/xvectors.h5', spk2utt_dataset_trial) if spk2utt_dataset_trial else {}

    # Assemble x_vectors_for_dataset merged
    x_vectors_for_dataset = {}
    for s in dataset_spk2utt_merged:
        x_vectors_for_dataset[s] = {}
        if s in x_vectors_enroll_ds:
            x_vectors_for_dataset[s].update(x_vectors_enroll_ds[s])
        if s in x_vectors_trial_ds:
            x_vectors_for_dataset[s].update(x_vectors_trial_ds[s])

    x_vectors_for_predicates = x_vectors_trial_pred

    logger.info(f'[L={L}] x_vectors_for_predicates speakers: {len(x_vectors_for_predicates)}')
    logger.info(f'[L={L}] x_vectors_for_dataset speakers: {len(x_vectors_for_dataset)}')

    # Run per geo level
    for geo_level in geo_levels:
        results_file_plot2 = f'{results_dir}/plot2.L{L}.{geo_level}.knowns_{args.geo_known_fracs}.{timestamp}.csv'
        logger.info(f'PLOT 2 geo_level={geo_level} -> {results_file_plot2}')

        geo_index = None
        if geo_level != "none":
            geo_index = build_index(geo_df, geo_level)

        t0 = time.time()
        plot2(
            x_vectors_for_predicates, N_PREDICATES,
            x_vectors_for_dataset, N_CONVERSATIONS_PER_SPEAKER,
            list(VALUES_FOR_N), L, N_RUNS,
            results_file_plot2,
            geo_level=geo_level,
            geo_df=geo_df,
            geo_index=geo_index,
            known_fracs=known_fracs,
            seed=0,
            n_jobs=args.n_jobs
        )
        logger.info(f'Done geo_level={geo_level} in {time.time() - t0:.3f}s')