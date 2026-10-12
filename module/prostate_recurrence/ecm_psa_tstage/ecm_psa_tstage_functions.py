import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# RANKING FUNCTIONS
# ============================================================

def fit_probability_ranker(probabilities):
    """Fit an empirical ranker using finite training probabilities.

    Converts probabilities to a sorted NumPy array after removing
    non-finite values. The resulting array is used to map probabilities
    to their empirical rank proportions.

    Args:
        probabilities: Training probabilities to use for fitting.

    Returns:
        A sorted NumPy array of finite training probabilities.

    Raises:
        ValueError: If no finite probabilities are available.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities)]

    if len(probabilities) == 0:
        raise ValueError(
            "Cannot fit ranker: no finite probabilities."
        )

    return np.sort(probabilities)


def apply_probability_ranker(ranker, probabilities):
    """Convert probabilities to empirical rank proportions.

    Each probability is mapped to the proportion of training
    probabilities less than or equal to it. Values are calculated
    using right-sided insertion, so ties receive the rank after
    all equal values.

    Args:
        ranker: Sorted array of finite training probabilities,
            typically returned by ``fit_probability_ranker``.
        probabilities: Probabilities to transform.

    Returns:
        A NumPy array of rank proportions between 0 and 1.

    Raises:
        ValueError: If the ranker contains no training probabilities.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(ranker)

    if n == 0:
        raise ValueError(
            "Ranker contains no training probabilities."
        )

    ranks = np.searchsorted(ranker, probabilities, side="right")

    return ranks / n


def rank_threshold(ranker, threshold):
    """Convert a probability threshold to its empirical rank proportion.

    Calculates the proportion of training probabilities less than
    or equal to the specified threshold, using right-sided insertion
    so that values tied with the threshold are included.

    Args:
        ranker: Sorted array of finite training probabilities,
            typically returned by ``fit_probability_ranker``.
        threshold: Probability threshold to transform into a rank
            proportion.

    Returns:
        The empirical rank proportion of the threshold, between
        0 and 1.

    Raises:
        ValueError: If the ranker contains no training probabilities.
    """
    threshold = float(threshold)
    n = len(ranker)

    if n == 0:
        raise ValueError(
            "Ranker contains no training probabilities."
        )

    rank = np.searchsorted(ranker, threshold, side="right")

    return rank / n



# ============================================================
# LOAD INNER DATA
# ============================================================

def load_inner_data(
    psa_dir,
    ecm_dir,
    subfile_name,
    bootstrap,
    outer_fold
):
    """Load and combine PSA and ECM predictions from inner folds.

    Reads the selected-model prediction files for all five inner folds
    of a given outer fold and bootstrap. Retains the patient identifier,
    outcome label, recurrence probability, and classification threshold,
    renaming the probability and threshold columns to distinguish PSA
    from ECM.

    Args:
        psa_dir: Directory containing PSA prediction files.
        ecm_dir: Directory containing ECM prediction files.
        subfile_name: Subfile identifier used in the filenames.
        bootstrap: Bootstrap iteration number.
        outer_fold: Outer cross-validation fold number.

    Returns:
        A tuple containing:
            - psa_inner: Combined PSA predictions across the five
              inner folds.
            - ecm_inner: Combined ECM predictions across the five
              inner folds.

        Both DataFrames contain ``patient_id``, ``case``, a model-specific
        probability column, and a model-specific threshold column.
    """
    psa_inner = []
    ecm_inner = []

    for inner_fold in range(5):
        psa_file = os.path.join(
            psa_dir,
            f"outer_fold_{outer_fold:05d}_"
            f"inner_fold_{inner_fold:05d}_"
            f"{subfile_name}_"
            f"shuffle_{bootstrap:02d}_"
            f"selected_model_from_inner_"
            f"psa_patient_probabilities_ecm_threshold.csv"
        )

        ecm_file = os.path.join(
            ecm_dir,
            f"outer_fold_{outer_fold:05d}_"
            f"inner_fold_{inner_fold:05d}_"
            f"{subfile_name}_"
            f"shuffle_{bootstrap:02d}_"
            f"selected_model_from_inner_"
            f"ecm_patient_probabilities.csv"
        )

        psa_inner.append(pd.read_csv(psa_file))
        ecm_inner.append(pd.read_csv(ecm_file))

    psa_inner = pd.concat(
        psa_inner,
        ignore_index=True
    )

    ecm_inner = pd.concat(
        ecm_inner,
        ignore_index=True
    )

    psa_inner = (
        psa_inner[
            [
                "patient_id",
                "case",
                "recurrence_probability",
                "classification_probability_threshold",
            ]
        ]
        .rename(
            columns={
                "recurrence_probability": "psa_probability",
                "classification_probability_threshold": "psa_threshold",
            }
        )
    )

    ecm_inner = (
        ecm_inner[
            [
                "patient_id",
                "case",
                "recurrence_probability",
                "classification_probability_threshold",
            ]
        ]
        .rename(
            columns={
                "recurrence_probability": "ecm_probability",
                "classification_probability_threshold": "ecm_threshold",
            }
        )
    )

    return psa_inner, ecm_inner


# ============================================================
# LOAD OUTER DATA
# ============================================================

def load_outer_data(
    psa_dir,
    ecm_dir,
    subfile_name,
    bootstrap,
    outer_fold
):
    """Load and merge PSA and ECM predictions for an outer fold.

    Reads the selected-model prediction files for the specified outer
    fold and bootstrap. Retains the patient identifier, outcome label,
    recurrence probability, and classification threshold, renaming
    model-specific columns before merging the predictions by patient
    identifier and outcome label.

    Args:
        psa_dir: Directory containing PSA prediction files.
        ecm_dir: Directory containing ECM prediction files.
        subfile_name: Subfile identifier used in the filenames.
        bootstrap: Bootstrap iteration number.
        outer_fold: Outer cross-validation fold number.

    Returns:
        A DataFrame containing matched PSA and ECM predictions, with
        columns for patient ID, outcome, each model's recurrence
        probability, and each model's classification threshold.

    Raises:
        FileNotFoundError: If either prediction file does not exist.
        pandas.errors.MergeError: If the merge validation fails, if
            validation is added to the merge operation.
    """
    psa_file = os.path.join(
        psa_dir,
        f"outer_fold_{outer_fold:05d}_"
        f"{subfile_name}_"
        f"shuffle_{bootstrap:02d}_"
        f"selected_model_from_inner_"
        f"psa_patient_probabilities_ecm_threshold.csv"
    )

    ecm_file = os.path.join(
        ecm_dir,
        f"outer_fold_{outer_fold:05d}_"
        f"{subfile_name}_"
        f"shuffle_{bootstrap:02d}_"
        f"selected_model_from_inner_"
        f"ecm_patient_probabilities.csv"
    )

    psa_outer = pd.read_csv(psa_file)
    ecm_outer = pd.read_csv(ecm_file)

    psa_outer = (
        psa_outer[
            [
                "patient_id",
                "case",
                "recurrence_probability",
                "classification_probability_threshold",
            ]
        ]
        .rename(
            columns={
                "recurrence_probability": "psa_probability",
                "classification_probability_threshold": "psa_threshold",
            }
        )
    )

    ecm_outer = (
        ecm_outer[
            [
                "patient_id",
                "case",
                "recurrence_probability",
                "classification_probability_threshold",
            ]
        ]
        .rename(
            columns={
                "recurrence_probability": "ecm_probability",
                "classification_probability_threshold": "ecm_threshold",
            }
        )
    )

    outer = psa_outer.merge(
        ecm_outer,
        on=["patient_id", "case"],
    )

    return outer


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability,
    prediction
):
    """Calculate ROC-AUC and class-specific classification metrics.

    Computes ROC-AUC from predicted recurrence probabilities and
    calculates F1-score, precision, and recall separately for recurrent
    (class 1) and non-recurrent (class 0) patients.

    ROC-AUC is calculated only when both classes are present. If the
    input contains no patients, the function returns the default metrics
    with ``n = 0`` and all other values set to NaN. Undefined precision
    and recall values are set to zero by the scikit-learn metric
    functions.

    Args:
        y_true: True binary outcome labels, where 1 indicates recurrence
            and 0 indicates non-recurrence.
        probability: Predicted recurrence probabilities, used to
            calculate ROC-AUC.
        prediction: Predicted binary class labels, used to calculate
            F1-score, precision, and recall.

    Returns:
        A dictionary containing:
            - ``n``: Number of observations.
            - ``roc_auc``: Area under the ROC curve.
            - ``f1_recur``: F1-score for recurrence.
            - ``f1_nonrecur``: F1-score for non-recurrence.
            - ``precision_recur``: Precision for recurrence.
            - ``recall_recur``: Recall (sensitivity) for recurrence.
            - ``precision_nonrecur``: Precision for non-recurrence.
            - ``recall_nonrecur``: Recall (specificity) for
              non-recurrence.
    """
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    prediction = np.asarray(prediction)

    metrics = {
        "n": len(y_true),
        "roc_auc": np.nan,
        "f1_recur": np.nan,
        "f1_nonrecur": np.nan,
        "precision_recur": np.nan,
        "recall_recur": np.nan,
        "precision_nonrecur": np.nan,
        "recall_nonrecur": np.nan,
    }

    if len(y_true) == 0:
        return metrics

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probability,
        )

    # --------------------------------------------------------
    # F1-SCORE
    # --------------------------------------------------------

    metrics["f1_recur"] = f1_score(
        y_true,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    metrics["f1_nonrecur"] = f1_score(
        y_true,
        prediction,
        pos_label=0,
        zero_division=0,
    )

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    metrics["precision_recur"] = precision_score(
        y_true,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    metrics["precision_nonrecur"] = precision_score(
        y_true,
        prediction,
        pos_label=0,
        zero_division=0,
    )

    # --------------------------------------------------------
    # RECALL / SENSITIVITY
    # --------------------------------------------------------

    metrics["recall_recur"] = recall_score(
        y_true,
        prediction,
        pos_label=1,
        zero_division=0,
    )

    metrics["recall_nonrecur"] = recall_score(
        y_true,
        prediction,
        pos_label=0,
        zero_division=0,
    )

    return metrics



# ============================================================
# END OF HELPER FUNCTIONS
# ============================================================



def evaluate_roc_auc_by_ecm_probability_region(
    base_dir,
    bootstraps,
    outer_folds,
    subfile_name
):
    """Evaluate PSA and ECM ROC-AUC across ECM probability regions.

    For each bootstrap replicate and outer fold, loads the selected
    inner-fold PSA and ECM patient-level predictions, fits empirical
    probability rankers on the inner data, and compares the ROC-AUC
    of the two models within three ECM ranked-probability regions.

    Regions:
        (0.00, 0.33]
        (0.33, 0.67]
        (0.67, 1.00]

    The first region includes ranked probabilities equal to zero.

    Parameters
    ----------
    base_dir : str
        Base directory containing the PSA and ECM model outputs.
    bootstraps : iterable
        Bootstrap replicate identifiers.
    outer_folds : iterable
        Outer-fold identifiers.
    subfile_name : str
        Filename component passed to load_inner_data().

    Returns
    -------
    pd.DataFrame
        ROC-AUC results for each bootstrap, outer fold and region.
        Delta is ECM ROC-AUC minus PSA ROC-AUC.
    """

    # ============================================================
    # RESULTS
    # ============================================================

    results = []

    # ============================================================
    # BOOTSTRAP / OUTER-FOLD LOOP
    # ============================================================

    for bootstrap in bootstraps:

        for outer_fold in outer_folds:

            print(
                f"\nBootstrap {bootstrap:02d} | "
                f"Outer fold {outer_fold}"
            )

            # ====================================================
            # DIRECTORIES
            # ====================================================

            psa_dir = os.path.join(
                base_dir,
                f"gleason_7_patient_level_logistic_"
                f"psa_tstage_shuffle_"
                f"{bootstrap:02d}"
            )

            ecm_dir = os.path.join(
                base_dir,
                f"gleason_7_patient_level_logistic_"
                f"ecm_xgboost_shuffle_{bootstrap:02d}"
            )

            # ====================================================
            # LOAD INNER DATA
            # ====================================================

            psa_inner, ecm_inner = load_inner_data(
                psa_dir,
                ecm_dir,
                subfile_name,
                bootstrap,
                outer_fold
            )

            inner = psa_inner.merge(
                ecm_inner,
                on=[
                    "patient_id",
                    "case"
                ]
            )

            # ====================================================
            # FIT PROBABILITY RANKERS
            #
            # IMPORTANT:
            # Fit on inner data only.
            # ====================================================

            psa_ranker = fit_probability_ranker(
                inner["psa_probability"]
            )

            ecm_ranker = fit_probability_ranker(
                inner["ecm_probability"]
            )

            # ====================================================
            # RANK INNER PROBABILITIES
            # ====================================================

            inner["psa_ranked_probability"] = (
                apply_probability_ranker(
                    psa_ranker,
                    inner["psa_probability"]
                )
            )

            inner["ecm_ranked_probability"] = (
                apply_probability_ranker(
                    ecm_ranker,
                    inner["ecm_probability"]
                )
            )

            # ====================================================
            # ECM PROBABILITY REGIONS
            # ====================================================

            for lb, ub in zip(
                [0.00, 0.33, 0.67],
                [0.33, 0.67, 1.00]
            ):

                lb = np.round(lb, 2)
                ub = np.round(ub, 2)

                # ------------------------------------------------
                # SELECT ECM PROBABILITY REGION
                # ------------------------------------------------

                if lb == 0:

                    use_ecm = (
                        inner["ecm_ranked_probability"] <= ub
                    )

                else:

                    use_ecm = (
                        (
                            inner["ecm_ranked_probability"] > lb
                        )
                        &
                        (
                            inner["ecm_ranked_probability"] <= ub
                        )
                    )

                # ------------------------------------------------
                # REGION DATA
                # ------------------------------------------------

                region = inner.loc[use_ecm]

                y_region = region["case"]

                ecm_probability = region["ecm_probability"]
                psa_probability = region["psa_probability"]

                # ------------------------------------------------
                # ROC AUC
                # ------------------------------------------------

                if y_region.nunique() == 2:

                    ecm_roc_auc = roc_auc_score(
                        y_region,
                        ecm_probability
                    )

                    psa_roc_auc = roc_auc_score(
                        y_region,
                        psa_probability
                    )

                    delta = (
                        ecm_roc_auc - psa_roc_auc
                    )

                else:

                    ecm_roc_auc = np.nan
                    psa_roc_auc = np.nan
                    delta = np.nan

                # ------------------------------------------------
                # STORE RESULTS
                # ------------------------------------------------

                results.append({

                    "bootstrap": bootstrap,
                    "outer_fold": outer_fold,
                    "no_patients": len(inner),

                    "lb": lb,
                    "ub": ub,
                    "midpoint": (lb + ub) / 2,

                    "ecm_roc_auc": ecm_roc_auc,
                    "psa_roc_auc": psa_roc_auc,
                    "delta": delta
                })

    # ============================================================
    # RESULTS DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(results)

    return results_df








def evaluate_inner_psa_ecm_gate(
    psa_dir,
    ecm_dir,
    run_dir,
    subfile_name,
    outer_fold,
    bootstrap
):
    """Evaluate candidate ECM gate boundaries using inner-fold data.

    Loads the selected PSA and ECM patient-level probabilities from
    the five inner folds, fits empirical probability rankers using
    the inner data, and evaluates a combined PSA/ECM model across
    candidate ECM gate boundaries.

    The ECM model is used only for patients whose ranked ECM
    probability exceeds the candidate boundary. All other patients
    are classified using the PSA model.

    Parameters
    ----------
    psa_dir : str
        Directory containing selected PSA inner-fold predictions.
    ecm_dir : str
        Directory containing selected ECM inner-fold predictions.
    subfile_name : str
        Subfile name for PSA and ECM output files.    
    run_dir : str
        Directory in which to save the performance CSV.
    outer_fold : int
        Outer-fold identifier.
    bootstrap : int
        Bootstrap replicate identifier.

    Returns
    -------
    pd.DataFrame
        Performance metrics for each candidate gate boundary.
    """

    # ============================================================
    # LOAD INNER DATA
    # ============================================================
    psa_inner, ecm_inner =  load_inner_data(
        psa_dir,
        ecm_dir,
        subfile_name,
        bootstrap,
        outer_fold
    )

    # ============================================================
    # CLASSIFICATION THRESHOLDS
    # ============================================================

    mean_psa_probability_threshold = round(
        psa_inner["psa_threshold"].mean(),
        2
    )

    mean_ecm_probability_threshold = round(
        ecm_inner["ecm_threshold"].mean(),
        2
    )

    # ============================================================
    # FIT EMPIRICAL RANKERS
    # ============================================================

    psa_ranker = fit_probability_ranker(
        psa_inner["psa_probability"]
    )

    ecm_ranker = fit_probability_ranker(
        ecm_inner["ecm_probability"]
    )

    # ============================================================
    # RANK INNER PROBABILITIES
    # ============================================================

    psa_inner["psa_ranked_probability"] = (
        apply_probability_ranker(
            psa_ranker,
            psa_inner["psa_probability"]
        )
    )

    ecm_inner["ecm_ranked_probability"] = (
        apply_probability_ranker(
            ecm_ranker,
            ecm_inner["ecm_probability"]
        )
    )

    # ============================================================
    # RANK CLASSIFICATION THRESHOLDS
    # ============================================================

    psa_ranked_threshold = rank_threshold(
        psa_ranker,
        mean_psa_probability_threshold
    )

    ecm_ranked_threshold = rank_threshold(
        ecm_ranker,
        mean_ecm_probability_threshold
    )

    # ============================================================
    # INNER CLASSIFICATION
    # ============================================================

    psa_inner["psa_predicted_class"] = (
        psa_inner["psa_probability"]
        > mean_psa_probability_threshold
    ).astype(int)

    ecm_inner["ecm_predicted_class"] = (
        ecm_inner["ecm_probability"]
        > mean_ecm_probability_threshold
    ).astype(int)

    # ============================================================
    # MERGE PSA + ECM
    # ============================================================

    inner = psa_inner.merge(
        ecm_inner,
        on=["patient_id", "case"]
    )

    # ============================================================
    # CANDIDATE ECM GATE BOUNDARIES
    # ============================================================

    candidate_boundaries = np.round(
        np.arange(
            0.75,
            1.00,
            0.01
        ),
        2
    )

    # ============================================================
    # EVALUATE CANDIDATE BOUNDARIES
    # ============================================================

    rows = []

    for candidate_boundary in candidate_boundaries:

        # --------------------------------------------------------
        # DEFINE ECM / PSA REGIONS
        # --------------------------------------------------------

        use_ecm = (
            inner["ecm_ranked_probability"]
            > candidate_boundary
        )

        psa_region = ~use_ecm

        # --------------------------------------------------------
        # REGION SIZES
        # --------------------------------------------------------

        n_ecm = int(use_ecm.sum())
        n_psa = int(psa_region.sum())

        ecm_fraction = (
            n_ecm / len(inner)
            if len(inner) > 0
            else np.nan
        )

        psa_fraction = (
            n_psa / len(inner)
            if len(inner) > 0
            else np.nan
        )

        # --------------------------------------------------------
        # RECURRENCE COMPOSITION
        # --------------------------------------------------------

        n_ecm_recur = int(
            (
                (inner["case"] == 1) & use_ecm
            ).sum()
        )

        n_ecm_nonrecur = int(
            (
                (inner["case"] == 0) & use_ecm
            ).sum()
        )

        n_psa_recur = int(
            (
                (inner["case"] == 1) & psa_region
            ).sum()
        )

        n_psa_nonrecur = int(
            (
                (inner["case"] == 0) & psa_region
            ).sum()
        )

        # --------------------------------------------------------
        # RECURRENCE RATES
        # --------------------------------------------------------

        overall_recurrence_rate = inner["case"].mean()

        ecm_recurrence_rate = (
            n_ecm_recur / n_ecm
            if n_ecm > 0
            else np.nan
        )

        psa_recurrence_rate = (
            n_psa_recur / n_psa
            if n_psa > 0
            else np.nan
        )

        # --------------------------------------------------------
        # ENRICHMENT
        # --------------------------------------------------------

        ecm_recurrence_enrichment = (
            ecm_recurrence_rate / overall_recurrence_rate
            if (
                pd.notna(ecm_recurrence_rate)
                and overall_recurrence_rate > 0
            )
            else np.nan
        )

        ecm_vs_psa_recurrence_rate_difference = (
            ecm_recurrence_rate - psa_recurrence_rate
            if (
                pd.notna(ecm_recurrence_rate)
                and pd.notna(psa_recurrence_rate)
            )
            else np.nan
        )

        ecm_vs_psa_recurrence_rate_ratio = (
            ecm_recurrence_rate / psa_recurrence_rate
            if (
                pd.notna(ecm_recurrence_rate)
                and pd.notna(psa_recurrence_rate)
                and psa_recurrence_rate > 0
            )
            else np.nan
        )

        # ========================================================
        # COMBINED MODEL
        # ========================================================

        combined_probability = np.where(
            use_ecm,
            inner["ecm_ranked_probability"],
            inner["psa_ranked_probability"]
        )

        combined_prediction = np.where(
            use_ecm,
            inner["ecm_predicted_class"],
            inner["psa_predicted_class"]
        )

        # --------------------------------------------------------
        # OVERALL COMBINED PERFORMANCE
        # --------------------------------------------------------

        if inner["case"].nunique() == 2:

            roc_auc = roc_auc_score(
                inner["case"],
                combined_probability
            )

        else:
            roc_auc = np.nan

        f1_score_recur = f1_score(
            inner["case"],
            combined_prediction,
            pos_label=1,
            zero_division=0
        )

        f1_score_nonrecur = f1_score(
            inner["case"],
            combined_prediction,
            pos_label=0,
            zero_division=0
        )

        # ========================================================
        # ECM REGION PERFORMANCE
        # ========================================================

        if n_ecm > 0:

            ecm_case = inner.loc[
                use_ecm,
                "case"
            ]

            ecm_prediction = inner.loc[
                use_ecm,
                "ecm_predicted_class"
            ]

            ecm_probability = inner.loc[
                use_ecm,
                "ecm_ranked_probability"
            ]

            if ecm_case.nunique() == 2:

                ecm_region_roc_auc = roc_auc_score(
                    ecm_case,
                    ecm_probability
                )

            else:
                ecm_region_roc_auc = np.nan

            ecm_region_f1_recur = f1_score(
                ecm_case,
                ecm_prediction,
                pos_label=1,
                zero_division=0
            )

            ecm_region_f1_nonrecur = f1_score(
                ecm_case,
                ecm_prediction,
                pos_label=0,
                zero_division=0
            )

        else:

            ecm_region_roc_auc = np.nan
            ecm_region_f1_recur = np.nan
            ecm_region_f1_nonrecur = np.nan

        # ========================================================
        # PSA REGION PERFORMANCE
        # ========================================================

        if n_psa > 0:

            psa_case = inner.loc[
                psa_region,
                "case"
            ]

            psa_prediction = inner.loc[
                psa_region,
                "psa_predicted_class"
            ]

            psa_probability = inner.loc[
                psa_region,
                "psa_ranked_probability"
            ]

            if psa_case.nunique() == 2:

                psa_region_roc_auc = roc_auc_score(
                    psa_case,
                    psa_probability
                )

            else:
                psa_region_roc_auc = np.nan

            psa_region_f1_recur = f1_score(
                psa_case,
                psa_prediction,
                pos_label=1,
                zero_division=0
            )

            psa_region_f1_nonrecur = f1_score(
                psa_case,
                psa_prediction,
                pos_label=0,
                zero_division=0
            )

        else:

            psa_region_roc_auc = np.nan
            psa_region_f1_recur = np.nan
            psa_region_f1_nonrecur = np.nan

        # ========================================================
        # SAVE CANDIDATE RESULT
        # ========================================================

        rows.append({

            # Gate
            "candidate_boundary": candidate_boundary,

            # Region sizes
            "n_ecm": n_ecm,
            "n_psa": n_psa,
            "ecm_fraction": ecm_fraction,
            "psa_fraction": psa_fraction,

            # Class counts
            "n_ecm_recur": n_ecm_recur,
            "n_ecm_nonrecur": n_ecm_nonrecur,
            "n_psa_recur": n_psa_recur,
            "n_psa_nonrecur": n_psa_nonrecur,

            # Recurrence
            "overall_recurrence_rate": overall_recurrence_rate,
            "ecm_recurrence_rate": ecm_recurrence_rate,
            "psa_recurrence_rate": psa_recurrence_rate,

            # Enrichment
            "ecm_recurrence_enrichment":
                ecm_recurrence_enrichment,

            "ecm_vs_psa_recurrence_rate_difference":
                ecm_vs_psa_recurrence_rate_difference,

            "ecm_vs_psa_recurrence_rate_ratio":
                ecm_vs_psa_recurrence_rate_ratio,

            # Combined model
            "roc_auc": roc_auc,
            "f1_score_recur": f1_score_recur,
            "f1_score_nonrecur": f1_score_nonrecur,

            # ECM region
            "ecm_region_roc_auc": ecm_region_roc_auc,
            "ecm_region_f1_recur": ecm_region_f1_recur,
            "ecm_region_f1_nonrecur": ecm_region_f1_nonrecur,

            # PSA region
            "psa_region_roc_auc": psa_region_roc_auc,
            "psa_region_f1_recur": psa_region_f1_recur,
            "psa_region_f1_nonrecur": psa_region_f1_nonrecur
        })

    # ============================================================
    # FINAL INNER PERFORMANCE DATAFRAME
    # ============================================================

    df_inner_performance = pd.DataFrame(rows)

    # ============================================================
    # SAVE
    # ============================================================

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    save_name = (
        "outer_fold_test_"
        + str(outer_fold).zfill(5)
        + "_psa_ecm_gate_patient_level"
    )

    filename = os.path.join(
        run_dir,
        save_name + ".csv"
    )

    df_inner_performance.to_csv(
        filename,
        index=False
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\nSaved:", filename)

    print("\nCandidate boundaries:", candidate_boundaries)

    print("\nPSA probability threshold:",
          mean_psa_probability_threshold)

    print("PSA ranked threshold:", psa_ranked_threshold)

    print("\nECM probability threshold:",
          mean_ecm_probability_threshold)

    print("ECM ranked threshold:", ecm_ranked_threshold)

    print("\nInner gate performance:")

    print(
        df_inner_performance[
            [
                "candidate_boundary",
                "ecm_fraction",
                "ecm_recurrence_enrichment",
                "roc_auc",
                "f1_score_recur",
                "f1_score_nonrecur"
            ]
        ].to_string(index=False)
    )

    return df_inner_performance










def evaluate_psa_ecm_gate_outer_fold(
    base_dir,
    subfile_name,
    bootstrap,
    outer_fold
):
    """Evaluate the PSA/ECM gated model on one outer-test fold.

    Selects the gate boundary from the saved candidate-results CSV,
    fits probability rankers and obtains classification thresholds
    from inner data, evaluates on outer-test data, and saves
    patient-level predictions.

    Returns
    -------
    dict
        Summary of the model performance and gate composition.
    """

    print(
        f"\nBootstrap {bootstrap:02d} | "
        f"Outer fold {outer_fold}"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    psa_dir = os.path.join(
        base_dir,
        f"gleason_7_patient_level_logistic_"
        f"psa_tstage_shuffle_{bootstrap:02d}"
    )

    ecm_dir = os.path.join(
        base_dir,
        f"gleason_7_patient_level_logistic_"
        f"ecm_xgboost_shuffle_{bootstrap:02d}"
    )

    run_dir = os.path.join(
        base_dir,
        f"gleason_7_patient_level_psa_ecm_gate_"
        f"right_tail_shuffle_{bootstrap:02d}"
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    # ========================================================
    # SELECT GATE BOUNDARY
    # ========================================================

    save_name = (
        "outer_fold_test_"
        + str(outer_fold).zfill(5)
        + "_psa_ecm_gate_patient_level"
    )

    

    filename = os.path.join(
        run_dir,
        save_name + ".csv"
    )

    df = pd.read_csv(filename)

    df = df[
        df["f1_score_recur"] >=
        df["f1_score_nonrecur"]
    ].copy()

    best_row = None

    for f1_score_lower in [0.9, 0.7, 0.5, 0.3, 0.1]:

        df_lower = df[
            df["f1_score_nonrecur"] >=
            f1_score_lower
        ].copy()

        if len(df_lower) == 0:
            continue

        max_recur = df_lower[
            "f1_score_recur"
        ].max()

        df_lower = df_lower[
            df_lower["f1_score_recur"] ==
            max_recur
        ].copy()

        max_auc = df_lower[
            "roc_auc"
        ].max()

        df_lower = df_lower[
            df_lower["roc_auc"] ==
            max_auc
        ].copy()

        median_boundary = df_lower[
            "candidate_boundary"
        ].median()

        df_lower["dist"] = np.abs(
            df_lower["candidate_boundary"] -
            median_boundary
        )

        df_lower = (
            df_lower
            .sort_values(
                ["dist", "candidate_boundary"]
            )
            .reset_index(drop=True)
        )

        best_row = df_lower.iloc[0]

        break

    if best_row is None:
        raise ValueError(
            f"No eligible gate boundary found for "
            f"bootstrap={bootstrap:02d}, outer_fold={outer_fold}"
        )

    gate_boundary = float(
        best_row["candidate_boundary"]
    )

    # ========================================================
    # LOAD INNER DATA
    # ========================================================

    psa_inner, ecm_inner = load_inner_data(
        psa_dir,
        ecm_dir,
        subfile_name,
        bootstrap,
        outer_fold
    )

    # ========================================================
    # CLASSIFICATION THRESHOLDS
    # ========================================================

    psa_probability_threshold = round(
        psa_inner["psa_threshold"].mean(),
        2
    )

    ecm_probability_threshold = round(
        ecm_inner["ecm_threshold"].mean(),
        2
    )

    # ========================================================
    # FIT RANKERS ON INNER DATA
    # ========================================================

    psa_ranker = fit_probability_ranker(
        psa_inner["psa_probability"]
    )

    ecm_ranker = fit_probability_ranker(
        ecm_inner["ecm_probability"]
    )

    # ========================================================
    # CONVERT CLASSIFICATION THRESHOLDS TO RANK SCALE
    # ========================================================

    psa_ranked_threshold = rank_threshold(
        psa_ranker,
        psa_probability_threshold
    )

    ecm_ranked_threshold = rank_threshold(
        ecm_ranker,
        ecm_probability_threshold
    )

    # ========================================================
    # LOAD OUTER TEST DATA
    # ========================================================

    outer = load_outer_data(
        psa_dir,
        ecm_dir,
        subfile_name,
        bootstrap,
        outer_fold
    )

    y = outer["case"].to_numpy()

    # ========================================================
    # RANK OUTER PROBABILITIES
    # ========================================================

    outer["psa_ranked_probability"] = (
        apply_probability_ranker(
            psa_ranker,
            outer["psa_probability"]
        )
    )

    outer["ecm_ranked_probability"] = (
        apply_probability_ranker(
            ecm_ranker,
            outer["ecm_probability"]
        )
    )

    # ========================================================
    # MODEL CLASSIFICATIONS
    # ========================================================

    outer["psa_prediction"] = (
        outer["psa_ranked_probability"] >
        psa_ranked_threshold
    ).astype(int)

    outer["ecm_prediction"] = (
        outer["ecm_ranked_probability"] >
        ecm_ranked_threshold
    ).astype(int)

    # ========================================================
    # APPLY ECM GATE
    # ========================================================

    use_ecm = (
        outer["ecm_ranked_probability"] >
        gate_boundary
    )

    outer["use_ecm"] = use_ecm.astype(int)

    psa_region = ~use_ecm
    ecm_region = use_ecm

    # ========================================================
    # COMBINED GATED MODEL
    # ========================================================

    outer["combined_probability"] = np.where(
        use_ecm,
        outer["ecm_ranked_probability"],
        outer["psa_ranked_probability"]
    )

    outer["combined_prediction"] = np.where(
        use_ecm,
        outer["ecm_prediction"],
        outer["psa_prediction"]
    )

    # ========================================================
    # REGION COUNTS
    # ========================================================

    n_patients = len(outer)

    n_ecm = int(ecm_region.sum())
    n_psa = int(psa_region.sum())

    n_ecm_recur = int(
        ((y == 1) & ecm_region).sum()
    )

    n_ecm_nonrecur = int(
        ((y == 0) & ecm_region).sum()
    )

    n_psa_recur = int(
        ((y == 1) & psa_region).sum()
    )

    n_psa_nonrecur = int(
        ((y == 0) & psa_region).sum()
    )

    overall_recurrence_rate = np.mean(y)

    ecm_recurrence_rate = (
        n_ecm_recur / n_ecm
        if n_ecm > 0
        else np.nan
    )

    psa_recurrence_rate = (
        n_psa_recur / n_psa
        if n_psa > 0
        else np.nan
    )

    ecm_recurrence_enrichment = (
        ecm_recurrence_rate / overall_recurrence_rate
        if overall_recurrence_rate > 0
        else np.nan
    )

    psa_recurrence_enrichment = (
        psa_recurrence_rate / overall_recurrence_rate
        if overall_recurrence_rate > 0
        else np.nan
    )

    # ========================================================
    # OVERALL MODEL PERFORMANCE
    # ========================================================

    psa_all = calculate_metrics(
        y,
        outer["psa_ranked_probability"],
        outer["psa_prediction"]
    )

    ecm_all = calculate_metrics(
        y,
        outer["ecm_ranked_probability"],
        outer["ecm_prediction"]
    )

    combined = calculate_metrics(
        y,
        outer["combined_probability"],
        outer["combined_prediction"]
    )

    # ========================================================
    # REGION-SPECIFIC PERFORMANCE
    # ========================================================

    psa_in_psa = calculate_metrics(
        y[psa_region],
        outer.loc[psa_region, "psa_ranked_probability"],
        outer.loc[psa_region, "psa_prediction"]
    )

    ecm_in_ecm = calculate_metrics(
        y[ecm_region],
        outer.loc[ecm_region, "ecm_ranked_probability"],
        outer.loc[ecm_region, "ecm_prediction"]
    )

    psa_in_ecm = calculate_metrics(
        y[ecm_region],
        outer.loc[ecm_region, "psa_ranked_probability"],
        outer.loc[ecm_region, "psa_prediction"]
    )

    ecm_in_psa = calculate_metrics(
        y[psa_region],
        outer.loc[psa_region, "ecm_ranked_probability"],
        outer.loc[psa_region, "ecm_prediction"]
    )

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    result = {
        # Identifiers
        "bootstrap": bootstrap,
        "outer_fold": outer_fold,

        # Gate and thresholds
        "gate_boundary": gate_boundary,
        "psa_probability_threshold": psa_probability_threshold,
        "ecm_probability_threshold": ecm_probability_threshold,
        "psa_ranked_threshold": psa_ranked_threshold,
        "ecm_ranked_threshold": ecm_ranked_threshold,

        # Sample and gate composition
        "n_patients": n_patients,
        "n_ecm": n_ecm,
        "n_psa": n_psa,
        "ecm_fraction": n_ecm / n_patients,
        "psa_fraction": n_psa / n_patients,
        "n_ecm_recur": n_ecm_recur,
        "n_ecm_nonrecur": n_ecm_nonrecur,
        "n_psa_recur": n_psa_recur,
        "n_psa_nonrecur": n_psa_nonrecur,
        "overall_recurrence_rate": overall_recurrence_rate,
        "ecm_recurrence_rate": ecm_recurrence_rate,
        "psa_recurrence_rate": psa_recurrence_rate,
        "ecm_recurrence_enrichment": ecm_recurrence_enrichment,
        "psa_recurrence_enrichment": psa_recurrence_enrichment,
    }

    # Add all the performance metrics using the same naming
    # convention as the original script.
    metric_groups = {
        "psa": psa_all,
        "ecm": ecm_all,
        "combined": combined,
        "psa_in_psa": psa_in_psa,
        "ecm_in_ecm": ecm_in_ecm,
        "psa_in_ecm": psa_in_ecm,
        "ecm_in_psa": ecm_in_psa,
    }

    metric_names = [
        "roc_auc",
        "f1_recur",
        "f1_nonrecur",
        "precision_recur",
        "recall_recur",
        "precision_nonrecur",
        "recall_nonrecur",
    ]

    for group_name, group_metrics in metric_groups.items():
        for metric_name in metric_names:
            # Preserve the original output columns, which only
            # included five metrics for the regional comparisons.
            if group_name in {
                "psa_in_psa",
                "ecm_in_ecm",
                "psa_in_ecm",
                "ecm_in_psa",
            } and metric_name in {
                "precision_nonrecur",
                "recall_nonrecur",
            }:
                continue

            result[f"{group_name}_{metric_name}"] = (
                group_metrics[metric_name]
            )

    # ========================================================
    # SAVE PATIENT-LEVEL PREDICTIONS
    # ========================================================

    prediction_filename = os.path.join(
        run_dir,
        (
            "outer_fold_test_"
            + str(outer_fold).zfill(5)
            + "_psa_ecm_gate_patient_predictions.csv"
        )
    )

    outer.to_csv(
        prediction_filename,
        index=False
    )

    return result

