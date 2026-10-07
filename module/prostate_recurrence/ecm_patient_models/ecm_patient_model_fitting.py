import ast
import os
import sys

import numpy as np
import pandas as pd
from glob import glob

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    metrics,
    roc_auc_score,
)



module_path = os.path.abspath(
    os.path.join(
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/',
        f'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_psa_outer_fold_{20}/'
    )
)

if module_path not in sys.path:
    sys.path.append(module_path)

import tile_model_fitting as tmf





def performance_results_df_generator(
    csv_paths,
    df_logistic,
    tile_count_thresholds,
    output_dir
):

    probability_thresholds = np.linspace(
        0.99,
        0.01,
        99
    )

    performance_results_df = pd.DataFrame()

    last_metrics_location = None
    last_metrics_df = None


    # ========================================================
    # LOOP OVER F1 CSV FILES
    # ========================================================

    for csv_path in csv_paths:

        print(csv_path)


        # ====================================================
        # LOAD F1 CSV
        # ====================================================

        df_input = load_csv_data(
            csv_path
        )


        # ====================================================
        # GET RUN INFORMATION
        # ====================================================

        run_name, outer_fold, run, logistic_params = (
            get_run_fold_and_logistic_params(
                csv_path,
                df_logistic
            )
        )


        # ====================================================
        # FIND CORRESPONDING METRICS CSV
        # ====================================================

        fold_string = str(
            outer_fold
        ).zfill(5)

        run_string = str(
            run
        ).zfill(5)

        metrics_location = os.path.join(
            output_dir,
            f'fold_{fold_string}',
            f'run_{run_string}',
            f'outer_fold_{fold_string}_run_{run_string}_inner_cv_metrics_patient_level.csv'
        )


        # ====================================================
        # LOAD METRICS CSV
        # ====================================================

        if metrics_location != last_metrics_location:

            try:

                last_metrics_df = pd.read_csv(
                    metrics_location
                )

            except (
                FileNotFoundError,
                pd.errors.EmptyDataError
            ):

                last_metrics_df = None

            last_metrics_location = metrics_location


        # ====================================================
        # COMBINATIONS
        # ====================================================

        unique_combinations = df_input[
            'combination'
        ].unique()


        for unique_combination in unique_combinations:

            df_combination_subset = df_input[
                df_input['combination'] == unique_combination
            ].copy()


            # =================================================
            # TILE-COUNT THRESHOLDS
            # =================================================

            for tile_count_threshold in tile_count_thresholds:

                df_threshold_subset = df_combination_subset[
                    df_combination_subset[
                        'tile_count_threshold'
                    ] == tile_count_threshold
                ]


                # =================================================
                # F1 SCORES
                # =================================================

                mean_f1_score_recur = []
                mean_f1_score_nonrecur = []


                for classification_probability_threshold in (
                    probability_thresholds
                ):

                    mask = np.isclose(
                        df_threshold_subset[
                            'classification_probability_threshold'
                        ],
                        classification_probability_threshold,
                        atol=1e-8
                    )


                    mean_f1_score_recur.append(
                        df_threshold_subset.loc[
                            mask,
                            'f1_recur'
                        ].mean()
                    )

                    mean_f1_score_nonrecur.append(
                        df_threshold_subset.loc[
                            mask,
                            'f1_nonrecur'
                        ].mean()
                    )


                # =================================================
                # ROC-AUC
                # =================================================

                if last_metrics_df is not None:

                    mean_roc_auc = last_metrics_df.loc[
                        (
                            last_metrics_df['combination']
                            == unique_combination
                        ) &
                        (
                            last_metrics_df['tile_count_threshold']
                            == tile_count_threshold
                        ),
                        'roc_auc'
                    ].mean()

                else:

                    mean_roc_auc = np.nan


                # =================================================
                # CURRENT PERFORMANCE RESULTS
                # =================================================

                current_performance_df = pd.DataFrame({
                    'outer_fold':
                        outer_fold,

                    'run':
                        run,

                    'run_name':
                        run_name,

                    'combination':
                        unique_combination,

                    'tile_count_threshold':
                        tile_count_threshold,

                    'mean_f1_score_recur':
                        mean_f1_score_recur,

                    'mean_f1_score_nonrecur':
                        mean_f1_score_nonrecur,

                    'classification_probability_threshold':
                        probability_thresholds,

                    'mean_roc_auc':
                        mean_roc_auc
                })


                # =================================================
                # APPEND TO ALL PERFORMANCE RESULTS
                # =================================================

                performance_results_df = pd.concat(
                    [
                        performance_results_df,
                        current_performance_df
                    ],
                    axis=0,
                    ignore_index=True
                )


    return performance_results_df



def select_best_tile_threshold(
    performance_results_df,
    f1_score_thresholds
):
    """Select the best tile-count threshold.

    For each feature combination, selects the tile-count threshold
    using the specified nonrecurrent F1-score criterion, recurrent
    F1-score, ROC-AUC, and median tile-count threshold tie-breakers.
    """

    selected_rows = []

    combinations = (
        performance_results_df['combination']
        .unique()
    )

    for combination in combinations:

        df_combination = performance_results_df[
            performance_results_df['combination'] == combination
        ].copy()

        # ========================================================
        # PERFORMANCE MAXIMISATION
        # ========================================================

        df_threshold = df_combination.copy()

        df_threshold = df_threshold[df_threshold['mean_f1_score_recur'] >= df_threshold['mean_f1_score_nonrecur']]

        for f1_score_lower in f1_score_thresholds:

            df_lower = df_threshold[
                df_threshold['mean_f1_score_nonrecur']
                >= f1_score_lower
            ].copy()

            if len(df_lower) == 0:
                continue

            # ====================================================
            # MAXIMISE RECURRENCE F1 SCORE
            # ====================================================

            max_f1_score_recur = np.max(
                df_lower['mean_f1_score_recur']
            )

            df_lower = df_lower[
                df_lower['mean_f1_score_recur']
                == max_f1_score_recur
            ].copy()

            # ====================================================
            # BREAK TIES USING ROC-AUC
            # ====================================================

            max_roc_auc = np.max(
                df_lower['mean_roc_auc']
            )

            df_lower = df_lower[
                df_lower['mean_roc_auc']
                == max_roc_auc
            ].copy()

            # ====================================================
            # BREAK REMAINING TIES USING MEDIAN
            # TILE-COUNT THRESHOLD
            # ====================================================

            median_tile_count_threshold = (
                df_lower['tile_count_threshold']
                .median()
            )

            df_lower['distance_to_median'] = (
                df_lower['tile_count_threshold']
                - median_tile_count_threshold
            ).abs()

            df_lower = (
                df_lower
                .sort_values(
                    'distance_to_median'
                )
                .reset_index(
                    drop=True
                )
            )

            selected_rows.append(
                df_lower.iloc[0].copy()
            )

            break

    f1_score_df = pd.DataFrame(
        selected_rows
    )

    if len(f1_score_df) > 0:
        f1_score_df = f1_score_df.drop(
            columns=['distance_to_median']
        )

    return f1_score_df


#Also appears in PSA analysis
def create_logistic_model(df_row): # was get_logistic_model
    """Create a logistic regression model from run parameters."""
    return LogisticRegression(
        penalty=df_row['penalty'].iloc[0],
        C=df_row['C'].iloc[0],
        solver=df_row['solver'].iloc[0],
        l1_ratio=df_row['l1_ratio'].iloc[0],
        max_iter=1000,
        random_state=8,
    )



#Also appears in PSA code. Update and match variable names in both and store in one place.
def evaluate_logistic_model(# was called run_logistic_for_graded_f1_score
    performance_results_df,
    df_logistic,
    df_test_proba,
    df_train_proba,
    bootstrap,
    fold_n
):
    """Fit and evaluate patient-level logistic regression models.

    Fits the selected logistic regression model using the training
    patients and evaluates it on the outer test patients using the
    selected tile-count and classification probability thresholds.
    """
    results = []

    print('length of df is:')
    print(len(performance_results_df))

    for _, row in performance_results_df.iterrows():

        # ====================================================
        # RUN INFORMATION
        # ====================================================

        run_name = row['run_name']
        combination = row['combination']

        tile_count_threshold = row[
            'tile_count_threshold'
        ]

        classification_probability_threshold = row[
            'classification_probability_threshold'
        ]

        print(combination)

        # ====================================================
        # LOGISTIC MODEL
        # ====================================================

        df_logistic_row = df_logistic[
            df_logistic['run_name'] == run_name
        ].copy()

        logistic_model = create_logistic_model(
            df_logistic_row
        )

        feature_columns = list(
            ast.literal_eval(combination)
        )

        # ====================================================
        # TEST DATA
        # ====================================================

        df_test_thresholded = df_test_proba[
            df_test_proba.n_tiles >= tile_count_threshold
        ].copy()

        X_train = df_train_proba[
            feature_columns
        ]

        y_train = df_train_proba[
            'case'
        ]

        X_test = df_test_thresholded[
            feature_columns
        ]

        y_test = df_test_thresholded[
            'case'
        ]

        # ====================================================
        # FIT MODEL AND PREDICT
        # ====================================================

        logistic_model.fit(
            X_train,
            y_train
        )

        y_pred_probability = (
            logistic_model
            .predict_proba(X_test)[:, 1]
        )

        roc_auc = roc_auc_score(
            y_test,
            y_pred_probability
        )

        y_pred = np.zeros(
            len(y_test)
        )

        y_pred[
            y_pred_probability
            >= classification_probability_threshold
        ] = 1

        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        confusion_matrix_results = confusion_matrix(
            y_test,
            y_pred
        )

        precision, recall, fscore, _ = (
            metrics.precision_recall_fscore_support(
                y_test,
                y_pred
            )
        )

        true_recur = confusion_matrix_results[1][1]
        false_recur = confusion_matrix_results[0][1]
        true_nonrecur = confusion_matrix_results[0][0]
        false_nonrecur = confusion_matrix_results[1][0]

        precision_nonrecur = precision[0]
        recall_nonrecur = recall[0]
        f1_score_nonrecur = fscore[0]

        precision_recur = precision[1]
        recall_recur = recall[1]
        f1_score_recur = fscore[1]

        f1_score_average = np.mean(
            fscore
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        results.append({

            'bootstrap_rv':
                bootstrap,

            'fold':
                fold_n,

            'combination':
                combination,

            'logistic_model':
                logistic_model,

            'tile_count_threshold':
                tile_count_threshold,

            'classification_probability_threshold':
                classification_probability_threshold,

            'n_patients':
                len(df_test_thresholded),

            'roc_auc':
                roc_auc,

            'accuracy':
                accuracy,

            'f1_score_recur':
                f1_score_recur,

            'f1_score_nonrecur':
                f1_score_nonrecur,

            'f1_score_average':
                f1_score_average,

            'true_recur_1_1':
                true_recur,

            'false_recur_0_1':
                false_recur,

            'true_nonrecur_0_0':
                true_nonrecur,

            'false_nonrecur_1_0':
                false_nonrecur,

            'precision_nonrecur':
                precision_nonrecur,

            'recall_nonrecur':
                recall_nonrecur,

            'precision_recur':
                precision_recur,

            'recall_recur':
                recall_recur
        })

    return pd.DataFrame(
        results
    )





def aggregate_tile_probabilities_to_patients(# was calculate_patient_probabilities
    pred_proba,
    df,
    unique_patients
):
    """Aggregate tile-level prediction probabilities to patient level.

    Calculates the number of tiles and mean, standard deviation, and
    maximum prediction probability for each patient.
    """

    if isinstance(unique_patients, str):
        unique_patients = [unique_patients]

    patient_results = []

    for patient_id in unique_patients:

        patient_df = df[
            df['patient_id'] == patient_id
        ]

        patient_proba = pred_proba[
            df['patient_id'] == patient_id
        ]

        patient_results.append({
            'patient_id': patient_id,
            'gleason': patient_df['gleason'].unique()[0],
            'case': patient_df['case'].unique()[0],
            'n_tiles': len(patient_proba),
            'mean_prob': np.mean(patient_proba),
            'std_prob': np.std(patient_proba, ddof=1),
            'max_prob': np.max(patient_proba)
        })

    df_proba = pd.DataFrame(patient_results)

    df_proba['std_prob'] = df_proba['std_prob'].fillna(0)

    return df_proba

def select_best_probability_threshold(
    performance_results_df,
    f1_score_thresholds
):
    """Select the best classification probability threshold.

    For each outer fold, feature combination, and tile-count threshold,
    selects the probability threshold using the specified F1-score
    criterion, recurrent F1-score, and ROC-AUC tie-breakers.
    """

    selected_rows = []

    outer_folds = performance_results_df[
        'outer_fold'
    ].unique()

    combinations = performance_results_df[
        'combination'
    ].unique()

    tile_count_thresholds = performance_results_df[
        'tile_count_threshold'
    ].unique()

    for outer_fold in outer_folds:

        df_outer_fold = performance_results_df[
            performance_results_df['outer_fold'] == outer_fold
        ].copy()

        for combination in combinations:

            df_combination = df_outer_fold[
                df_outer_fold['combination'] == combination
            ].copy()

            for tile_count_threshold in tile_count_thresholds:

                df_threshold = df_combination[
                    df_combination['tile_count_threshold']
                    == tile_count_threshold
                ].copy()

                # Require recurrent F1 to be at least as high
                # as non-recurrent F1.
                df_threshold = df_threshold[
                    df_threshold['mean_f1_score_recur']
                    >= df_threshold['mean_f1_score_nonrecur']
                ].copy()

                for f1_score_lower in f1_score_thresholds:

                    df_lower = df_threshold[
                        df_threshold['mean_f1_score_nonrecur']
                        >= f1_score_lower
                    ].copy()

                    if len(df_lower) == 0:
                        continue

                    # Maximise recurrent F1.
                    max_f1_score_recur = np.max(
                        df_lower['mean_f1_score_recur']
                    )

                    df_lower = df_lower[
                        df_lower['mean_f1_score_recur']
                        == max_f1_score_recur
                    ].copy()

                    # Tie-break using ROC-AUC.
                    max_roc_auc = np.max(
                        df_lower['mean_roc_auc']
                    )

                    df_lower = df_lower[
                        df_lower['mean_roc_auc']
                        == max_roc_auc
                    ].copy()

                    # Final tie-break using the probability threshold
                    # closest to the median.
                    median_classification_probability_threshold = (
                        df_lower[
                            'classification_probability_threshold'
                        ].median()
                    )

                    df_lower['distance_to_median'] = (
                        df_lower[
                            'classification_probability_threshold'
                        ]
                        - median_classification_probability_threshold
                    ).abs()

                    df_lower = (
                        df_lower
                        .sort_values('distance_to_median')
                        .reset_index(drop=True)
                    )

                    selected_row = df_lower.iloc[0].copy()

                    selected_rows.append(selected_row)

                    break

    f1_score_df = pd.DataFrame(selected_rows)

    if len(f1_score_df) == 0:
        return f1_score_df

    if 'distance_to_median' in f1_score_df.columns:
        f1_score_df = f1_score_df.drop(
            columns=['distance_to_median']
        )

    return f1_score_df



#END OF HELPER FUNCTIONS



def write_logistic_analysis_script(#was generate_logistic_script
    keyfilename,
    outer_fold,
    run_number,
    penalty,
    C,
    solver,
    l1_ratio,
    main_path,
    file_input_name,
    patient_output_dir,
    bootstrap,
    fold_dir
):
    """Write a patient-level logistic regression analysis script.

    Creates a standalone script that trains an XGBoost tile-level model,
    aggregates tile predictions to patient level, fits logistic
    regression models, evaluates them across tile-count and
    classification probability thresholds, and saves the results.
    """

    # --------------------------------------------------------
    # Create run directory
    # --------------------------------------------------------

    run_dir = os.path.join(
        fold_dir,
        f'run_{run_number:05d}'
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Record run parameters
    # --------------------------------------------------------

    with open(keyfilename, 'a') as f:

        f.write(
            f'Fold_{outer_fold:05d}_'
            f'Run_{run_number:05d}'
        )

        l1_ratio_string = (
            str(l1_ratio)
            if l1_ratio is not None
            else 'None'
        )

        f.write(
            f' {outer_fold} '
            f'{penalty} '
            f'{C:f} '
            f'{solver} '
            f'{l1_ratio_string}\n'
        )

    # --------------------------------------------------------
    # Create script filename
    # --------------------------------------------------------

    script_filename = (
        main_path +
        file_input_name +
        f'{run_number:05d}.py'
    )

    # --------------------------------------------------------
    # Write script
    # --------------------------------------------------------

    with open(script_filename, 'w') as f:

        f.write(f"""\
# ============================================================
# IMPORTS
# ============================================================

import itertools
import os
import pickle
import re

from glob import glob

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

import tile_model_fitting as tmf


# ============================================================
# PATHS AND SETTINGS
# ============================================================

main_path = (
    '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
    'prostate_recurrence/model_evaluation/recurrence_status_tile/'
    'rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
)

output_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
    'feature_analysis_v3/rfe_runs/'
    'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
)

patient_output_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
    'feature_analysis_v3/rfe_runs/'
    'gleason_7_quad_1000_v8a_patient_level_logistic_plus_'
    'tile_threshold_f1score_removed_xgboost_outer_fold_{bootstrap}/'
)

input_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
    'feature_analysis_v3/'
)

code_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
    'prostate_recurrence/model_evaluation/recurrence_status_tile/'
    'rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
)

run_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
    'feature_analysis_v3/rfe_runs/'
    'gleason_7_quad_1000_v8a_patient_level_logistic_plus_'
    'tile_threshold_f1score_removed_xgboost_outer_fold_{bootstrap}/'
    'fold_{outer_fold:05d}/run_{run_number:05d}'
)

data_dir = os.path.join(
    input_dir,
    'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
)


# ============================================================
# MODEL AND ANALYSIS SETTINGS
# ============================================================

outer_fold = {outer_fold}
fold_assignment_bootstrap = {bootstrap}

n_inner_folds = 5

model_random_seed = 31
random_seed = 0

tumour_proportion = 0.7
tissue_proportion = 0.7

tile_count_thresholds = [
    0, 2, 5, 10, 15, 20, 25, 30, 35
]


# ============================================================
# HYPERPARAMETER FILE
# ============================================================

hyperparameters_df = pd.read_csv(
    os.path.join(
        code_dir,
        'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_runkey.txt'
    ),
    sep=' ',
    header=None
)

hyperparameters_df = hyperparameters_df.rename(columns={{
    0: 'name',
    1: 'fold',
    2: 'n_estimator',
    3: 'max_depth',
    4: 'learning_rate',
    5: 'subsample',
    6: 'colsample_bytree',
    7: 'gamma',
    8: 'min_child_weight'
}})

# Preserve original behaviour:
# the XGBoost run-key entries used here are from fold 0.
hyperparameters_df = hyperparameters_df[
    hyperparameters_df['fold'] == 0
]

hyperparameters_df = hyperparameters_df.loc[0:288]


# ============================================================
# SELECT BEST XGBOOST MODEL
# ============================================================

folds_csv_file = (
    code_dir +
    'folds_random_' +
    str(fold_assignment_bootstrap).zfill(3) +
    '.csv'
)

bootstrap_csv_name = (
    'xgboost_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_' +
    str(fold_assignment_bootstrap)
)

csv_structure = os.path.join(
    output_dir,
    bootstrap_csv_name +
    '_outer_fold_test_' +
    str(outer_fold).zfill(5) +
    '_run_*0_7*.csv'
)

csv_paths = sorted(
    glob(csv_structure)
)

test_scores = []
candidate_csv_paths = []

for csv_path in csv_paths:

    df_score = pd.read_csv(
        csv_path,
        low_memory=False
    )

    if not test_scores:

        test_scores.append(
            np.max(df_score['Mean_Test_Scores'])
        )

        candidate_csv_paths.append(
            csv_path
        )

    elif (
        np.max(df_score['Mean_Test_Scores'])
        >= np.max(test_scores)
    ):

        test_scores.append(
            np.max(df_score['Mean_Test_Scores'])
        )

        candidate_csv_paths.append(
            csv_path
        )


sav_file = candidate_csv_paths[-1].replace(
    '_mean_test_scores.csv',
    '.sav'
)

match = re.search(
    r'(?<=run_)\\d+(?=_roc)',
    candidate_csv_paths[-1]
)

run = int(
    match.group()
)

with open(sav_file, 'rb') as file:
    fold_model = pickle.load(file)


# ============================================================
# XGBOOST TILE MODEL
# ============================================================

xg_classifier_params = {{
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'n_jobs': -1,
    'random_state': 8,
    'n_estimators': hyperparameters_df.loc[run].n_estimator,
    'max_depth': hyperparameters_df.loc[run].max_depth,
    'learning_rate': hyperparameters_df.loc[run].learning_rate,
    'subsample': hyperparameters_df.loc[run].subsample,
    'colsample_bytree': hyperparameters_df.loc[run].colsample_bytree,
    'gamma': hyperparameters_df.loc[run].gamma,
    'min_child_weight': hyperparameters_df.loc[run].min_child_weight,
}}

for param, value in xg_classifier_params.items():

    print(param)
    print(value)

    if isinstance(value, str):

        try:
            xg_classifier_params[param] = float(value)

        except ValueError:
            pass


xg_classifier_tile = XGBClassifier(
    **xg_classifier_params
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    penalty={penalty!r},
    C={C!r},
    solver={solver!r},
    l1_ratio={l1_ratio!r},
    max_iter=1000,
    random_state=8
)

patient_model_features = [
    'mean_prob',
    'std_prob',
    'max_prob'
]

max_rank = 1


# ============================================================
# LOAD AND FILTER TILE DATA
# ============================================================

df_tiles = pd.read_csv(
    data_dir,
    low_memory=False
)

df_tiles = df_tiles[
    (df_tiles['tissue_proportion'] >= tissue_proportion) &
    (df_tiles['tumour_proportion'] >= tumour_proportion) &
    (df_tiles['gleason'].isin([1, 2]))
].reset_index(drop=True)

print(df_tiles.head())


# ============================================================
# CLINICAL SCORE
# ============================================================

conditions = [
    (df_tiles['gleason'] == 1) & (df_tiles['case'] == 1.0),
    (df_tiles['gleason'] == 2) & (df_tiles['case'] == 1.0),
    (df_tiles['gleason'] == 1) & (df_tiles['case'] == 0.0),
    (df_tiles['gleason'] == 2) & (df_tiles['case'] == 0.0),
]

df_tiles['clinical_score'] = np.select(
    conditions,
    range(4)
)


# ============================================================
# OUTER-FOLD SPLIT
# ============================================================

df_patients = tmf.create_patient_dataframe(
    df_tiles
)

df_stratified_patients = (
    df_patients
    .sort_values(
        by=['clinical_score', 'total_tiles'],
        ascending=[True, False]
    )
    .reset_index()
)

_, _, predefined_splits_array = tmf.patient_tile_stratifier(
    df_stratified_patients,
    df_tiles,
    5,
    folds_csv_file
)

df_tiles = df_tiles[
    predefined_splits_array != outer_fold
].reset_index(drop=True)


# ============================================================
# INNER-FOLD SETUP
# ============================================================

df_patients = tmf.create_patient_dataframe(
    df_tiles
)

df_stratified_patients = (
    df_patients
    .sort_values(
        by=['clinical_score', 'total_tiles'],
        ascending=[True, False]
    )
    .reset_index()
)

folds_csv_file = (
    main_path +
    'inner_folds_{outer_fold}_data.csv'
)

metrics_results = []
f1_threshold_results = []
prediction_results = []

inner_folds = np.arange(
    n_inner_folds
)


# ============================================================
# INNER CROSS-VALIDATION
# ============================================================

for inner_fold in inner_folds:

    print(inner_fold)


    # --------------------------------------------------------
    # Create patient/tile split
    # --------------------------------------------------------

    df_patients = tmf.create_patient_dataframe(
        df_tiles
    )

    df_stratified_patients = (
        df_patients
        .sort_values(
            by=['clinical_score', 'total_tiles'],
            ascending=[True, False]
        )
        .reset_index()
    )

    _, _, predefined_splits_array = tmf.patient_tile_stratifier(
        df_stratified_patients,
        df_tiles,
        n_inner_folds,
        folds_csv_file
    )

    df_train_tiles = df_tiles[
        predefined_splits_array != inner_fold
    ].reset_index(drop=True)

    df_test_tiles = df_tiles[
        predefined_splits_array == inner_fold
    ].reset_index(drop=True)


    # --------------------------------------------------------
    # Tile-level feature matrices
    # --------------------------------------------------------

    tile_columns_to_drop = [
        'patient_id',
        'slide_id',
        'area',
        'decision',
        'case',
        'scene',
        'tile',
        'quadrant',
        'clinical_score',
        'tissue_proportion',
        'tumour_proportion'
    ]

    X_train_tiles = df_train_tiles.drop(
        columns=tile_columns_to_drop
    ).copy()

    y_train_tiles = df_train_tiles['case']

    X_test_tiles = df_test_tiles.drop(
        columns=tile_columns_to_drop
    ).copy()

    y_test_tiles = df_test_tiles['case']


    # --------------------------------------------------------
    # RFE feature selection
    # --------------------------------------------------------

    include_columns = list(
        X_train_tiles.columns[
            fold_model.ranking_ <= max_rank
        ]
    )

    print('number of columns: ')
    print(len(include_columns))

    X_train_tiles = X_train_tiles[
        include_columns
    ].copy()

    X_test_tiles = X_test_tiles[
        include_columns
    ].copy()

    print(X_train_tiles)


    # --------------------------------------------------------
    # Fit tile-level XGBoost model
    # --------------------------------------------------------

    xg_classifier_tile.fit(
        X_train_tiles,
        y_train_tiles
    )

    train_pred_prob = xg_classifier_tile.predict_proba(
        X_train_tiles
    )[:, 1]

    test_pred_prob = xg_classifier_tile.predict_proba(
        X_test_tiles
    )[:, 1]


    # ========================================================
    # AGGREGATE TILE PREDICTIONS TO PATIENT LEVEL
    # ========================================================

    test_patient_ids = list(
        df_test_tiles['patient_id'].unique()
    )

    train_patient_ids = list(
        df_train_tiles['patient_id'].unique()
    )


    # --------------------------------------------------------
    # Test patients
    # --------------------------------------------------------

    test_cases = []
    test_gleasons = []
    test_n_tiles = []
    test_mean_prob = []
    test_std_prob = []
    test_max_prob = []

    for patient_id in test_patient_ids:

        patient_probabilities = test_pred_prob[
            df_test_tiles['patient_id'] == patient_id
        ]

        test_cases.append(
            df_test_tiles[
                df_test_tiles['patient_id'] == patient_id
            ]['case'].unique()[0]
        )

        test_gleasons.append(
            df_test_tiles[
                df_test_tiles['patient_id'] == patient_id
            ]['gleason'].unique()[0]
        )

        test_n_tiles.append(
            len(patient_probabilities)
        )

        test_mean_prob.append(
            np.mean(patient_probabilities)
        )

        test_std_prob.append(
            np.std(
                patient_probabilities,
                ddof=1
            )
        )

        test_max_prob.append(
            np.max(patient_probabilities)
        )


    df_test_patients = pd.DataFrame({{
        'patient_id': test_patient_ids,
        'gleason': test_gleasons,
        'case': test_cases,
        'n_tiles': test_n_tiles,
        'mean_prob': test_mean_prob,
        'std_prob': test_std_prob,
        'max_prob': test_max_prob
    }})

    df_test_patients['std_prob'] = (
        df_test_patients['std_prob']
        .fillna(0)
    )


    # --------------------------------------------------------
    # Train patients
    # --------------------------------------------------------

    train_cases = []
    train_gleasons = []
    train_n_tiles = []
    train_mean_prob = []
    train_std_prob = []
    train_max_prob = []

    for patient_id in train_patient_ids:

        patient_probabilities = train_pred_prob[
            df_train_tiles['patient_id'] == patient_id
        ]

        train_cases.append(
            df_train_tiles[
                df_train_tiles['patient_id'] == patient_id
            ]['case'].unique()[0]
        )

        train_gleasons.append(
            df_train_tiles[
                df_train_tiles['patient_id'] == patient_id
            ]['gleason'].unique()[0]
        )

        train_n_tiles.append(
            len(patient_probabilities)
        )

        train_mean_prob.append(
            np.mean(patient_probabilities)
        )

        train_std_prob.append(
            np.std(
                patient_probabilities,
                ddof=1
            )
        )

        train_max_prob.append(
            np.max(patient_probabilities)
        )


    df_train_patients = pd.DataFrame({{
        'patient_id': train_patient_ids,
        'gleason': train_gleasons,
        'case': train_cases,
        'n_tiles': train_n_tiles,
        'mean_prob': train_mean_prob,
        'std_prob': train_std_prob,
        'max_prob': train_max_prob
    }})

    df_train_patients['std_prob'] = (
        df_train_patients['std_prob']
        .fillna(0)
    )


    # ========================================================
    # STANDARDISE PATIENT-LEVEL XGBOOST FEATURES
    # ========================================================

    feature_means = df_train_patients[
        patient_model_features
    ].mean()

    feature_stds = df_train_patients[
        patient_model_features
    ].std()

    df_train_patients[
        patient_model_features
    ] = (
        df_train_patients[patient_model_features]
        - feature_means
    ) / feature_stds

    df_test_patients[
        patient_model_features
    ] = (
        df_test_patients[patient_model_features]
        - feature_means
    ) / feature_stds


    # ========================================================
    # FIT PATIENT-LEVEL LOGISTIC REGRESSION
    # ========================================================

    for r in range(
        1,
        len(patient_model_features) + 1
    ):

        for combination in itertools.combinations(
            patient_model_features,
            r
        ):

            X_patient_train = df_train_patients[
                list(combination)
            ].copy()

            y_train = df_train_patients['case']

            logistic_model.fit(
                X_patient_train,
                y_train
            )


            # =================================================
            # EVALUATE ACROSS TILE-COUNT THRESHOLDS
            # =================================================

            for tile_count_threshold in tile_count_thresholds:

                test_patients_filtered = df_test_patients[
                    df_test_patients['n_tiles'] >=
                    tile_count_threshold
                ].copy()

                X_patient_test = test_patients_filtered[
                    list(combination)
                ]

                y_test = test_patients_filtered['case']

                n_test_patients = len(
                    X_patient_test
                )

                if n_test_patients < 5:
                    continue


                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                y_pred_prob = logistic_model.predict_proba(
                    X_patient_test
                )[:, 1]

                y_pred = logistic_model.predict(
                    X_patient_test
                )


                # ------------------------------------------------
                # F1 across probability thresholds
                # ------------------------------------------------

                classification_probability_thresholds = np.linspace(
                    0.01,
                    0.99,
                    99
                )

                f1_recurrence = []
                f1_nonrecurrence = []

                for probability_threshold in (
                    classification_probability_thresholds
                ):

                    predicted_y = (
                        y_pred_prob >= probability_threshold
                    ).astype(int)

                    f1_recurrence.append(
                        f1_score(
                            y_test,
                            predicted_y,
                            pos_label=1
                        )
                    )

                    f1_nonrecurrence.append(
                        f1_score(
                            y_test,
                            predicted_y,
                            pos_label=0
                        )
                    )


                df_f1_scores = pd.DataFrame({{
                    'combination': [combination] *
                                   len(
                                       classification_probability_thresholds
                                   ),
                    'fold': [inner_fold] *
                            len(
                                classification_probability_thresholds
                            ),
                    'tile_count_threshold':
                        [tile_count_threshold] *
                        len(
                            classification_probability_thresholds
                        ),
                    'test_size':
                        [n_test_patients] *
                        len(
                            classification_probability_thresholds
                        ),
                    'classification_probability_threshold':
                        classification_probability_thresholds,
                    'f1_recur': f1_recurrence,
                    'f1_nonrecur': f1_nonrecurrence
                }})

                f1_threshold_results.append(
                    df_f1_scores
                )


                # ------------------------------------------------
                # Standard metrics
                # ------------------------------------------------

                f1_derivative_values = np.diff(
                    np.asarray(f1_recurrence)
                )

                max_f1_derivative = np.max(
                    np.abs(f1_derivative_values)
                )

                roc_auc = roc_auc_score(
                    y_test,
                    y_pred_prob
                )

                f1_recurrence_score = f1_score(
                    y_test,
                    y_pred,
                    pos_label=1
                )

                f1_nonrecurrence_score = f1_score(
                    y_test,
                    y_pred,
                    pos_label=0
                )

                metrics_results.append({{
                    'combination': combination,
                    'fold': inner_fold,
                    'roc_auc': roc_auc,
                    'f1_score_recur':
                        f1_recurrence_score,
                    'f1_score_nonrecur':
                        f1_nonrecurrence_score,
                    'f1_score_average': np.mean([
                        f1_recurrence_score,
                        f1_nonrecurrence_score
                    ]),
                    'f1_score_max_derivative':
                        max_f1_derivative,
                    'accuracy': accuracy_score(
                        y_test,
                        y_pred
                    ),
                    'tile_count_threshold':
                        tile_count_threshold,
                    'test_size':
                        n_test_patients
                }})


                # ------------------------------------------------
                # Store predictions
                # ------------------------------------------------

                prediction_results.append(
                    pd.DataFrame({{
                        'y_true': y_test.values,
                        'y_pred_prob': y_pred_prob,
                        'combination': [combination] *
                                       len(y_test),
                        'fold': [inner_fold] *
                                len(y_test),
                        'tile_count_threshold':
                            [tile_count_threshold] *
                            len(y_test)
                    }})
                )


# ============================================================
# COMBINE RESULTS
# ============================================================

f1_thresholds_df = pd.concat(
    f1_threshold_results,
    ignore_index=True
)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)

metrics_df = pd.DataFrame(
    metrics_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

run_prefix = (
    f'outer_fold_{{outer_fold:05d}}'
    f'_run_{{run_number:05d}}'
    f'_inner_cv_'
)

output_files = {{
    'metrics_patient_level':
        metrics_df,
    'f1_probability_threshold_patient_level':
        f1_thresholds_df,
    'probability_predictions_patient_level':
        predictions_df,
}}

for suffix, output_df in output_files.items():

    filename = os.path.join(
        run_dir,
        f'{{run_prefix}}_{{suffix}}.csv'
    )

    output_df.to_csv(
        filename,
        index=False
    )
""")


def generate_f1_score_csv(bootstrap):
    """Combine patient-level F1 results across all folds and runs.

    Collects the probability-threshold performance CSVs generated by the
    patient-level logistic analysis and combines them into a single
    performance-results CSV for subsequent model and threshold selection.
    """

    # ========================================================
    # SETTINGS
    # ========================================================

    tile_count_thresholds = [
        0, 2, 5, 10, 15, 20, 25, 30, 35
    ]

    run_name = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_plus_'
        'tile_threshold_f1score_removed_xgboost_outer_fold_'
        f'{bootstrap}'
    )

    # ========================================================
    # PATHS
    # ========================================================

    base_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/'
        'slide_20X/feature_analysis_v3/rfe_runs/'
    )

    output_dir = os.path.join(
        base_dir,
        run_name
    )

    code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/'
        'recurrence_status_tile/rfe_runs/'
    )

    logistic_runkey = os.path.join(
        code_dir,
        run_name,
        f'{run_name}_fold_runkey.txt'
    )

    # ========================================================
    # LOAD LOGISTIC RUN KEY
    # ========================================================

    df_logistic = pd.read_table(
        logistic_runkey,
        sep=' ',
        header=None
    )

    df_logistic = df_logistic.rename(
        columns={
            0: 'run_name',
            1: 'fold',
            2: 'penalty',
            3: 'C',
            4: 'solver',
            5: 'l1_ratio'
        }
    )

    df_logistic = df_logistic.replace(
        {np.nan: None}
    )

    # ========================================================
    # FIND GENERATED F1 CSV FILES
    # ========================================================

    csv_pattern = os.path.join(
        output_dir,
        'fold_*',
        'run_*',
        'outer_fold_*run_*_inner_cv_'
        'f1_probability_threshold_patient_level.csv'
    )



    csv_paths = sorted(
        glob(csv_pattern)
    )

    if len(csv_paths) == 0:
        raise FileNotFoundError(
            'No patient-level F1 probability-threshold CSV files '
            f'found in {output_dir}'
        )

    print(
        f'Found {len(csv_paths)} F1 performance CSV files.'
    )

    # ========================================================
    # GENERATE COMBINED PERFORMANCE RESULTS
    # ========================================================

    performance_results_df = (
        performance_results_df_generator(
            csv_paths,
            df_logistic,
            tile_count_thresholds,
            output_dir
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        output_dir,
        'mean_inner_f1_score_plus_roc_auc_'
        'all_probability_thresholds.csv'
    )

    performance_results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f'Saved performance results to:\n{output_file}'
    )





















def run_ecm_patient_level_nested_cv(
    bootstrap,
    fold_n,
    subfile_name,
    f1_score_thresholds
):
    """Run nested cross-validation for the ECM patient-level model.

    Outer fold:
        - Held completely out during model/hyperparameter selection.
        - Used only for final outer-test evaluation.

    Inner folds:
        - Used to select the XGBoost model.
        - Used to select the patient-level feature combination,
          tile-count threshold, and classification probability threshold.

    Final outer evaluation:
        - XGBoost is fitted on all outer-training patients.
        - Patient-level logistic regression is fitted on outer-training
          patients using the configuration selected entirely from inner CV.
        - Performance is then evaluated once on the untouched outer-test
          patients.
    """

    # ========================================================
    # RUN NAME
    # ========================================================

    run_name = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_plus_'
        'tile_threshold_f1score_removed_xgboost_outer_fold_'
        f'{bootstrap}'
    )

    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    # Contains the logistic-regression hyperparameters corresponding
    # to the patient-level runs.
    #
    # These parameters are predetermined from the model-selection
    # pipeline and are not fitted using the outer-test patients.
    python_code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        f'{run_name}/{run_name}_fold_runkey.txt'
    )

    df_logistic = pd.read_table(
        python_code_dir,
        delimiter=" ",
        header=None
    )

    df_logistic = df_logistic.rename(columns={
        0: "run_name",
        1: "fold",
        2: "penalty",
        3: "C",
        4: "solver",
        5: "l1_ratio"
    })

    df_logistic = df_logistic.replace({
        np.nan: None
    })

    # ========================================================
    # PATHS
    # ========================================================

    output_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/rfe_runs/'
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    )

    patient_output_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/rfe_runs/'
        f'{run_name}/'
    )

    input_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/'
    )

    code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    )

    data_dir = os.path.join(
        input_dir,
        'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
    )

    folds_csv_file = os.path.join(
        code_dir,
        f'folds_random_{bootstrap:03d}.csv'
    )

    # ========================================================
    # XGBOOST SETTINGS
    # ========================================================

    tumour_proportion = 0.7
    tissue_proportion = 0.7
    max_rank = 1

    # ========================================================
    # XGBOOST HYPERPARAMETERS
    # ========================================================

    # These are the XGBoost hyperparameters corresponding to the
    # previously generated nested-CV runs.
    df_hp = pd.read_csv(
        os.path.join(
            code_dir,
            'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_runkey.txt'
        ),
        sep=' ',
        header=None
    )

    df_hp = df_hp.rename(columns={
        0: 'name',
        1: 'fold',
        2: 'n_estimator',
        3: 'max_depth',
        4: 'learning_rate',
        5: 'subsample',
        6: 'colsample_bytree',
        7: 'gamma',
        8: 'min_child_weight'
    })

    df_hp = df_hp[
        df_hp.fold == 0
    ]

    df_hp = df_hp.loc[0:288]

    # ========================================================
    # SELECT XGBOOST MODEL USING INNER-CV PERFORMANCE
    # ========================================================

    # IMPORTANT:
    # Mean_Test_Scores in these files represent the inner-CV
    # model-selection scores, despite the historical filename.
    #
    # Therefore the outer-test fold is NOT being used here to
    # select the XGBoost model.
    bootstrap_csv_name = (
        'xgboost_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_'
        f'{bootstrap}'
    )

    csv_structure = os.path.join(
        output_dir,
        bootstrap_csv_name +
        '_outer_fold_test_' +
        f'{fold_n:05d}' +
        '_run_*0_7*.csv'
    )

    csv_paths = sorted(
        glob(csv_structure)
    )

    test_score = []
    candidate_csv_paths = []

    for csv_path in csv_paths:

        df_score = pd.read_csv(
            csv_path,
            low_memory=False
        )

        current_score = np.max(
            df_score.Mean_Test_Scores
        )

        if not test_score:

            test_score.append(
                current_score
            )

            candidate_csv_paths.append(
                csv_path
            )

        elif current_score >= np.max(test_score):

            test_score.append(
                current_score
            )

            candidate_csv_paths.append(
                csv_path
            )

    # Select the corresponding trained XGBoost model.
    sav_file = (
        candidate_csv_paths[-1]
        .replace(
            '_mean_test_scores.csv',
            '.sav'
        )
    )

    match = re.search(
        r"(?<=run_)\d+(?=_roc)",
        candidate_csv_paths[-1]
    )

    run = int(
        match.group()
    )

    fold_model = pickle.load(
        open(
            sav_file,
            'rb'
        )
    )

    # ========================================================
    # XGBOOST CLASSIFIER
    # ========================================================

    xg_classifier_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 8,
        'n_estimators': df_hp.loc[run].n_estimator,
        'max_depth': df_hp.loc[run].max_depth,
        'learning_rate': df_hp.loc[run].learning_rate,
        'subsample': df_hp.loc[run].subsample,
        'colsample_bytree': df_hp.loc[run].colsample_bytree,
        'gamma': df_hp.loc[run].gamma,
        'min_child_weight': df_hp.loc[run].min_child_weight
    }

    for param, value in xg_classifier_params.items():

        if isinstance(value, str):

            try:
                xg_classifier_params[param] = float(value)

            except ValueError:
                pass

    xg_classifier_tile = XGBClassifier(
        **xg_classifier_params
    )

    # ========================================================
    # LOAD AND FILTER TILE DATA
    # ========================================================

    df = pd.read_csv(
        data_dir,
        low_memory=False
    )

    df = df[
        df['tissue_proportion'] >= tissue_proportion
    ]

    df = df[
        df['tumour_proportion'] >= tumour_proportion
    ]

    df = df[
        (df['gleason'] == 1) |
        (df['gleason'] == 2)
    ]

    df = df.reset_index(
        drop=True
    )

    # ========================================================
    # PATIENT-LEVEL STRATIFICATION DATA
    # ========================================================

    patient_ids = list(
        df.patient_id.unique()
    )

    conditions = [
        (df['gleason'] == 1) & (df['case'] == 1.0),
        (df['gleason'] == 2) & (df['case'] == 1.0),
        (df['gleason'] == 1) & (df['case'] == 0.0),
        (df['gleason'] == 2) & (df['case'] == 0.0)
    ]

    values = range(0, 4)

    df['clinical_score'] = np.select(
        conditions,
        values
    )

    gleasons = []
    cases = []
    total_tiles = []
    clinical_scores = []

    for patient_id in patient_ids:

        patient_df = df[
            df.patient_id == patient_id
        ]

        gleasons.append(
            patient_df.gleason.unique()[0]
        )

        cases.append(
            patient_df.case.unique()[0]
        )

        clinical_scores.append(
            patient_df.clinical_score.unique()[0]
        )

        total_tiles.append(
            len(patient_df.case)
        )

    df_patient = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'clinical_score': clinical_scores,
        'total_tiles': total_tiles
    })

    df_strat = (
        df_patient
        .sort_values(
            by=[
                'clinical_score',
                'total_tiles'
            ],
            ascending=[
                True,
                False
            ]
        )
        .copy()
        .reset_index()
    )

    df_strat, predefined_splits, predefined_splits_array = (
        tmf.patient_tile_stratifier(
            df_strat,
            df,
            5,
            folds_csv_file
        )
    )

    # ========================================================
    # OUTER TRAIN / OUTER TEST SPLIT
    # ========================================================

    # IMPORTANT:
    # The outer-test patients are identified here and must remain
    # completely untouched during all model/hyperparameter selection.
    df_train = df[
        predefined_splits_array != fold_n
    ].reset_index(
        drop=True
    )

    df_test = df[
        predefined_splits_array == fold_n
    ].reset_index(
        drop=True
    )

    # ========================================================
    # XGBOOST OUTER-TRAINING DATA
    # ========================================================

    # IMPORTANT:
    # XGBoost is fitted only on the outer-training patients.
    X_train = df_train.drop(
        [
            'patient_id',
            'slide_id',
            'area',
            'decision',
            'case',
            'scene',
            'tile',
            'quadrant',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion'
        ],
        axis=1
    ).copy()

    y_train = df_train["case"]

    include_columns = list(
        X_train.columns[
            fold_model.ranking_ <= max_rank
        ]
    )

    X_train_subset_df = X_train[
        include_columns
    ].copy()

    # ========================================================
    # XGBOOST OUTER-TEST DATA
    # ========================================================

    # IMPORTANT:
    # These patients are not used to select the XGBoost model
    # or any patient-level hyperparameters.
    X_test = df_test.drop(
        [
            'patient_id',
            'slide_id',
            'area',
            'decision',
            'case',
            'scene',
            'tile',
            'quadrant',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion'
        ],
        axis=1
    ).copy()

    X_test_subset_df = X_test[
        include_columns
    ].copy()

    # ========================================================
    # FIT XGBOOST ON OUTER TRAINING DATA
    # ========================================================

    # IMPORTANT:
    # This is the final XGBoost fit for this outer fold.
    # It is fitted using all available outer-training patients,
    # but no outer-test patients.
    xg_classifier_tile.fit(
        X_train_subset_df,
        y_train
    )

    # ========================================================
    # TILE-LEVEL PROBABILITIES
    # ========================================================

    train_pred_proba = (
        xg_classifier_tile
        .predict_proba(X_train_subset_df)[:, 1]
    )

    test_pred_proba = (
        xg_classifier_tile
        .predict_proba(X_test_subset_df)[:, 1]
    )

    # ========================================================
    # AGGREGATE TILE PREDICTIONS TO PATIENT LEVEL
    # ========================================================

    test_unique_patients = df_test.patient_id.unique()
    train_unique_patients = df_train.patient_id.unique()

    df_test_proba = aggregate_tile_probabilities_to_patients(
        test_pred_proba,
        df_test,
        test_unique_patients
    )

    df_train_proba = aggregate_tile_probabilities_to_patients(
        train_pred_proba,
        df_train,
        train_unique_patients
    )

    # ========================================================
    # STANDARDISE PATIENT-LEVEL FEATURES
    # ========================================================

    columns = [
        'mean_prob',
        'std_prob',
        'max_prob'
    ]

    # IMPORTANT:
    # Scaling parameters are calculated from outer-training patients
    # only. The outer-test patients therefore cannot influence
    # standardisation.
    means = df_train_proba[
        columns
    ].mean()

    stds = df_train_proba[
        columns
    ].std()

    df_train_proba[
        columns
    ] = (
        df_train_proba[columns] - means
    ) / stds

    df_test_proba[
        columns
    ] = (
        df_test_proba[columns] - means
    ) / stds

    # ========================================================
    # LOAD INNER-CV PERFORMANCE LANDSCAPE
    # ========================================================

    # IMPORTANT:
    # This file contains performance generated from the inner-CV
    # procedure. It is used exclusively for model/hyperparameter
    # selection.
    performance_file = (
        'mean_inner_f1_score_plus_roc_auc_all_probability_thresholds.csv'
    )

    performance_results_df = pd.read_csv(
        os.path.join(
            patient_output_dir,
            performance_file
        )
    )

    # ========================================================
    # SELECT PROBABILITY THRESHOLD USING INNER CV
    # ========================================================

    # IMPORTANT:
    # This selection uses INNER-CV performance only.
    #
    # For each combination and tile-count threshold:
    #
    #   1. Apply the recurrent/non-recurrent F1 constraint.
    #   2. Apply the minimum non-recurrent F1 criterion.
    #   3. Maximise recurrent F1.
    #   4. Use inner-CV ROC-AUC as the tie-break.
    #   5. Use the median probability threshold as the final
    #      deterministic tie-break.
    #
    # No outer-test performance is accessed here.
    f1_score_df = select_best_probability_threshold(
        performance_results_df,
        f1_score_thresholds
    )

    # Keep only the configuration corresponding to the current
    # outer fold.
    f1_score_df = f1_score_df[
        f1_score_df['outer_fold'] == fold_n
    ].copy()

    print(
        'The input f1_score_df is given by:'
    )

    print(
        f1_score_df
    )

    # ========================================================
    # SELECT TILE-COUNT THRESHOLD USING INNER CV
    # ========================================================

    # IMPORTANT:
    # At this point the probability threshold has already been
    # selected separately for each combination/tile threshold.
    #
    # select_best_tile_threshold() therefore performs the next
    # stage of INNER-CV model selection.
    #
    # The outer-test patients are still untouched.
    selected_tile_threshold_df = select_best_tile_threshold(
        f1_score_df,
        f1_score_thresholds
    )

    # ========================================================
    # FINAL OUTER-FOLD LOGISTIC REGRESSION
    # ========================================================

    # IMPORTANT:
    # All model-selection decisions have now been completed using
    # inner-CV information.
    #
    # The selected configuration is now applied to the outer fold:
    #
    #   - logistic regression is fitted on outer-training patients
    #   - predictions are generated for outer-test patients
    #   - outer-test performance is calculated
    #
    # This is the first point at which the outer-test outcomes
    # are legitimately used for performance evaluation.
    outer_logistic_f1_score_df = (
        evaluate_logistic_model(
            selected_tile_threshold_df,
            df_logistic,
            df_test_proba,
            df_train_proba,
            bootstrap,
            fold_n
        )
    )

    # ========================================================
    # SAVE OUTER-TEST RESULTS
    # ========================================================

    # This file contains the genuine OUTER-TEST performance.

    save_name = (
        f'outer_fold_{fold_n:05d}'
        f'_outer_test_{subfile_name}'
        '_performance_roc_auc_added.csv'
    )

    outer_logistic_f1_score_df.to_csv(
        os.path.join(
            patient_output_dir,
            save_name
        ),
        index=False
    )

    # ========================================================
    # SAVE SELECTED INNER-CV CONFIGURATION
    # ========================================================

    # This file records the configuration selected using INNER CV.
    #
    # It should therefore be interpreted as model-selection output,
    # not as an independent outer-test performance estimate.

    save_name = (
        f'outer_fold_{fold_n:05d}'
        f'_inner_cv_selected_configuration_{subfile_name}'
        '_performance_roc_auc_added.csv'
    )



    selected_tile_threshold_df.to_csv(
        os.path.join(
            patient_output_dir,
            save_name
        ),
        index=False
    )
    
def generate_optimal_inner_model_predictions(#was generate_inner_probabilities_for_optimum_inner_model
    bootstrap,
    fold_n,
    subfile_name,
    f1_score_thresholds
):
    """Generate patient-level probabilities from the optimal inner model.

    Selects the optimal feature combination, tile-count threshold, and
    classification probability threshold from the inner-fold results,
    fits the corresponding patient-level logistic regression model,
    and generates predictions for the inner test patients.
    """
    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    run_name_prefix = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_plus_tile_threshold_'
        'f1score_removed_xgboost_outer_fold_'
        f'{bootstrap}'
    )

    python_code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        f'{run_name_prefix}/{run_name_prefix}_fold_runkey.txt'
    )

    df_logistic = pd.read_table(
        python_code_dir,
        delimiter=' ',
        header=None
    )

    df_logistic = df_logistic.rename(
        columns={
            0: 'run_name',
            1: 'fold',
            2: 'penalty',
            3: 'C',
            4: 'solver',
            5: 'l1_ratio'
        }
    )

    df_logistic = df_logistic.replace(
        {np.nan: None}
    )

    # ========================================================
    # PATHS
    # ========================================================

    patient_output_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/rfe_runs/'
        f'{run_name_prefix}/'
    )

    input_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/'
    )

    xgboost_code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    )

    xgboost_output_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/rfe_runs/'
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    )

    data_dir = os.path.join(
        input_dir,
        'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
    )

    folds_csv_file = os.path.join(
        xgboost_code_dir,
        f'folds_random_{bootstrap:03d}.csv'
    )

    # ========================================================
    # XGBOOST SETTINGS
    # ========================================================

    tumour_proportion = 0.7
    tissue_proportion = 0.7
    max_rank = 1
    n_inner_folds = 5

    # ========================================================
    # XGBOOST HYPERPARAMETERS
    # ========================================================

    df_hp = pd.read_csv(
        os.path.join(
            xgboost_code_dir,
            'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_runkey.txt'
        ),
        sep=' ',
        header=None
    )

    df_hp = df_hp.rename(
        columns={
            0: 'name',
            1: 'fold',
            2: 'n_estimator',
            3: 'max_depth',
            4: 'learning_rate',
            5: 'subsample',
            6: 'colsample_bytree',
            7: 'gamma',
            8: 'min_child_weight'
        }
    )

    df_hp = df_hp[
        df_hp['fold'] == 0
    ].loc[0:288]

    # ========================================================
    # SELECT BEST OUTER-FOLD XGBOOST MODEL
    # ========================================================

    bootstrap_csv_name = (
        'xgboost_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_'
        f'{bootstrap}'
    )

    csv_structure = os.path.join(
        xgboost_output_dir,
        f'{bootstrap_csv_name}_outer_fold_test_'
        f'{fold_n:05d}_run_*0_7*.csv'
    )

    csv_paths = sorted(
        glob(csv_structure)
    )

    best_score = None
    best_csv_path = None

    for csv_path in csv_paths:

        df_score = pd.read_csv(
            csv_path,
            low_memory=False
        )

        current_score = np.max(
            df_score['Mean_Test_Scores']
        )

        if (
            best_score is None
            or current_score >= best_score
        ):
            best_score = current_score
            best_csv_path = csv_path

    sav_file = best_csv_path.replace(
        '_mean_test_scores.csv',
        '.sav'
    )

    match = re.search(
        r'(?<=run_)\d+(?=_roc)',
        best_csv_path
    )

    xgboost_run = int(
        match.group()
    )

    fold_model = pickle.load(
        open(
            sav_file,
            'rb'
        )
    )

    # ========================================================
    # XGBOOST CLASSIFIER
    # ========================================================

    xg_classifier_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 8,
        'n_estimators': df_hp.loc[
            xgboost_run,
            'n_estimator'
        ],
        'max_depth': df_hp.loc[
            xgboost_run,
            'max_depth'
        ],
        'learning_rate': df_hp.loc[
            xgboost_run,
            'learning_rate'
        ],
        'subsample': df_hp.loc[
            xgboost_run,
            'subsample'
        ],
        'colsample_bytree': df_hp.loc[
            xgboost_run,
            'colsample_bytree'
        ],
        'gamma': df_hp.loc[
            xgboost_run,
            'gamma'
        ],
        'min_child_weight': df_hp.loc[
            xgboost_run,
            'min_child_weight'
        ]
    }

    for param, value in xg_classifier_params.items():

        if isinstance(value, str):

            try:
                xg_classifier_params[param] = float(value)

            except ValueError:
                pass

    xg_classifier_tile = XGBClassifier(
        **xg_classifier_params
    )

    # ========================================================
    # LOAD INNER-FOLD PERFORMANCE RESULTS
    # ========================================================

    input_file = (
        f'outer_fold_{fold_n:05d}_'
        'mean_inner_f1_score_plus_roc_auc_'
        'all_probability_thresholds.csv'
    )

    input_path = os.path.join(
        patient_output_dir,
        input_file
    )

    df_in = pd.read_csv(
        input_path
    )

    # ========================================================
    # SELECT MODEL
    # ========================================================

    # Apply the same selection rule as
    # select_best_probability_threshold():
    #
    # 1. Recurrent F1 must be >= non-recurrent F1.
    # 2. Find the first achievable non-recurrent F1 lower bound.
    # 3. Maximise recurrent F1.
    # 4. Maximise mean ROC-AUC.
    # 5. Break any remaining tie using the classification
    #    probability threshold closest to its median.

    df_threshold = df_in.copy()

    df_threshold = df_threshold[
        df_threshold['mean_f1_score_recur']
        >=
        df_threshold['mean_f1_score_nonrecur']
    ].copy()

    row = None

    for f1_score_lower in f1_score_thresholds:

        df_lower = df_threshold[
            df_threshold['mean_f1_score_nonrecur']
            >= f1_score_lower
        ].copy()

        if len(df_lower) == 0:
            continue

        max_recur = np.max(
            df_lower['mean_f1_score_recur']
        )

        df_lower = df_lower[
            df_lower['mean_f1_score_recur'] == max_recur
        ].copy()

        max_roc_auc = np.max(
            df_lower['mean_roc_auc']
        )

        df_lower = df_lower[
            df_lower['mean_roc_auc'] == max_roc_auc
        ].copy()

        median_probability_threshold = (
            df_lower[
                'classification_probability_threshold'
            ].median()
        )

        df_lower['threshold_distance'] = (
            df_lower[
                'classification_probability_threshold'
            ]
            - median_probability_threshold
        ).abs()

        df_lower = df_lower[
            df_lower['threshold_distance']
            ==
            df_lower['threshold_distance'].min()
        ].copy()

        row = df_lower.iloc[0]

        break

    if row is None:
        raise ValueError(
            f'No valid inner model found for outer fold {fold_n}.'
        )

    # ========================================================
    # SELECTED MODEL PARAMETERS
    # ========================================================

    run = row['run']
    run_name = row['run_name']
    combination = row['combination']

    # In the new performance-results format:
    # tile_count_threshold replaces the old 'threshold'
    # classification_probability_threshold replaces
    # the old 'threshold_probability'.

    tile_count_threshold = row[
        'tile_count_threshold'
    ]

    classification_probability_threshold = row[
        'classification_probability_threshold'
    ]

    combination_list = list(
        ast.literal_eval(combination)
    )

    df_logistic_row = df_logistic[
        df_logistic['run_name'] == run_name
    ].copy()

    logistic_model = create_logistic_model(
        df_logistic_row
    )

    # ========================================================
    # TILE DATA
    # ========================================================

    df = pd.read_csv(
        data_dir,
        low_memory=False
    )

    df = df[
        (df['tissue_proportion'] >= tissue_proportion) &
        (df['tumour_proportion'] >= tumour_proportion) &
        (df['gleason'].isin([1, 2]))
    ].reset_index(
        drop=True
    )

    # ========================================================
    # OUTER-TRAINING POPULATION
    # ========================================================

    patient_ids = list(
        df.patient_id.unique()
    )

    conditions = [
        (df['gleason'] == 1) & (df['case'] == 1.0),
        (df['gleason'] == 2) & (df['case'] == 1.0),
        (df['gleason'] == 1) & (df['case'] == 0.0),
        (df['gleason'] == 2) & (df['case'] == 0.0)
    ]

    df['clinical_score'] = np.select(
        conditions,
        range(4)
    )

    gleasons = []
    cases = []
    clinical_scores = []
    total_tiles = []

    for patient_id in patient_ids:

        patient_df = df[
            df.patient_id == patient_id
        ]

        gleasons.append(
            patient_df.gleason.unique()[0]
        )

        cases.append(
            patient_df.case.unique()[0]
        )

        clinical_scores.append(
            patient_df.clinical_score.unique()[0]
        )

        total_tiles.append(
            len(patient_df)
        )

    df_patient = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'clinical_score': clinical_scores,
        'total_tiles': total_tiles
    })

    df_strat = (
        df_patient
        .sort_values(
            by=[
                'clinical_score',
                'total_tiles'
            ],
            ascending=[
                True,
                False
            ]
        )
        .copy()
        .reset_index()
    )

    _, _, predefined_splits_array = (
        tmf.patient_tile_stratifier(
            df_strat,
            df,
            5,
            folds_csv_file
        )
    )

    df = df[
        predefined_splits_array != fold_n
    ].reset_index(
        drop=True
    )

    # ========================================================
    # INNER FOLDS
    # ========================================================

    patient_ids = list(
        df.patient_id.unique()
    )

    inner_folds_file = os.path.join(
        main_path,
        f'inner_folds_{fold_n}_data.csv'
    )

    inner_folds = np.arange(
        n_inner_folds
    )

    for inner_fold in inner_folds:

        print(
            inner_fold
        )

        # ================================================
        # INNER-FOLD PATIENT STRATIFICATION
        # ================================================

        gleasons = []
        cases = []
        clinical_scores = []
        total_tiles = []

        for patient_id in patient_ids:

            patient_df = df[
                df.patient_id == patient_id
            ]

            gleasons.append(
                patient_df.gleason.unique()[0]
            )

            cases.append(
                patient_df.case.unique()[0]
            )

            clinical_scores.append(
                patient_df.clinical_score.unique()[0]
            )

            total_tiles.append(
                len(patient_df)
            )

        df_patient = pd.DataFrame({
            'patient_id': patient_ids,
            'gleason': gleasons,
            'case': cases,
            'clinical_score': clinical_scores,
            'total_tiles': total_tiles
        })

        df_strat = (
            df_patient
            .sort_values(
                by=[
                    'clinical_score',
                    'total_tiles'
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .copy()
            .reset_index()
        )

        _, _, predefined_splits_array = (
            tmf.patient_tile_stratifier(
                df_strat,
                df,
                5,
                inner_folds_file
            )
        )

        # ================================================
        # INNER TRAIN / TEST DATA
        # ================================================

        df_train = df[
            predefined_splits_array != inner_fold
        ].reset_index(
            drop=True
        )

        df_test = df[
            predefined_splits_array == inner_fold
        ].reset_index(
            drop=True
        )

        # ================================================
        # XGBOOST TILE FEATURES
        # ================================================

        drop_columns = [
            'patient_id',
            'slide_id',
            'area',
            'decision',
            'case',
            'scene',
            'tile',
            'quadrant',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion'
        ]

        X_train = df_train.drop(
            drop_columns,
            axis=1
        ).copy()

        y_train = df_train[
            'case'
        ]

        X_test = df_test.drop(
            drop_columns,
            axis=1
        ).copy()

        # ================================================
        # RFE FEATURE SELECTION
        # ================================================

        include_columns = list(
            X_train.columns[
                fold_model.ranking_ <= max_rank
            ]
        )

        print(
            'number of columns:'
        )

        print(
            len(include_columns)
        )

        X_train_subset = X_train[
            include_columns
        ].copy()

        X_test_subset = X_test[
            include_columns
        ].copy()

        # ================================================
        # FIT XGBOOST AND PREDICT TILE PROBABILITIES
        # ================================================

        xg_classifier_tile.fit(
            X_train_subset,
            y_train
        )

        train_pred_proba = (
            xg_classifier_tile
            .predict_proba(X_train_subset)[:, 1]
        )

        test_pred_proba = (
            xg_classifier_tile
            .predict_proba(X_test_subset)[:, 1]
        )

        # ================================================
        # PATIENT-LEVEL ECM FEATURES
        # ================================================

        train_unique_patients = (
            df_train.patient_id.unique()
        )

        test_unique_patients = (
            df_test.patient_id.unique()
        )

        df_train_proba = aggregate_tile_probabilities_to_patients(
            train_pred_proba,
            df_train,
            train_unique_patients
        )

        df_test_proba = aggregate_tile_probabilities_to_patients(
            test_pred_proba,
            df_test,
            test_unique_patients
        )

        # ================================================
        # STANDARDISE ECM FEATURES USING TRAINING DATA ONLY
        # ================================================

        probability_columns = [
            'mean_prob',
            'std_prob',
            'max_prob'
        ]

        probability_means = (
            df_train_proba[
                probability_columns
            ].mean()
        )

        probability_stds = (
            df_train_proba[
                probability_columns
            ].std()
        )

        df_train_proba[
            probability_columns
        ] = (
            df_train_proba[probability_columns]
            - probability_means
        ) / probability_stds

        df_test_proba[
            probability_columns
        ] = (
            df_test_proba[probability_columns]
            - probability_means
        ) / probability_stds

        # ================================================
        # TEST PATIENTS ABOVE TILE-COUNT THRESHOLD
        # ================================================

        df_test_thresholded = df_test_proba[
            df_test_proba.n_tiles >= tile_count_threshold
        ].copy()

        X_patient_train = df_train_proba[
            combination_list
        ]

        y_patient_train = df_train_proba[
            'case'
        ]

        X_patient_test = df_test_thresholded[
            combination_list
        ]

        # ================================================
        # FIT LOGISTIC MODEL AND PREDICT
        # ================================================

        logistic_model.fit(
            X_patient_train,
            y_patient_train
        )

        y_pred_probability = (
            logistic_model
            .predict_proba(X_patient_test)[:, 1]
        )

        # ================================================
        # ADD PREDICTIONS / METADATA
        # ================================================

        df_test_thresholded[
            'recurrence_probability'
        ] = y_pred_probability

        df_test_thresholded['fold'] = fold_n
        df_test_thresholded['bootstrap'] = bootstrap
        df_test_thresholded['run'] = run
        df_test_thresholded['run_name'] = run_name
        df_test_thresholded['combination'] = combination
        df_test_thresholded[
            'tile_count_threshold'
        ] = tile_count_threshold
        df_test_thresholded[
            'classification_probability_threshold'
        ] = classification_probability_threshold

        # Convert ECM features back to original units
        df_test_thresholded[
            probability_columns
        ] = (
            df_test_thresholded[
                probability_columns
            ] * probability_stds
        ) + probability_means

        # ================================================
        # SAVE PATIENT-LEVEL PREDICTIONS
        # ================================================

        selected_model_save_name = (
            f'outer_fold_{fold_n:05d}'
            f'_inner_fold_{inner_fold:05d}_'
            f'{subfile_name}_'
            f'bootstrap_{bootstrap:03d}_'
            'inner_cv_selected_model_patient_probability_predictions.csv'
        )


        selected_model_path = os.path.join(
            patient_output_dir,
            selected_model_save_name
        )

        print(
            selected_model_path
        )

        try:

            df_test_thresholded.to_csv(
                selected_model_path,
                index=False
            )

            print(
                'Successfully saved CSV'
            )

        except Exception as e:

            print(
                f'Error saving CSV: {e}'
            )