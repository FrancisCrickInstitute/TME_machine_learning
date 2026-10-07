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
    'tile_threshold_f1score_removed_xgboost_outer_fold_20/'
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
    'tile_threshold_f1score_removed_xgboost_outer_fold_20/'
    'fold_00000/run_00000'
)

data_dir = os.path.join(
    input_dir,
    'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
)


# ============================================================
# MODEL AND ANALYSIS SETTINGS
# ============================================================

outer_fold = 0
fold_assignment_bootstrap = 20
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

hyperparameters_df = hyperparameters_df.rename(columns={
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
    r'(?<=run_)\d+(?=_roc)',
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

xg_classifier_params = {
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
}

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
    penalty='l1',
    C=0.000100,
    solver='liblinear',
    l1_ratio=None,
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
    'inner_folds_0_data.csv'
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


    df_test_patients = pd.DataFrame({
        'patient_id': test_patient_ids,
        'gleason': test_gleasons,
        'case': test_cases,
        'n_tiles': test_n_tiles,
        'mean_prob': test_mean_prob,
        'std_prob': test_std_prob,
        'max_prob': test_max_prob
    })

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


    df_train_patients = pd.DataFrame({
        'patient_id': train_patient_ids,
        'gleason': train_gleasons,
        'case': train_cases,
        'n_tiles': train_n_tiles,
        'mean_prob': train_mean_prob,
        'std_prob': train_std_prob,
        'max_prob': train_max_prob
    })

    df_train_patients['std_prob'] = (
        df_train_patients['std_prob']
        .fillna(0)
    )


    # ========================================================
    # STANDARDISE PATIENT-LEVEL XGBOOST FEATURES
    # ========================================================

    patient_model_features = [
        'mean_prob',
        'std_prob',
        'max_prob'
    ]

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


                df_f1_scores = pd.DataFrame({
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
                })

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

                metrics_results.append({
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
                })


                # ------------------------------------------------
                # Store predictions
                # ------------------------------------------------

                prediction_results.append(
                    pd.DataFrame({
                        'y_true': y_test.values,
                        'y_pred_prob': y_pred_prob,
                        'combination': [combination] *
                                       len(y_test),
                        'fold': [inner_fold] *
                                len(y_test),
                        'n_tiles_threshold':
                            [tile_count_threshold] *
                            len(y_test)
                    })
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
    f'outer_fold_{outer_fold:05d}'
    f'_run_{run_number:05d}'
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
        f'{run_prefix}_{suffix}.csv'
    )

    output_df.to_csv(
        filename,
        index=False
    )