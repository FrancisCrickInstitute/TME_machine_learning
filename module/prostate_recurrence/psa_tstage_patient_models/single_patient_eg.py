# ============================================================
# IMPORTS
# ============================================================

import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

import tile_model_fitting as tmf
import psa_tstage_model_fitting as ptmf


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
    'gleason_7_quad_1000_v8a_patient_level_logistic_removed_'
    'xgboost_psa_tstage_only_outer_fold_20/'
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

psa_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/raw_data/clinical/'
    'prostate/chiip_cohort/TPSAdata/'
)

run_dir = (
    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
    'feature_analysis_v3/rfe_runs/'
    'gleason_7_quad_1000_v8a_patient_level_logistic_removed_'
    'xgboost_psa_tstage_only_outer_fold_20/'
    'fold_00000/run_00011'
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
    0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50
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

hyperparameters_df = hyperparameters_df[
    hyperparameters_df['fold'] == outer_fold
]

hyperparameters_df = hyperparameters_df.loc[0:288]


# ============================================================
# PATIENT-LEVEL CLINICAL DATA
# ============================================================

df_clinical = pd.read_csv(
    os.path.join(
        psa_dir,
        'baseline_fullchhipcohort_PSATstage.csv'
    ),
    low_memory=False
)

df_clinical['patient_id'] = (
    df_clinical['trialno']
    .astype(str)
    .str.zfill(6)
    .radd('C')
)

df_clinical = df_clinical.drop(
    [
        'case',
        'trialno',
        'prehrmpsa',
        'clint_clean',
        'prehcat2',
        'prehcat1',
    ],
    axis=1
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    penalty='l1',
    C=10.0,
    solver='saga',
    l1_ratio=None,
    max_iter=1000,
    random_state=8
)

clinical_model_features = ['psa', 't2', 't3']


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

df_patients = ptmf.create_patient_dataframe(df_tiles)

folds_csv_file = (
    code_dir +
    'folds_random_' +
    str(fold_assignment_bootstrap).zfill(3) +
    '.csv'
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

df_patients = ptmf.create_patient_dataframe(df_tiles)

df_stratified_patients = (
    df_patients
    .sort_values(
        by=['clinical_score', 'total_tiles'],
        ascending=[True, False]
    )
    .reset_index()
)

folds_csv_file = main_path + 'inner_folds_0_data.csv'

metrics_results = []
f1_threshold_results = []
prediction_results = []

inner_folds = np.arange(n_inner_folds)


# ============================================================
# INNER CROSS-VALIDATION
# ============================================================

for inner_fold in inner_folds:

    print(inner_fold)

    # --------------------------------------------------------
    # Create patient/tile split
    # --------------------------------------------------------

    df_patients = ptmf.create_patient_dataframe(df_tiles)

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
    # Patient-level clinical data
    # --------------------------------------------------------

    excluded_patient_ids = []

    train_patients = ptmf.get_patient_splits(
        df_train_tiles,
        df_clinical,
        patient_ids_to_exclude=excluded_patient_ids
    )

    test_patients = ptmf.get_patient_splits(
        df_test_tiles,
        df_clinical,
        patient_ids_to_exclude=excluded_patient_ids
    )


    # --------------------------------------------------------
    # Standardise PSA using training data only
    # --------------------------------------------------------

    psa_mean = train_patients['psa'].mean()
    psa_std = train_patients['psa'].std()

    train_patients['psa'] = (
        train_patients['psa'] - psa_mean
    ) / psa_std

    test_patients['psa'] = (
        test_patients['psa'] - psa_mean
    ) / psa_std


    # --------------------------------------------------------
    # Fit patient-level logistic regression
    # --------------------------------------------------------

    X_clinical_train = train_patients[
        clinical_model_features
    ]

    y_train = train_patients['case']

    logistic_model.fit(
        X_clinical_train,
        y_train
    )


    # ========================================================
    # EVALUATE ACROSS TILE-COUNT THRESHOLDS
    # ========================================================

    for tile_count_threshold in tile_count_thresholds:

        test_patients_filtered = test_patients[
            test_patients['total_tiles'] >= tile_count_threshold
        ].copy()

        X_clinical_test = test_patients_filtered[
            clinical_model_features
        ]

        y_test = test_patients_filtered['case']

        n_test_patients = len(X_clinical_test)

        if n_test_patients < 5:
            continue


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred_prob = logistic_model.predict_proba(
            X_clinical_test
        )[:, 1]

        y_pred = logistic_model.predict(
            X_clinical_test
        )


        # ----------------------------------------------------
        # F1 across probability thresholds
        # ----------------------------------------------------

        classification_probability_thresholds = np.linspace(
            0.01,
            0.99,
            99
        )

        f1_recurrence = []
        f1_nonrecurrence = []

        for probability_threshold in classification_probability_thresholds:

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
            'combination': [clinical_model_features] *
                           len(classification_probability_thresholds),
            'fold': [inner_fold] *
                    len(classification_probability_thresholds),
            'tile_count_threshold': [tile_count_threshold] *
                                    len(classification_probability_thresholds),
            'test_size': [n_test_patients] *
                         len(classification_probability_thresholds),
            'classification_probability_threshold':
                classification_probability_thresholds,
            'f1_recur': f1_recurrence,
            'f1_nonrecur': f1_nonrecurrence
        })

        f1_threshold_results.append(
            df_f1_scores
        )


        # ----------------------------------------------------
        # Standard metrics
        # ----------------------------------------------------

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
            'combination': clinical_model_features,
            'fold': inner_fold,
            'roc_auc': roc_auc,
            'f1_score_recur': f1_recurrence_score,
            'f1_score_nonrecur': f1_nonrecurrence_score,
            'f1_score_average': np.mean([
                f1_recurrence_score,
                f1_nonrecurrence_score
            ]),
            'f1_score_max_derivative': max_f1_derivative,
            'accuracy': accuracy_score(
                y_test,
                y_pred
            ),
            'tile_count_threshold': tile_count_threshold,
            'test_size': n_test_patients
        })


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        prediction_results.append(
            pd.DataFrame({
                'y_true': y_test.values,
                'y_pred_prob': y_pred_prob,
                'combination': [clinical_model_features] *
                               len(y_test),
                'fold': [inner_fold] *
                        len(y_test),
                'n_tiles_threshold': [tile_count_threshold] *
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
    f'outer_fold_test_{outer_fold:05d}'
    f'run_{11:05d}'
)

output_files = {
    'metrics_patient_level':
        metrics_df,
    'f1_probability_threshold_patient_level':
        f1_thresholds_df,
    'all_probability_predictions':
        predictions_df,
}

for suffix, output_df in output_files.items():

    filename = os.path.join(
        run_dir,
        f'{run_prefix}_{suffix}.csv'
    )

    output_df.to_csv(
        filename,
        index=False
    )

