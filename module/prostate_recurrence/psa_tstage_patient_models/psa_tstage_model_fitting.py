import os
import sys
import ast
import numpy as np
import pandas as pd
from glob import glob

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


def generate_logistic_script(
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
    """Write one logistic-regression analysis script and record its parameters."""

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
    'xgboost_psa_tstage_only_outer_fold_{bootstrap}/'
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
    'xgboost_psa_tstage_only_outer_fold_{bootstrap}/'
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
    penalty={penalty!r},
    C={C!r},
    solver={solver!r},
    l1_ratio={l1_ratio!r},
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


        df_f1_scores = pd.DataFrame({{
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
        }})

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

        metrics_results.append({{
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
        }})


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        prediction_results.append(
            pd.DataFrame({{
                'y_true': y_test.values,
                'y_pred_prob': y_pred_prob,
                'combination': [clinical_model_features] *
                               len(y_test),
                'fold': [inner_fold] *
                        len(y_test),
                'n_tiles_threshold': [tile_count_threshold] *
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
    f'outer_fold_test_{{outer_fold:05d}}'
    f'run_{{run_number:05d}}'
)

output_files = {{
    'metrics_patient_level':
        metrics_df,
    'f1_probability_threshold_patient_level':
        f1_thresholds_df,
    'all_probability_predictions':
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

    return script_filename


def load_csv_data(csv_path):

    return pd.read_csv(
        csv_path,
        low_memory=False
    )


def get_run_fold_and_logistic_params(
    csv_path,
    df_logistic
):

    fold = int(
        csv_path.split('fold_test_')[1].split('run_')[0]
    )

    run = int(
        csv_path.split('run_')[2].split('_')[0]
    )

    run_name = (
        f'Fold_{fold}_Run_{run}'
    )

    print('run_name:')
    print(run_name)

    df_row = df_logistic[
        df_logistic['run_name'] == run_name
    ]

    logistic_params = {
        'penalty':
            df_row['penalty'].iloc[0],

        'C':
            df_row['C'].iloc[0],

        'solver':
            df_row['solver'].iloc[0],

        'l1_ratio':
            df_row['l1_ratio'].iloc[0],

        'max_iter':
            1000,

        'random_state':
            8
    }

    return (
        run_name,
        fold,
        run,
        logistic_params
    )



def create_patient_dataframe(df_tiles):
    """Create one row per patient with clinical and tile-count data."""

    return (
        df_tiles.groupby('patient_id', as_index=False)
        .agg(
            gleason=('gleason', 'first'),
            case=('case', 'first'),
            clinical_score=('clinical_score', 'first'),
            total_tiles=('case', 'size')
        )
    )


def add_clinical_features(df_patients, df_clinical):
    """Add PSA and one-hot encoded T-stage to patient-level data."""

    df_patients = df_patients.copy()

    df_patients = df_patients.merge(
        df_clinical[['patient_id', 'psa_liz', 'Tstage_group']],
        on='patient_id',
        how='left'
    )

    df_patients['psa'] = df_patients['psa_liz']

    df_patients['t2'] = (
        df_patients['Tstage_group'] == 'T2'
    ).astype(int)

    df_patients['t3'] = (
        df_patients['Tstage_group'] == 'T3'
    ).astype(int)

    return df_patients.drop(
        columns=['psa_liz', 'Tstage_group']
    )


def prepare_patient_data(df_tiles, df_clinical):
    """Create the patient-level dataset used by the logistic model."""

    df_patients = create_patient_dataframe(df_tiles)

    return add_clinical_features(
        df_patients,
        df_clinical
    )


def get_patient_splits(
    df_tiles,
    df_clinical,
    patient_ids_to_exclude=None
):
    """Create patient-level clinical data for a tile dataframe."""

    df_patients = (
        create_patient_dataframe(df_tiles)
        .loc[lambda x: ~x['patient_id'].isin(
            patient_ids_to_exclude or []
        )]
        .reset_index(drop=True)
    )

    return add_clinical_features(
        df_patients,
        df_clinical
    )




def generate_performance_csv(
    python_code_dir,
    output_dir
):

    tile_count_thresholds = [
        0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50
    ][::-1]


    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

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
    # FIND F1 CSV FILES
    # ========================================================

    csv_structure = os.path.join(
        output_dir,
        'fold_*',
        'run_*',
        'outer_fold_test_*run_*_f1_probability_threshold_patient_level.csv'
    )

    csv_paths = sorted(
        glob(
            csv_structure
            .replace('/', os.sep)
            .replace('\\', os.sep)
        )
    )


    # ========================================================
    # GENERATE PERFORMANCE RESULTS
    # ========================================================

    performance_results_df = performance_results_df_generator(
        csv_paths,
        df_logistic,
        tile_count_thresholds,
        output_dir
    )


    # ========================================================
    # SAVE
    # ========================================================

    performance_results_df.to_csv(
        os.path.join(
            output_dir,
            'mean_inner_f1_score_plus_roc_auc_all_probability_thresholds.csv'
        ),
        index=False
    )


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
            f'outer_fold_test_{fold_string}run_{run_string}_metrics_patient_level.csv'
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



def generate_inner_outer_scores_for_graded_f1_score_psa_only(
    bootstrap,
    fold_n,
    subfile_name,
    f1_score_thresholds
):

    # ========================================================
    # RUN NAME
    # ========================================================

    run_name = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        f'psa_tstage_only_outer_fold_{bootstrap}'
    )

    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    python_code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/'
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

    print(df_logistic.head())

    # ========================================================
    # PATHS
    # ========================================================

    code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/'
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

    data_dir = os.path.join(
        input_dir,
        'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
    )

    psa_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/raw_data/clinical/'
        'prostate/chiip_cohort/TPSAdata/'
    )

    ecm_threshold_file = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/'
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        'psa_tstage_only_outer_fold_20/selected_thresholds_from_ecm.csv'
    )

    folds_csv_file = os.path.join(
        code_dir,
        f'folds_random_{bootstrap:03d}.csv'
    )

    # ========================================================
    # ECM THRESHOLD
    # ========================================================

    df_ecm_threshold = pd.read_csv(
        ecm_threshold_file
    )

    df_ecm_threshold = df_ecm_threshold[
        (df_ecm_threshold['bootstrap'] == bootstrap) &
        (df_ecm_threshold['outer_fold'] == fold_n)
    ].copy()

    ecm_threshold = (
        df_ecm_threshold['threshold'].iloc[0]
    )

    # ========================================================
    # PSA DATA
    # ========================================================

    df_psa = pd.read_csv(
        os.path.join(
            psa_dir,
            'baseline_fullchhipcohort_PSATstage.csv'
        ),
        low_memory=False
    )

    df_psa["patient_id"] = (
        df_psa["trialno"]
        .astype(str)
        .str.zfill(6)
        .radd("C")
    )

    df_psa = df_psa.drop(
        [
            "case",
            "trialno",
            "prehrmpsa",
            "clint_clean",
            "prehcat2",
            "prehcat1"
        ],
        axis=1
    )

    # ========================================================
    # TILE DATA
    # ========================================================

    df = pd.read_csv(
        data_dir,
        low_memory=False
    )

    df = df[
        df['tissue_proportion'] >= 0.7
    ]

    df = df[
        df['tumour_proportion'] >= 0.7
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

        gleasons.append(
            df[
                df.patient_id == patient_id
            ].gleason.unique()[0]
        )

        cases.append(
            df[
                df.patient_id == patient_id
            ].case.unique()[0]
        )

        clinical_scores.append(
            df[
                df.patient_id == patient_id
            ].clinical_score.unique()[0]
        )

        total_tiles.append(
            len(
                df[
                    df.patient_id == patient_id
                ].case
            )
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
    # TRAIN / TEST PATIENTS
    # ========================================================

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

    test_unique_patients = [
        patient_id
        for patient_id in df_test.patient_id.unique()
        if patient_id != ""
    ]

    train_unique_patients = [
        patient_id
        for patient_id in df_train.patient_id.unique()
        if patient_id != ""
    ]

    # ========================================================
    # PSA PROBABILITIES
    # ========================================================

    df_test_proba = calculate_patient_psa(
        df_test,
        test_unique_patients,
        df_psa
    )

    df_train_proba = calculate_patient_psa(
        df_train,
        train_unique_patients,
        df_psa
    )

    psa_columns = ['psa']

    psa_means = df_train_proba[
        psa_columns
    ].mean()

    psa_stds = df_train_proba[
        psa_columns
    ].std()

    df_train_proba[
        psa_columns
    ] = (
        df_train_proba[psa_columns] - psa_means
    ) / psa_stds

    df_test_proba[
        psa_columns
    ] = (
        df_test_proba[psa_columns] - psa_means
    ) / psa_stds

    # ========================================================
    # PERFORMANCE RESULTS
    # ========================================================

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
    # PSA / ECM THRESHOLD ANALYSES
    # ========================================================

    threshold_analyses = [
        (
            0,
            'psa_zero_threshold'
        ),
        (
            ecm_threshold,
            'psa_ecm_threshold'
        )
    ]

    for tile_count_threshold, output_suffix in threshold_analyses:

        # ====================================================
        # SELECT MAXIMUM F1 SCORE THRESHOLD
        # ====================================================

        f1_score_df = max_f1score_graded_single_threshold(
            performance_results_df,
            patient_output_dir,
            f1_score_thresholds,
            tile_count_threshold
        )

        f1_score_df = f1_score_df[
            f1_score_df['outer_fold'] == fold_n
        ].copy()

        print(
            'The input f1_score_df is given by:'
        )

        print(
            f1_score_df
        )

        print(
            'end'
        )

        print(
            f1_score_df['combination'].unique()
        )

        # ====================================================
        # OUTER-FOLD LOGISTIC MODEL
        # ====================================================

        outer_logistic_f1_score_df = (
            run_logistic_for_graded_f1_score(
                f1_score_df,
                df_logistic,
                df_test_proba,
                df_train_proba,
                bootstrap,
                fold_n
            )
        )

        # ====================================================
        # SAVE OUTER-FOLD RESULTS
        # ====================================================

        save_name = (
            f'outer_fold_{fold_n:05d}'
            f'_graded_f1_score_outer_fold_{subfile_name}'
            f'_{output_suffix}.csv'
        )

        print(
            os.path.join(
                patient_output_dir,
                save_name
            )
        )

        try:

            outer_logistic_f1_score_df.to_csv(
                os.path.join(
                    patient_output_dir,
                    save_name
                ),
                index=False
            )

            print(
                "Successfully saved CSV"
            )

        except Exception as e:

            print(
                f"Error saving CSV: {e}"
            )

        # ====================================================
        # SAVE INNER-FOLD PERFORMANCE
        # ====================================================

        save_name = (
            f'outer_fold_{fold_n:05d}'
            f'_graded_f1_score_inner_fold_{subfile_name}'
            f'_performance_{output_suffix}.csv'
        )

        print(
            os.path.join(
                patient_output_dir,
                save_name
            )
        )

        f1_score_df.to_csv(
            os.path.join(
                patient_output_dir,
                save_name
            ),
            index=False
        )


def run_logistic_for_graded_f1_score(
    f1_score_df,
    df_logistic,
    df_test_proba,
    df_train_proba,
    bootstrap,
    fold_n
):

    results = []

    print('length of df is:')
    print(len(f1_score_df))

    for _, row in f1_score_df.iterrows():

        # ====================================================
        # RUN INFORMATION
        # ====================================================

        run_name = row['run_name']
        combination = row['combination']
        threshold = row['threshold']
        threshold_probability = row['threshold_probability']
        max_threshold = row.get(
            'max_threshold',
            None
        )

        print(combination)

        # ====================================================
        # LOGISTIC MODEL
        # ====================================================

        df_logistic_row = df_logistic[
            df_logistic['run_name'] == run_name
        ].copy()

        logistic_model = get_logistic_model(
            df_logistic_row
        )

        feature_columns = list(
            ast.literal_eval(combination)
        )

        # ====================================================
        # TEST DATA
        # ====================================================

        df_test_thresholded = df_test_proba[
            df_test_proba.n_tiles >= threshold
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

        y_pred_probability = logistic_model.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_pred_probability
        )

        y_pred = np.zeros(
            len(y_test)
        )

        y_pred[
            y_pred_probability >= threshold_probability
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

            'threshold':
                threshold,

            'max_threshold':
                max_threshold,

            'threshold_probability':
                threshold_probability,

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

















    
def calculate_patient_psa(df, unique_patients, df_psa):
    """Calculate probabilities for each patient and return a DataFrame."""
    patient_ids = []
    cases = []
    gleasons = []
    n_tiles = []
    psa = []
    t2=[]
    t3=[]

    print('unqie_patients in function :')
    print(unique_patients)
    if isinstance(unique_patients, str):
        unique_patients = [unique_patients]

    for unique_patient in unique_patients:
        print(unique_patient)
        #patient_proba = pred_proba[df.patient_id == unique_patient]
        #print(patient_proba)
        patient_ids.append(unique_patient)
        print(df[df.patient_id == unique_patient])
        cases.append(df[df.patient_id == unique_patient].case.unique()[0])
        gleasons.append(df[df.patient_id == unique_patient].gleason.unique()[0])
        n_tiles.append(len(df[df.patient_id == unique_patient]))
   
        single_patient = df_psa[df_psa['patient_id']==unique_patient].copy()
        psa.append(single_patient['psa_liz'].iloc[0])
        tstage = single_patient['Tstage_group'].iloc[0]
        if tstage == "T1":
            t2.append(0)
            t3.append(0)
        elif tstage == "T2":
            t2.append(1)
            t3.append(0)
        elif tstage == "T3":
            t2.append(0)
            t3.append(1)

    # Create a DataFrame for patient probabilities
    df_proba = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'n_tiles': n_tiles,
        'psa': psa,
        't2': t2,
        't3': t3
    })

    return df_proba

   
def max_f1score_graded_single_threshold(df, input_dir, f1_score_thresholds,threshold_in):
    outer_folds = df['outer_fold'].unique()
    combinations = df['combination'].unique()
    thresholds = df['threshold'].unique()

    rows = []
    roc_aucs = []

    for outer_fold in outer_folds:
        df_outer = df[df['outer_fold'] == outer_fold].copy()

        df_combination = df_outer.copy()

        df_threshold = df_combination[df_combination['threshold'] == threshold_in].copy()
        df_threshold = df_threshold[df_threshold['mean_f1_score_recur'] >= df_threshold['mean_f1_score_nonrecur']].copy()

        for f1_score_lower in f1_score_thresholds:
            df_lower = df_threshold[df_threshold['mean_f1_score_nonrecur'] >= f1_score_lower].copy()
            if len(df_lower) == 0:
                continue
            max_recur = np.max(df_lower['mean_f1_score_recur'])
            df_lower = df_lower[df_lower['mean_f1_score_recur']==max_recur].copy()
            max_roc_auc = np.max(df_lower['mean_roc_auc'])
            df_lower = df_lower[df_lower['mean_roc_auc']==max_roc_auc].copy()
            
            median_prob = df_lower['threshold_probability'].median()
            df_lower['dist_to_median'] = np.abs(df_lower['threshold_probability'] - median_prob)
            df_lower = df_lower.sort_values('dist_to_median').reset_index(drop=True)
            row = df_lower.iloc[0]

            fold_string = str(row['outer_fold']).zfill(5)
            run_string = str(row['run']).zfill(5)
            csv_location = os.path.join(
                input_dir,
                f'fold_{fold_string}',
                f'run_{run_string}',
                f'outer_fold_test_{fold_string}run_{run_string}_roc_auc_svm_patient_level.csv'
            )

            roc_auc_df = pd.read_csv(csv_location)
            roc_value = roc_auc_df[
                (roc_auc_df['combination'] == row['combination']) &
                (roc_auc_df['threshold'] == row['threshold'])
            ]['roc_auc'].mean()

            rows.append(row)
            roc_aucs.append(roc_value)
            break

    f1_score_df = pd.DataFrame(rows)
    f1_score_df['mean_roc_auc'] = roc_aucs

    numeric_cols = f1_score_df.select_dtypes(include='object').columns
    f1_score_df[numeric_cols] = f1_score_df[numeric_cols].apply(pd.to_numeric, errors='ignore')
    f1_score_df  = f1_score_df.drop(['dist_to_median'],axis=1)
    return f1_score_df


def get_logistic_model(df_row):
    """Create and return an SVM model based on the parameters from df_row."""
    return LogisticRegression(
        penalty=df_row['penalty'].iloc[0],
        C=df_row['C'].iloc[0],
        solver=df_row['solver'].iloc[0],
        l1_ratio=df_row['l1_ratio'].iloc[0],
        max_iter = 1000,
        random_state=8,
    )






def generate_final_model_from_inner_psa_predicted_probabilities(
    bootstrap,
    fold_n,
    subfile_name,
    f1_score_thresholds
):


    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    run_name_prefix = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        f'psa_tstage_only_outer_fold_{bootstrap}'
    )

    python_code_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        f'{run_name_prefix}/{run_name_prefix}_fold_runkey.txt'
    )

    df_logistic = pd.read_table(
        python_code_dir,
        delimiter=" ",
        header=None
    )

    df_logistic = df_logistic.rename(
        columns={
            0: "run_name",
            1: "fold",
            2: "penalty",
            3: "C",
            4: "solver",
            5: "l1_ratio"
        }
    )

    df_logistic = df_logistic.replace({
        np.nan: None
    })

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

    psa_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/raw_data/clinical/'
        'prostate/chiip_cohort/TPSAdata/'
    )

    ecm_threshold_file = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        'psa_tstage_only_outer_fold_20/selected_thresholds_from_ecm.csv'
    )

    folds_csv_file = os.path.join(
        code_dir,
        f'folds_random_{bootstrap:03d}.csv'
    )

    # ========================================================
    # ECM THRESHOLD
    # ========================================================

    df_ecm_threshold = pd.read_csv(
        ecm_threshold_file
    )

    df_ecm_threshold = df_ecm_threshold[
        (df_ecm_threshold['bootstrap'] == bootstrap) &
        (df_ecm_threshold['outer_fold'] == fold_n)
    ].copy()

    ecm_threshold = (
        df_ecm_threshold['threshold'].iloc[0]
    )

    # ========================================================
    # PSA DATA
    # ========================================================

    df_psa = pd.read_csv(
        os.path.join(
            psa_dir,
            'baseline_fullchhipcohort_PSATstage.csv'
        ),
        low_memory=False
    )

    df_psa["patient_id"] = (
        df_psa["trialno"]
        .astype(str)
        .str.zfill(6)
        .radd("C")
    )

    df_psa = df_psa.drop(
        [
            "case",
            "trialno",
            "prehrmpsa",
            "clint_clean",
            "prehcat2",
            "prehcat1"
        ],
        axis=1
    )

    # ========================================================
    # TILE DATA
    # ========================================================

    df = pd.read_csv(
        data_dir,
        low_memory=False
    )

    df = df[
        (df['tissue_proportion'] >= 0.7) &
        (df['tumour_proportion'] >= 0.7) &
        (df['gleason'].isin([1, 2]))
    ].reset_index(
        drop=True
    )

    # ========================================================
    # PATIENT-LEVEL STRATIFICATION
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

    # ========================================================
    # TRAIN / TEST DATA
    # ========================================================

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

    test_unique_patients = [
        patient_id
        for patient_id in df_test.patient_id.unique()
        if patient_id != "" #Removed for GitHub
    ]

    train_unique_patients = [
        patient_id
        for patient_id in df_train.patient_id.unique()
        if patient_id != ""
    ]

    # ========================================================
    # PATIENT-LEVEL PSA / T-STAGE DATA
    # ========================================================

    df_test_proba = calculate_patient_psa(
        df_test,
        test_unique_patients,
        df_psa
    )

    df_train_proba = calculate_patient_psa(
        df_train,
        train_unique_patients,
        df_psa
    )

    # ========================================================
    # STANDARDISE PSA USING TRAINING DATA ONLY
    # ========================================================

    psa_columns = ['psa']

    psa_means = df_train_proba[
        psa_columns
    ].mean()

    psa_stds = df_train_proba[
        psa_columns
    ].std()

    df_train_proba[
        psa_columns
    ] = (
        df_train_proba[psa_columns] - psa_means
    ) / psa_stds

    df_test_proba[
        psa_columns
    ] = (
        df_test_proba[psa_columns] - psa_means
    ) / psa_stds

    clinical_model_features = [
        'psa',
        't2',
        't3'
    ]

    # ========================================================
    # PSA / ECM THRESHOLD ANALYSES
    # ========================================================

    threshold_analyses = [
        (
            'psa_zero_threshold',
            'psa_zero_threshold'
        ),
        (
            'psa_ecm_threshold',
            'psa_ecm_threshold'
        )
    ]

    for output_suffix, model_type in threshold_analyses:

        # ====================================================
        # INPUT FILE
        # ====================================================

        input_file = (
            f'outer_fold_{fold_n:05d}'
            '_graded_f1_score_inner_fold_'
            f'{subfile_name}_performance_{output_suffix}.csv'
        )

        input_path = os.path.join(
            patient_output_dir,
            input_file
        )

        print(
            "Trying:",
            input_path
        )

        print(
            "Exists:",
            os.path.exists(input_path)
        )

        print(
            "Directory exists:",
            os.path.exists(patient_output_dir)
        )

        print(
            "Files in directory:"
        )

        print(
            os.listdir(patient_output_dir)
        )

        # ====================================================
        # LOAD SELECTED INNER-FOLD MODEL
        # ====================================================

        df_in = pd.read_csv(
            input_path
        )

        row = df_in.iloc[0]

        run = row['run']
        run_name = row['run_name']
        combination = row['combination']
        threshold = row['threshold']
        threshold_probability = row['threshold_probability']

        df_logistic_row = df_logistic[
            df_logistic['run_name'] == run_name
        ].copy()

        logistic_model = get_logistic_model(
            df_logistic_row
        )

        # ====================================================
        # TEST DATA ABOVE TILE-COUNT THRESHOLD
        # ====================================================

        df_test_thresholded = df_test_proba[
            df_test_proba.n_tiles >= threshold
        ].copy()

        X_train = df_train_proba[
            clinical_model_features
        ].copy()

        y_train = df_train_proba[
            'case'
        ].copy()

        X_test = df_test_thresholded[
            clinical_model_features
        ].copy()

        y_test = df_test_thresholded[
            'case'
        ].copy()

        # ====================================================
        # FIT MODEL AND PREDICT
        # ====================================================

        logistic_model.fit(
            X_train,
            y_train
        )

        y_pred_probability = logistic_model.predict_proba(
            X_test
        )[:, 1]

        # ====================================================
        # MODEL COEFFICIENTS / ODDS RATIOS
        # ====================================================

        coefficients = logistic_model.coef_[0]

        coef_dict = dict(
            zip(
                X_train.columns,
                coefficients
            )
        )

        coef_psa = coef_dict['psa']
        coef_t2 = coef_dict['t2']
        coef_t3 = coef_dict['t3']

        or_psa = np.exp(
            coef_psa
        )

        or_t2 = np.exp(
            coef_t2
        )

        or_t3 = np.exp(
            coef_t3
        )

        psa_sd = psa_stds['psa']

        coef_psa_per_unit = (
            coef_psa / psa_sd
        )

        or_psa_per_unit = np.exp(
            coef_psa_per_unit
        )

        # ====================================================
        # PATIENT-LEVEL LOG-ODDS CONTRIBUTIONS
        # ====================================================

        df_test_thresholded["psa_contribution"] = (
            df_test_thresholded["psa"] * coef_psa
        )

        df_test_thresholded["t2_contribution"] = (
            df_test_thresholded["t2"] * coef_t2
        )

        df_test_thresholded["t3_contribution"] = (
            df_test_thresholded["t3"] * coef_t3
        )

        df_test_thresholded["clinical_contribution"] = (
            df_test_thresholded["psa_contribution"]
            + df_test_thresholded["t2_contribution"]
            + df_test_thresholded["t3_contribution"]
        )

        # ====================================================
        # MODEL-INFLUENCE DATAFRAME
        # ====================================================

        df_model_influence = pd.DataFrame({

            'bootstrap': [
                bootstrap
            ],

            'fold': [
                fold_n
            ],

            'model_type': [
                model_type
            ],

            'run': [
                run
            ],

            'run_name': [
                run_name
            ],

            'combination': [
                combination
            ],

            'threshold': [
                threshold
            ],

            'threshold_probability': [
                threshold_probability
            ],

            'n_train_patients': [
                len(df_train_proba)
            ],

            'n_test_patients': [
                len(df_test_thresholded)
            ],

            'psa_sd': [
                psa_sd
            ],

            'coef_psa': [
                coef_psa
            ],

            'coef_psa_per_unit': [
                coef_psa_per_unit
            ],

            'coef_t2': [
                coef_t2
            ],

            'coef_t3': [
                coef_t3
            ],

            'or_psa': [
                or_psa
            ],

            'or_psa_per_unit': [
                or_psa_per_unit
            ],

            'or_t2': [
                or_t2
            ],

            'or_t3': [
                or_t3
            ],

            'penalty': [
                logistic_model.penalty
            ],

            'C': [
                logistic_model.C
            ],

            'solver': [
                logistic_model.solver
            ],

            'max_iter': [
                logistic_model.max_iter
            ],

            'l1_ratio': [
                getattr(
                    logistic_model,
                    'l1_ratio',
                    None
                )
            ]
        })

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\n============================================")
        print(
            f"MODEL INFLUENCE - {model_type.upper()}"
        )
        print("============================================")

        print(
            f"PSA coefficient: {coef_psa:+.4f}"
        )

        print(
            f"PSA OR:          {or_psa:.4f}"
        )

        print(
            f"T2 coefficient:  {coef_t2:+.4f}"
        )

        print(
            f"T2 OR:           {or_t2:.4f}"
        )

        print(
            f"T3 coefficient:  {coef_t3:+.4f}"
        )

        print(
            f"T3 OR:           {or_t3:.4f}"
        )

        # ====================================================
        # SAVE MODEL-INFLUENCE CSV
        # ====================================================

        feature_influence_save_name = (
            f'outer_fold_{fold_n:05d}_'
            f'{subfile_name}_'
            f'bootstrap_{bootstrap:03d}_'
            f'model_influence_{model_type}.csv'
        )

        feature_influence_path = os.path.join(
            patient_output_dir,
            feature_influence_save_name
        )

        print(
            "Saving model influence:",
            feature_influence_path
        )

        df_model_influence.to_csv(
            feature_influence_path,
            index=False
        )

        # ====================================================
        # SAVE PATIENT-LEVEL PREDICTIONS
        # ====================================================

        df_test_thresholded[
            'recurrence_probability'
        ] = y_pred_probability

        df_test_thresholded['fold'] = fold_n
        df_test_thresholded['bootstrap'] = bootstrap
        df_test_thresholded['run'] = run
        df_test_thresholded['run_name'] = run_name
        df_test_thresholded['combination'] = combination
        df_test_thresholded['threshold'] = threshold
        df_test_thresholded[
            'threshold_probability'
        ] = threshold_probability

        # Convert PSA back to original units
        df_test_thresholded[
            psa_columns
        ] = (
            df_test_thresholded[psa_columns] * psa_stds
        ) + psa_means

        selected_model_save_name = (
            f'outer_fold_{fold_n:05d}_'
            f'{subfile_name}_'
            f'bootstrap_{bootstrap:03d}_'
            f'selected_model_from_inner_{model_type}.csv'
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
                "Successfully saved CSV"
            )

        except Exception as e:

            print(
                "Error saving CSV"
            )



