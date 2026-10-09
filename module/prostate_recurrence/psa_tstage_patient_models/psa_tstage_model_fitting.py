import os
import sys
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import re
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

#duplicate with ecm or analagous

# called by performance_results_df_generator. Not really needed
def load_csv_data(csv_path):

    return pd.read_csv(
        csv_path,
        low_memory=False
    )


#called by generate_performance_csv
def performance_results_df_generator(
    csv_paths,
    df_logistic,
    tile_count_thresholds,
    output_dir,
    data_type
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
            f'outer_fold_{fold_string}_run_{run_string}_inner_cv_metrics_patient_level_{data_type}.csv'
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





# called by generate_inner_outer_scores_for_graded_f1_score_psa_only
def evaluate_logistic_model(
    performance_results_df,
    df_logistic,
    df_test_proba,
    df_train_proba,
    bootstrap,
    fold_n
):

    results = []

    print('length of df is:')
    print(len(performance_results_df))

    for _, row in performance_results_df.iterrows():

        # ====================================================
        # RUN INFORMATION
        # ====================================================

        run_name = row['run_name']
        combination = row['combination']
        tile_count_threshold = row['tile_count_threshold']
        classification_probability_threshold = row['classification_probability_threshold']

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
            y_pred_probability >= classification_probability_threshold
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

#Duplicate with ECM. Called by evaluate_logistic_model, generate_inner_probabilities_for_optimum_psa_inner_model, generate_final_model_from_inner_psa_predicted_probabilities, 
def create_logistic_model(df_row):
    """Create a logistic regression model from run parameters."""

    return LogisticRegression(
        penalty=df_row['penalty'].iloc[0],
        C=df_row['C'].iloc[0],
        solver=df_row['solver'].iloc[0],
        l1_ratio=df_row['l1_ratio'].iloc[0],
        max_iter=1000,
        random_state=8,
    )



#called by generate_inner_outer_scores_for_graded_f1_score_psa_only   was called max_f1score_graded_single_threshold
def select_best_probability_threshold_single_tile_threshold(performance_results_df, f1_score_thresholds,threshold_in):
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

        #select simply the tile input threshold rather than cycling through
        df_threshold = df_outer_fold[df_outer_fold['tile_count_threshold'] == threshold_in].copy()
        df_threshold = df_threshold[df_threshold['mean_f1_score_recur'] >= df_threshold['mean_f1_score_nonrecur']].copy()

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
    numeric_cols = f1_score_df.select_dtypes(include='object').columns
    f1_score_df[numeric_cols] = f1_score_df[numeric_cols].apply(pd.to_numeric, errors='ignore')

    if 'distance_to_median' in f1_score_df.columns:
        f1_score_df = f1_score_df.drop(
            columns=['distance_to_median']
        )




    return f1_score_df



# functions can be added to ecm


#called by performance_results_df_generator
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


#appears in get_patient_splits, generate_logistic_script
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

#psa specific
# appears in get_patient_splits
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


#appears in generate_logistic_script
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


# called by generate_inner_outer_scores_for_graded_f1_score_psa_only, generate_inner_probabilities_for_optimum_psa_inner_model, generate_final_model_from_inner_psa_predicted_probabilities    
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






    

#END OF HELPER FUNCTIONS




def write_logistic_analysis_script(
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
    """
    Write a patient-level logistic regression analysis script.

    Creates a standalone script that loads PSA/T-stage clinical data,
    performs patient-level logistic regression, evaluates the model
    across tile-count and classification probability thresholds,
    and saves the results.
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
        main_path
        + file_input_name
        + f'{run_number:05d}.py'
    )

    # --------------------------------------------------------
    # Write script
    # --------------------------------------------------------

    with open(script_filename, 'w') as f:

        f.write(f"""
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
    'xgboost_psa_tstage_only_outer_fold_{{bootstrap}}/'
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
    'xgboost_psa_tstage_only_outer_fold_{{bootstrap}}/'
    'fold_{{outer_fold:05d}}/run_{{run_number:05d}}'
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
    0,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
    35
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

hyperparameters_df = hyperparameters_df.rename(
    columns={{
        0: 'name',
        1: 'fold',
        2: 'n_estimator',
        3: 'max_depth',
        4: 'learning_rate',
        5: 'subsample',
        6: 'colsample_bytree',
        7: 'gamma',
        8: 'min_child_weight'
    }}
)

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

patient_model_features = [
    'psa',
    't2',
    't3'
]


# ============================================================
# LOAD AND FILTER TILE DATA
# ============================================================

df_tiles = pd.read_csv(
    data_dir,
    low_memory=False
)

df_tiles = df_tiles[
    (df_tiles['tissue_proportion'] >= tissue_proportion)
    & (df_tiles['tumour_proportion'] >= tumour_proportion)
    & (df_tiles['gleason'].isin([1, 2]))
].reset_index(drop=True)

print(df_tiles.head())


# ============================================================
# CLINICAL SCORE
# ============================================================

conditions = [
    (
        (df_tiles['gleason'] == 1)
        & (df_tiles['case'] == 1.0)
    ),
    (
        (df_tiles['gleason'] == 2)
        & (df_tiles['case'] == 1.0)
    ),
    (
        (df_tiles['gleason'] == 1)
        & (df_tiles['case'] == 0.0)
    ),
    (
        (df_tiles['gleason'] == 2)
        & (df_tiles['case'] == 0.0)
    ),
]

df_tiles['clinical_score'] = np.select(
    conditions,
    range(4)
)


# ============================================================
# OUTER-FOLD SPLIT
# ============================================================

df_patients = ptmf.create_patient_dataframe(
    df_tiles
)

folds_csv_file = (
    code_dir
    + 'folds_random_'
    + str(fold_assignment_bootstrap).zfill(3)
    + '.csv'
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

df_patients = ptmf.create_patient_dataframe(
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

folds_csv_file = main_path + 'inner_folds_{outer_fold}_data.csv'

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

    df_patients = ptmf.create_patient_dataframe(
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

    X_train_tiles = (
        df_train_tiles
        .drop(columns=tile_columns_to_drop)
        .copy()
    )

    y_train_tiles = df_train_tiles['case']

    X_test_tiles = (
        df_test_tiles
        .drop(columns=tile_columns_to_drop)
        .copy()
    )

    y_test_tiles = df_test_tiles['case']


    # --------------------------------------------------------
    # Patient-level clinical data
    # --------------------------------------------------------

    excluded_patient_ids = []

    df_train_patients = ptmf.get_patient_splits(
        df_train_tiles,
        df_clinical,
        patient_ids_to_exclude=excluded_patient_ids
    )

    df_test_patients = ptmf.get_patient_splits(
        df_test_tiles,
        df_clinical,
        patient_ids_to_exclude=excluded_patient_ids
    )


    # --------------------------------------------------------
    # Standardise PSA using training data only
    # --------------------------------------------------------

    psa_mean = df_train_patients['psa'].mean()
    psa_std = df_train_patients['psa'].std()

    df_train_patients['psa'] = (
        df_train_patients['psa'] - psa_mean
    ) / psa_std

    df_test_patients['psa'] = (
        df_test_patients['psa'] - psa_mean
    ) / psa_std


    # --------------------------------------------------------
    # Fit patient-level logistic regression
    # --------------------------------------------------------

    X_patient_train = (
        df_train_patients[
            patient_model_features
        ]
    )

    y_train = df_train_patients['case']

    logistic_model.fit(
        X_patient_train,
        y_train
    )


    # ========================================================
    # EVALUATE ACROSS TILE-COUNT THRESHOLDS
    # ========================================================

    for tile_count_threshold in tile_count_thresholds:

        test_patients_filtered = df_test_patients[
            df_test_patients['total_tiles']
            >= tile_count_threshold
        ].copy()

        X_patient_test = (
            test_patients_filtered[
                patient_model_features
            ]
        )

        y_test = test_patients_filtered['case']

        n_test_patients = len(
            X_patient_test
        )

        if n_test_patients < 5:
            continue


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred_prob = (
            logistic_model
            .predict_proba(X_patient_test)[:, 1]
        )

        y_pred = logistic_model.predict(
            X_patient_test
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
            'combination': [
                patient_model_features
            ] * len(
                classification_probability_thresholds
            ),
            'fold': [
                inner_fold
            ] * len(
                classification_probability_thresholds
            ),
            'tile_count_threshold': [
                tile_count_threshold
            ] * len(
                classification_probability_thresholds
            ),
            'test_size': [
                n_test_patients
            ] * len(
                classification_probability_thresholds
            ),
            'classification_probability_threshold':
                classification_probability_thresholds,
            'f1_recur':
                f1_recurrence,
            'f1_nonrecur':
                f1_nonrecurrence
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
            'combination':
                patient_model_features,
            'fold':
                inner_fold,
            'roc_auc':
                roc_auc,
            'f1_score_recur':
                f1_recurrence_score,
            'f1_score_nonrecur':
                f1_nonrecurrence_score,
            'f1_score_average':
                np.mean([
                    f1_recurrence_score,
                    f1_nonrecurrence_score
                ]),
            'f1_score_max_derivative':
                max_f1_derivative,
            'accuracy':
                accuracy_score(
                    y_test,
                    y_pred
                ),
            'tile_count_threshold':
                tile_count_threshold,
            'test_size':
                n_test_patients
        }})


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        prediction_results.append(
            pd.DataFrame({{
                'y_true':
                    y_test.values,
                'y_pred_prob':
                    y_pred_prob,
                'combination':
                    [
                        patient_model_features
                    ] * len(y_test),
                'fold':
                    [
                        inner_fold
                    ] * len(y_test),
                'tile_count_threshold':
                    [
                        tile_count_threshold
                    ] * len(y_test)
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
    'metrics_patient_level_psa':
        metrics_df,

    'f1_probability_threshold_patient_level_psa':
        f1_thresholds_df,

    'probability_predictions_patient_level_psa':
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




#duplicate with ecm
def generate_f1_score_csv(
    code_dir,
    run_name,
    base_dir,
    data_type):
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


    # ========================================================
    # PATHS
    # ========================================================


    output_dir = os.path.join(
        base_dir,
        run_name
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
        'outer_fold_*_run_*_inner_cv_'
        f'f1_probability_threshold_patient_level_{data_type}.csv'
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
            output_dir,
            data_type
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        output_dir,
        'mean_inner_f1_score_plus_roc_auc_'
        f'all_probability_thresholds_{data_type}.csv'
    )

    performance_results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f'Saved performance results to:\n{output_file}'
    )






def run_psa_patient_level_nested_cv(
    run_name_prefix,
    bootstrap,
    base_code_dir,
    base_data_dir,
    fold_n,
    subfile_name,
    f1_score_thresholds
):
    """Run outer-fold evaluation for the PSA patient-level model.

    Outer fold:
        - Uses the predefined patient-level split shared with the ECM
          pipeline.
        - Is used for final outer-test evaluation.
        - Does not participate in selection of the PSA probability
          threshold, which is selected from inner-CV performance.

    Inner folds:
        - Provide the performance results used to select the PSA
          classification probability threshold.
        - Are evaluated at two tile-count thresholds: zero and the
          threshold selected by the ECM pipeline for this bootstrap
          and outer fold.

    Final outer evaluation:
        - PSA values are obtained for the outer-training and outer-test
          patients.
        - PSA is standardised using the outer-training patients only.
        - The patient-level logistic-regression model is evaluated using
          the selected probability-threshold configuration.
        - Results are saved separately for the zero and ECM-derived
          tile-count thresholds.

    Unlike the ECM pipeline, this function does not fit an XGBoost
    tile classifier or derive patient-level features from tile
    probabilities. It uses PSA as the patient-level predictor and
    imports the ECM-selected tile-count threshold for comparison.
    """

    # ========================================================
    # RUN NAME
    # ========================================================

    run_name = f'{run_name_prefix}{bootstrap}'

    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    # Contains the logistic-regression hyperparameters corresponding
    # to the patient-level runs.
    logistic_run_key = os.path.join(
        base_code_dir,
        run_name,
        f'{run_name}_fold_runkey.txt'
    )

    df_logistic = pd.read_table(
        logistic_run_key,
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

    patient_output_dir = os.path.join(
        base_data_dir,
        'rfe_runs',
        run_name
    )

    code_dir = os.path.join(
        base_code_dir,
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost'
    )

    data_dir = os.path.join(
        base_data_dir,
        'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
    )

    folds_csv_file = os.path.join(
        code_dir,
        f'folds_random_{bootstrap:03d}.csv'
    )

    psa_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/raw_data/clinical/'
        'prostate/chiip_cohort/TPSAdata/'
    )

    # Match the ECM-threshold file used by the CSV-generation pipeline.
    ecm_threshold_file = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/'
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        'psa_tstage_only_outer_fold_20/selected_thresholds_from_ecm.csv'
    )

    # ========================================================
    # LOAD ECM-SELECTED TILE-COUNT THRESHOLD
    # ========================================================

    # The ECM-selected threshold is used for one of the PSA analyses.
    # The other analysis uses a tile-count threshold of zero.
    df_ecm_threshold = pd.read_csv(
        ecm_threshold_file
    )

    df_ecm_threshold = df_ecm_threshold[
        (df_ecm_threshold['bootstrap'] == bootstrap) &
        (df_ecm_threshold['outer_fold'] == fold_n)
    ].copy()

    tile_count_threshold = (
        df_ecm_threshold['tile_count_threshold'].iloc[0]
    )

    # ========================================================
    # LOAD PSA DATA
    # ========================================================

    df_psa = pd.read_csv(
        os.path.join(
            psa_dir,
            'baseline_fullchhipcohort_PSATstage.csv'
        ),
        low_memory=False
    )

    # Construct patient IDs to match the tile-level dataset.
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
    # LOAD AND FILTER TILE DATA
    # ========================================================

    # Tile data are used to construct the patient-level stratification
    # and reproduce the predefined outer-fold split.
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
    # CALCULATE PATIENT-LEVEL PSA VALUES
    # ========================================================

    # Obtain PSA features for the outer-test and outer-training patients.
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
    # STANDARDISE PSA USING OUTER-TRAINING PATIENTS
    # ========================================================

    psa_columns = ['psa']

    # IMPORTANT:
    # Calculate the scaling parameters using outer-training patients
    # only, then apply those same parameters to both datasets.
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
    # LOAD INNER-CV PERFORMANCE RESULTS
    # ========================================================

    # These results are used to select the probability threshold
    # for each of the two tile-count-threshold analyses.
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
    # ANALYSE BOTH TILE-COUNT THRESHOLDS
    # ========================================================

    # Compare PSA model performance at:
    #   1. A tile-count threshold of zero.
    #   2. The threshold selected by the ECM pipeline.
    threshold_analyses = [
        (
            0,
            'psa_zero_threshold'
        ),
        (
            tile_count_threshold,
            'psa_ecm_threshold'
        )
    ]

    for tile_count_threshold, output_suffix in threshold_analyses:

        # ====================================================
        # SELECT PROBABILITY THRESHOLD USING INNER-CV RESULTS
        # ====================================================

        f1_score_df = select_best_probability_threshold_single_tile_threshold(
            performance_results_df,
            f1_score_thresholds,
            tile_count_threshold
        )

        # Retain the results corresponding to the current outer fold.
        f1_score_df = f1_score_df[
            f1_score_df['outer_fold'] == fold_n
        ].copy()

        # ====================================================
        # FINAL OUTER-FOLD LOGISTIC REGRESSION EVALUATION
        # ====================================================

        # Apply the selected configuration to the outer fold.
        # evaluate_logistic_model handles fitting on the training
        # data and evaluation using the held-out test data.
        outer_logistic_f1_score_df = (
            evaluate_logistic_model(
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

            print("Successfully saved CSV")

        except Exception as e:
            print(
                f"Error saving CSV: {e}"
            )

        # ====================================================
        # SAVE SELECTED INNER-CV PERFORMANCE RESULTS
        # ====================================================

        # This contains the selected probability-threshold results
        # for the current outer fold and tile-count threshold.
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




def generate_optimal_psa_inner_model_predictions(#was generate_inner_probabilities_for_optimum_psa_inner_model
    bootstrap,
    fold_n,
    subfile_name,
    f1_score_thresholds
):
"""Generate patient-level PSA/T-stage probabilities using the selected inner model.

```
Loads the optimal model configuration and classification probability
threshold selected from the inner-fold results for each of two analyses:
a zero tile-count threshold and an ECM-derived tile-count threshold.

Reconstructs the predefined outer-training cohort and its five inner
folds, calculates patient-level PSA and T-stage features, and standardises
PSA using the training patients' mean and standard deviation within each
inner fold. Fits the selected logistic regression model on the inner-training
patients and predicts recurrence probabilities for eligible inner-test
patients meeting the selected tile-count threshold.

Saves a CSV of patient-level predictions and associated model metadata for
each inner fold and threshold analysis. PSA values are converted back to
their original units before the predictions are saved.

Args:
    bootstrap: Bootstrap replicate identifier used to select the predefined
        outer-fold split and identify the model run.
    fold_n: Index of the held-out outer fold. Only the remaining outer-fold
        patients are used to construct the inner training and test folds.
    subfile_name: Identifier used to locate the selected inner-model
        performance files and name the prediction output files.
    f1_score_thresholds: Probability-threshold values associated with the
        graded F1-score analysis. This parameter is retained for
        consistency with the model-evaluation pipeline but is not directly
        used within this function.

Returns:
    None. Saves one patient-level prediction CSV per inner fold for each
    threshold analysis to the configured patient output directory.
```

"""

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

    main_path = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
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

    folds_csv_file = os.path.join(
        code_dir,
        f'folds_random_{bootstrap:03d}.csv'
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
    # OUTER TRAINING DATA
    # ========================================================

    df = df[
        predefined_splits_array != fold_n
    ].reset_index(
        drop=True
    )

    patient_ids = list(
        df.patient_id.unique()
    )

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    patient_model_features = [
        'psa',
        't2',
        't3'
    ]

    # ========================================================
    # INNER FOLDS
    # ========================================================

    inner_folds = np.arange(5)

    inner_folds_file = os.path.join(
        main_path,
        f'inner_folds_{fold_n}_data.csv'
    )

    # ========================================================
    # PSA / ECM THRESHOLD ANALYSES
    # ========================================================

    threshold_analyses = [
        'psa_zero_threshold',
        'psa_ecm_threshold'
    ]

    for output_suffix in threshold_analyses:

        # ====================================================
        # LOAD SELECTED INNER-FOLD MODEL
        # ====================================================

        input_file = (
            f'outer_fold_{fold_n:05d}_'
            'graded_f1_score_inner_fold_'
            f'{subfile_name}_performance_{output_suffix}.csv'
        )

        input_path = os.path.join(
            patient_output_dir,
            input_file
        )

        df_in = pd.read_csv(
            input_path
        )

        row = df_in.iloc[0]

        run = row['run']
        run_name = row['run_name']
        combination = row['combination']
        tile_count_threshold = row['tile_count_threshold']
        classification_probability_threshold = row['classification_probability_threshold']

        df_logistic_row = df_logistic[
            df_logistic['run_name'] == run_name
        ].copy()

        logistic_model = create_logistic_model(
            df_logistic_row
        )

        # ====================================================
        # INNER-FOLD MODELS
        # ====================================================

        for inner_fold in inner_folds:

            print(inner_fold)

            # ================================================
            # INNER-FOLD PATIENT STRATIFICATION
            # ================================================

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
            # PATIENT-LEVEL PSA / T-STAGE DATA
            # ================================================

            test_unique_patients = [
                patient_id
                for patient_id in df_test.patient_id.unique()
                if patient_id != "" # Removed for GitHub
            ]

            train_unique_patients = [
                patient_id
                for patient_id in df_train.patient_id.unique()
                if patient_id != ""
            ]

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

            # ================================================
            # STANDARDISE PSA USING TRAINING DATA ONLY
            # ================================================

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

            # ================================================
            # TEST PATIENTS ABOVE TILE-COUNT THRESHOLD
            # ================================================

            df_test_thresholded = df_test_proba[
                df_test_proba.tile_count >= tile_count_threshold
            ].copy()

            X_train = df_train_proba[
                patient_model_features
            ]

            y_train = df_train_proba[
                'case'
            ]

            X_test = df_test_thresholded[
                patient_model_features
            ]

            # ================================================
            # FIT MODEL AND PREDICT
            # ================================================

            logistic_model.fit(
                X_train,
                y_train
            )

            y_pred_probability = logistic_model.predict_proba(
                X_test
            )[:, 1]

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
            df_test_thresholded['tile_count_threshold'] = tile_count_threshold
            df_test_thresholded[
                'classification_probability_threshold'
            ] = classification_probability_threshold

            # Convert PSA back to original units
            df_test_thresholded[
                psa_columns
            ] = (
                df_test_thresholded[psa_columns] * psa_stds
            ) + psa_means

            # ================================================
            # SAVE PATIENT-LEVEL PREDICTIONS
            # ================================================

            selected_model_save_name = (
                f'outer_fold_{fold_n:05d}'
                f'inner_fold_{inner_fold:05d}_'
                f'{subfile_name}_'
                f'bootstrap_{bootstrap:03d}_'
                f'selected_model_from_inner_'
                f'psa_patient_probabilities_{output_suffix.replace("psa_", "")}.csv'
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
                    f"Error saving CSV: {e}"
                )








def collect_selected_thresholds(
    base_dir,
    folder_pattern,
    output_file,
):
    """Collect selected thresholds from inner-fold model-selection CSVs.

    Searches matching experiment folders directly under base_dir.
    Extracts outer_fold, inner_fold and bootstrap from each filename,
    validates the threshold column, and saves a summary CSV.
    """
    base_dir = Path(base_dir)
    output_file = Path(output_file)

    file_pattern = (
        "outer_fold_*_inner_fold_*_f1_score_increment_20_bootstrap_*"
        "_selected_model_from_inner_patient_probabilities.csv"
    )

    file_regex = re.compile(
        r"outer_fold_(\d+)_inner_fold_(\d+)_"
        r"f1_score_increment_20_bootstrap_(\d+)_"
        r"selected_model_from_inner_patient_probabilities\.csv$"
    )

    rows = []

    for folder in sorted(base_dir.glob(folder_pattern)):
        for file in sorted(folder.glob(file_pattern)):
            match = file_regex.fullmatch(file.name)

            if match is None:
                continue

            outer_fold, inner_fold, bootstrap = map(
                int, match.groups()
            )

            df = pd.read_csv(file)

            if "threshold" not in df.columns:
                raise ValueError(
                    f"'threshold' column not found in {file}"
                )

            thresholds = df["threshold"].dropna().unique()

            if len(thresholds) != 1:
                raise ValueError(
                    f"Expected exactly one non-null threshold in "
                    f"{file}, found {thresholds}"
                )

            rows.append({
                "bootstrap": bootstrap,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "threshold": thresholds[0],
                "file": str(file),
            })

    columns = [
        "bootstrap",
        "outer_fold",
        "inner_fold",
        "threshold",
        "file",
    ]

    summary_df = pd.DataFrame(rows, columns=columns)

    summary_df = summary_df.sort_values(
        ["bootstrap", "outer_fold", "inner_fold"]
    ).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    print(f"Found {len(summary_df)} matching files.")
    print(f"Saved summary to:\n{output_file}")

    return summary_df




def fit_and_interpret_final_psa_model(
    bootstrap,
    fold_n,
    subfile_name,
):
    """Fit and interpret final PSA/T-stage models for an outer CV fold.

    Selected configurations come from inner-fold performance CSVs.
    PSA is standardised using the outer-training cohort only, and the
    fitted model is evaluated on outer-test patients meeting the selected
    tile-count threshold.

    Saves model interpretation and patient predictions separately for
    the zero-threshold and ECM-threshold analyses.
    """

    # ========================================================
    # LOGISTIC REGRESSION RUN KEY
    # ========================================================

    run_name_prefix = (
        "gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_"
        f"psa_tstage_only_outer_fold_{bootstrap}"
    )

    logistic_run_key_path = (
        "/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/"
        "prostate_recurrence/model_evaluation/recurrence_status_tile/"
        "rfe_runs/"
        f"{run_name_prefix}/{run_name_prefix}_fold_runkey.txt"
    )

    df_logistic_run_key = (
        pd.read_table(
            logistic_run_key_path,
            delimiter=" ",
            header=None,
        )
        .rename(
            columns={
                0: "run_name",
                1: "fold",
                2: "penalty",
                3: "C",
                4: "solver",
                5: "l1_ratio",
            }
        )
        .replace({np.nan: None})
    )

    # ========================================================
    # PATHS
    # ========================================================

    patient_output_dir = (
        "/nemo/project/proj-sahai-tme-ml/working/processed_data/"
        "feature_engineering/clinical/prostate/chiip_cohort/slide_20X/"
        "feature_analysis_v3/rfe_runs/"
        f"{run_name_prefix}/"
    )

    input_dir = (
        "/nemo/project/proj-sahai-tme-ml/working/processed_data/"
        "feature_engineering/clinical/prostate/chiip_cohort/slide_20X/"
        "feature_analysis_v3/"
    )

    code_dir = (
        "/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/"
        "prostate_recurrence/model_evaluation/recurrence_status_tile/"
        "rfe_runs/"
        "gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/"
    )

    data_dir = os.path.join(
        input_dir,
        "updated_tumour_boundary_quadrants_double_are_removed_df.csv",
    )

    psa_dir = (
        "/nemo/project/proj-sahai-tme-ml/working/raw_data/clinical/"
        "prostate/chiip_cohort/TPSAdata/"
    )

    ecm_threshold_file = (
        "/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/"
        "prostate_recurrence/model_evaluation/recurrence_status_tile/"
        "rfe_runs/"
        "gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_"
        "psa_tstage_only_outer_fold_20/selected_thresholds_from_ecm.csv"
    )

    folds_csv_file = os.path.join(
        code_dir,
        f"folds_random_{bootstrap:03d}.csv",
    )

    # ========================================================
    # ECM THRESHOLD SUMMARY
    # ========================================================

    df_selected_ecm_threshold = pd.read_csv(ecm_threshold_file)

    df_selected_ecm_threshold = df_selected_ecm_threshold[
        (df_selected_ecm_threshold["bootstrap"] == bootstrap)
        & (df_selected_ecm_threshold["outer_fold"] == fold_n)
    ].copy()

    if len(df_selected_ecm_threshold) != 1:
        raise ValueError(
            "Expected exactly one ECM threshold row for "
            f"bootstrap={bootstrap}, outer_fold={fold_n}; "
            f"found {len(df_selected_ecm_threshold)}."
        )

    selected_ecm_tile_count_threshold = (
        df_selected_ecm_threshold["tile_count_threshold"].iloc[0]
    )

    # ========================================================
    # PSA DATA
    # ========================================================

    df_psa = pd.read_csv(
        os.path.join(
            psa_dir,
            "baseline_fullchhipcohort_PSATstage.csv",
        ),
        low_memory=False,
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
            "prehcat1",
        ],
        axis=1,
    )

    # ========================================================
    # TILE DATA
    # ========================================================

    df = pd.read_csv(
        data_dir,
        low_memory=False,
    )

    df = df[
        (df["tissue_proportion"] >= 0.7)
        & (df["tumour_proportion"] >= 0.7)
        & (df["gleason"].isin([1, 2]))
    ].reset_index(drop=True)

    # ========================================================
    # PATIENT-LEVEL STRATIFICATION
    # ========================================================

    patient_ids = list(df.patient_id.unique())

    conditions = [
        (df["gleason"] == 1) & (df["case"] == 1.0),
        (df["gleason"] == 2) & (df["case"] == 1.0),
        (df["gleason"] == 1) & (df["case"] == 0.0),
        (df["gleason"] == 2) & (df["case"] == 0.0),
    ]

    df["clinical_score"] = np.select(
        conditions,
        range(4),
    )

    gleasons = []
    cases = []
    total_tiles = []
    clinical_scores = []

    for patient_id in patient_ids:
        patient_df = df[df.patient_id == patient_id]

        gleasons.append(patient_df.gleason.unique()[0])
        cases.append(patient_df.case.unique()[0])
        clinical_scores.append(patient_df.clinical_score.unique()[0])
        total_tiles.append(len(patient_df))

    df_patient = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "gleason": gleasons,
            "case": cases,
            "clinical_score": clinical_scores,
            "total_tiles": total_tiles,
        }
    )

    df_strat = (
        df_patient.sort_values(
            by=["clinical_score", "total_tiles"],
            ascending=[True, False],
        )
        .copy()
        .reset_index()
    )

    _, _, predefined_splits_array = tmf.patient_tile_stratifier(
        df_strat,
        df,
        5,
        folds_csv_file,
    )

    # ========================================================
    # OUTER TRAIN / TEST DATA
    # ========================================================

    df_train = df[
        predefined_splits_array != fold_n
    ].reset_index(drop=True)

    df_test = df[
        predefined_splits_array == fold_n
    ].reset_index(drop=True)

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
    # PATIENT-LEVEL PSA / T-STAGE DATA
    # ========================================================

    df_outer_test_patients = calculate_patient_psa(
        df_test,
        test_unique_patients,
        df_psa,
    )

    df_outer_train_patients = calculate_patient_psa(
        df_train,
        train_unique_patients,
        df_psa,
    )

    # ========================================================
    # STANDARDISE PSA USING OUTER-TRAINING DATA ONLY
    # ========================================================

    psa_columns = ["psa"]

    psa_means = df_outer_train_patients[psa_columns].mean()
    psa_stds = df_outer_train_patients[psa_columns].std()

    if psa_stds.isna().any() or (psa_stds == 0).any():
        raise ValueError(
            "PSA standard deviation is missing or zero "
            "in outer training data."
        )

    df_outer_train_patients[psa_columns] = (
        df_outer_train_patients[psa_columns] - psa_means
    ) / psa_stds

    df_outer_test_patients[psa_columns] = (
        df_outer_test_patients[psa_columns] - psa_means
    ) / psa_stds

    patient_model_features = ["psa", "t2", "t3"]

    threshold_analyses = [
        "psa_zero_threshold",
        "psa_ecm_threshold",
    ]

    # ========================================================
    # FIT / INTERPRET EACH SELECTED THRESHOLD ANALYSIS
    # ========================================================

    for threshold_analysis in threshold_analyses:

        # ====================================================
        # LOAD SELECTED INNER-CV CONFIGURATION
        # ====================================================

        input_filename = (
            f"outer_fold_{fold_n:05d}"
            "_graded_f1_score_inner_fold_"
            f"{subfile_name}_performance_{threshold_analysis}.csv"
        )

        input_path = os.path.join(
            patient_output_dir,
            input_filename,
        )

        print("Loading selected inner-CV results:", input_path)

        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"Selected inner-CV results not found: {input_path}"
            )

        df_inner_cv_results = pd.read_csv(input_path)

        if df_inner_cv_results.empty:
            raise ValueError(
                f"Selected inner-CV results file is empty: {input_path}"
            )

        selected_model_row = df_inner_cv_results.iloc[0]

        run = selected_model_row["run"]
        run_name = selected_model_row["run_name"]
        combination = selected_model_row["combination"]

        tile_count_threshold = selected_model_row["tile_count_threshold"]

        classification_probability_threshold = (
            selected_model_row["classification_probability_threshold"]
        )

        # Check the selected ECM threshold against the summary file.
        if threshold_analysis == "psa_ecm_threshold":
            if not np.isclose(
                float(tile_count_threshold),
                float(selected_ecm_tile_count_threshold),
                equal_nan=True,
            ):
                print(
                    "Warning: ECM threshold summary has "
                    f"tile_count_threshold={selected_ecm_tile_count_threshold}, "
                    "but the selected inner-CV row has "
                    f"tile_count_threshold={tile_count_threshold}. "
                    "Using the inner-CV selected value."
                )

        # ====================================================
        # LOAD LOGISTIC REGRESSION CONFIGURATION
        # ====================================================

        df_logistic_row = df_logistic_run_key[
            df_logistic_run_key["run_name"] == run_name
        ].copy()

        if df_logistic_row.empty:
            raise ValueError(
                f"No logistic run-key row found for run_name={run_name!r} "
                f"in {logistic_run_key_path}"
            )

        if len(df_logistic_row) > 1:
            raise ValueError(
                f"Expected one logistic run-key row for "
                f"run_name={run_name!r}; found {len(df_logistic_row)}."
            )

        final_model = create_logistic_model(df_logistic_row)

        # ====================================================
        # SELECT ELIGIBLE OUTER-TEST PATIENTS
        # ====================================================

        df_eligible_outer_test_patients = df_outer_test_patients[
            df_outer_test_patients.tile_count >= tile_count_threshold
        ].copy()

        X_outer_train = df_outer_train_patients[
            patient_model_features
        ].copy()

        y_outer_train = df_outer_train_patients["case"].copy()

        X_outer_test = df_eligible_outer_test_patients[
            patient_model_features
        ].copy()

        y_outer_test = df_eligible_outer_test_patients["case"].copy()

        if X_outer_train.empty:
            raise ValueError(
                "Outer-training patient feature matrix is empty."
            )

        if X_outer_test.empty:
            raise ValueError(
                "No outer-test patients meet "
                f"tile_count_threshold={tile_count_threshold} "
                f"for {threshold_analysis}."
            )

        # ====================================================
        # FIT MODEL AND PREDICT OUTER-TEST PATIENTS
        # ====================================================

        final_model.fit(
            X_outer_train,
            y_outer_train,
        )

        outer_test_recurrence_probabilities = (
            final_model.predict_proba(X_outer_test)[:, 1]
        )

        # ====================================================
        # MODEL COEFFICIENTS / ODDS RATIOS
        # ====================================================

        coefficients = final_model.coef_[0]

        coef_dict = dict(
            zip(X_outer_train.columns, coefficients)
        )

        # PSA coefficient is per one training-set SD of PSA.
        coef_psa = coef_dict["psa"]
        coef_t2 = coef_dict["t2"]
        coef_t3 = coef_dict["t3"]

        intercept = final_model.intercept_[0]

        or_psa_per_sd = np.exp(coef_psa)
        or_t2 = np.exp(coef_t2)
        or_t3 = np.exp(coef_t3)

        psa_sd = psa_stds["psa"]

        coef_psa_per_unit = coef_psa / psa_sd
        or_psa_per_unit = np.exp(coef_psa_per_unit)

        # ====================================================
        # PATIENT-LEVEL LOG-ODDS CONTRIBUTIONS
        # ====================================================

        # Contributions use standardised PSA, matching model fitting.
        df_eligible_outer_test_patients["psa_contribution"] = (
            df_eligible_outer_test_patients["psa"] * coef_psa
        )

        df_eligible_outer_test_patients["t2_contribution"] = (
            df_eligible_outer_test_patients["t2"] * coef_t2
        )

        df_eligible_outer_test_patients["t3_contribution"] = (
            df_eligible_outer_test_patients["t3"] * coef_t3
        )

        df_eligible_outer_test_patients["clinical_contribution"] = (
            df_eligible_outer_test_patients["psa_contribution"]
            + df_eligible_outer_test_patients["t2_contribution"]
            + df_eligible_outer_test_patients["t3_contribution"]
        )

        # ====================================================
        # MODEL INTERPRETATION SUMMARY
        # ====================================================

        df_model_interpretation = pd.DataFrame(
            {
                "bootstrap": [bootstrap],
                "fold": [fold_n],
                "threshold_analysis": [threshold_analysis],
                "run": [run],
                "run_name": [run_name],
                "combination": [combination],
                "tile_count_threshold": [tile_count_threshold],
                "classification_probability_threshold": [
                    classification_probability_threshold
                ],
                "n_train_patients": [len(df_outer_train_patients)],
                "n_test_patients": [
                    len(df_eligible_outer_test_patients)
                ],
                "psa_sd": [psa_sd],
                "intercept": [intercept],
                "coef_psa": [coef_psa],
                "coef_psa_per_unit": [coef_psa_per_unit],
                "coef_t2": [coef_t2],
                "coef_t3": [coef_t3],
                "or_psa_per_sd": [or_psa_per_sd],
                "or_psa_per_unit": [or_psa_per_unit],
                "or_t2": [or_t2],
                "or_t3": [or_t3],
                "penalty": [final_model.penalty],
                "C": [final_model.C],
                "solver": [final_model.solver],
                "max_iter": [final_model.max_iter],
                "l1_ratio": [
                    getattr(final_model, "l1_ratio", None)
                ],
            }
        )

        # ====================================================
        # PRINT MODEL INTERPRETATION
        # ====================================================

        print("\n============================================")
        print(f"MODEL INTERPRETATION - {threshold_analysis.upper()}")
        print("============================================")

        print(f"PSA coefficient (per SD):  {coef_psa:+.4f}")
        print(f"PSA OR (per SD):           {or_psa_per_sd:.4f}")
        print(f"PSA coefficient (per unit): {coef_psa_per_unit:+.4f}")
        print(f"PSA OR (per unit):          {or_psa_per_unit:.4f}")
        print(f"T2 coefficient:             {coef_t2:+.4f}")
        print(f"T2 OR:                      {or_t2:.4f}")
        print(f"T3 coefficient:             {coef_t3:+.4f}")
        print(f"T3 OR:                      {or_t3:.4f}")

        # ====================================================
        # SAVE MODEL INTERPRETATION CSV
        # ====================================================

        model_interpretation_filename = (
            f"outer_fold_{fold_n:05d}_"
            f"{subfile_name}_"
            f"bootstrap_{bootstrap:03d}_"
            f"model_interpretation_{threshold_analysis}.csv"
        )

        model_interpretation_path = os.path.join(
            patient_output_dir,
            model_interpretation_filename,
        )

        print(
            "Saving model interpretation:",
            model_interpretation_path,
        )

        df_model_interpretation.to_csv(
            model_interpretation_path,
            index=False,
        )

        # ====================================================
        # SAVE OUTER-TEST PATIENT PREDICTIONS
        # ====================================================

        df_eligible_outer_test_patients["recurrence_probability"] = (
            outer_test_recurrence_probabilities
        )

        df_eligible_outer_test_patients["fold"] = fold_n
        df_eligible_outer_test_patients["bootstrap"] = bootstrap
        df_eligible_outer_test_patients["run"] = run
        df_eligible_outer_test_patients["run_name"] = run_name
        df_eligible_outer_test_patients["combination"] = combination

        df_eligible_outer_test_patients["tile_count_threshold"] = (
            tile_count_threshold
        )

        df_eligible_outer_test_patients[
            "classification_probability_threshold"
        ] = classification_probability_threshold

        df_eligible_outer_test_patients["threshold_analysis"] = (
            threshold_analysis
        )

        # Restore PSA to original units for saved predictions.
        # Contributions above remain on the standardised scale.
        df_eligible_outer_test_patients[psa_columns] = (
            df_eligible_outer_test_patients[psa_columns] * psa_stds
        ) + psa_means

        outer_test_predictions_filename = (
            f"outer_fold_{fold_n:05d}_"
            f"{subfile_name}_"
            f"bootstrap_{bootstrap:03d}_"
            f"outer_test_predictions_{threshold_analysis}.csv"
        )

        outer_test_predictions_path = os.path.join(
            patient_output_dir,
            outer_test_predictions_filename,
        )

        print(
            "Saving outer-test predictions:",
            outer_test_predictions_path,
        )

        df_eligible_outer_test_patients.to_csv(
            outer_test_predictions_path,
            index=False,
        )
