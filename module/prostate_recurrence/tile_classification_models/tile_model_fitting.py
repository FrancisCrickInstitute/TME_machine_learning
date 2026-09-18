import os
import sys
import re
import glob
import pickle

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

from sklearn.feature_selection import RFECV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score


def rfecv(output_dir,df_train,df_strat,folds_csv_file,X_train,y_train,model,model_label,tumour_proportion=0,save_prefix='',steps=50,random_state=0,scorer='accuracy',no_folds = 5,n_splits=5):

    subfile_name = save_prefix + 'tile_'+ model_label
    df_strat, predefined_splits,predefined_splits_array = patient_tile_stratifier(df_strat,df_train,no_folds,folds_csv_file)
    subfile_name = subfile_name + '_rfe'
  
    search = RFECV(estimator=model, step=steps, cv=predefined_splits, scoring=scorer,min_features_to_select=10)
    
    search.fit(X_train, y_train)
    filename = output_dir + subfile_name +'.sav'
    with open(filename, 'wb') as handle:
        pickle.dump(search, handle)
 
    # Get mean cross-validation scores
    mean_test_scores = search.cv_results_['mean_test_score']
    std_test_scores = search.cv_results_['std_test_score']

    # Create a DataFrame with the information
    data = {'Mean_Test_Scores': mean_test_scores,'Std_Test_Scores': std_test_scores}
    df_rfecv_info = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    filename = output_dir + subfile_name +'_mean_test_scores.csv'
    df_rfecv_info.to_csv(filename, index=False)    

    for i, (train_index, test_index) in enumerate(predefined_splits.split()):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

def patient_tile_stratifier(df_strat,df,n_splits,folds_csv_file):
    folds_df = pd.read_csv(folds_csv_file,low_memory=False)
    folds = list(folds_df.folds)[:len(df_strat)]
  
    # Assign the array to a new column in the DataFrame
    df_strat['fold'] = folds
    df_strat = df_strat.sort_values(by='index')
    df_strat.reset_index(inplace=True,drop=True)
    df_strat=df_strat.drop('index',axis=1)
    df_tile_strat = df.copy()
    patient_ids=df.patient_id.unique()
    predefined_splits_array = np.zeros(len(df_tile_strat))
    for patient_id in patient_ids:
        predefined_splits_array[list(df_tile_strat[df_tile_strat.patient_id==patient_id].index)] = df_strat[df_strat.patient_id==patient_id].fold

    predefined_splits = PredefinedSplit(predefined_splits_array)
    return df_strat, predefined_splits,predefined_splits_array





def select_best_xgboost_model(csv_paths, valid_runs=None):
    test_score = []
    candidate_csv_paths = []

    for csv_path in csv_paths:

        if valid_runs is not None:
            match = re.search(r'_run_(\d+)_', csv_path)

            if not match:
                continue

            run = int(match.group())

            if run not in valid_runs:
                continue

        df_score = pd.read_csv(csv_path)

        score = df_score['Mean_Test_Scores'].max()

        if not test_score or score >= np.max(test_score):
            test_score.append(score)
            candidate_csv_paths.append(csv_path)

    sav_file = candidate_csv_paths[-1].replace(
        'mean_test_scores.csv',
        'mean_test_scores.sav'
    )

    match = re.search(r'_run_(\d+)_', sav_file)
    run = int(match.group())

    with open(sav_file, 'rb') as f:
        fold_model = pickle.load(f)

    return run, fold_model


def select_rfe_features(X_train, X_test, fold_model, max_rank):
    include_columns = X_train.columns[
        fold_model.ranking_ <= max_rank
    ]

    X_train_subset_df = X_train[include_columns]
    X_test_subset_df = X_test[include_columns]

    return X_train_subset_df, X_test_subset_df


def fit_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)

    test_pred_proba = model.predict_proba(X_test)[:, 1]

    return test_pred_proba


def calculate_and_save_shap(model, X_test, shap_save_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test.values,
        feature_names=X_test.columns.tolist()
    )

    with open(shap_save_path, 'wb') as f:
        pickle.dump(shap_explanation, f)



def generate_tumour_tile_probabilities(fold_n):

    # ============================================================
    # Paths and setup
    # ============================================================

    feature_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/'
    output_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/tumour_proportion_2026/xgboost_v2/'
    code_dir = '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/tumour_proportion_2026/xgboost_v2/'

    outer_fold_output_dir = output_dir + 'outer_fold_output/'
    os.makedirs(outer_fold_output_dir, exist_ok=True)

    max_rank = 1


    # ============================================================
    # Load hyperparameters and select model
    # ============================================================

    hyperparameters_df = pd.read_csv(
        os.path.join(
            code_dir,
            'xg_quad_1000_normal_vs_tumour_nested_5_folds_v2_runkey.txt'
        ),
        sep=' ',
        header=None
    )

    hyperparameters_df.drop(columns=[10], inplace=True)

    hyperparameters_df = hyperparameters_df.rename(
        columns={
            0: 'name',
            1: 'fold',
            2: 'n_estimator',
            3: 'max_depth',
            4: 'learning_rate',
            5: 'subsample',
            6: 'colsample_bytree',
            7: 'gamma',
            8: 'scale_pos_weight',
            9: 'min_child_weight'
        }
    )

    hyperparameters_df = hyperparameters_df[
        hyperparameters_df.fold == 0
    ]

    hyperparameters_df = hyperparameters_df.loc[0:3888]

    valid_runs = set(hyperparameters_df.index)
    #CHECK THIS FOR OUTPUT FROM OTHER SCRIPT HERE##
    csv_pattern = (
        output_dir
        + 'xgboost_normal_vs_tumour_quadrant_1000_total_outer_folds_5_'
        + 'outer_fold_test_'
        + str(fold_n).zfill(5)
        + '_run_*_roc_auc_tile_xgboosttumour_gr_0_rfe_mean_test_scores.csv'
    )

    score_csv_paths = glob(csv_pattern)
    score_csv_paths = sorted(score_csv_paths)

    run_id, rfe_model = select_best_xgboost_model(
        score_csv_paths,
        valid_runs
    )


    # ============================================================
    # Build XGBoost model
    # ============================================================

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 8,
        'n_estimators': hyperparameters_df.loc[run_id].n_estimator,
        'max_depth': hyperparameters_df.loc[run_id].max_depth,
        'learning_rate': hyperparameters_df.loc[run_id].learning_rate,
        'subsample': hyperparameters_df.loc[run_id].subsample,
        'colsample_bytree': hyperparameters_df.loc[run_id].colsample_bytree,
        'gamma': hyperparameters_df.loc[run_id].gamma,
        'scale_pos_weight': hyperparameters_df.loc[run_id].scale_pos_weight,
        'min_child_weight': hyperparameters_df.loc[run_id].min_child_weight
    }

    for param, value in xgb_params.items():

        print(param)
        print(value)

        if isinstance(value, str):
            try:
                xgb_params[param] = float(value)
            except ValueError:
                pass

    xgb_model = XGBClassifier(**xgb_params)


    # ============================================================
    # Load and prepare data
    # ============================================================

    tissue_proportion = 0.7
    lower_tumour_proportion = 0.1
    upper_tumour_proportion = 0.9

    data_path = (
        feature_dir
        + 'updated_tumour_boundary_quadrants_double_are_recurrent_df.csv'
    )

    df = pd.read_csv(
        data_path,
        low_memory=False
    )

    df = df[df['tissue_proportion'] >= tissue_proportion]

    df = df[
        (df['tumour_proportion'] >= upper_tumour_proportion)
        | (df['tumour_proportion'] <= lower_tumour_proportion)
    ]

    df = df[
        (df['gleason'] == 1)
        | (df['gleason'] == 2)
    ]

    df = df.reset_index(drop=True)

    print(df.head())

    patient_ids = list(df.patient_id.unique())

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

    df.loc[
        df['tumour_proportion'] <= 0.1,
        'tumour_proportion'
    ] = 0

    df.loc[
        df['tumour_proportion'] >= 0.9,
        'tumour_proportion'
    ] = 1

    gleasons = []
    cases = []
    total_tiles = []
    clinical_scores = []

    for patient_id in patient_ids:

        gleasons.append(
            df[df.patient_id == patient_id].gleason.unique()[0]
        )

        cases.append(
            df[df.patient_id == patient_id].case.unique()[0]
        )

        clinical_scores.append(
            df[
                df.patient_id == patient_id
            ].clinical_score.unique()[0]
        )

        total_tiles.append(
            len(df[df.patient_id == patient_id].case)
        )

    patient_df = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'clinical_score': clinical_scores,
        'total_tiles': total_tiles
    })

    stratification_df = (
        patient_df
        .sort_values(
            by=['clinical_score', 'total_tiles'],
            ascending=[True, False]
        )
        .copy()
        .reset_index()
    )


    # ============================================================
    # Patient-level stratification
    # ============================================================

    folds_csv_file = code_dir + 'outer_folds_data.csv'

    (
        stratification_df,
        _,
        fold_assignments
    ) = patient_tile_stratifier(
        stratification_df,
        df,
        5,
        folds_csv_file
    )


    # ============================================================
    # Train/test split
    # ============================================================

    df_train = df[
        fold_assignments != fold_n
    ]

    df_train = df_train.reset_index(drop=True)

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
            'gleason',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion'
        ],
        axis=1
    ).copy()

    y_train = df_train["tumour_proportion"]

    df_test = df[
        fold_assignments == fold_n
    ]

    df_test = df_test.reset_index(drop=True)

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
            'gleason',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion'
        ],
        axis=1
    ).copy()

    y_test = df_test["tumour_proportion"]

    X_train_rfe, X_test_rfe = select_rfe_features(
        X_train,
        X_test,
        rfe_model,
        max_rank
    )


    # ============================================================
    # Fit model and predict
    # ============================================================

    test_probabilities = fit_and_predict(
        xgb_model,
        X_train_rfe,
        y_train,
        X_test_rfe
    )


    # ============================================================
    # Save tile-level predictions
    # ============================================================

    tumour_probabilities_df = df_test[
        [
            'patient_id',
            'slide_id',
            'case',
            'scene',
            'tile',
            'quadrant',
            'clinical_score',
            'gleason',
            'tissue_proportion',
            'tumour_proportion'
        ]
    ].copy()

    tumour_probabilities_df.loc[
        :,
        'tumour_probability'
    ] = test_probabilities

    output_filename = (
        '/outer_fold_'
        + str(fold_n).zfill(5)
        + 'tile_level_tumour_probabilities.csv'
    )

    print(outer_fold_output_dir + output_filename)

    tumour_probabilities_df.to_csv(
        outer_fold_output_dir + output_filename,
        index=False
    )


    # ============================================================
    # Calculate ROC AUC
    # ============================================================

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print('roc_auc:')
    print(test_roc_auc)


    # ============================================================
    # Calculate and save SHAP values
    # ============================================================

    print("Fitting SHAP values...")

    shap_output_path = os.path.join(
        outer_fold_output_dir,
        f"outer_fold_{str(fold_n).zfill(5)}_shap_output.pkl"
    )

    calculate_and_save_shap(
        xgb_model,
        X_test_rfe,
        shap_output_path
    )


# ================================================================
# Gleason 3 vs 4
# ================================================================

def generate_gleason_tile_probabilities(fold_n):

    # ============================================================
    # Paths and setup
    # ============================================================

    feature_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/'
    output_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/gleason_3_vs_4_with_sara/xgboost_v2/'
    code_dir = '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/gleason_3_vs_4_with_sara/xgboost_v2/'
    data_path = feature_dir + 'updated_tumour_boundary_quadrants_double_are_recurrent_df.csv'
    gleason_annotation_parent_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/pre_processed_data/clinical/prostate/chiip_cohort/overlay_tissue_tumour_gleason_annotations/'

    outer_fold_output_dir = output_dir + 'outer_fold_output/'
    os.makedirs(outer_fold_output_dir, exist_ok=True)


    tumour_proportion = 0.7
    tissue_proportion = 0.7
    max_rank = 1


    # ============================================================
    # Load hyperparameters and select model
    # ============================================================

    hyperparameters_df = pd.read_csv(
        os.path.join(
            code_dir,
            'xg_quad_1000_gleason_3_vs_4_nested_5_folds_v2_runkey.txt'
        ),
        sep=' ',
        header=None
    )

    hyperparameters_df.drop(columns=[10], inplace=True)

    hyperparameters_df = hyperparameters_df.rename(
        columns={
            0: 'name',
            1: 'fold',
            2: 'n_estimator',
            3: 'max_depth',
            4: 'learning_rate',
            5: 'subsample',
            6: 'colsample_bytree',
            7: 'gamma',
            8: 'scale_pos_weight',
            9: 'min_child_weight'
        }
    )

    hyperparameters_df = hyperparameters_df[
        hyperparameters_df.fold == 0
    ]

    hyperparameters_df = hyperparameters_df.loc[0:4374]

    valid_runs = set(hyperparameters_df.index)

    csv_pattern = os.path.join(
        output_dir,
        'xgboost_gleason_3_vs_4_quadrant_1000_total_outer_folds_5_outer_fold_test_'
        + str(fold_n).zfill(5)
        + '_run_*_roc_auc_tile_xgboosttumour_gr_0_7_rfe_mean_test_scores.csv'
    )

    score_csv_paths = glob(csv_pattern)
    score_csv_paths = sorted(score_csv_paths)

    run_id, rfe_model = select_best_xgboost_model(
        score_csv_paths,
        valid_runs
    )


    # ============================================================
    # Build XGBoost model
    # ============================================================

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 8,
        'n_estimators': hyperparameters_df.loc[run_id].n_estimator,
        'max_depth': hyperparameters_df.loc[run_id].max_depth,
        'learning_rate': hyperparameters_df.loc[run_id].learning_rate,
        'subsample': hyperparameters_df.loc[run_id].subsample,
        'colsample_bytree': hyperparameters_df.loc[run_id].colsample_bytree,
        'gamma': hyperparameters_df.loc[run_id].gamma,
        'scale_pos_weight': hyperparameters_df.loc[run_id].scale_pos_weight,
        'min_child_weight': hyperparameters_df.loc[run_id].min_child_weight
    }

    for param, value in xgb_params.items():

        print(param)
        print(value)

        if isinstance(value, str):
            try:
                xgb_params[param] = float(value)
            except ValueError:
                pass

    xgb_model = XGBClassifier(**xgb_params)


    # ============================================================
    # Load and prepare data
    # ============================================================

    feature_df = pd.read_csv(
        data_path,
        low_memory=False
    )

    feature_df = feature_df.reset_index(drop=True)

    conditions = [
        (feature_df['gleason'] == 0) & (feature_df['case'] == 1.0),
        (feature_df['gleason'] == 1) & (feature_df['case'] == 1.0),
        (feature_df['gleason'] == 2) & (feature_df['case'] == 1.0),
        (feature_df['gleason'] == 3) & (feature_df['case'] == 1.0),
        (feature_df['gleason'] == 0) & (feature_df['case'] == 0.0),
        (feature_df['gleason'] == 1) & (feature_df['case'] == 0.0),
        (feature_df['gleason'] == 2) & (feature_df['case'] == 0.0),
        (feature_df['gleason'] == 3) & (feature_df['case'] == 0.0)
    ]

    values = range(0, 8)

    feature_df['clinical_score'] = np.select(
        conditions,
        values
    )

    gleason_6_8_df = feature_df[
        (feature_df['tissue_proportion'] > tissue_proportion)
        & (feature_df['tumour_proportion'] > tumour_proportion)
        & (
            (feature_df['gleason'] == 0)
            | (feature_df['gleason'] == 3)
        )
    ].copy()

    gleason_6_8_df['gleason_4'] = (
        gleason_6_8_df['gleason'] == 3
    ).astype(int)

    gleason_6_8_df['gleason_3'] = (
        gleason_6_8_df['gleason'] == 0
    ).astype(int)

    gleason_7_df = feature_df[
        (feature_df['tissue_proportion'] > tissue_proportion)
        & (feature_df['tumour_proportion'] > tumour_proportion)
        & (
            (feature_df['gleason'] == 1)
            | (feature_df['gleason'] == 2)
        )
    ].copy()


    # ============================================================
    # Load Gleason annotation data
    # ============================================================

    csv_files = [
        os.path.join(gleason_annotation_parent_dir, f)
        for f in os.listdir(gleason_annotation_parent_dir)
        if f.endswith('_1000.csv')
    ]

    all_dfs = []

    for file_path in csv_files:

        print(f"Loading: {file_path}")

        df = pd.read_csv(file_path)
        all_dfs.append(df)

    combined_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    combined_df.rename(
        columns={
            'tumour_gleason4_fraction_of_tissue_area': 'gleason_4_proportion',
            'tumour_gleason3_fraction_of_tissue_area': 'gleason_3_proportion'
        },
        inplace=True
    )

    gleason_scoring_df = combined_df[
        (combined_df['tissue_fraction_of_tile_area'] > tissue_proportion)
        & (
            (combined_df['gleason_4_proportion'] > tumour_proportion)
            | (combined_df['gleason_3_proportion'] > tumour_proportion)
        )
    ].copy()

    merged_7_df = gleason_7_df.merge(
        gleason_scoring_df,
        on=['slide_id', 'scene', 'tile', 'quadrant'],
        suffixes=('_df7', '_df_scoring'),
        how='inner'
    )

    merged_7_df.drop(
        columns=[
            'tumour_total_fraction_of_tissue_area',
            'tumour_other_fraction_of_tissue_area',
            'tissue_fraction_of_tile_area',
            'row',
            'col',
            'quadrant_row',
            'quadrant_col'
        ],
        inplace=True
    )

    merged_7_df['gleason_4'] = np.round(
        merged_7_df['gleason_4_proportion']
    )

    merged_7_df['gleason_3'] = np.round(
        merged_7_df['gleason_3_proportion']
    )

    merged_7_df.drop(
        columns=[
            'gleason_4_proportion',
            'gleason_3_proportion'
        ],
        inplace=True
    )

    gleason_df = pd.concat(
        [merged_7_df, gleason_6_8_df],
        ignore_index=True
    )

    gleason_df.drop(
        columns=['area', 'decision', 'gleason_3'],
        inplace=True
    )


    # ============================================================
    # Patient-level stratification
    # ============================================================

    unique_patients = gleason_df['patient_id'].unique()

    patient_records = []

    for unique_patient in unique_patients:

        single_patient_df = gleason_df[
            gleason_df['patient_id'] == unique_patient
        ]

        total_tiles = len(single_patient_df)

        clinical_score = single_patient_df[
            'clinical_score'
        ].unique()[0]

        total_gleason_4 = np.sum(
            single_patient_df['gleason_4']
        )

        total_gleason_3 = total_tiles - total_gleason_4

        patient_records.append({
            "patient_id": unique_patient,
            "total_tiles": total_tiles,
            "total_gleason_4": total_gleason_4,
            "total_gleason_3": total_gleason_3,
            "proportion_3": total_gleason_3 / total_tiles,
            "clinical_score": clinical_score
        })

    patient_df = pd.DataFrame(patient_records)

    stratification_df = (
        patient_df
        .copy()
        .sort_values(
            by=['clinical_score', 'total_gleason_3', 'total_tiles'],
            ascending=[True, False, False]
        )
        .reset_index()
    )

    folds_csv_file = code_dir + 'outer_folds_data.csv'

    (
        stratification_df,
        _,
        fold_assignments
    ) = patient_tile_stratifier(
        stratification_df,
        gleason_df,
        5,
        folds_csv_file
    )


    # ============================================================
    # Train/test split
    # ============================================================

    df = gleason_df.copy()

    df_train = df[
        fold_assignments != fold_n
    ]

    df_train = df_train.reset_index(drop=True)

    X_train = df_train.drop(
        [
            'patient_id',
            'slide_id',
            'gleason',
            'case',
            'scene',
            'tile',
            'quadrant',
            'tissue_proportion',
            'tumour_proportion',
            'clinical_score',
            'gleason_4'
        ],
        axis=1
    ).copy()

    y_train = df_train['gleason_4']

    df_test = df[
        fold_assignments == fold_n
    ]

    df_test = df_test.reset_index(drop=True)

    X_test = df_test.drop(
        [
            'patient_id',
            'slide_id',
            'gleason',
            'case',
            'scene',
            'tile',
            'quadrant',
            'tissue_proportion',
            'tumour_proportion',
            'clinical_score',
            'gleason_4'
        ],
        axis=1
    ).copy()

    y_test = df_test['gleason_4']

    X_train_rfe, X_test_rfe = select_rfe_features(
        X_train,
        X_test,
        rfe_model,
        max_rank
    )


    # ============================================================
    # Fit model and predict
    # ============================================================

    test_probabilities = fit_and_predict(
        xgb_model,
        X_train_rfe,
        y_train,
        X_test_rfe
    )


    # ============================================================
    # Save tile-level predictions
    # ============================================================

    gleason_probabilities_df = df_test[
        [
            'patient_id',
            'slide_id',
            'case',
            'scene',
            'tile',
            'quadrant',
            'clinical_score',
            'gleason',
            'gleason_4',
            'tissue_proportion',
            'tumour_proportion'
        ]
    ].copy()

    gleason_probabilities_df.loc[
        :,
        'gleason_4_probability'
    ] = test_probabilities

    output_filename = (
        '/outer_fold_'
        + str(fold_n).zfill(5)
        + 'tile_level_gleason_probabilities.csv'
    )

    print(outer_fold_output_dir + output_filename)

    gleason_probabilities_df.to_csv(
        outer_fold_output_dir + output_filename,
        index=False
    )


    # ============================================================
    # Calculate ROC AUC
    # ============================================================

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print('roc_auc:')
    print(test_roc_auc)


    # ============================================================
    # Calculate and save SHAP values
    # ============================================================

    print("Fitting SHAP values...")

    shap_output_path = os.path.join(
        outer_fold_output_dir,
        f"outer_fold_{str(fold_n).zfill(5)}_shap_output.pkl"
    )

    calculate_and_save_shap(
        xgb_model,
        X_test_rfe,
        shap_output_path
    )


# ================================================================
# Recurrent
# ================================================================

def generate_recurrent_tile_probabilities(bootstrap, fold_n):

    # ============================================================
    # Paths and setup
    # ============================================================

    outer_fold_output_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/rfe_runs/gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_f1score_removed_xgboost_outer_fold_' + str(bootstrap)

    os.makedirs(outer_fold_output_dir, exist_ok=True)

    data_exploration_module_path = os.path.abspath(
        os.path.join(
            '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/data_exploration/'
        )
    )

    if data_exploration_module_path not in sys.path:
        sys.path.append(data_exploration_module_path)


    model_results_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    feature_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/'
    code_dir = '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
    data_path = feature_dir + 'updated_tumour_boundary_quadrants_double_are_removed_df.csv'

    tumour_proportion = 0.7
    tissue_proportion = 0.7
    max_rank = 1


    # ============================================================
    # Load hyperparameters and select model
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

    hyperparameters_df = hyperparameters_df[
        hyperparameters_df.fold == 0
    ]

    hyperparameters_df = hyperparameters_df.loc[0:288]

    folds_csv_file = (
        code_dir
        + 'folds_random_'
        + str(bootstrap).zfill(3)
        + '.csv'
    )

    bootstrap_csv_name = (
        'xgboost_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_'
        + str(bootstrap)
    )

    csv_pattern = os.path.join(
        model_results_dir,
        bootstrap_csv_name
        + '_outer_fold_test_'
        + str(fold_n).zfill(5)
        + '_run_*0_7*.csv'
    )

    score_csv_paths = glob(csv_pattern)
    score_csv_paths = sorted(score_csv_paths)

    run_id, rfe_model = select_best_xgboost_model(
        score_csv_paths
    )


    # ============================================================
    # Build XGBoost model
    # ============================================================

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 8,
        'n_estimators': hyperparameters_df.loc[run_id].n_estimator,
        'max_depth': hyperparameters_df.loc[run_id].max_depth,
        'learning_rate': hyperparameters_df.loc[run_id].learning_rate,
        'subsample': hyperparameters_df.loc[run_id].subsample,
        'colsample_bytree': hyperparameters_df.loc[run_id].colsample_bytree,
        'gamma': hyperparameters_df.loc[run_id].gamma,
        'min_child_weight': hyperparameters_df.loc[run_id].min_child_weight
    }

    for param, value in xgb_params.items():

        print(param)
        print(value)

        if isinstance(value, str):

            try:
                xgb_params[param] = float(value)

            except ValueError:
                pass

    xgb_model = XGBClassifier(
        **xgb_params
    )


    # ============================================================
    # Load and prepare data
    # ============================================================

    df = pd.read_csv(
        data_path,
        low_memory=False
    )

    df = df[
        df['tissue_proportion'] >= tissue_proportion
    ]

    df = df[
        df['tumour_proportion'] >= tumour_proportion
    ]

    df = df[
        (df['gleason'] == 1)
        | (df['gleason'] == 2)
    ]

    df = df.reset_index(drop=True)

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
            df[df.patient_id == patient_id].gleason.unique()[0]
        )

        cases.append(
            df[df.patient_id == patient_id].case.unique()[0]
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

    patient_df = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'clinical_score': clinical_scores,
        'total_tiles': total_tiles
    })

    stratification_df = (
        patient_df
        .sort_values(
            by=['clinical_score', 'total_tiles'],
            ascending=[True, False]
        )
        .copy()
        .reset_index()
    )


    # ============================================================
    # Patient-level stratification
    # ============================================================

    (
        stratification_df,
        _,
        fold_assignments
    ) = patient_tile_stratifier(
        stratification_df,
        df,
        5,
        folds_csv_file
    )


    # ============================================================
    # Train/test split
    # ============================================================

    df_train = df[
        fold_assignments != fold_n
    ]

    df_train = df_train.reset_index(drop=True)

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

    df_test = df[
        fold_assignments == fold_n
    ]

    df_test = df_test.reset_index(drop=True)

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

    y_test = df_test["case"]

    X_train_rfe, X_test_rfe = select_rfe_features(
        X_train,
        X_test,
        rfe_model,
        max_rank
    )


    # ============================================================
    # Fit model and predict
    # ============================================================

    test_probabilities = fit_and_predict(
        xgb_model,
        X_train_rfe,
        y_train,
        X_test_rfe
    )


    # ============================================================
    # Save tile-level predictions
    # ============================================================

    recurrence_probabilities_df = df_test[
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
        ]
    ].copy()

    recurrence_probabilities_df.loc[
        :,
        'prob_recurrence'
    ] = test_probabilities

    output_filename = (
        '/outer_fold_'
        + str(fold_n).zfill(5)
        + 'tile_level_recur_probabilities_xgboost.csv'
    )

    print(outer_fold_output_dir + output_filename)

    recurrence_probabilities_df.to_csv(
        outer_fold_output_dir + output_filename,
        index=False
    )


    # ============================================================
    # Calculate ROC AUC
    # ============================================================

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    print('roc_auc:')
    print(test_roc_auc)


    # ============================================================
    # Calculate and save SHAP values
    # ============================================================

    print("Fitting SHAP values...")

    shap_output_path = os.path.join(
        outer_fold_output_dir,
        f"outer_fold_{str(fold_n).zfill(5)}_shap_output.pkl"
    )

    calculate_and_save_shap(
        xgb_model,
        X_test_rfe,
        shap_output_path
    )




```python
def calculate_and_save_shap(model, X_test, shap_save_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test.values,
        feature_names=X_test.columns.tolist()
    )

    with open(shap_save_path, 'wb') as f:
        pickle.dump(shap_explanation, f)


def generate_recurrent_tile_probabilities(bootstrap, fold_n, model_type):

    # ============================================================
    # PATHS AND SETTINGS
    # ============================================================

    feature_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/'
    )

    if model_type == 'xgboost':

        outer_fold_output_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_'
            'f1score_removed_xgboost_outer_fold_' +
            str(bootstrap) +
            '/'
        )

        model_results_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
        )

        code_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'
        )

    elif model_type == 'forest':

        outer_fold_output_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_'
            'f1score_removed_outer_fold_' +
            str(bootstrap) +
            '/'
        )

        model_results_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap/'
        )

        code_dir = (
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap/'
        )

    else:
        raise ValueError(
            "model_type must be either 'xgboost' or 'forest'"
        )

    os.makedirs(
        outer_fold_output_dir,
        exist_ok=True
    )

    data_path = (
        feature_dir +
        'updated_tumour_boundary_quadrants_double_are_removed_df.csv'
    )

    folds_csv_file = (
        code_dir +
        'folds_random_' +
        str(bootstrap).zfill(3) +
        '.csv'
    )

    tumour_proportion = 0.7
    tissue_proportion = 0.7
    max_rank = 1


    # ============================================================
    # MODEL-SPECIFIC SETUP
    # ============================================================

    if model_type == 'xgboost':

        hyperparameters_df = pd.read_csv(
            os.path.join(
                code_dir,
                'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_runkey.txt'
            ),
            sep=' ',
            header=None
        )

        hyperparameters_df = hyperparameters_df.rename(
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

        hyperparameters_df = hyperparameters_df[
            hyperparameters_df.fold == 0
        ]

        hyperparameters_df = hyperparameters_df.loc[0:288]

        bootstrap_csv_name = (
            'xgboost_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_' +
            str(bootstrap)
        )

        csv_pattern = os.path.join(
            model_results_dir,
            bootstrap_csv_name +
            '_outer_fold_test_' +
            str(fold_n).zfill(5) +
            '_run_*0_7*.csv'
        )

        score_csv_paths = sorted(
            glob(csv_pattern)
        )

        run_id, rfe_model = select_best_xgboost_model(
            score_csv_paths
        )

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_jobs': -1,
            'random_state': 8,
            'n_estimators': hyperparameters_df.loc[run_id].n_estimator,
            'max_depth': hyperparameters_df.loc[run_id].max_depth,
            'learning_rate': hyperparameters_df.loc[run_id].learning_rate,
            'subsample': hyperparameters_df.loc[run_id].subsample,
            'colsample_bytree': hyperparameters_df.loc[run_id].colsample_bytree,
            'gamma': hyperparameters_df.loc[run_id].gamma,
            'min_child_weight': hyperparameters_df.loc[run_id].min_child_weight
        }

        for param, value in xgb_params.items():
            print(param)
            print(value)

            if isinstance(value, str):
                try:
                    xgb_params[param] = float(value)
                except ValueError:
                    pass

        model = XGBClassifier(
            **xgb_params
        )

    else:

        module_path = os.path.abspath(
            feature_dir +
            'rfe_runs/'
            'gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_'
            'f1score_removed_outer_fold_' +
            str(bootstrap) +
            '/'
        )

        if module_path not in sys.path:
            sys.path.append(module_path)

        import model_fitting_rfe as mf

        hyperparameters_df = pd.read_csv(
            os.path.join(
                code_dir,
                'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_runkey.txt'
            ),
            sep=' ',
            header=None
        )

        hyperparameters_df.drop(
            columns=[7],
            inplace=True
        )

        hyperparameters_df = hyperparameters_df.rename(
            columns={
                0: 'name',
                1: 'fold',
                2: 'n_estimator',
                3: 'max_depth',
                4: 'min_samples_split',
                5: 'min_samples_leaf',
                6: 'max_feature'
            }
        )

        hyperparameters_df = hyperparameters_df[
            hyperparameters_df.fold == 0
        ]

        hyperparameters_df = hyperparameters_df.loc[0:162]

        bootstrap_csv_name = (
            'forest_recurrence_quadrant_1000_total_outer_folds_5_bootstrap_' +
            str(bootstrap)
        )

        csv_pattern = os.path.join(
            model_results_dir,
            bootstrap_csv_name +
            '_outer_fold_test_' +
            str(fold_n).zfill(5) +
            '_run_*0_7*.csv'
        )

        score_csv_paths = sorted(
            glob(csv_pattern)
        )

        test_score = []
        candidate_csv_paths = []

        for csv_path in score_csv_paths:

            with open(csv_path, 'r') as file:
                df_score = pd.read_csv(
                    file,
                    low_memory=False
                )

            if not test_score:

                test_score.append(
                    np.max(df_score.Mean_Test_Scores)
                )

                candidate_csv_paths.append(
                    csv_path
                )

            elif np.max(
                df_score.Mean_Test_Scores
            ) >= np.max(test_score):

                test_score.append(
                    np.max(df_score.Mean_Test_Scores)
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

        with open(sav_file, 'rb') as f:
            fold_model = pickle.load(f)

        rf_classifier_params = {
            'n_estimators': hyperparameters_df.loc[run].n_estimator,
            'max_depth': hyperparameters_df.loc[run].max_depth,
            'min_samples_split': hyperparameters_df.loc[run].min_samples_split,
            'min_samples_leaf': hyperparameters_df.loc[run].min_samples_leaf,
            'max_features': hyperparameters_df.loc[run].max_feature,
            'random_state': 8
        }

        for param, value in rf_classifier_params.items():

            if isinstance(value, str):
                try:
                    rf_classifier_params[param] = float(value)
                except ValueError:
                    pass

        model = RandomForestClassifier(
            **rf_classifier_params
        )


    # ============================================================
    # LOAD AND FILTER DATA
    # ============================================================

    df = pd.read_csv(
        data_path,
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


    # ============================================================
    # CREATE CLINICAL SCORE
    # ============================================================

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


    # ============================================================
    # CREATE PATIENT-LEVEL STRATIFICATION DATA
    # ============================================================

    patient_ids = list(
        df.patient_id.unique()
    )

    gleasons = []
    cases = []
    total_tiles = []
    clinical_scores = []

    for patient_id in patient_ids:

        gleasons.append(
            df[df.patient_id == patient_id].gleason.unique()[0]
        )

        cases.append(
            df[df.patient_id == patient_id].case.unique()[0]
        )

        clinical_scores.append(
            df[df.patient_id == patient_id].clinical_score.unique()[0]
        )

        total_tiles.append(
            len(
                df[df.patient_id == patient_id].case
            )
        )

    patient_df = pd.DataFrame({
        'patient_id': patient_ids,
        'gleason': gleasons,
        'case': cases,
        'total_tiles': total_tiles,
        'clinical_score': clinical_scores
    })

    stratification_df = (
        patient_df
        .sort_values(
            by=['clinical_score', 'total_tiles'],
            ascending=[True, False]
        )
        .copy()
        .reset_index()
    )


    # ============================================================
    # CREATE FOLD ASSIGNMENTS
    # ============================================================

    (
        stratification_df,
        _,
        fold_assignments
    ) = patient_tile_stratifier(
        stratification_df,
        df,
        5,
        folds_csv_file
    )



    # ============================================================
    # CREATE OUTER-FOLD TRAINING AND TEST DATA
    # ============================================================

    df_train = df[
        fold_assignments != fold_n
    ].reset_index(
        drop=True
    )

    df_test = df[
        fold_assignments == fold_n
    ].reset_index(
        drop=True
    )


    # ============================================================
    # PREPARE TRAINING AND TEST DATA
    # ============================================================

    metadata_columns = [
        'patient_id',
        'slide',
        'case',
        'gleason',
        'tissue_proportion',
        'tumour_proportion',
        'clinical_score'
    ]

    X_train = df_train.drop(
        metadata_columns,
        axis=1
    ).copy()

    y_train = df_train['case']

    X_test = df_test.drop(
        metadata_columns,
        axis=1
    ).copy()


    # ============================================================
    # SELECT RFE FEATURES
    # ============================================================

    X_train_subset_df, X_test_subset_df = select_rfe_features(
        X_train,
        X_test,
        rfe_model,
        max_rank
    )


    # ============================================================
    # FIT MODEL AND PREDICT
    # ============================================================

    if model_type == 'xgboost':

        test_probabilities = fit_and_predict(
            model,
            X_train_subset_df,
            y_train,
            X_test_subset_df
        )

    else:

        model.fit(
            X_train_subset_df,
            y_train
        )

        test_probabilities = model.predict_proba(
            X_test_subset_df
        )[:, 1]


    # ============================================================
    # SAVE TILE-LEVEL RECURRENCE PROBABILITIES
    # ============================================================

    metadata_columns_output = [
        'patient_id',
        'slide',
        'x',
        'y',
        'case',
        'gleason',
        'tissue_proportion',
        'tumour_proportion'
    ]

    recurrence_probabilities_df = df_test[
        metadata_columns_output
    ].copy()

    recurrence_probabilities_df.loc[:, 'prob_recurrence'] = (
        test_probabilities
    )

    if model_type == 'xgboost':

        output_filename = (
            'outer_fold_' +
            str(fold_n).zfill(5) +
            'tile_level_recur_probabilities_xgboost.csv'
        )

    else:

        output_filename = (
            'outer_fold_' +
            str(fold_n).zfill(5) +
            'tile_level_recur_probabilities_forest.csv'
        )

    print(
        os.path.join(
            outer_fold_output_dir,
            output_filename
        )
    )

    recurrence_probabilities_df.to_csv(
        os.path.join(
            outer_fold_output_dir,
            output_filename
        ),
        index=False
    )



    # ============================================================
    # SHAP
    # ============================================================

    shap_save_path = os.path.join(
        outer_fold_output_dir,
        (
            f'outer_fold_{str(fold_n).zfill(5)}'
            f'_shap_output.pkl'
        )
    )

    calculate_and_save_shap(
        model,
        X_test_subset_df,
        shap_save_path
    )

