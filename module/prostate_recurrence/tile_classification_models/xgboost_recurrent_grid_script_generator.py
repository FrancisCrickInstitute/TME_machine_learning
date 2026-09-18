# ============================================================
# IMPORTS
# ============================================================

import itertools
import numpy as np


# ============================================================
# PATHS AND SETTINGS
# ============================================================

main_path = (
    "/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/"
    "prostate_recurrence/model_evaluation/recurrence_status_tile/"
    "rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/"
)

n_bootstraps = 30
n_outer_folds = 5

file_input_name = (
    'quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_'
)

keyfilename = (
    main_path
    + "/"
    + file_input_name
    + "runkey.txt"
)


# ============================================================
# OUTER FOLDS
# ============================================================

bootstraps = np.arange(20,20+n_bootstraps)
outer_folds = np.arange(n_outer_folds)


# ============================================================
# XGBOOST PARAMETER GRID
# ============================================================

n_estimators       = [100, 200]        
max_depths         = [3, 5, 7]         
learning_rates     = [0.05, 0.1]       
subsamples         = [0.7, 1.0]        
colsample_bytrees  = [0.7, 1.0]        
gammas             = [0, 1]            
min_child_weights  = [1, 5, 10]        
scale_pos_weights  = [1.0]                


# ============================================================
# GENERATE ALL PARAMETER COMBINATIONS
# ============================================================

parameter_combinations = list(
    itertools.product(
        n_estimators,
        max_depths,
        learning_rates,
        subsamples,
        colsample_bytrees,
        gammas,
        scale_pos_weights,
        min_child_weights,
    )
)

print(
    f"Number of parameter combinations: "
    f"{len(parameter_combinations)}"
)


# ============================================================
# GENERATE RUN FILES
# ============================================================






for bootstrap in bootstraps:
    col_name = f'folds_random_{str(bootstrap).zfill(3)}'
 
    for outer_fold in outer_folds:

        counter = 0

        file_input_name = (
            'xg_boost_nested_cv_gleason_7_quad_1000_v8a_ambiguous_removed_forest_outer_bootstrap_'
            + str(bootstrap).zfill(3)
            + '_fold_'
            + str(outer_fold).zfill(5)
            + '_v0_run_'
        )

        for (
            n_estimator,
            max_depth,
            learning_rate,
            subsample,
            colsample_bytree,
            gamma,
            scale_pos_weight,
            min_child_weight,
        ) in parameter_combinations:

            # ----------------------------------------------------
            # Write run information to key file
            # ----------------------------------------------------

            with open(keyfilename, "a") as f:

                f.write(
                    "Fold_"
                    + str(outer_fold).zfill(5)
                    + "_Run_"
                    + str(counter).zfill(5)
                )

                f.write(
                    " %d %d %d %f %f %f %f %f %d\n"
                    % (
                        outer_fold,
                        n_estimator,
                        max_depth,
                        learning_rate,
                        subsample,
                        colsample_bytree,
                        gamma,
                        scale_pos_weight,
                        min_child_weight,
                    )
                )


            # ----------------------------------------------------
            # Create Python run file
            # ----------------------------------------------------

            filename = (
                main_path
                + file_input_name
                + str(counter).zfill(5)
                + ".py"
            )

            with open(filename, "w") as f:






                f.write("# ============================================================\n")
                f.write("# IMPORTS\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("import os\n")
                f.write("\n")
                f.write("import pandas as pd\n")
                f.write("import numpy as np\n")
                f.write("\n")
                f.write("from sklearn.metrics import make_scorer, roc_auc_score\n")
                f.write("from xgboost import XGBClassifier\n")
                f.write("\n")
                f.write("import tile_model_fitting as tmf\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# PATHS\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("main_path = (\n")
                f.write("    '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'\n")
                f.write("    'prostate_recurrence/model_evaluation/recurrence_status_tile/'\n")
                f.write("    'rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'\n")
                f.write(")\n")
                f.write("\n")

                f.write("output_dir = (\n")
                f.write("    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'\n")
                f.write("    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'\n")
                f.write("    'feature_analysis_v3/rfe_runs/'\n")
                f.write("    'gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap_xgboost/'\n")
                f.write(")\n")
                f.write("\n")

                f.write("input_dir = (\n")
                f.write("    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'\n")
                f.write("    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'\n")
                f.write("    'feature_analysis_v3/'\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# PARAMETERS\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("steps = 10\n")
                f.write("no_inner_folds = 5\n")
                f.write("model_random_seed = 31\n")
                f.write("\n")
                f.write("tumour_proportion = 0.7\n")
                f.write("tissue_proportion = 0.7\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# LOAD AND FILTER DATA\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("data_dir = (\n")
                f.write("    input_dir\n")
                f.write("    + 'updated_tumour_boundary_quadrants_double_are_removed_df.csv'\n")
                f.write(")\n")
                f.write("\n")

                f.write("df = pd.read_csv(\n")
                f.write("    data_dir,\n")
                f.write("    low_memory=False\n")
                f.write(")\n")
                f.write("\n")

                f.write("df = df[df['tissue_proportion'] >= tissue_proportion]\n")
                f.write("\n")
                f.write("df = df[df['tumour_proportion'] >= tumour_proportion]\n")
                f.write("\n")

                f.write("df = df[\n")
                f.write("    (df['gleason'] == 1)\n")
                f.write("    | (df['gleason'] == 2)\n")
                f.write("]\n")
                f.write("\n")

                f.write("df = df.reset_index(drop=True)\n")
                f.write("\n")

                f.write("print(df.head())\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# CREATE CLINICAL SCORE\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("patient_ids = list(df.patient_id.unique())\n")
                f.write("\n")

                f.write("conditions = [\n")
                f.write("    (df['gleason'] == 1) & (df['case'] == 1.0),\n")
                f.write("    (df['gleason'] == 2) & (df['case'] == 1.0),\n")
                f.write("    (df['gleason'] == 1) & (df['case'] == 0.0),\n")
                f.write("    (df['gleason'] == 2) & (df['case'] == 0.0),\n")
                f.write("]\n")
                f.write("\n")

                f.write("values = range(0, 4)\n")
                f.write("\n")

                f.write("df['clinical_score'] = np.select(\n")
                f.write("    conditions,\n")
                f.write("    values\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# CREATE PATIENT-LEVEL STRATIFICATION DATA\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("gleasons = []\n")
                f.write("cases = []\n")
                f.write("total_tiles = []\n")
                f.write("clinical_scores = []\n")
                f.write("\n")

                f.write("for patient_id in patient_ids:\n")
                f.write("\n")

                f.write("    gleasons.append(\n")
                f.write("        df[df.patient_id == patient_id].gleason.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    cases.append(\n")
                f.write("        df[df.patient_id == patient_id].case.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    clinical_scores.append(\n")
                f.write("        df[df.patient_id == patient_id].clinical_score.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    total_tiles.append(\n")
                f.write("        len(df[df.patient_id == patient_id].case)\n")
                f.write("    )\n")
                f.write("\n")

                f.write("\n")

                f.write("df_patient = pd.DataFrame({\n")
                f.write("    'patient_id': patient_ids,\n")
                f.write("    'gleason': gleasons,\n")
                f.write("    'case': cases,\n")
                f.write("    'clinical_score': clinical_scores,\n")
                f.write("    'total_tiles': total_tiles,\n")
                f.write("})\n")
                f.write("\n")

                f.write("df_strat = (\n")
                f.write("    df_patient\n")
                f.write("    .sort_values(\n")
                f.write("        by=['clinical_score', 'total_tiles'],\n")
                f.write("        ascending=[True, False]\n")
                f.write("    )\n")
                f.write("    .copy()\n")
                f.write("    .reset_index()\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# OUTER FOLD\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write(f"folds_csv_file = main_path + '{col_name}.csv'\n")
                f.write("\n")

                f.write("df_strat, predefined_splits, predefined_splits_array = (\n")
                f.write("    tmf.patient_tile_stratifier(\n")
                f.write("        df_strat,\n")
                f.write("        df,\n")
                f.write("        5,\n")
                f.write("        folds_csv_file\n")
                f.write("    )\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# CREATE OUTER-FOLD TRAINING DATA\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("df_train = df[predefined_splits_array != ")
                f.write("%d" % outer_fold)
                f.write("]\n")
                f.write("\n")

                f.write("df_train = df_train.reset_index(drop=True)\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# CREATE INNER-FOLD PATIENT STRATIFICATION DATA\n")
                f.write("# ============================================================\n")
                f.write("\n")
                f.write("patient_ids = list(df_train.patient_id.unique())\n")
                f.write("\n")

                f.write("gleasons = []\n")
                f.write("cases = []\n")
                f.write("clinical_scores = []\n")
                f.write("total_tiles = []\n")
                f.write("\n")

                f.write("for patient_id in patient_ids:\n")
                f.write("\n")

                f.write("    gleasons.append(\n")
                f.write("        df_train[df_train.patient_id == patient_id].gleason.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    cases.append(\n")
                f.write("        df_train[df_train.patient_id == patient_id].case.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    clinical_scores.append(\n")
                f.write("        df_train[\n")
                f.write("            df_train.patient_id == patient_id\n")
                f.write("        ].clinical_score.unique()[0]\n")
                f.write("    )\n")
                f.write("\n")

                f.write("    # Deliberately uses df rather than df_train to preserve\n")
                f.write("    # the original implementation exactly.\n")
                f.write("    total_tiles.append(\n")
                f.write("        len(df[df.patient_id == patient_id].case)\n")
                f.write("    )\n")
                f.write("\n")
                f.write("\n")

                f.write("df_patient = pd.DataFrame({\n")
                f.write("    'patient_id': patient_ids,\n")
                f.write("    'gleason': gleasons,\n")
                f.write("    'case': cases,\n")
                f.write("    'clinical_score': clinical_scores,\n")
                f.write("    'total_tiles': total_tiles,\n")
                f.write("})\n")
                f.write("\n")

                f.write("df_strat = (\n")
                f.write("    df_patient\n")
                f.write("    .sort_values(\n")
                f.write("        by=['clinical_score', 'total_tiles'],\n")
                f.write("        ascending=[True, False]\n")
                f.write("    )\n")
                f.write("    .copy()\n")
                f.write("    .reset_index()\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# INNER FOLD\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("folds_csv_file = main_path + 'inner_folds_")
                f.write("%d" % outer_fold)
                f.write("_data.csv'\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# PREPARE TRAINING DATA\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("X_train = df_train.drop(\n")
                f.write("    [\n")
                f.write("        'patient_id',\n")
                f.write("        'slide_id',\n")
                f.write("        'area',\n")
                f.write("        'decision',\n")
                f.write("        'case',\n")
                f.write("        'scene',\n")
                f.write("        'tile',\n")
                f.write("        'quadrant',\n")
                f.write("        'clinical_score',\n")
                f.write("        'tissue_proportion',\n")
                f.write("        'tumour_proportion',\n")
                f.write("    ],\n")
                f.write("    axis=1\n")
                f.write(").copy()\n")
                f.write("\n")

                f.write("y_train = df_train['case']\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# XGBOOST MODEL\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("xg_classifier = XGBClassifier(\n")
                f.write("    objective='binary:logistic',\n")
                f.write("    eval_metric='logloss',\n")
                f.write("    use_label_encoder=False,\n")
                f.write("    n_jobs=-1,\n")
                f.write("    random_state=8,\n")
                f.write("    n_estimators=")
                f.write("%d" % n_estimator)
                f.write(",\n")
                f.write("    max_depth=")
                f.write("%d" % max_depth)
                f.write(",\n")
                f.write("    learning_rate=")
                f.write("%f" % learning_rate)
                f.write(",\n")
                f.write("    subsample=")
                f.write("%f" % subsample)
                f.write(",\n")
                f.write("    colsample_bytree=")
                f.write("%f" % colsample_bytree)
                f.write(",\n")
                f.write("    gamma=")
                f.write("%f" % gamma)
                f.write(",\n")
                f.write("    scale_pos_weight=")
                f.write("%f" % scale_pos_weight)
                f.write(",\n")
                f.write("    min_child_weight=")
                f.write("%d" % min_child_weight)
                f.write("\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# SCORING\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("rocauc = make_scorer(\n")
                f.write("    roc_auc_score,\n")
                f.write("    needs_threshold=True\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")

                f.write("# ============================================================\n")
                f.write("# OUTPUT FILE NAME\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("save_name = (\n")
                f.write("\'xgboost_recurrence_quadrant_1000_total_outer_folds_\'")
                f.write("+ str(")
                f.write("%d" % n_outer_folds)
                f.write(")"
                f.write("+ \'_bootstrap_\'")
                f.write(" + str(")
                f.write("%d" % bootstrap)
                f.write(")")
                f.write(" + \'_outer_fold_test_\'") 
                f.write("+ str(")
                f.write("%d" % outer_fold)
                f.write(").zfill(5)")
                f.write("+\'_run_\'")
                f.write("+ str(")
                f.write("%d" % counter)
                f.write(").zfill(5)")
                f.write("+\'_roc_auc_seed_\'")
                f.write("+str(model_random_seed)")
                f.write("+\'_\')\n")
                f.write(")\n")
                f.write("\n")
                f.write("\n")


                f.write("# ============================================================\n")
                f.write("# RFECV\n")
                f.write("# ============================================================\n")
                f.write("\n")

                f.write("print(X_train)\n")
                f.write("\n")

                f.write("tmf.rfecv(\n")
                f.write("    output_dir=output_dir,\n")
                f.write("    df_train=df_train,\n")
                f.write("    df_strat=df_strat,\n")
                f.write("    folds_csv_file=folds_csv_file,\n")
                f.write("    X_train=X_train,\n")
                f.write("    y_train=y_train,\n")
                f.write("    model=xg_classifier,\n")
                f.write("    model_label='xgboost',\n")
                f.write("    save_prefix=save_name,\n")
                f.write("    steps=steps,\n")
                f.write("    scorer=rocauc,\n")
                f.write("    no_folds=no_inner_folds\n")
                f.write(")\n")
                counter += 1




















