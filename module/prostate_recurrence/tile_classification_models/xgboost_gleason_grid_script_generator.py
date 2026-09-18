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
    "prostate_recurrence/model_evaluation/gleason_3_vs_4_with_sara/"
    "xgboost_v2/"
)

n_outer_folds = 5

file_input_name = (
    'xg_quad_1000_gleason_3_vs_4_nested_5_folds_v2_'
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

outer_folds = np.arange(n_outer_folds)


# ============================================================
# XGBOOST PARAMETER GRID
# ============================================================

n_estimators = [100, 200]
max_depths = [3, 5, 7]
learning_rates = [0.01, 0.05, 0.1]
subsamples = [0.7, 0.85, 1.0]
colsample_bytrees = [0.7, 0.85, 1.0]
gammas = [0, 1, 5]
scale_pos_weights = [0.1, 0.2, 1.0]
min_child_weights = [1, 5, 10]



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

for outer_fold in outer_folds:

    counter = 0

    file_input_name = (
        'xg_quad_1000_gleason_3_vs_4_nested_5_folds_v2_fold_'
        + str(outer_fold).zfill(5)
        + '_v2_run_'
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
            f.write("feature_dir = (\n")
            f.write("    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'\n")
            f.write("    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'\n")
            f.write("    'feature_analysis_v3/'\n")
            f.write(")\n")
            f.write("\n")

            f.write("gleason_annotation_parent_dir = (\n")
            f.write("    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'\n")
            f.write("    'pre_processed_data/clinical/prostate/chiip_cohort/'\n")
            f.write("    'overlay_tissue_tumour_gleason_annotations/'\n")
            f.write(")\n")
            f.write("\n")

            f.write("main_path = (\n")
            f.write("    \"/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/\"\n")
            f.write("    \"prostate_recurrence/model_evaluation/gleason_3_vs_4_with_sara/\"\n")
            f.write("    \"xgboost_v2/\"\n")
            f.write(")\n")
            f.write("\n")

            f.write("output_dir = (\n")
            f.write("    '/nemo/project/proj-sahai-tme-ml/working/processed_data/'\n")
            f.write("    'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'\n")
            f.write("    'feature_analysis_v3/gleason_3_vs_4_with_sara/xgboost_v2/'\n")
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
            f.write("# LOAD FEATURE DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("data_dir = (\n")
            f.write("    feature_dir\n")
            f.write("    + 'updated_tumour_boundary_quadrants_double_are_recurrent_df.csv'\n")
            f.write(")\n")
            f.write("\n")

            f.write("feature_df1 = pd.read_csv(\n")
            f.write("    data_dir,\n")
            f.write("    low_memory=False\n")
            f.write(")\n")
            f.write("\n")

            f.write("feature_df1 = feature_df1.reset_index(drop=True)\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# CREATE CLINICAL SCORE\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("conditions = [\n")
            f.write("    (feature_df1['gleason'] == 0) & (feature_df1['case'] == 1.0),\n")
            f.write("    (feature_df1['gleason'] == 1) & (feature_df1['case'] == 1.0),\n")
            f.write("    (feature_df1['gleason'] == 2) & (feature_df1['case'] == 1.0),\n")
            f.write("    (feature_df1['gleason'] == 3) & (feature_df1['case'] == 1.0),\n")
            f.write("    (feature_df1['gleason'] == 0) & (feature_df1['case'] == 0.0),\n")
            f.write("    (feature_df1['gleason'] == 1) & (feature_df1['case'] == 0.0),\n")
            f.write("    (feature_df1['gleason'] == 2) & (feature_df1['case'] == 0.0),\n")
            f.write("    (feature_df1['gleason'] == 3) & (feature_df1['case'] == 0.0),\n")
            f.write("]\n")
            f.write("\n")

            f.write("values = range(0, 8)\n")
            f.write("\n")

            f.write("feature_df1['clinical_score'] = np.select(\n")
            f.write("    conditions,\n")
            f.write("    values\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# SELECT GLEASON 6–8 TILES\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("gleason_6_8_df = feature_df1[\n")
            f.write("    (feature_df1['tissue_proportion'] > tissue_proportion)\n")
            f.write("    & (feature_df1['tumour_proportion'] > tumour_proportion)\n")
            f.write("    & (\n")
            f.write("        (feature_df1['gleason'] == 0)\n")
            f.write("        | (feature_df1['gleason'] == 3)\n")
            f.write("    )\n")
            f.write("].copy()\n")
            f.write("\n")

            f.write("gleason_6_8_df['gleason_4'] = (\n")
            f.write("    gleason_6_8_df['gleason'] == 3\n")
            f.write(").astype(int)\n")
            f.write("\n")

            f.write("gleason_6_8_df['gleason_3'] = (\n")
            f.write("    gleason_6_8_df['gleason'] == 0\n")
            f.write(").astype(int)\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# SELECT GLEASON 7 TILES\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("gleason_7_df = feature_df1[\n")
            f.write("    (feature_df1['tissue_proportion'] > tissue_proportion)\n")
            f.write("    & (feature_df1['tumour_proportion'] > tumour_proportion)\n")
            f.write("    & (\n")
            f.write("        (feature_df1['gleason'] == 1)\n")
            f.write("        | (feature_df1['gleason'] == 2)\n")
            f.write("    )\n")
            f.write("].copy()\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# LOAD GLEASON ANNOTATION DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("csv_files = [\n")
            f.write("    os.path.join(gleason_annotation_parent_dir, f)\n")
            f.write("    for f in os.listdir(gleason_annotation_parent_dir)\n")
            f.write("    if f.endswith('_1000.csv')\n")
            f.write("]\n")
            f.write("\n")

            f.write("all_dfs = []\n")
            f.write("\n")

            f.write("for file_path in csv_files:\n")
            f.write("    print(f\"Loading: {file_path}\")\n")
            f.write("    df = pd.read_csv(file_path)\n")
            f.write("    all_dfs.append(df)\n")
            f.write("\n")

            f.write("combined_df = pd.concat(\n")
            f.write("    all_dfs,\n")
            f.write("    ignore_index=True\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# PREPARE GLEASON SCORING DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("combined_df.rename(\n")
            f.write("    columns={\n")
            f.write("        'tumour_gleason4_fraction_of_tissue_area':\n")
            f.write("            'gleason_4_proportion',\n")
            f.write("        'tumour_gleason3_fraction_of_tissue_area':\n")
            f.write("            'gleason_3_proportion',\n")
            f.write("    },\n")
            f.write("    inplace=True\n")
            f.write(")\n")
            f.write("\n")

            f.write("gleason_scoring_df = combined_df[\n")
            f.write("    (combined_df['tissue_fraction_of_tile_area'] > tissue_proportion)\n")
            f.write("    & (\n")
            f.write("        (combined_df['gleason_4_proportion'] > tumour_proportion)\n")
            f.write("        | (combined_df['gleason_3_proportion'] > tumour_proportion)\n")
            f.write("    )\n")
            f.write("].copy()\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# MERGE GLEASON 7 WITH ANNOTATION DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("merged_7_df = gleason_7_df.merge(\n")
            f.write("    gleason_scoring_df,\n")
            f.write("    on=['slide_id', 'scene', 'tile', 'quadrant'],\n")
            f.write("    suffixes=('_df7', '_df_scoring'),\n")
            f.write("    how='inner'\n")
            f.write(")\n")
            f.write("\n")

            f.write("merged_7_df.drop(\n")
            f.write("    columns=[\n")
            f.write("        'tumour_total_fraction_of_tissue_area',\n")
            f.write("        'tumour_other_fraction_of_tissue_area',\n")
            f.write("        'tissue_fraction_of_tile_area',\n")
            f.write("        'row',\n")
            f.write("        'col',\n")
            f.write("        'quadrant_row',\n")
            f.write("        'quadrant_col',\n")
            f.write("    ],\n")
            f.write("    inplace=True\n")
            f.write(")\n")
            f.write("\n")

            f.write("merged_7_df['gleason_4'] = np.round(\n")
            f.write("    merged_7_df['gleason_4_proportion']\n")
            f.write(")\n")
            f.write("\n")

            f.write("merged_7_df['gleason_3'] = np.round(\n")
            f.write("    merged_7_df['gleason_3_proportion']\n")
            f.write(")\n")
            f.write("\n")

            f.write("merged_7_df.drop(\n")
            f.write("    columns=[\n")
            f.write("        'gleason_4_proportion',\n")
            f.write("        'gleason_3_proportion',\n")
            f.write("    ],\n")
            f.write("    inplace=True\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# COMBINE GLEASON GROUPS\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("gleason_df = pd.concat(\n")
            f.write("    [merged_7_df, gleason_6_8_df],\n")
            f.write("    ignore_index=True\n")
            f.write(")\n")
            f.write("\n")

            f.write("gleason_df.drop(\n")
            f.write("    columns=[\n")
            f.write("        'area',\n")
            f.write("        'decision',\n")
            f.write("        'gleason_3',\n")
            f.write("    ],\n")
            f.write("    inplace=True\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# CREATE PATIENT-LEVEL STRATIFICATION DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("unique_patients = gleason_df['patient_id'].unique()\n")
            f.write("\n")
            f.write("patient_df = []\n")
            f.write("\n")

            f.write("for unique_patient in unique_patients[0:]:\n")
            f.write("\n")
            f.write("    single_patient_df = (\n")
            f.write("        gleason_df[\n")
            f.write("            gleason_df['patient_id'] == unique_patient\n")
            f.write("        ]\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_tiles = len(single_patient_df)\n")
            f.write("\n")

            f.write("    clinical_score = (\n")
            f.write("        single_patient_df['clinical_score'].unique()[0]\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_gleason_4 = np.sum(\n")
            f.write("        single_patient_df['gleason_4']\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_gleason_3 = total_tiles - total_gleason_4\n")
            f.write("\n")

            f.write("    patient_df.append({\n")
            f.write("        \"patient_id\": unique_patient,\n")
            f.write("        \"total_tiles\": total_tiles,\n")
            f.write("        \"total_gleason_4\": total_gleason_4,\n")
            f.write("        \"total_gleason_3\": total_gleason_3,\n")
            f.write("        \"proportion_3\": total_gleason_3 / total_tiles,\n")
            f.write("        \"clinical_score\": clinical_score,\n")
            f.write("    })\n")
            f.write("\n")
            f.write("\n")

            f.write("patient_df = pd.DataFrame(patient_df)\n")
            f.write("\n")

            f.write("df_strat = (\n")
            f.write("    patient_df\n")
            f.write("    .copy()\n")
            f.write("    .sort_values(\n")
            f.write("        by=[\n")
            f.write("            'clinical_score',\n")
            f.write("            'total_gleason_3',\n")
            f.write("            'total_tiles',\n")
            f.write("        ],\n")
            f.write("        ascending=[True, False, False]\n")
            f.write("    )\n")
            f.write("    .reset_index()\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# OUTER FOLD\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("folds_csv_file = main_path + 'outer_folds_data.csv'\n")
            f.write("\n")

            f.write("df_strat, predefined_splits, predefined_splits_array = (\n")
            f.write("    tmf.patient_tile_stratifier(\n")
            f.write("        df_strat,\n")
            f.write("        gleason_df,\n")
            f.write("        %d,\n" % n_outer_folds)
            f.write("        folds_csv_file\n")
            f.write("    )\n")

            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# CREATE OUTER-FOLD TRAINING DATA\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("df = gleason_df.copy()\n")
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
            f.write("unique_patients = df_train['patient_id'].unique()\n")
            f.write("\n")
            f.write("patient_df = []\n")
            f.write("\n")

            f.write("for unique_patient in unique_patients[0:]:\n")
            f.write("\n")
            f.write("    single_patient_df = (\n")
            f.write("        df_train[\n")
            f.write("            df_train['patient_id'] == unique_patient\n")
            f.write("        ]\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_tiles = len(single_patient_df)\n")
            f.write("\n")

            f.write("    clinical_score = (\n")
            f.write("        single_patient_df['clinical_score'].unique()[0]\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_gleason_4 = np.sum(\n")
            f.write("        single_patient_df['gleason_4']\n")
            f.write("    )\n")
            f.write("\n")

            f.write("    total_gleason_3 = total_tiles - total_gleason_4\n")
            f.write("\n")

            f.write("    patient_df.append({\n")
            f.write("        \"patient_id\": unique_patient,\n")
            f.write("        \"total_tiles\": total_tiles,\n")
            f.write("        \"total_gleason_4\": total_gleason_4,\n")
            f.write("        \"total_gleason_3\": total_gleason_3,\n")
            f.write("        \"proportion_3\": total_gleason_3 / total_tiles,\n")
            f.write("        \"clinical_score\": clinical_score,\n")
            f.write("    })\n")
            f.write("\n")
            f.write("\n")

            f.write("patient_df = pd.DataFrame(patient_df)\n")
            f.write("\n")

            f.write("df_strat = (\n")
            f.write("    patient_df\n")
            f.write("    .copy()\n")
            f.write("    .sort_values(\n")
            f.write("        by=[\n")
            f.write("            'clinical_score',\n")
            f.write("            'total_gleason_3',\n")
            f.write("            'total_tiles',\n")
            f.write("        ],\n")
            f.write("        ascending=[True, False, False]\n")
            f.write("    )\n")
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
            f.write("        'gleason',\n")
            f.write("        'case',\n")
            f.write("        'scene',\n")
            f.write("        'tile',\n")
            f.write("        'quadrant',\n")
            f.write("        'tissue_proportion',\n")
            f.write("        'tumour_proportion',\n")
            f.write("        'clinical_score',\n")
            f.write("        'gleason_4',\n")
            f.write("    ],\n")
            f.write("    axis=1,\n")
            f.write(").copy()\n")
            f.write("\n")

            f.write("y_train = df_train['gleason_4']\n")
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
            f.write("    n_estimators = ")
            f.write("%d" % n_estimator)
            f.write(",\n")
            f.write("    max_depth = ")
            f.write("%d" % max_depth)
            f.write(",\n")
            f.write("    learning_rate = ")
            f.write("%f" % learning_rate)
            f.write(",\n")
            f.write("    subsample = ")
            f.write("%f" % subsample)
            f.write(",\n")
            f.write("    colsample_bytree = ")
            f.write("%f" % colsample_bytree)
            f.write(",\n")
            f.write("    gamma = ")
            f.write("%f" % gamma)
            f.write(",\n")
            f.write("    scale_pos_weight = ")
            f.write("%f" % scale_pos_weight)
            f.write(",\n")
            f.write("    min_child_weight = ")
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
            f.write("    'xgboost_gleason_3_vs_4_quadrant_1000_total_outer_folds_'\n")
            f.write("    + str(")
            f.write("%d" % n_outer_folds)
            f.write(")\n")
            f.write("    + '_outer_fold_test_'\n")
            f.write("    + str(")
            f.write("%d" % outer_fold)
            f.write(").zfill(5)\n")
            f.write("    + '_run_'\n")
            f.write("    + str(")
            f.write("%d" % counter)
            f.write(").zfill(5)\n")
            f.write("    + '_roc_auc_seed_'\n")
            f.write("    + str(model_random_seed)\n")
            f.write("    + '_'\n")
            f.write(")\n")
            f.write("\n")
            f.write("\n")

            f.write("# ============================================================\n")
            f.write("# RFECV\n")
            f.write("# ============================================================\n")
            f.write("\n")
            f.write("tmf.rfecv(\n")
            f.write("    output_dir=output_dir,\n")
            f.write("    df_train=df_train,\n")
            f.write("    df_strat=df_strat,\n")
            f.write("    folds_csv_file=folds_csv_file,\n")
            f.write("    X_train=X_train,\n")
            f.write("    y_train=y_train,\n")
            f.write("    model=xg_classifier,\n")
            f.write("    model_label=\'xgboost\',\n")
            f.write("    save_prefix=save_name,\n")
            f.write("    steps=steps,\n")
            f.write("    scorer=rocauc,\n")
            f.write("    no_folds=no_inner_folds\n")
            f.write(")\n")

            counter += 1

            


