# ============================================================
# IMPORTS
# ============================================================

import numpy as np


# ============================================================
# PATHS AND SETTINGS
# ============================================================

main_path = "/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap/"
n_outer_folds = 5
n_bootstraps = 30
file_input_name = "quad_1000_v8a_gleason_7_nested_5_folds_v8a_run_"

keyfilename = main_path + "/" + file_input_name + "runkey.txt"
counter = 0


# ============================================================
# OUTER FOLDS AND BOOTSTRAPS
# ============================================================

outer_folds = np.arange(n_outer_folds)
bootstraps = np.arange(20, 20 + n_bootstraps)


# ============================================================
# RANDOM FOREST PARAMETER GRID
# ============================================================

n_estimators = [100, 200]
max_depths = [5, 20, 50]
min_samples_splits = [50, 100, 200]
min_samples_leafs = [10, 20, 50]
max_features = [0.1, "sqrt", "log2"]


# ============================================================
# GENERATE RUN FILES
# ============================================================

for bootstrap in bootstraps:

    col_name = f"folds_random_{str(bootstrap).zfill(3)}"

    for outer_fold in outer_folds:

        counter = 0

        file_input_name = (
            "nested_cv_gleason_7_quad_1000_v8a_ambiguous_removed_"
            "forest_outer_bootstrap_"
            + str(bootstrap).zfill(3)
            + "_fold_"
            + str(outer_fold).zfill(5)
            + "_v1_run_"
        )

        for n_estimator in n_estimators:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    for min_samples_leaf in min_samples_leafs:
                        for max_feature in max_features:

                            # ============================================================
                            # RUN KEY
                            # ============================================================

                            with open(keyfilename, "a") as f:
                                f.write(
                                    "Fold_"
                                    + str(outer_fold).zfill(5)
                                    + "_Run_"
                                    + str(counter).zfill(5)
                                )
                                f.write(
                                    " %d %d %s %d %d %s \n"
                                    % (
                                        outer_fold,
                                        n_estimator,
                                        max_depth,
                                        min_samples_split,
                                        min_samples_leaf,
                                        max_feature,
                                    )
                                )

                            filename = (
                                main_path
                                + file_input_name
                                + str(counter).zfill(5)
                                + ".py"
                            )

                            # ============================================================
                            # GENERATE RUN FILE
                            # ============================================================

                            with open(filename, "w") as f:

                                # ============================================================
                                # IMPORTS
                                # ============================================================

                                f.write("import pandas as pd\n")
                                f.write("import numpy as np\n")
                                f.write("\n")
                                f.write("from sklearn.ensemble import RandomForestClassifier\n")
                                f.write("from sklearn.metrics import make_scorer, roc_auc_score\n")
                                f.write("\n")
                                f.write("import tile_model_fitting as tmf\n")
                                f.write("\n")

                                # ============================================================
                                # PATHS
                                # ============================================================

                                f.write(
                                    "main_path = '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/prostate_recurrence/model_evaluation/recurrence_status_tile/rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap/'\n"
                                )
                                f.write(
                                    "output_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/rfe_runs/gleason_7_quad_1000_v8a_ambiguous_removed_bootstrap/'\n"
                                )
                                f.write(
                                    "input_dir = '/nemo/project/proj-sahai-tme-ml/working/processed_data/feature_engineering/clinical/prostate/chiip_cohort/slide_20X/feature_analysis_v3/'\n"
                                )
                                f.write("\n")

                                # ============================================================
                                # PARAMETERS
                                # ============================================================

                                f.write("steps = 10\n")
                                f.write("no_inner_folds = 5\n")
                                f.write("model_random_seed = 31\n")
                                f.write("tumour_proportion = 0.7\n")
                                f.write("tissue_proportion = 0.7\n")
                                f.write("\n")

                                # ============================================================
                                # LOAD AND FILTER DATA
                                # ============================================================

                                f.write(
                                    "data_dir = input_dir + 'updated_tumour_boundary_quadrants_double_are_removed_df.csv'\n"
                                )
                                f.write(
                                    "df = pd.read_csv(data_dir, low_memory=False)\n"
                                )
                                f.write(
                                    "df = df[df['tissue_proportion'] >= tissue_proportion]\n"
                                )
                                f.write(
                                    "df = df[df['tumour_proportion'] >= tumour_proportion]\n"
                                )
                                f.write(
                                    "df = df[(df['gleason'] == 1) | (df['gleason'] == 2)]\n"
                                )
                                f.write("df = df.reset_index(drop=True)\n")
                                f.write("print(df.head())\n")
                                f.write("\n")

                                # ============================================================
                                # CREATE CLINICAL SCORE
                                # ============================================================

                                f.write(
                                    "conditions = [\n"
                                )
                                f.write(
                                    "    (df['gleason'] == 1) & (df['case'] == 1.0),\n"
                                )
                                f.write(
                                    "    (df['gleason'] == 2) & (df['case'] == 1.0),\n"
                                )
                                f.write(
                                    "    (df['gleason'] == 1) & (df['case'] == 0.0),\n"
                                )
                                f.write(
                                    "    (df['gleason'] == 2) & (df['case'] == 0.0)\n"
                                )
                                f.write("]\n")
                                f.write("values = range(0, 4)\n")
                                f.write(
                                    "df['clinical_score'] = np.select(conditions, values)\n"
                                )
                                f.write("\n")

                                # ============================================================
                                # CREATE PATIENT-LEVEL STRATIFICATION DATA
                                # ============================================================

                                f.write(
                                    "patient_ids = list(df.patient_id.unique())\n"
                                )
                                f.write("gleasons = []\n")
                                f.write("cases = []\n")
                                f.write("total_tiles = []\n")
                                f.write("clinical_scores = []\n")
                                f.write("\n")

                                f.write("for patient_id in patient_ids:\n")
                                f.write(
                                    "    gleasons.append(df[df.patient_id == patient_id].gleason.unique()[0])\n"
                                )
                                f.write(
                                    "    cases.append(df[df.patient_id == patient_id].case.unique()[0])\n"
                                )
                                f.write(
                                    "    clinical_scores.append(df[df.patient_id == patient_id].clinical_score.unique()[0])\n"
                                )
                                f.write(
                                    "    total_tiles.append(len(df[df.patient_id == patient_id].case))\n"
                                )
                                f.write("\n")

                                f.write(
                                    "df_patient = pd.DataFrame({'patient_id': patient_ids, 'gleason': gleasons, 'case': cases, 'clinical_score': clinical_scores, 'total_tiles': total_tiles})\n"
                                )
                                f.write(
                                    "df_strat = df_patient.sort_values(by=['clinical_score', 'total_tiles'], ascending=[True, False]).copy().reset_index()\n"
                                )
                                f.write(
                                    f"folds_csv_file = main_path + '{col_name}.csv'\n"
                                )
                                f.write(
                                    "df_strat, predefined_splits, predefined_splits_array = tmf.patient_tile_stratifier(df_strat, df,"
                                )
                                f.write("%d" % n_outer_folds)
                                f.write(", folds_csv_file)\n")
                                f.write("\n")

                                # ============================================================
                                # OUTER FOLD
                                # ============================================================

                                f.write(
                                    "df_train = df[predefined_splits_array != "
                                )
                                f.write("%d" % outer_fold)
                                f.write("]\n")
                                f.write(
                                    "df_train = df_train.reset_index(drop=True)\n"
                                )
                                f.write("\n")

                                # ============================================================
                                # CREATE INNER-FOLD PATIENT STRATIFICATION DATA
                                # ============================================================

                                f.write(
                                    "patient_ids = list(df_train.patient_id.unique())\n"
                                )
                                f.write("gleasons = []\n")
                                f.write("cases = []\n")
                                f.write("clinical_scores = []\n")
                                f.write("total_tiles = []\n")
                                f.write("\n")

                                f.write("for patient_id in patient_ids:\n")
                                f.write(
                                    "    gleasons.append(df_train[df_train.patient_id == patient_id].gleason.unique()[0])\n"
                                )
                                f.write(
                                    "    cases.append(df_train[df_train.patient_id == patient_id].case.unique()[0])\n"
                                )
                                f.write(
                                    "    clinical_scores.append(df_train[df_train.patient_id == patient_id].clinical_score.unique()[0])\n"
                                )
                                f.write(
                                    "    total_tiles.append(len(df_train[df_train.patient_id == patient_id].case))\n"
                                )
                                f.write("\n")

                                f.write(
                                    "df_patient = pd.DataFrame({'patient_id': patient_ids, 'gleason': gleasons, 'case': cases, 'clinical_score': clinical_scores, 'total_tiles': total_tiles})\n"
                                )
                                f.write(
                                    "df_strat = df_patient.sort_values(by=['clinical_score', 'total_tiles'], ascending=[True, False]).copy().reset_index()\n"
                                )
                                f.write(
                                    "folds_csv_file = main_path + 'inner_folds_"
                                )
                                f.write("%d" % outer_fold)
                                f.write("_data.csv'\n")
                                f.write("\n")

                                # ============================================================
                                # PREPARE TRAINING DATA
                                # ============================================================

                                f.write(
                                    "X_train = df_train.drop(['patient_id', 'slide_id', 'area', 'decision', 'case', 'scene', 'tile', 'quadrant', 'clinical_score', 'tissue_proportion', 'tumour_proportion'], axis=1).copy()\n"
                                )
                                f.write(
                                    "y_train = df_train['case']\n"
                                )
                                f.write("\n")

                                # ============================================================
                                # RANDOM FOREST MODEL
                                # ============================================================

                                f.write("rf_classifier = RandomForestClassifier(\n")
                                f.write("    n_estimators = ")
                                f.write("%d" % n_estimator)
                                f.write(",\n")
                                f.write("    max_depth = ")
                                if max_depth is not None:
                                    f.write("%d" % max_depth)
                                else:
                                    f.write("None")
                                f.write(",\n")
                                f.write("    min_samples_split = ")
                                f.write("%d" % min_samples_split)
                                f.write(",\n")
                                f.write("    min_samples_leaf = ")
                                f.write("%d" % min_samples_leaf)
                                f.write(",\n")
                                f.write("    max_features = ")
                                if isinstance(max_feature, str):
                                    f.write("'")
                                    f.write("%s" % max_feature)
                                    f.write("'")
                                else:
                                    f.write("%f" % max_feature)
                                f.write(",\n")
                                f.write("    random_state = 8\n")
                                f.write(")\n")
                                f.write("\n")

                                # ============================================================
                                # SCORING
                                # ============================================================

                                f.write(
                                    "rocauc = make_scorer(roc_auc_score, needs_threshold=True)\n"
                                )
                                f.write("\n")

                                # ============================================================
                                # OUTPUT FILE NAME
                                # ============================================================

                                f.write(
                                    "save_name = 'forest_recurrence_quadrant_1000_total_outer_folds_' + str("
                                )
                                f.write("%d" % n_outer_folds)
                                f.write(
                                    ") + '_bootstrap_' + str("
                                )
                                f.write("%d" % bootstrap)
                                f.write(
                                    ") + '_outer_fold_test_' + str("
                                )
                                f.write("%d" % outer_fold)
                                f.write(
                                    ").zfill(5) + '_run_' + str("
                                )
                                f.write("%d" % counter)
                                f.write(
                                    ").zfill(5) + '_roc_auc_seed_' + str(model_random_seed) + '_'\n"
                                )
                                f.write("\n")

                                f.write("print(X_train)\n")
                                f.write("\n")

                                # ============================================================
                                # RFECV
                                # ============================================================

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
                                f.write("    model=rf_classifier,\n")
                                f.write("    model_label='forest',\n")
                                f.write("    save_prefix=save_name,\n")
                                f.write("    steps=steps,\n")
                                f.write("    scorer=rocauc,\n")
                                f.write("    no_folds=no_inner_folds\n")
                                f.write(")\n")

                            counter += 1