import os
import numpy as np

import outer_and_inner_fold_performance_generator as pg


# ============================================================
# SETTINGS
# ============================================================

n_outer_folds = 5
subfile_name = 'f1_score_increment_20'
f1_score_thresholds = np.linspace(0.9, 0.1, 5)


# ============================================================
# GENERATE FOR EACH BOOTSTRAP
# ============================================================

for bootstrap in range(20, 50):

    print(f'\n{"=" * 60}')
    print(f'BOOTSTRAP {bootstrap}')
    print(f'{"=" * 60}\n')


    # ========================================================
    # PATHS
    # ========================================================

    run_name = (
        'gleason_7_quad_1000_v8a_patient_level_logistic_removed_xgboost_'
        'psa_tstage_only_outer_fold_'
        f'{bootstrap}'
    )

    main_path = (
        '/nemo/project/proj-sahai-tme-ml/working/codebase/local/module/'
        'prostate_recurrence/model_evaluation/recurrence_status_tile/'
        'rfe_runs/'
        f'{run_name}/'
    )

    patient_output_dir = (
        '/nemo/project/proj-sahai-tme-ml/working/processed_data/'
        'feature_engineering/clinical/prostate/chiip_cohort/slide_20X/'
        'feature_analysis_v3/rfe_runs/'
        f'{run_name}/'
    )

    keyfilename = os.path.join(
        main_path,
        f'{run_name}_fold_runkey.txt'
    )


    # ========================================================
    # GENERATE LOGISTIC MODEL SCRIPTS
    # ========================================================

    penalties = ['l1', 'l2', 'elasticnet']
    Cs = [10**x for x in range(-4, 5)]
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for outer_fold in range(n_outer_folds):

        counter = 0

        file_input_name = (
            'nested_cv_'
            f'{run_name}_outer_fold_{outer_fold:05d}_v1_run_'
        )

        fold_dir = os.path.join(
            patient_output_dir,
            f'fold_{outer_fold:05d}'
        )

        os.makedirs(fold_dir, exist_ok=True)

        for penalty in penalties:
            for C in Cs:

                valid_solvers = (
                    ['saga']
                    if penalty == 'elasticnet'
                    else ['liblinear', 'saga']
                )

                for solver in valid_solvers:

                    if penalty == 'elasticnet':
                        for l1_ratio in l1_ratios:

                            counter = generate_logistic_scripts(
                                keyfilename,
                                outer_fold,
                                counter,
                                penalty,
                                C,
                                solver,
                                l1_ratio,
                                main_path,
                                file_input_name,
                                patient_output_dir,
                                bootstrap,
                                fold_dir
                            )

                    else:

                        counter = generate_logistic_scripts(
                            keyfilename,
                            outer_fold,
                            counter,
                            penalty,
                            C,
                            solver,
                            None,
                            main_path,
                            file_input_name,
                            patient_output_dir,
                            bootstrap,
                            fold_dir
                        )


    # ========================================================
    # GENERATE OUTER-FOLD PERFORMANCE CSV
    # ========================================================

    python_code_dir = os.path.join(
        main_path,
        f'{run_name}_fold_runkey.txt'
    )

    pg.generate_performance_csv(
        python_code_dir,
        patient_output_dir
    )


    # ========================================================
    # GENERATE INNER-FOLD PERFORMANCE
    # ========================================================

    for fold_n in range(n_outer_folds):

        pg.generate_inner_outer_scores_for_graded_f1_score_psa_only(
            bootstrap,
            fold_n=fold_n,
            subfile_name=subfile_name,
            f1_score_thresholds=f1_score_thresholds
        )


    # ========================================================
    # GENERATE INNER OUT-OF-FOLD PATIENT PROBABILITIES
    # ========================================================

    for fold_n in range(n_outer_folds):

        pg.generate_inner_probabilities_for_optimum_psa_inner_model(
            bootstrap,
            fold_n,
            subfile_name,
            f1_score_thresholds
        )


    # ========================================================
    # GENERATE FINAL OUTER-TEST PATIENT PROBABILITIES
    # ========================================================
    #generate better function names
    for fold_n in range(n_outer_folds):

        pg.generate_final_model_from_inner_psa_predicted_probabilities(
            bootstrap,
            fold_n,
            subfile_name,
            f1_score_thresholds
        )

