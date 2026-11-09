
# =============================================================================
# Imports
# =============================================================================

import os

import explainability as exp


# =============================================================================
# General paths and settings
# =============================================================================

feature_dir = (
    r'Y:\processed_data\feature_engineering\clinical\prostate'
    r'\chiip_cohort\slide_20X\feature_analysis'
)

gleason_annotation_parent_dir = (
    r'Y:\processed_data\pre_processed_data\clinical\prostate'
    r'\chiip_cohort\overlay_tissue_tumour_gleason_annotations'
)

parent_dir = os.path.join(
    feature_dir,
    'rfe_runs'
)

presentation_dir = (
    r'Y:\paper_draft_figures'
)

fig4_dir = os.path.join(presentation_dir, 'fig4')
fig6_dir = os.path.join(presentation_dir, 'fig6')
suppfig7_dir = os.path.join(presentation_dir, 'suppfig7')

os.makedirs(fig4_dir, exist_ok=True)
os.makedirs(fig6_dir, exist_ok=True)
os.makedirs(suppfig7_dir, exist_ok=True)


# =============================================================================
# Model and analysis settings
# =============================================================================

dir_prefix = (
    'gleason_7_quad_1000_v8a_patient_level_svm_plus_'
    'tile_threshold_f1score_removed_xgboost_outer_fold_'
)

tissue_proportion = 0.7
tumour_proportion = 0.7

lower_tumour_proportion = 0.1
upper_tumour_proportion = 0.9

frequency = 20


# =============================================================================
# 1. Gleason 3+4 versus 4+3 SHAP analysis
# =============================================================================

gleason_df = exp.prepare_gleason_dataset(
    feature_dir=feature_dir,
    gleason_annotation_parent_dir=gleason_annotation_parent_dir,
    tumour_proportion=tumour_proportion,
    tissue_proportion=tissue_proportion
)

gleason_input_dir = os.path.join(
    feature_dir,
    'gleason_3_vs_4_with_sara',
    'xgboost_v2',
    'outer_fold_output'
)

shap_gleason_explanations, all_gleason_features = (
    exp.get_all_shap_features(gleason_input_dir)
)

(
    shap_gleason_explanation,
    shap_gleason_values,
    shap_gleason_data,
    mean_base_value
) = exp.combine_shap_explanations(
    shap_gleason_explanations,
    all_gleason_features
)

gleason_output_path = os.path.join(
    fig4_dir,
    'shap_values_gleason_3_vs_4_10_features.tiff'
)

fig_gleason, ax_gleason, gleason_top_features = (
    exp.plot_shap_summary(
        shap_explanation=shap_gleason_explanation,
        feature_df=gleason_df,
        max_display=10,
        output_path=gleason_output_path,
        x_ticks=(-1, 0, 1, 2),
        bottom_margin=0.3,
        random_seed=0
    )
)


# =============================================================================
# 2. Normal versus tumour SHAP analysis
# =============================================================================

tumour_df = exp.prepare_tumour_proportion_dataset(
    feature_dir=feature_dir,
    tissue_proportion=tissue_proportion,
    lower_tumour_proportion=lower_tumour_proportion,
    upper_tumour_proportion=upper_tumour_proportion
)

tumour_input_dir = os.path.join(
    feature_dir,
    'tumour_proportion_2026',
    'xgboost_v2',
    'outer_fold_output'
)

shap_tumour_explanations, all_tumour_features = (
    exp.get_all_shap_features(tumour_input_dir)
)

(
    shap_tumour_explanation,
    shap_tumour_values,
    shap_tumour_data,
    mean_base_value
) = exp.combine_shap_explanations(
    shap_tumour_explanations,
    all_tumour_features
)

tumour_output_path = os.path.join(
    fig4_dir,
    'shap_values_normal_vs_tumour_10_features.svg'
)

fig_tumour, ax_tumour, tumour_top_features = (
    exp.plot_shap_summary(
        shap_explanation=shap_tumour_explanation,
        feature_df=tumour_df,
        max_display=10,
        output_path=tumour_output_path,
        x_ticks=(-2, -1, 0, 1),
        bottom_margin=0.2,
        random_seed=0
    )
)


# =============================================================================
# 3. Prepare recurrence datasets
# =============================================================================

recur_feature_df = exp.prepare_tumour_feature_dataset(
    input_dir=feature_dir,
    tissue_proportion=tissue_proportion,
    tumour_proportion=tumour_proportion
)

df_all_shuffles = exp.load_shuffle_fold_results(
    parent_dir=parent_dir
)

df_shuffle_probabilities = exp.combine_shuffle_probabilities(
    parent_dir=parent_dir
)

shuffle_groups = exp.select_recurrence_groups(
    df_shuffle_probabilities=df_shuffle_probabilities,
    feature_df=recur_feature_df
)

df_recurrent = shuffle_groups['df_recurrent']
df_non_recurrent = shuffle_groups['df_non_recurrent']
df_gleason_4_non_recurrent = shuffle_groups[
    'df_gleason_4_non_recurrent'
]


# =============================================================================
# 4. Recurrent sample SHAP analysis
# =============================================================================

(
    shap_recur_search_df,
    unique_recur_quadrants_df,
    shap_recur_feature_df
) = exp.prepare_shap_search_datasets(
    df_all_shuffles=df_all_shuffles,
    feature_df=recur_feature_df,
    df_condition=df_recurrent
)

(
    shap_explanations_by_recur_sample,
    all_recur_features
) = exp.get_selected_shap_features(
    shap_search_df=shap_recur_search_df,
    selected_samples_df=df_recurrent,
    parent_dir=parent_dir,
    dir_prefix=dir_prefix
)

recur_shap_output_dir = os.path.join(
    parent_dir,
    'high_recurrence_shap_xgboost'
)

os.makedirs(recur_shap_output_dir, exist_ok=True)

shap_explanations_by_recur_sample = (
    exp.combine_selected_sample_shap_explanations(
        shap_search_df=shap_recur_search_df,
        selected_samples_df=df_recurrent,
        all_features=all_recur_features,
        parent_dir=parent_dir,
        dir_prefix=dir_prefix,
        output_dir=recur_shap_output_dir
    )
)

shap_all_recur_explanation = (
    exp.combine_all_sample_shap_explanations(
        shap_output_dir=recur_shap_output_dir
    )
)

recur_output_path = os.path.join(
    fig6_dir,
    'mean_shap_values_highly_recurrent_tiles_only_10_features.svg'
)

fig_recur, ax_recur, recur_top_features = (
    exp.plot_shap_summary(
        shap_explanation=shap_all_recur_explanation,
        feature_df=recur_feature_df,
        max_display=10,
        output_path=recur_output_path,
        x_ticks=(-1, 0, 1, 2),
        left_margin=0.7,
        bottom_margin=0.2
    )
)


# =============================================================================
# 5. Compare recurrence-specific SHAP feature importance
# =============================================================================

recur_comparison_df = exp.compare_shap_feature_specificity(
    condition_explanation=shap_all_recur_explanation,
    tumour_explanation=shap_tumour_explanation,
    gleason_explanation=shap_gleason_explanation
)

recur_specificity_output_path = os.path.join(
    fig6_dir,
    'recurrent_shap_feature_specificity.svg'
)

exp.plot_shap_specificity(
    comparison_df=recur_comparison_df,
    condition='recur',
    top_n=10,
    output_path=recur_specificity_output_path
)


# =============================================================================
# 6. Identify external neighbouring quadrants
# =============================================================================

external_neighbours_df = exp.get_external_neighbours(
    df_shuffle_probabilities=df_shuffle_probabilities,
    df_recurrent=df_recurrent
)

external_neighbours_binned_df = (
    exp.calculate_neighbour_cell_frequencies(
        external_neighbours_df
    )
)


# =============================================================================
# 7. Prepare neighbour SHAP inputs
# =============================================================================

(
    selected_neighbours_df,
    neighbour_shap_search_df,
    neighbour_feature_df,
    unique_neighbours_df,
    all_neighbour_features,
    frequency_label,
    neighbour_shap_output_dir
) = exp.prepare_neighbour_shap_inputs(
    external_neighbours_df=external_neighbours_binned_df,
    df_all_shuffles=df_all_shuffles,
    feature_df=recur_feature_df,
    parent_dir=parent_dir,
    frequency=frequency,
    dir_prefix=dir_prefix
)

os.makedirs(neighbour_shap_output_dir, exist_ok=True)


# =============================================================================
# 8. Neighbour SHAP analysis
# =============================================================================

shap_explanations_by_neighbour = (
    exp.combine_selected_sample_shap_explanations(
        shap_search_df=neighbour_shap_search_df,
        selected_samples_df=unique_neighbours_df,
        all_features=all_neighbour_features,
        parent_dir=parent_dir,
        output_dir=neighbour_shap_output_dir,
        dir_prefix=dir_prefix
    )
)

shap_all_neighbour_explanation = (
    exp.combine_all_sample_shap_explanations(
        shap_output_dir=neighbour_shap_output_dir
    )
)


# =============================================================================
# 9. Compare recurrent versus neighbouring SHAP importance
# =============================================================================

recur_neighbour_comparison_df = (
    exp.compare_recur_neighbour_shap_importance(
        shap_all_recur_explanation=shap_all_recur_explanation,
        shap_all_neighbour_explanation=shap_all_neighbour_explanation
    )
)


# =============================================================================
# 10. Plot recurrent versus neighbouring feature comparison
# =============================================================================

recur_neighbour_output_df = (
    exp.plot_recur_neighbour_shap_comparison(
        comparison_df=recur_neighbour_comparison_df,
        recur_feature_df=recur_feature_df,
        df_recurrent=df_recurrent,
        neighbour_feature_df=neighbour_feature_df,
        str_freq=frequency_label,
        output_dir=suppfig7_dir,
        top_n=5
    )
)
