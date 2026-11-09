import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import shap

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize




def prepare_gleason_dataset(
    feature_dir,
    gleason_annotation_parent_dir,
    tumour_proportion=0.7,
    tissue_proportion=0.7,
    return_intermediates=False
):
    """
    Prepare the Gleason 3 vs Gleason 4 dataset.

    Processing:
        1. Load feature data.
        2. Assign clinical scores.
        3. Filter Gleason 6/8 and Gleason 7 tiles.
        4. Load and combine Gleason annotation files.
        5. Filter annotation data by tissue and tumour proportions.
        6. Merge Gleason 7 features with annotation scores.
        7. Assign binary Gleason 3/4 classifications.
        8. Combine the Gleason groups.

    Returns:
        gleason_df, or a dictionary of intermediate DataFrames
        if return_intermediates=True.
    """

    # -------------------------------------------------------------
    # 1. Load feature data
    # -------------------------------------------------------------

    data_file = os.path.join(
        feature_dir,
        'updated_tumour_boundary_quadrants_double_are_recurrent_df.csv'
    )

    feature_df = pd.read_csv(
        data_file,
        low_memory=False
    ).reset_index(drop=True)


    # -------------------------------------------------------------
    # 2. Assign clinical scores
    # -------------------------------------------------------------

    clinical_score_conditions = [
        (feature_df['gleason'] == gleason) &
        (feature_df['case'] == case)
        for case in (1.0, 0.0)
        for gleason in range(4)
    ]

    feature_df['clinical_score'] = np.select(
        clinical_score_conditions,
        range(8)
    )


    # -------------------------------------------------------------
    # 3. Filter Gleason groups
    # -------------------------------------------------------------

    quality_filter = (
        (feature_df['tissue_proportion'] > tissue_proportion) &
        (feature_df['tumour_proportion'] > tumour_proportion)
    )

    # Gleason 6 and 8
    gleason_6_8_df = feature_df[
        quality_filter &
        feature_df['gleason'].isin([0, 3])
    ].copy()

    gleason_6_8_df['gleason_4'] = (
        gleason_6_8_df['gleason'] == 3
    ).astype(int)

    gleason_6_8_df['gleason_3'] = (
        gleason_6_8_df['gleason'] == 0
    ).astype(int)


    # Gleason 7
    gleason_7_df = feature_df[
        quality_filter &
        feature_df['gleason'].isin([1, 2])
    ].copy()


    # -------------------------------------------------------------
    # 4. Load Gleason annotation data
    # -------------------------------------------------------------

    annotation_files = [
        os.path.join(gleason_annotation_parent_dir, filename)
        for filename in os.listdir(gleason_annotation_parent_dir)
        if filename.endswith('_1000.csv')
    ]

    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation files ending in '_1000.csv' found in "
            f"{gleason_annotation_parent_dir}"
        )

    annotation_dfs = [
        pd.read_csv(file_path)
        for file_path in annotation_files
    ]

    combined_annotation_df = pd.concat(
        annotation_dfs,
        ignore_index=True
    )

    combined_annotation_df.rename(
        columns={
            'tumour_gleason4_fraction_of_tissue_area': 'gleason_4_proportion',
            'tumour_gleason3_fraction_of_tissue_area': 'gleason_3_proportion',
        },
        inplace=True
    )


    # -------------------------------------------------------------
    # 5. Filter annotation data
    # -------------------------------------------------------------

    gleason_scoring_df = combined_annotation_df[
        (
            combined_annotation_df['tissue_fraction_of_tile_area']
            > tissue_proportion
        ) &
        (
            (
                combined_annotation_df['gleason_4_proportion']
                > tumour_proportion
            ) |
            (
                combined_annotation_df['gleason_3_proportion']
                > tumour_proportion
            )
        )
    ].copy()


    # -------------------------------------------------------------
    # 6. Merge Gleason 7 features with annotation scores
    # -------------------------------------------------------------

    merge_columns = [
        'slide_id',
        'scene',
        'tile',
        'quadrant'
    ]

    merged_7_df = gleason_7_df.merge(
        gleason_scoring_df,
        on=merge_columns,
        suffixes=('_df7', '_scoring'),
        how='inner'
    )

    columns_to_drop = [
        'tumour_total_fraction_of_tissue_area',
        'tumour_other_fraction_of_tissue_area',
        'tissue_fraction_of_tile_area',
        'row',
        'col',
        'quadrant_row',
        'quadrant_col',
    ]

    merged_7_df.drop(
        columns=columns_to_drop,
        inplace=True
    )


    # -------------------------------------------------------------
    # 7. Assign binary Gleason classifications
    # -------------------------------------------------------------

    # Explicit 50% threshold rather than np.round()
    merged_7_df['gleason_4'] = (
        merged_7_df['gleason_4_proportion'] >= 0.5
    ).astype(int)

    merged_7_df['gleason_3'] = (
        merged_7_df['gleason_3_proportion'] >= 0.5
    ).astype(int)

    merged_7_df.drop(
        columns=[
            'gleason_4_proportion',
            'gleason_3_proportion'
        ],
        inplace=True
    )


    # -------------------------------------------------------------
    # 8. Combine Gleason groups
    # -------------------------------------------------------------

    gleason_df = pd.concat(
        [merged_7_df, gleason_6_8_df],
        ignore_index=True
    )


    # -------------------------------------------------------------
    # 9. Return results
    # -------------------------------------------------------------

    if return_intermediates:
        return {
            'feature_df': feature_df,
            'gleason_6_8_df': gleason_6_8_df,
            'gleason_7_df': gleason_7_df,
            'combined_annotation_df': combined_annotation_df,
            'gleason_scoring_df': gleason_scoring_df,
            'merged_7_df': merged_7_df,
            'gleason_df': gleason_df
        }

    return gleason_df



def prepare_tumour_proportion_dataset(
    feature_dir,
    tissue_proportion=0.7,
    lower_tumour_proportion=0.1,
    upper_tumour_proportion=0.9,
    return_intermediates=False
):
    """
    Prepare the Gleason 7 dataset for tumour proportion analysis.

    Processing:
        1. Load feature data.
        2. Filter by tissue proportion.
        3. Retain tiles with low or high tumour proportions.
        4. Retain Gleason 7 tiles.
        5. Assign clinical scores.
        6. Recode tumour proportions as binary values.

    Returns:
        df, or a dictionary of intermediate results
        if return_intermediates=True.
    """

    # -------------------------------------------------------------
    # 1. Load feature data
    # -------------------------------------------------------------

    data_file = os.path.join(
        feature_dir,
        'updated_tumour_boundary_quadrants_double_are_recurrent_df.csv'
    )

    feature_df = pd.read_csv(
        data_file,
        low_memory=False
    )


    # -------------------------------------------------------------
    # 2. Filter feature data
    # -------------------------------------------------------------

    tissue_filter = (
        feature_df['tissue_proportion'] >= tissue_proportion
    )

    tumour_filter = (
        (feature_df['tumour_proportion'] >= upper_tumour_proportion) |
        (feature_df['tumour_proportion'] <= lower_tumour_proportion)
    )

    gleason_filter = feature_df['gleason'].isin([1, 2])

    df = feature_df[
        tissue_filter & tumour_filter & gleason_filter
    ].copy()

    df.reset_index(drop=True, inplace=True)


    # -------------------------------------------------------------
    # 3. Extract patient IDs
    # -------------------------------------------------------------

    patient_ids = list(df['patient_id'].unique())


    # -------------------------------------------------------------
    # 4. Assign clinical scores
    # -------------------------------------------------------------

    clinical_score_conditions = [
        (df['gleason'] == gleason) &
        (df['case'] == case)
        for case in (1.0, 0.0)
        for gleason in (1, 2)
    ]

    df['clinical_score'] = np.select(
        clinical_score_conditions,
        range(4)
    )


    # -------------------------------------------------------------
    # 5. Recode tumour proportions
    # -------------------------------------------------------------

    df.loc[
        df['tumour_proportion'] <= lower_tumour_proportion,
        'tumour_proportion'
    ] = 0

    df.loc[
        df['tumour_proportion'] >= upper_tumour_proportion,
        'tumour_proportion'
    ] = 1


    # -------------------------------------------------------------
    # 6. Return results
    # -------------------------------------------------------------

    if return_intermediates:
        return {
            'feature_df': feature_df,
            'df': df,
            'patient_ids': patient_ids
        }

    return df



def get_all_shap_features(input_dir, n_folds=5):
    shap_explanations = []
    all_features = set()

    for fold in range(n_folds):
        filename = os.path.join(
            input_dir,
            f'outer_fold_{fold:05d}_xgboost_shap_output.pkl'
        )

        with open(filename, 'rb') as f:
            shap_explanation = pickle.load(f)

        shap_explanations.append(shap_explanation)
        all_features.update(shap_explanation.feature_names)

    return shap_explanations, sorted(all_features)



def combine_shap_explanations(shap_explanations, all_features):
    shap_all_values = []
    shap_all_data = []
    base_values_all = []

    for explanation in shap_explanations:
        values = explanation.values
        data = explanation.data
        feature_names = explanation.feature_names

        # Handle 3D multi-class SHAP values
        if values.ndim == 3:
            n_classes, n_samples, n_features = values.shape
            values = values.transpose(1, 0, 2).reshape(
                n_classes * n_samples, n_features
            )
            data = np.repeat(data, n_classes, axis=0)

        # Align features to the unified feature order
        feature_idx_map = {
            feature: i for i, feature in enumerate(feature_names)
        }

        unified_values = np.zeros((values.shape[0], len(all_features)))
        unified_data = np.zeros((data.shape[0], len(all_features)))

        for j, feature in enumerate(all_features):
            if feature in feature_idx_map:
                idx = feature_idx_map[feature]
                unified_values[:, j] = values[:, idx]
                unified_data[:, j] = data[:, idx]

        shap_all_values.append(unified_values)
        shap_all_data.append(unified_data)
        base_values_all.append(explanation.base_values)

    # Combine folds
    shap_all_values = np.vstack(shap_all_values)
    shap_all_data = np.vstack(shap_all_data)
    mean_base_value = np.mean(base_values_all)

    # Create unified SHAP explanation
    shap_all_explanation = shap.Explanation(
        values=shap_all_values,
        base_values=mean_base_value,
        data=shap_all_data,
        feature_names=all_features
    )

    print(f"Combined SHAP array shape: {shap_all_values.shape}")
    print(f"Mean base value: {mean_base_value:.4f}")

    return shap_all_explanation, shap_all_values, shap_all_data, mean_base_value



def plot_shap_summary(
    shap_explanation,
    feature_df,
    max_display=10,
    output_path=None,
    output_format=None,
    x_ticks=(-1, 0, 1, 2),
    left_margin=0.7,
    bottom_margin=0.3,
    random_seed=0
):
    # Extract SHAP values and feature data
    shap_values = shap_explanation.values
    feature_data = shap_explanation.data
    feature_names = shap_explanation.feature_names

    # Compute padded feature ranges from the full dataset
    feature_mins = feature_df[feature_names].min().values
    feature_maxs = feature_df[feature_names].max().values

    feature_ranges = {}

    for i, feature in enumerate(feature_names):
        feature_range = feature_maxs[i] - feature_mins[i]
        feature_ranges[feature] = (
            feature_mins[i] - 0.1 * feature_range,
            feature_maxs[i] + 0.1 * feature_range
        )

    # Select top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-max_display:]

    # Figure setup
    fig, ax = plt.subplots(
        figsize=(10, max_display * 0.4 + 2)
    )

    rng = np.random.default_rng(random_seed)

    # Plot each feature
    for rank, feature_idx in enumerate(top_features_idx):
        feature = feature_names[feature_idx]
        shap_col = shap_values[:, feature_idx]
        feature_col = feature_data[:, feature_idx]

        # Normalise feature values for colouring
        vmin, vmax = feature_ranges[feature]
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        colors = cm.plasma(norm(feature_col))

        # Add vertical jitter
        y_jitter = rng.normal(
            loc=rank,
            scale=0.1,
            size=len(shap_col)
        )

        ax.scatter(
            shap_col,
            y_jitter,
            color=colors,
            s=25,
            edgecolor='k',
            linewidth=0.0,
            alpha=0.8,
            zorder=3,
            rasterized=True
        )

    # Axes formatting
    ax.set_yticks(range(len(top_features_idx)))
    ax.set_yticklabels(
        [feature_names[i] for i in top_features_idx],
        fontsize=20,
        fontweight='bold'
    )

    ax.set_xlabel(
        "SHAP value",
        fontsize=20,
        fontweight='bold'
    )

    ax.grid(
        axis='x',
        linestyle='--',
        alpha=0.5,
        zorder=1
    )

    ax.tick_params(axis='x', labelsize=20)
    ax.set_xticks(x_ticks)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    ax.tick_params(width=2.5, length=6)

    # Layout
    plt.subplots_adjust(
        left=left_margin,
        bottom=bottom_margin
    )

    # Save figure if requested
    if output_path is not None:
        if output_format is None:
            output_format = output_path.rsplit('.', 1)[-1].lower()

        if output_format == 'svg':
            mpl.rcParams['svg.fonttype'] = 'none'

        fig.savefig(
            output_path,
            format=output_format,
            bbox_inches='tight',
            dpi=600 if output_format in ('tiff', 'tif') else 300,
            metadata={'Date': None} if output_format == 'svg' else None
        )

    plt.show()

    return fig, ax, [feature_names[i] for i in top_features_idx]






def prepare_tumour_feature_dataset(
    input_dir,
    tissue_proportion=0.7,
    tumour_proportion=0.7
):
    data_file = os.path.join(
        input_dir,
        'quadrant_features.csv'
    )

    feature_df = pd.read_csv(data_file, low_memory=False)

    # Filter by tissue and tumour proportions
    feature_df = feature_df[
        (feature_df['tissue_proportion'] >= tissue_proportion) &
        (feature_df['tumour_proportion'] >= tumour_proportion) &
        (feature_df['gleason'].isin([1, 2]))
    ].copy()

    feature_df.reset_index(drop=True, inplace=True)

    # Assign clinical scores
    conditions = [
        (feature_df['gleason'] == gleason) & (feature_df['case'] == case)
        for case in (1.0, 0.0)
        for gleason in (1, 2)
    ]

    feature_df['clinical_score'] = np.select(conditions, range(4))

    return feature_df



def load_shuffle_fold_results(
    parent_dir,
    shuffle_start=20,
    shuffle_end=50,
    n_folds=5,
    shuffle_dir_prefix='gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_f1score_removed_outer_fold_',
    columns_to_drop=None
):
    """
    Load and combine tile-level recurrence probabilities across
    shuffle iterations and outer folds.

    Returns:
        df_all_shuffles: Combined DataFrame containing all shuffle/fold results.
    """

    if columns_to_drop is None:
        columns_to_drop = [
            'area',
            'decision',
            'clinical_score',
            'tissue_proportion',
            'tumour_proportion',
            'prob_recurrence'
        ]

    all_shuffles = []

    for shuffle in range(shuffle_start, shuffle_end):

        shuffle_dir = (
            f'{shuffle_dir_prefix}{shuffle}'
        )

        for fold in range(n_folds):

            fold_file = (
                f'outer_fold_{fold:05d}'
                f'tile_level_recur_probabilities.csv'
            )

            file_path = os.path.join(
                parent_dir,
                shuffle_dir,
                fold_file
            )

            # Load fold results
            df_shuffle_fold = pd.read_csv(
                file_path,
                delimiter=','
            )

            # Remove unnecessary columns
            df_shuffle_fold.drop(
                columns=columns_to_drop,
                inplace=True
            )

            # Record original index and identifiers
            df_shuffle_fold['idx'] = df_shuffle_fold.index
            df_shuffle_fold['fold'] = fold
            df_shuffle_fold['shuffle'] = shuffle

            all_shuffles.append(df_shuffle_fold)

    # Combine all shuffle/fold results
    df_all_shuffles = pd.concat(
        all_shuffles,
        ignore_index=True
    )

    return df_all_shuffles



def combine_shuffle_probabilities(
    parent_dir,
    shuffle_start=20,
    shuffle_end=50,
    n_folds=5,
    shuffle_dir_prefix=(
        'gleason_7_quad_1000_v8a_patient_level_svm_plus_tile_threshold_f1score_removed'
        '_xgboost_outer_fold_'
    )
):
    """
    Load and merge tile-level recurrence probabilities across
    shuffle iterations and outer folds.

    Calculates the mean and standard deviation of recurrence
    probabilities across shuffle iterations.

    Returns:
        df_shuffle_probabilities: Combined DataFrame containing
        individual shuffle probabilities and summary statistics.
    """

    merge_columns = [
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

    shuffle_dfs = []

    # -------------------------------------------------------------
    # 1. Load and combine folds within each shuffle
    # -------------------------------------------------------------

    for shuffle in range(shuffle_start, shuffle_end):

        print(shuffle)

        shuffle_dir = os.path.join(
            parent_dir,
            f'{shuffle_dir_prefix}{shuffle}'
        )

        fold_dfs = []

        for fold in range(n_folds):

            filename = (
                f'outer_fold_{fold:05d}'
                f'tile_level_recur_probabilities_xgboost.csv'
            )

            file_path = os.path.join(
                shuffle_dir,
                filename
            )

            df_fold = pd.read_csv(
                file_path,
                low_memory=False
            )

            df_fold.rename(
                columns={
                    'prob_recurrence':
                    f'prob_recurrence_shuffle_{shuffle}'
                },
                inplace=True
            )

            fold_dfs.append(df_fold)

        df_shuffle = pd.concat(
            fold_dfs,
            ignore_index=True
        )

        shuffle_dfs.append(df_shuffle)


    # -------------------------------------------------------------
    # 2. Merge shuffle iterations
    # -------------------------------------------------------------

    df_shuffle_probabilities = shuffle_dfs[0]

    for df_shuffle in shuffle_dfs[1:]:
        df_shuffle_probabilities = df_shuffle_probabilities.merge(
            df_shuffle,
            on=merge_columns,
            how='inner'
        )


    # -------------------------------------------------------------
    # 3. Calculate summary statistics
    # -------------------------------------------------------------

    probability_columns = [
        f'prob_recurrence_shuffle_{shuffle}'
        for shuffle in range(shuffle_start, shuffle_end)
    ]

    df_shuffle_probabilities['mean_prob'] = (
        df_shuffle_probabilities[probability_columns].mean(axis=1)
    )

    df_shuffle_probabilities['std_prob'] = (
        df_shuffle_probabilities[probability_columns].std(axis=1)
    )

    return df_shuffle_probabilities



def select_recurrence_groups(
    df_shuffle_probabilities,
    feature_df,
    merge_columns=None
):
    """
    Select recurrence and non-recurrence groups based on
    mean and standard deviation of shuffled recurrence probabilities.

    Also merges the selected groups with the original feature data.

    Returns:
        Dictionary containing the selected and merged DataFrames.
    """

    if merge_columns is None:
        merge_columns = [
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

    # -------------------------------------------------------------
    # 1. Select recurrence groups
    # -------------------------------------------------------------

    df_recurrent = df_shuffle_probabilities[
        (df_shuffle_probabilities['mean_prob'] > 0.7) &
        (df_shuffle_probabilities['std_prob'] < 0.08)
    ].copy()


    # -------------------------------------------------------------
    # 2. Select non-recurrence groups
    # -------------------------------------------------------------

    df_non_recurrent = df_shuffle_probabilities[
        (df_shuffle_probabilities['mean_prob'] <= 0.2) &
        (df_shuffle_probabilities['std_prob'] < 0.1)
    ].copy()


    # -------------------------------------------------------------
    # 3. Select Gleason 4 tiles
    # -------------------------------------------------------------

    df_gleason_4 = df_shuffle_probabilities[
        df_shuffle_probabilities['clinical_score'].isin([1, 3])
    ].copy()


    # -------------------------------------------------------------
    # 4. Select Gleason 4 non-recurrent tiles
    # -------------------------------------------------------------

    df_gleason_4_non_recurrent = df_gleason_4[
        (df_gleason_4['mean_prob'] <= 0.6) &
        (df_gleason_4['std_prob'] < 0.1)
    ].copy()


    # -------------------------------------------------------------
    # 5. Merge selected groups with original feature data
    # -------------------------------------------------------------

    df_recurrent = feature_df.merge(
        df_recurrent,
        on=merge_columns,
        how='inner'
    )

    df_non_recurrent = feature_df.merge(
        df_non_recurrent,
        on=merge_columns,
        how='inner'
    )

    df_gleason_4_non_recurrent = feature_df.merge(
        df_gleason_4_non_recurrent,
        on=merge_columns,
        how='inner'
    )


    # -------------------------------------------------------------
    # 6. Return results
    # -------------------------------------------------------------

    return {
        'df_recurrent': df_recurrent,
        'df_non_recurrent': df_non_recurrent,
        'df_gleason_4': df_gleason_4,
        'df_gleason_4_non_recurrent': df_gleason_4_non_recurrent
    }

def prepare_shap_search_datasets(df_all_shuffles, feature_df, df_condition):
    merge_columns = [
        'patient_id',
        'slide_id',
        'case',
        'scene',
        'tile',
        'quadrant'
    ]

    condition_quadrants = df_condition[merge_columns].drop_duplicates()

    # SHAP results restricted to condition quadrants
    shap_search_df = df_all_shuffles.merge(
        condition_quadrants,
        on=merge_columns,
        how='inner'
    )

    # Unique recurrent quadrants
    unique_quadrants_df = condition_quadrants.reset_index(drop=True)

    # Original feature data restricted to recurrent quadrants
    shap_feature_df = feature_df.merge(
        condition_quadrants,
        on=merge_columns,
        how='inner'
    )

    return shap_search_df, unique_quadrants_df, shap_feature_df

def get_selected_shap_features(
    shap_search_df,
    selected_samples_df,
    parent_dir,
    dir_prefix=(
        'gleason_7_quad_1000_v8a_patient_level_svm_plus_'
        'tile_threshold_f1score_removed_xgboost_outer_fold_'
    )
):
    all_features = set()
    shap_explanations_by_sample = {}

    sample_columns = ['slide_id', 'scene', 'tile', 'quadrant']

    for sample in selected_samples_df[sample_columns].itertuples(
        index=False,
        name=None
    ):
        slide_id, scene, tile, quadrant = sample

        shap_sample_df = shap_search_df[
            (shap_search_df['slide_id'] == slide_id) &
            (shap_search_df['scene'] == scene) &
            (shap_search_df['tile'] == tile) &
            (shap_search_df['quadrant'] == quadrant)
        ]

        shap_explanations = []

        for row in shap_sample_df.itertuples(index=False):
            shuffle = row.shuffle
            fold = row.fold

            input_dir = os.path.join(
                parent_dir,
                f'{dir_prefix}{shuffle}'
            )

            filename = os.path.join(
                input_dir,
                f'outer_fold_{fold:05d}_xgboost_shap_output.pkl'
            )

            with open(filename, 'rb') as f:
                shap_explanation = pickle.load(f)

            shap_explanations.append(shap_explanation)
            all_features.update(shap_explanation.feature_names)

        shap_explanations_by_sample[sample] = shap_explanations

    return shap_explanations_by_sample, sorted(all_features)

def combine_selected_sample_shap_explanations(
    shap_search_df,
    selected_samples_df,
    all_features,
    parent_dir,
    output_dir,
    dir_prefix=(
        'gleason_7_quad_1000_v8a_patient_level_svm_plus_'
        'tile_threshold_f1score_removed_xgboost_outer_fold_'
    )
):
    os.makedirs(output_dir, exist_ok=True)

    sample_columns = [
        'patient_id', 'slide_id', 'case',
        'scene', 'tile', 'quadrant'
    ]

    shap_explanations_by_sample = {}

    # Map unified feature names to their positions
    feature_indices = {
        feature: idx
        for idx, feature in enumerate(all_features)
    }

    # Process each selected sample
    for sample_idx, sample in enumerate(
        selected_samples_df[sample_columns].itertuples(
            index=False,
            name=None
        )
    ):
        patient_id, slide_id, case, scene, tile, quadrant = sample

        print(f'Processing sample {sample_idx}...')

        # Select SHAP results for this sample
        sample_mask = np.ones(
            len(shap_search_df),
            dtype=bool
        )

        for column, value in zip(sample_columns, sample):
            sample_mask &= (
                shap_search_df[column] == value
            ).to_numpy()

        shap_sample_df = shap_search_df.loc[sample_mask]

        n_models = len(shap_sample_df)
        n_features = len(all_features)

        shap_values = np.zeros((n_models, n_features))
        feature_data = np.zeros((n_models, n_features))
        base_values = []

        # Load and combine SHAP values across shuffle/fold models
        for model_idx, row in enumerate(
            shap_sample_df.itertuples(index=False)
        ):
            input_dir = os.path.join(
                parent_dir,
                f'{dir_prefix}{row.shuffle}'
            )

            filename = os.path.join(
                input_dir,
                f'outer_fold_{row.fold:05d}_xgboost_shap_output.pkl'
            )

            with open(filename, 'rb') as f:
                shap_explanation = pickle.load(f)

            sample_shap_values = shap_explanation.values[row.idx, :]
            sample_feature_data = shap_explanation.data[row.idx, :]

            # Align SHAP values using feature names
            for source_idx, feature in enumerate(
                shap_explanation.feature_names
            ):
                if feature in feature_indices:
                    target_idx = feature_indices[feature]

                    shap_values[model_idx, target_idx] = (
                        sample_shap_values[source_idx]
                    )

                    feature_data[model_idx, target_idx] = (
                        sample_feature_data[source_idx]
                    )

            # Collect model base value
            base_value = shap_explanation.base_values

            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.asarray(base_value).flatten()[0]

            base_values.append(base_value)

        # Create combined SHAP explanation
        shap_explanation_combined = shap.Explanation(
            values=shap_values,
            base_values=np.mean(base_values),
            data=feature_data,
            feature_names=all_features
        )

        shap_explanations_by_sample[sample] = shap_explanation_combined

        # Save combined explanation
        filename = (
            f'slide_{slide_id}_scene_{scene}_tile_{tile}_'
            f'quadrant_{quadrant}_xgboost_shap_output.pkl'
        )

        save_path = os.path.join(output_dir, filename)

        with open(save_path, 'wb') as f:
            pickle.dump(shap_explanation_combined, f)

        print(
            f'Saved SHAP data for slide {slide_id}, '
            f'scene {scene}, tile {tile}, quadrant {quadrant}'
        )

    return shap_explanations_by_sample



def combine_all_sample_shap_explanations(shap_output_dir):

    shap_values_all = []
    feature_data_all = []
    base_values_all = []

    feature_names = None

    pkl_files = glob(os.path.join(shap_output_dir, '*.pkl'))

    if not pkl_files:
        raise FileNotFoundError(
            f'No SHAP files found in {shap_output_dir}'
        )

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            shap_explanation = pickle.load(f)

        shap_values = shap_explanation.values
        feature_data = shap_explanation.data

        # Handle multiclass SHAP explanations
        if shap_values.ndim == 3:
            n_classes, n_samples, n_features = shap_values.shape

            shap_values = shap_values.transpose(1, 0, 2).reshape(
                n_classes * n_samples,
                n_features
            )

            feature_data = np.repeat(
                feature_data,
                n_classes,
                axis=0
            )

        shap_values_all.append(shap_values)
        feature_data_all.append(feature_data)
        base_values_all.append(shap_explanation.base_values)

        if feature_names is None:
            feature_names = shap_explanation.feature_names

    # Combine all samples
    shap_values_all = np.vstack(shap_values_all)
    feature_data_all = np.vstack(feature_data_all)
    mean_base_value = np.mean(base_values_all)

    # Create combined SHAP explanation
    shap_all_samples_explanation = shap.Explanation(
        values=shap_values_all,
        base_values=mean_base_value,
        data=feature_data_all,
        feature_names=feature_names
    )

    print(f'Loaded {len(pkl_files)} SHAP files')
    print(f'Combined SHAP array shape: {shap_values_all.shape}')
    print(f'Mean base value: {mean_base_value:.4f}')

    return shap_all_samples_explanation



def compare_shap_feature_specificity(
    condition_explanation,
    tumour_explanation,
    gleason_explanation
):
    # Calculate mean absolute SHAP values
    condition_mean_abs = np.abs(condition_explanation.values).mean(axis=0)
    tumour_mean_abs = np.abs(tumour_explanation.values).mean(axis=0)
    gleason_mean_abs = np.abs(gleason_explanation.values).mean(axis=0)

    # Normalise within each model
    condition_norm = condition_mean_abs / condition_mean_abs.sum()
    tumour_norm = tumour_mean_abs / tumour_mean_abs.sum()
    gleason_norm = gleason_mean_abs / gleason_mean_abs.sum()

    # Create feature-to-importance dictionaries
    condition_dict = dict(zip(
        condition_explanation.feature_names, condition_norm
    ))
    tumour_dict = dict(zip(
        tumour_explanation.feature_names, tumour_norm
    ))
    gleason_dict = dict(zip(
        gleason_explanation.feature_names, gleason_norm
    ))

    # Compare features present in the condition explanation
    rows = []

    for feature in condition_explanation.feature_names:
        condition_val = condition_dict[feature]
        tumour_val = tumour_dict.get(feature, 0)
        gleason_val = gleason_dict.get(feature, 0)

        competitor_max = max(tumour_val, gleason_val)

        rows.append({
            'feature': feature,
            'condition_norm': condition_val,
            'tumour_norm': tumour_val,
            'gleason_norm': gleason_val,
            'competitor_max': competitor_max,
            'specificity': condition_val - competitor_max
        })

    comparison_df = pd.DataFrame(rows)


    comparison_df = (
        comparison_df[
            (comparison_df['specificity'] > 0) &
            (comparison_df['feature'] != 'gleason')
        ]
        .sort_values('specificity', ascending=False)
    )

    return comparison_df



def plot_shap_specificity(
    comparison_df,
    condition,
    output_path,
    top_n=10    
):
    # SVG and rendering settings
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['agg.path.chunksize'] = 10000

    # Select and order top features
    plot_df = (
        comparison_df
        .head(top_n)
        .sort_values('specificity')
    )

    features = plot_df['feature'].values
    condition_vals = plot_df['condition_norm'].values
    competitor_vals = plot_df['competitor_max'].values

    y_pos = np.arange(len(features))

    # Set condition dot colour
    condition_colour = {
        'recur': 'grey',
        'non_recur': 'black'
    }

    if condition not in condition_colour:
        raise ValueError(
            "condition must be either 'recur' or 'non_recur'"
        )

    # Create figure
    fig, ax = plt.subplots(
        figsize=(18, 0.5 * len(features) + 2)
    )

    # Competitor dots
    ax.scatter(
        competitor_vals,
        y_pos,
        s=600,
        c='white',
        edgecolors='black',
        linewidths=1.5,
        rasterized=True
    )

    # Condition dots
    ax.scatter(
        condition_vals,
        y_pos,
        s=600,
        c=condition_colour[condition],
        edgecolors='black',
        linewidths=1,
        rasterized=True
    )

    # Arrows
    for i in range(len(features)):
        ax.annotate(
            '',
            xy=(condition_vals[i], y_pos[i]),
            xytext=(competitor_vals[i], y_pos[i]),
            arrowprops=dict(
                arrowstyle='->',
                linewidth=3,
                mutation_scale=30
            )
        )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_xticks([0, 0.05, 0.1])
    ax.set_yticklabels(features)
    ax.axvline(0, linestyle='--', linewidth=0.8)

    ax.set_xlabel(
        'Normalized SHAP importance',
        fontsize=20,
        fontweight='bold'
    )

    ax.tick_params(axis='both', labelsize=20)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Layout
    fig.subplots_adjust(bottom=0.3, left=0.6)

    # Save figure

    output_format = output_path.rsplit('.', 1)[-1].lower()

    fig.savefig(
        output_path,
        format=output_format
    )

    plt.show()







def generate_neighbour_coords(row, max_quad_row=1, max_quad_col=1):
    neighbours = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue

            neighbour_row = row['quad_row'] + dr
            neighbour_col = row['quad_col'] + dc

            tile_row = row['tile_row']
            tile_col = row['tile_col']

            # Handle quadrant overflow into adjacent tiles
            if neighbour_row < 0:
                neighbour_row = max_quad_row
                tile_row -= 1
            elif neighbour_row > max_quad_row:
                neighbour_row = 0
                tile_row += 1

            if neighbour_col < 0:
                neighbour_col = max_quad_col
                tile_col -= 1
            elif neighbour_col > max_quad_col:
                neighbour_col = 0
                tile_col += 1

            neighbours.append((
                row['scene'],
                row['slide_id'],
                tile_row,
                tile_col,
                neighbour_row,
                neighbour_col
            ))

    return neighbours

def get_external_neighbours(
    df_shuffle_probabilities,
    df_recurrent,
    max_quad_row=1,
    max_quad_col=1
):
    # Parse tile and quadrant coordinates
    df_coordinates = df_shuffle_probabilities.copy()

    df_coordinates[['tile_row', 'tile_col']] = (
        df_coordinates['tile'].str.split('_', expand=True).astype(int)
    )

    df_coordinates[['quad_row', 'quad_col']] = (
        df_coordinates['quadrant'].str.split('_', expand=True).astype(int)
    )

    # Generate neighbouring coordinates for recurrent quadrants
    all_neighbours = []

    for _, row in df_recurrent.iterrows():
        all_neighbours.extend(
            generate_neighbour_coords(
                row,
                max_quad_row=max_quad_row,
                max_quad_col=max_quad_col
            )
        )

    # Convert neighbour coordinates to DataFrame
    neighbour_columns = [
        'scene', 'slide_id', 'tile_row', 'tile_col',
        'quad_row', 'quad_col'
    ]

    neighbour_df = pd.DataFrame(
        all_neighbours,
        columns=neighbour_columns
    )

    # Match neighbours against all available quadrants
    neighbour_df = neighbour_df.merge(
        df_coordinates,
        on=neighbour_columns,
        how='inner'
    )

    # Remove the original recurrent quadrants
    recurrent_keys = df_recurrent[
        ['scene', 'slide_id', 'tile', 'quadrant']
    ].drop_duplicates()

    external_neighbours_df = (
        neighbour_df
        .merge(
            recurrent_keys.assign(is_recurrent=True),
            on=['scene', 'slide_id', 'tile', 'quadrant'],
            how='left'
        )
        .loc[lambda df: df['is_recurrent'].isna()]
        .drop(columns=['is_recurrent'])
        .drop_duplicates(
            subset=['scene', 'slide_id', 'tile', 'quadrant']
        )
        .drop(columns=['tile_row', 'tile_col', 'quad_row', 'quad_col'])
        .reset_index(drop=True)
    )

    return external_neighbours_df

def calculate_neighbour_cell_frequencies(
    external_neighbours_df,
    std_thresholds=None,
    mean_thresholds=None
):
    if std_thresholds is None:
        std_thresholds = np.linspace(0.30, 0.08, num=12)

    if mean_thresholds is None:
        mean_thresholds = np.linspace(0.1, 0.7, num=7)

    # Select relevant columns
    external_binned_df = external_neighbours_df[
        [
            'scene', 'slide_id', 'patient_id', 'case',
            'tile', 'quadrant', 'mean_prob', 'std_prob'
        ]
    ].copy()

    # Assign grid cells
    external_binned_df['mean_bin'] = np.digitize(
        external_binned_df['mean_prob'],
        mean_thresholds
    )

    external_binned_df['std_bin'] = np.digitize(
        external_binned_df['std_prob'],
        std_thresholds
    )

    # Count observations in each grid cell
    cell_counts = (
        external_binned_df
        .groupby(['mean_bin', 'std_bin'])
        .size()
        .reset_index(name='frequency')
    )

    # Add cell frequencies to each observation
    external_binned_df = external_binned_df.merge(
        cell_counts,
        on=['mean_bin', 'std_bin'],
        how='left'
    )

    return external_binned_df



def prepare_neighbour_shap_inputs(
    external_neighbours_df,
    df_all_shuffles,
    feature_df,
    parent_dir,
    frequency=20,
    dir_prefix=(
        'gleason_7_quad_1000_v8a_patient_level_svm_plus_'
        'tile_threshold_f1score_removed_xgboost_outer_fold_'
    )
):
    merge_columns = [
        'patient_id', 'slide_id', 'case',
        'scene', 'tile', 'quadrant'
    ]

    # Select neighbours by frequency
    if frequency <= 20:
        frequency_label = 'sparse'
        selected_neighbours_df = external_neighbours_df[
            external_neighbours_df['frequency'] <= frequency
        ].copy()
    else:
        frequency_label = str(frequency)
        selected_neighbours_df = external_neighbours_df[
            external_neighbours_df['frequency'] == frequency
        ].copy()

    # Get unique sample identifiers
    selected_sample_keys = selected_neighbours_df[
        merge_columns
    ].drop_duplicates()

    # Match selected neighbours to shuffle/fold results
    neighbour_shap_search_df = df_all_shuffles.merge(
        selected_sample_keys,
        on=merge_columns,
        how='inner'
    )

    # Match selected neighbours to the feature dataset
    neighbour_feature_df = feature_df.merge(
        selected_sample_keys,
        on=merge_columns,
        how='inner'
    )

    # Identify unique selected samples
    unique_neighbours_df = (
        neighbour_shap_search_df[merge_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Identify the union of features across selected SHAP explanations
    all_neighbour_features = set()

    for row in neighbour_shap_search_df.itertuples(index=False):
        shuffle_dir = os.path.join(
            parent_dir,
            f'{dir_prefix}{row.shuffle}'
        )

        shap_file = os.path.join(
            shuffle_dir,
            f'outer_fold_{row.fold:05d}_xgboost_shap_output.pkl'
        )

        with open(shap_file, 'rb') as f:
            shap_explanation = pickle.load(f)

        all_neighbour_features.update(
            shap_explanation.feature_names
        )

    all_neighbour_features = np.sort(
        list(all_neighbour_features)
    )

    # Define output directory
    shap_output_dir = os.path.join(
        parent_dir,
        'high_recurrence_neighbours_shap',
        f'freq_{frequency_label}_neighbours'
    )

    print(f'Frequency selection: {frequency_label}')
    print(f'Selected neighbours: {len(selected_neighbours_df)}')
    print(f'Unique neighbours: {len(unique_neighbours_df)}')
    print(f'Number of SHAP features: {len(all_neighbour_features)}')

    return (
        selected_neighbours_df,
        neighbour_shap_search_df,
        neighbour_feature_df,
        unique_neighbours_df,
        all_neighbour_features,
        frequency_label,
        shap_output_dir
    )


def compare_recur_neighbour_shap_importance(
    shap_all_recur_explanation,
    shap_all_neighbour_explanation
):
    recur_features = np.array(
        shap_all_recur_explanation.feature_names
    )
    neighbour_features = np.array(
        shap_all_neighbour_explanation.feature_names
    )

    # Calculate mean absolute SHAP importance
    mean_abs_recur = np.abs(
        shap_all_recur_explanation.values
    ).mean(axis=0)

    mean_abs_neighbour = np.abs(
        shap_all_neighbour_explanation.values
    ).mean(axis=0)

    recur_importance = dict(zip(recur_features, mean_abs_recur))
    neighbour_importance = dict(zip(neighbour_features, mean_abs_neighbour))

    # Build comparison dataframe using recurrent features as reference
    rows = []

    for feature in recur_features:
        recur_val = recur_importance.get(feature, 0)
        neighbour_val = neighbour_importance.get(feature, 0)

        rows.append({
            'feature': feature,
            'recur_importance_raw': recur_val,
            'neighbour_importance_raw': neighbour_val
        })

    comparison_df = pd.DataFrame(rows)

    # Calculate raw differences
    comparison_df['diff_raw'] = (
        comparison_df['neighbour_importance_raw'] -
        comparison_df['recur_importance_raw']
    )

    comparison_df['abs_diff_raw'] = np.abs(
        comparison_df['diff_raw']
    )

    comparison_df = comparison_df.sort_values(
        'abs_diff_raw',
        ascending=False
    ).reset_index(drop=True)

    return comparison_df






def plot_recur_neighbour_shap_comparison(
    comparison_df,
    recur_feature_df,
    df_recurrent,
    neighbour_feature_df,
    str_freq,
    output_dir,
    top_n=5
):
    # --- Select top recurrence-specific features ---
    plot_df = (
        comparison_df.loc[
            ~comparison_df['feature'].str.contains(
                'gleason',
                case=False,
                na=False
            )
        ]
        .head(top_n)
        .sort_values('abs_diff_raw', ascending=True)
        .copy()
        .reset_index(drop=True)
    )

    features = plot_df['feature'].values
    recur_values = plot_df['recur_importance_raw'].values
    neighbour_values = plot_df['neighbour_importance_raw'].values

    y_positions = np.arange(len(features))

    # --- Calculate feature-based colour scaling ---
    feature_mins = recur_feature_df[features].quantile(0.05)
    feature_maxs = recur_feature_df[features].quantile(0.95)

    recur_feature_means = df_recurrent[features].mean()
    neighbour_feature_means = neighbour_feature_df[features].mean()

    # Normalise feature values to [0, 1]
    feature_ranges = (feature_maxs - feature_mins).replace(0, np.nan)

    recur_norm = (
        (recur_feature_means - feature_mins) / feature_ranges
    ).fillna(0.5).clip(0, 1)

    neighbour_norm = (
        (neighbour_feature_means - feature_mins) / feature_ranges
    ).fillna(0.5).clip(0, 1)

    # Align colour values with plotted feature order
    recur_norm = recur_norm.loc[features].values
    neighbour_norm = neighbour_norm.loc[features].values

    # Map normalised values to colours
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=1)

    recur_colors = cmap(norm(recur_norm))
    neighbour_colors = cmap(norm(neighbour_norm))

    # --- Create plot ---
    fig, ax = plt.subplots(
        figsize=(10, 0.5 * len(features) + 1)
    )

    # Recurrent points
    ax.scatter(
        recur_values,
        y_positions,
        s=600,
        c=recur_colors,
        edgecolors='grey',
        linewidths=2
    )

    # Neighbour points
    ax.scatter(
        neighbour_values,
        y_positions,
        s=600,
        c=neighbour_colors,
        edgecolors='black',
        linewidths=2
    )

    # Arrows from recurrent to neighbour group
    for i in range(len(features)):
        ax.annotate(
            '',
            xy=(neighbour_values[i], y_positions[i]),
            xytext=(recur_values[i], y_positions[i]),
            arrowprops=dict(
                arrowstyle='->',
                linewidth=3,
                mutation_scale=30
            )
        )

    # --- Format plot ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.set_ylim(-0.5, len(features) - 0.5)

    ax.set_xlabel(
        'Mean absolute SHAP value',
        fontsize=20,
        fontweight='bold'
    )

    ax.tick_params(axis='both', labelsize=20)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    fig.tight_layout()

    # --- Save figures ---
    os.makedirs(output_dir, exist_ok=True)

    save_base = os.path.join(
        output_dir,
        f'shap_values_change_highly_recurrent_neighbours_tiles_freq_{str_freq}_only'
    )

    # Editable SVG
    fig.savefig(
        save_base + '.svg',
        format='svg',
        bbox_inches='tight'
    )

    # TIFF
    fig.savefig(
        save_base + '.tiff',
        format='tiff',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()

    return plot_df