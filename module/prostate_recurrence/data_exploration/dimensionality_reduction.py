"""# dimensionality reduction

## read_data(...) to read tabulated data that record a list of quantative features.

## process_data(...) to process the data prior to performing dimensionality reduction.

## perform_pca(...) to perform principal component analysis.

## perform_tsne(...) to perform t-SNE.

## perform_umap(...) to perform UMAP.

## save_plots(...) to save plots.

"""

from typing import Dict, List, Tuple

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_data(paths_to_data: List[str]) -> pd.DataFrame:
    """read tabulated data that contain quantitative features
    This function reads a list of paths to data that contain quantitative
    features, extracts labels that reflect the names of slides and image
    tiles, and returns a data frame of annotated data with labels.

    Parameters
    ----------
    paths_to_data : List[str]
        a list of paths to tabulated data
    Returns
    -------
    pd.DataFrame
        a data frame with column names ["slide", "image_tile", "label",
        "[feature1]", ...] that records features values as well as the
        information about slide and image tile, which is used as a unique
        label for data points.

    """
    all_data_labeled = pd.DataFrame()
    for path_to_data in paths_to_data:
        path_elements = path_to_data.split("/")
        index_string_slide = path_elements.index("slide")
        slide_name = path_elements[index_string_slide + 1]
        index_string_tile_level_features = path_elements.index("tile_level_features")
        image_tile_name = path_elements[index_string_tile_level_features + 1]
        label = ":".join([slide_name, image_tile_name])

        data_point_information = pd.DataFrame(
            {"slide": [slide_name], "image_tile": [image_tile_name], "label": [label]}
        )

        data = pd.read_csv(
            path_to_data, header=None, names=["feature", "feature_value"]
        )

        data_T = data.set_index("feature").T

        data_T.reset_index(drop=True, inplace=True),
        data_point_information.reset_index(drop=True, inplace=True),

        data_T_labeled = pd.concat(
            [data_T, data_point_information],
            axis=1,
        )
        all_data_labeled = all_data_labeled.append(data_T_labeled)

    return all_data_labeled


def process_data(
    data: pd.DataFrame, type_of_processing: str = "standardisation"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """process data via standardisation
    This function expects a data frame that records quantitative features,
    with its header being the names of features, along with the type of
    processing as input parameters and returns both processed data and
    the attributes of the data transformation.

    Parameters
    ----------
    data : pd.DataFrame
        a data frame that records quantitative features, with its header
        being the names of features.
    type_of_processing : str, optional
        the type of data processing, by default "standardisation"

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        a tuple of two data frames. The first data frame contains the processed
        data with the header being the names of features. The second data frame
        contains the attributes of transformation. For standardisation, attributes
        include the mean, variance and scale (i.e., standard deviation) of each
        feature.
    """

    feature_names = data.columns
    feature_values = data.values
    if type_of_processing == "standardisation":
        scaler = StandardScaler()
        transfored_feature_values = scaler.fit_transform(feature_values)

        standardised_data = pd.DataFrame(
            columns=feature_names, data=transfored_feature_values
        )

        standardisation_attributes = pd.DataFrame(
            {
                "feature": feature_names,
                "scaler_mean_": scaler.mean_,
                "scaler_var_": scaler.var_,
                "scaler_scale_": scaler.scale_,
            }
        )

        return (standardised_data, standardisation_attributes)

    return (None, None)


def perform_pca(
    data: pd.DataFrame,
    slide_names: np.ndarray,
    output_directory: str,
    n_components: int = None,
    save_plot: bool = True,
    save_array: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components)
    results_pca = pca.fit_transform(data)

    pca_components = pca.components_
    pca_explained_variance_ratio = pca.explained_variance_ratio_
    pca_explained_variance = pca.explained_variance_
    loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])

    if save_array:
        path_save_pca_components = os.path.join(
            os.path.join(output_directory, "pca_components.npy")
        )
        with open(path_save_pca_components, "wb") as f:
            np.save(f, pca_components)

        path_save_pca_explained_variance_ratio = os.path.join(
            os.path.join(output_directory, "pca_explained_variance_ratio.npy")
        )
        with open(path_save_pca_explained_variance_ratio, "wb") as f:
            np.save(f, pca_explained_variance_ratio)

        path_save_pca_explained_variance = os.path.join(
            os.path.join(output_directory, "pca_explained_variance.npy")
        )
        with open(path_save_pca_explained_variance, "wb") as f:
            np.save(f, pca_explained_variance)

    if save_plot:
        path_save_plot_pca_explained_variance_ratio = os.path.join(
            os.path.join(output_directory, "pca_explained_variance_ratio.pdf")
        )
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1)
        ax1.set_xticks(np.arange(1, pca_explained_variance_ratio.size + 1, 5))
        ax1.set_xlabel("Principal components")
        ax1.set_ylabel("Fraction of variance explained", c="b")
        ax2.set_ylabel("Cumulative fraction of variance explained", c="r")
        ax1.bar(
            np.arange(1, pca_explained_variance_ratio.size + 1),
            height=pca_explained_variance_ratio,
            color="b",
        )
        ax2.plot(
            np.arange(1, pca_explained_variance_ratio.size + 1),
            np.cumsum(pca_explained_variance_ratio),
            c="r",
            marker="o",
            markerfacecolor="none",
        )
        plt.savefig(path_save_plot_pca_explained_variance_ratio, dpi=300)
        plt.close()

        path_save_plot_pca_components = os.path.join(
            os.path.join(output_directory, "pca_components.pdf")
        )
        unique_slide_names = np.unique(slide_names).tolist()
        colorvals = np.linspace(0, 1, len(unique_slide_names))

        cs = [
            colorvals[unique_slide_names.index(slide_name)]
            for slide_name in slide_names
        ]

        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        scatter = ax1.scatter(results_pca[:, 0], results_pca[:, 1], c=cs, s=3)
        ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_slide_names,
            prop={"size": 6},
        )
        ax1.set_xlabel(
            f"PC1 ({round(pca_explained_variance_ratio[0]*100, 2)}% variance explained)"
        )
        ax1.set_ylabel(
            f"PC2 ({round(pca_explained_variance_ratio[1]*100, 2)}% variance explained)"
        )
        plt.savefig(path_save_plot_pca_components, dpi=300)

        path_save_plot_pca_components_with_loading = os.path.join(
            os.path.join(output_directory, "pca_components_with_loading.pdf")
        )
        for pc1, pc2 in loadings:
            ax1.plot([0, pc1], [0, pc2])
        plt.savefig(path_save_plot_pca_components_with_loading, dpi=300)

        plt.close()


def perform_tsne():
    pass


def perform_umap():
    pass


def save_plots():
    pass
