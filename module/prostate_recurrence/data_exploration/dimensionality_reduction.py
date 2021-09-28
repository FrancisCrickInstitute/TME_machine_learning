"""# dimensionality reduction

## read_data(...) to read tabulated data that record a list of quantative features.
This function expects a list of paths to data that contain the quantitative features
as input and returns combined feature values for all image tiles in a data frame.
The header of the data frame includes "slide", "image_tile", "label", which combines
slide and tile information, and the names of all features. The rows of the data frame
record readouts from individual image tiles.

## process_data(...) to process the data prior to performing dimensionality reduction.
This function expects combined feature values for all image tiles in a data frame and
the type of processing as input and returns a tuple of two data frames. The first data
frame contains the processed feature values for all image tiles. The second data frame
contains the attributes of the processing procedure.

## perform_pca(...) to perform principal component analysis.
This function expects as input the processed feature values for all image tiles in a
data frame, the directory to write outputs into, the number of components to keep for
PCA, and two boolean variables indicating whether to save plots and arrays. By default,
all PCA components are kept, and plots and arrays are saved into the designated directory.
This function returns a dictionary recording the results of PCA together with other useful
outputs.

## perform_tsne(...) to perform t-SNE.
This function expects as inputs the processed feature values for all image tiles in a
data frame, the directory to write outputs into, the number of components to keep for
t-SNE, perplexity parameter, and two boolean variables indicating whether to save plots
and arrays. By default, the low-dimensional space is represented with respect to two t-SNE
components, and plots and arrays are saved into the designated directory. This function
returns a dictionary recording the results of t-SNE embedding.

## perform_umap(...) to perform UMAP.
This function expects as inputs the processed feature values for all image tiles in a
data frame, the directory to write outputs into, settings for UMAP analysis, and two
boolean variables indicating whether to save plots and arrays. By default, plots and arrays
are saved into the designated directory. This function returns a dictionary recording the
results of UMAP embedding and other attributes.

"""

from typing import Dict, List, Tuple

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


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
    output_directory: str,
    n_components: int = None,
    save_plot: bool = True,
    save_array: bool = True,
) -> Dict[str, np.ndarray]:
    """perform PCA
    This function expects a data frame, which records quantitative features
    and labels reflecting slide and image tile identifiers, and an output
    directory as input parameters. Optional parameters include the number
    of principal components to use for PCA and flags indicating whether to
    save PCA outputs, including plots and numpy arrays. This function returns
    a dictionary of attributes of the fit PCA model.

    Parameters
    ----------
    data : pd.DataFrame
        a data frame of quantiative features and labels.
    output_directory : str
        an output directory to save PCA outputs into.
    n_components : int, optional
        the number of principal components to keep in PCA, by default None
    save_plot : bool, optional
        a boolean variable indicating whether to save plots, by default True
    save_array : bool, optional
        a boolean variable indicating whether to save arrays, by default True

    Returns
    -------
    Dict[str, np.ndarray]
        a dictionary of attributes of PCA, including the following:
        "results_pca": n_samples x n_components array
        "components_": n_components x n_features array
        "explained_variance_ratio_": n_components x 1 array
        "explained_variance_": n_components x 1 array
        "loadings": n_features x 2 array
    """

    features = data[
        [col for col in data.columns if col not in ["slide", "image_tile", "label"]]
    ]

    pca = PCA(n_components=n_components)
    results_pca = pca.fit_transform(features)

    pca_components = pca.components_
    pca_explained_variance_ratio = pca.explained_variance_ratio_
    pca_explained_variance = pca.explained_variance_
    loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])

    pca_attributes = {
        "results_pca": results_pca,
        "components_": pca_components,
        "explained_variance_ratio_": pca_explained_variance_ratio,
        "explained_variance_": pca_explained_variance,
        "loadings": loadings,
    }

    if save_array:
        path_save_pca_results = os.path.join(
            os.path.join(output_directory, "results_pca.npy")
        )
        with open(path_save_pca_results, "wb") as f:
            np.save(f, results_pca)

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
        ax1.set_xlabel("Principal components", size=8)
        ax1.set_ylabel("Fraction of variance explained", c="b", size=8)
        ax2.set_ylabel("Cumulative fraction of variance explained", c="r", size=8)

        for ax in [ax1, ax2]:
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)
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
            ms=4,
        )
        plt.savefig(path_save_plot_pca_explained_variance_ratio, dpi=300)
        plt.close()

        path_save_plot_pca_components = os.path.join(
            os.path.join(output_directory, "pca_components.pdf")
        )
        unique_slide_names = np.unique(data.slide).tolist()
        colorvals = np.linspace(0, 1, len(unique_slide_names))

        cs = [
            colorvals[unique_slide_names.index(slide_name)] for slide_name in data.slide
        ]

        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        scatter = ax1.scatter(
            results_pca[:, 0],
            results_pca[:, 1],
            c=cs,
            s=3,
            cmap=plt.cm.gist_rainbow,
        )
        ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_slide_names,
            prop={"size": 6},
        )
        ax1.set_xlabel(
            f"PC1 ({round(pca_explained_variance_ratio[0]*100, 1)}% variance explained)",
            size=8,
        )
        ax1.set_ylabel(
            f"PC2 ({round(pca_explained_variance_ratio[1]*100, 1)}% variance explained)",
            size=8,
        )
        for ax in [ax1]:
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)
        plt.savefig(path_save_plot_pca_components, dpi=300)
        plt.close()

        path_save_plot_pca_components_with_loading = os.path.join(
            os.path.join(output_directory, "pca_components_scaled_with_loading.pdf")
        )
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        scale_x = 1 / (results_pca[:, 0].max() - results_pca[:, 0].min())
        scale_y = 1 / (results_pca[:, 1].max() - results_pca[:, 1].min())
        scatter = ax1.scatter(
            results_pca[:, 0] * scale_x,
            results_pca[:, 1] * scale_y,
            c=cs,
            s=3,
            cmap=plt.cm.gist_rainbow,
        )
        ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_slide_names,
            prop={"size": 6},
        )
        ax1.set_xlabel(
            f"PC1 ({round(pca_explained_variance_ratio[0]*100, 1)}% variance explained)",
            size=8,
        )
        ax1.set_ylabel(
            f"PC2 ({round(pca_explained_variance_ratio[1]*100, 1)}% variance explained)",
            size=8,
        )
        for ax in [ax1]:
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)
        for pc1, pc2 in loadings:
            ax1.plot([0, pc1], [0, pc2])
        plt.savefig(path_save_plot_pca_components_with_loading, dpi=300)

        plt.close()

    return pca_attributes


def perform_tsne(
    data: pd.DataFrame,
    output_directory: str,
    n_components: int = 2,
    perplexity: float = 30,
    save_plot: bool = True,
    save_array: bool = True,
) -> Dict[str, np.ndarray]:
    """perform t-SNE
    This function expects a data frame, which records quantitative features
    and labels reflecting slide and image tile identifiers, an output directory,
    and settings as input parameters. Optional parameters include the number
    of components with respect to which low-dimensional embedding is constructed,
    perplexity parameter, and flags indicating whether to save t-SNE outputs,
    including plots and numpy arrays. This function returns a dictionary of attributes
    of the fit t-SNE embedding.

    Parameters
    ----------
    data : pd.DataFrame
        a data frame of quantiative features and labels.
    output_directory : str
        an output directory to save t-SNE outputs into.
    n_components : int, optional
        the number of components with repect to which t-SNE embedding is constructed,
        by default 2
    perplexity : float, optional
        parameter balancing local and global effects, by default 30
    save_plot : bool, optional
        a boolean variable indicating whether to save plots, by default True
    save_array : bool, optional
        a boolean variable indicating whether to save arrays, by default True

    Returns
    -------
    Dict[str, np.ndarray]
        a dictionary of attributes of t-SNE, including the following:
        "embedding_": n_samples x n_components array
    """

    features = data[
        [col for col in data.columns if col not in ["slide", "image_tile", "label"]]
    ]

    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    results_tsne = tsne.fit_transform(features)
    tsne_attributes = {"embedding_": tsne.embedding_}

    if save_array:
        path_save_tsne_results = os.path.join(
            os.path.join(output_directory, "results_tsne.npy")
        )
        with open(path_save_tsne_results, "wb") as f:
            np.save(f, results_tsne)

        path_save_tsne_embedding = os.path.join(
            os.path.join(output_directory, "tsne_embedding.npy")
        )
        with open(path_save_tsne_embedding, "wb") as f:
            np.save(f, tsne.embedding_)

    if save_plot:
        path_save_plot_tsne_embedding = os.path.join(
            os.path.join(output_directory, "tsne_embedding.pdf")
        )
        unique_slide_names = np.unique(data.slide).tolist()
        colorvals = np.linspace(0, 1, len(unique_slide_names))

        cs = [
            colorvals[unique_slide_names.index(slide_name)] for slide_name in data.slide
        ]

        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        scatter = ax1.scatter(
            results_tsne[:, 0], results_tsne[:, 1], c=cs, s=3, cmap=plt.cm.gist_rainbow
        )
        ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_slide_names,
            prop={"size": 6},
        )
        ax1.set_xlabel(
            "t-SNE 1",
            size=8,
        )
        ax1.set_ylabel(
            "t-SNE 2",
            size=8,
        )
        for ax in [ax1]:
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)
        plt.savefig(path_save_plot_tsne_embedding, dpi=300)
        plt.close()

    return tsne_attributes


def perform_umap(
    data: pd.DataFrame,
    output_directory: str,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    save_plot: bool = True,
    save_array: bool = True,
) -> Dict[str, np.ndarray]:
    """perform UMAP
    This function expects a data frame, which records quantitative features
    and labels reflecting slide and image tile identifiers, an output directory,
    and settings as input parameters. Optional parameters include settings as
    described below and flags indicating whether to save UMAP outputs, including
    plots and numpy arrays. This function returns a dictionary of attributes
    of the fit UMAP embedding.

    Parameters
    ----------
    data : pd.DataFrame
        a data frame of quantiative features and labels.
    output_directory : str
        an output directory to save UMAP outputs into.
    n_neighbors : int, optional
        parameter balancing local and global effects, by default 10
    min_dist : float, optional
        parameter controlling how tightly data points are packed, by default 0.1
    metric : str, optional
        parameter indicating how distance is computed in the ambient space of
        input data, by default "euclidean"
    save_plot : bool, optional
        a boolean variable indicating whether to save plots, by default True
    save_array : bool, optional
        a boolean variable indicating whether to save arrays, by default True

    Returns
    -------
    Dict[str, np.ndarray]
        a dictionary of attributes of UMAP, including the following:
        "embedding_": n_samples x n_components array
        "graph_": [... not explored ...]
        "graph_dists": [... not explored ...]
    """
    features = data[
        [col for col in data.columns if col not in ["slide", "image_tile", "label"]]
    ]

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    results_umap = umap.fit_transform(features)

    umap_embedding = umap.embedding_
    umap_graph = umap.graph_
    umap_graph_dists = umap.graph_dists_

    umap_attributes = {
        "embedding_": umap_embedding,
        "graph_": umap_graph,
        "graph_dists": umap_graph_dists,
    }

    if save_array:
        path_save_umap_results = os.path.join(
            os.path.join(output_directory, "results_umap.npy")
        )
        with open(path_save_umap_results, "wb") as f:
            np.save(f, results_umap)

        path_save_umap_embedding = os.path.join(
            os.path.join(output_directory, "umap_embedding.npy")
        )
        with open(path_save_umap_embedding, "wb") as f:
            np.save(f, umap.embedding_)

        path_save_umap_graph = os.path.join(
            os.path.join(output_directory, "umap_graph.npy")
        )
        with open(path_save_umap_graph, "wb") as f:
            np.save(f, umap.graph_)

        path_save_umap_graph_dist = os.path.join(
            os.path.join(output_directory, "umap_graph_dist.npy")
        )
        with open(path_save_umap_graph_dist, "wb") as f:
            np.save(f, umap.graph_dists_)

    if save_plot:
        path_save_plot_umap_embedding = os.path.join(
            os.path.join(output_directory, "umap_embedding.pdf")
        )
        unique_slide_names = np.unique(data.slide).tolist()
        colorvals = np.linspace(0, 1, len(unique_slide_names))

        cs = [
            colorvals[unique_slide_names.index(slide_name)] for slide_name in data.slide
        ]

        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax1 = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        scatter = ax1.scatter(
            results_umap[:, 0], results_umap[:, 1], c=cs, s=3, cmap=plt.cm.gist_rainbow
        )
        ax1.legend(
            handles=scatter.legend_elements()[0],
            labels=unique_slide_names,
            prop={"size": 6},
        )
        ax1.set_xlabel(
            "UMAP 1",
            size=8,
        )
        ax1.set_ylabel(
            "UMAP 2",
            size=8,
        )
        for ax in [ax1]:
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.tick_params(axis="both", which="minor", labelsize=6)
        plt.savefig(path_save_plot_umap_embedding, dpi=300)
        plt.close()

    return umap_attributes
