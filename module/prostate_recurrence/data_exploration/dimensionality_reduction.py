"""# dimensionality reduction

## read_data(...) to read tabulated data that record a list of quantative features.

## process_data(...) to process the data prior to performing dimensionality reduction.

## perform_pca(...) to perform principal component analysis.

## perform_tsne(...) to perform t-SNE.

## perform_umap(...) to perform UMAP.

## save_plots(...) to save plots.

"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler


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
        scaler.fit_transform(feature_values)

        standardised_data = pd.DataFrame(columns=feature_names, data=feature_values)

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


def perform_pca():
    pass


def perform_tsne():
    pass


def perform_umap():
    pass


def save_plots():
    pass
