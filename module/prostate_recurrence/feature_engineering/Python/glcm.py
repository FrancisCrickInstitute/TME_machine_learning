"""# a set of functions for extracting GLCM features

## GLCM is short for gray scale co-occurrence matrix

## read_image(...) to read image using PIL library.
This function expects the full absolute path to an input image file as input
parameters. In practice, the input image will be an image patch after tiling
of a whole slide image. This function returns a numpy array of the image.

## construct_glcm(...) to construct the GLCM using skimage library.
This function expects an image in the format of a numpy array and settings
for GLCM construction as input parameters. The settings include a list of pixel-
pair distances, a list of pixel-pair angles, number of gray levels, whether
GLCM is symmetric, and whether the elements in GLCM are normalised. This function
returns GLCM, namely, P[i,j,d,theta] with respect to different combinations of
distances and angles, in the format of a numpy array.

## extract_glcm_features(...) to extract quantitative features from GLCM, partly
using skimage library.
This function expects the GLCM and a tuple of quantitative features as input
parameters. Note that skimage function covers only six GLCM quantitative features.
Customised implementation of other features will be developed. This function returns
a dictionary of [feature name: feature value].

## save_glcm_features(...) to save GLCM and quantitative features.
This function expects the GLCM, a dictionary of the extracted features, some settings
for GLCM, and the directory for saving outputs into as input parameters. Settings include
a list of pixel-pair distances and a list of pixel-pair angles. GLCM is saved into an
image using matplotlib library and into a numpy array. GLCM quantitative features are saved
in a .csv file.

"""

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from skimage.feature import greycomatrix, greycoprops


def read_image(path_to_img: str) -> np.ndarray:
    """read gray scale image patch into a numpy array
    This function reads a gray scale image using PIL library
    and returns a numpy array

    Parameters
    ----------
    path_to_img : str
        The full path to the image file

    Returns
    -------
    np.ndarray
        The image patch as numpy array
    """

    image = Image.open(path_to_img)
    image_arr = np.array(image)

    return image_arr


def construct_glcm(
    image: np.array,
    distances: List[int],
    angles: List[float],
    levels: int = 256,
    symmetric: bool = True,
    normed: bool = True,
) -> np.ndarray:
    """construct GLCM for the input image
    This function inputs an input image in a format of a numpy
    array and constructs the GLCM. The GLCM, P[i,j,d,theta] is
    returned as a numpy array.

    Parameters
    ----------
    image : np.array
        An input gray scale image as numpy array
    distances : List[int]
        A list of pixel-pair offset distances
    angles : List[float]
        A list of pixel-pair angles in radians
    levels : int, optional
        The number of gray levels, by default 256
    symmetric : bool, optional
        A boolean variable indicating if GLCM is symmetric,
        by default True
    normed : bool, optional
        A boolean variable indicating if GLCM is normalised,
        by default True

    Returns
    -------
    np.ndarray
        The GLCM with respect to different levels of distances and angles,
        i.e., P[i,j,d,theta].
    """

    glcm = greycomatrix(
        image=image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )

    return glcm


def extract_glcm_features(
    matrix: np.array, features: Tuple[str]
) -> Dict[str, np.ndarray]:
    """extract quantitative features based on the input GLCM
    This function inputs the GLCM in the format of a numpy array, namely,
    P[i,j,d,theta], and extracts a set of quantitative features as instructed
    by the user.
    Note: Scikit Image library provides implementation of an imcomplete set of
    GLCM features, so this function contains customised implementation of the
    other GLCM features not covered by Scikit Image.

    Parameters
    ----------
    matrix : np.array
        The GLCM with respect to different levels of distances and angles,
        i.e., P[i,j,d,theta]
    features : Tuple[str]
        A tuple containing a set of quantitative features to extract based on
        GLCM

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of [feature name : feature value] with feature value stored
        with respect to different distances and angles
    """

    features_implemented_in_skimage: Tuple = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    )

    features_implemented_extra = ()

    glcm_features_output = dict()
    for feature in features:
        if (
            feature not in features_implemented_in_skimage
            and feature not in features_implemented_extra
        ):
            print(f"feature {feature} has not been implemented yet. skip...")
            continue
        glcm_features_output[feature] = greycoprops(matrix, feature)

    return glcm_features_output


def save_glcm_features(
    matrix: np.ndarray,
    glcm_features_output: Dict[str, np.ndarray],
    distances: List[int],
    angles: List[float],
    output_directory: str,
) -> None:
    """save GLCM and quantitative features
    This function inputs the constructed GLCM and extracted quantitative features
    and outputs these results into numpy arrays and tables, respectively.

    Parameters
    ----------
    matrix : np.ndarray
        The GLCM with respect to different levels of distances and angles,
        i.e., P[i,j,d,theta]
    glcm_features_output : Dict[str, np.ndarray]
        A dictionary of [feature name : feature value] with feature value stored
        with respect to different distances and angles
    distances : List[int]
        A list of pixel-pair offset distances
    angles : List[float]
        A list of pixel-pair angles in radians
    output_directory: str
        Directory to save outputs into
    """

    def save_glcm_as_image() -> None:
        """save GLCM as image
        This function outputs GLCM as images using matplotlib library, with respect
        to different distances and angles
        """
        for i, distance in enumerate(distances):
            for j, angle in enumerate(angles):
                figure_name = (
                    f"glcm_distance_{distance}_angle_{int(angle/np.pi*180)}.pdf"
                )
                fig = plt.figure(figsize=(4, 4), dpi=300)
                ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
                ax_cbar = fig.add_axes([0.8, 0.15, 0.05, 0.6])
                ms = ax.matshow(
                    matrix[:, :, i, j],
                    cmap=plt.cm.Greys,
                    norm=LogNorm(vmin=1e-6, vmax=1e-1),
                )
                ax.set_xticks([0, 64, 128, 192, 255])
                ax.set_yticks([0, 64, 128, 192, 255])
                plt.colorbar(ms, cax=ax_cbar, orientation="vertical")
                plt.savefig(
                    os.path.join(output_directory, figure_name),
                    dpi=300,
                    transparent=True,
                )
                plt.close()

    def save_glcm_as_array() -> None:
        """save GLCM as numpy array
        This function outputs GLCM as a numpy array, with distances and angles reflected
        in the 3rd and 4th dimension of the array, respectively.
        """
        array_name = "glcm_array.npy"
        with open(os.path.join(output_directory, array_name), "wb") as fout:
            np.save(fout, matrix)

    def save_glcm_features_as_csv() -> None:
        """save GLCM quantitative features
        This function outputs GLCM quantitative features into a .csv file.
        """
        file_name = "glcm_features.csv"
        columns = ["feature", "value"]
        data_rows = []
        for feature_name, feature_value in glcm_features_output.items():
            for i, distance in enumerate(distances):
                for j, angle in enumerate(angles):
                    feature_name_this_analysis = f"glcm_{feature_name}_distance_{distance}_angle_{int(angle/np.pi*180)}"
                    feature_value_this_analysis = feature_value[i, j]
                    data_rows.append(
                        (feature_name_this_analysis, feature_value_this_analysis)
                    )
        data_frame = pd.DataFrame(columns=columns, data=data_rows)
        data_frame.to_csv(os.path.join(output_directory, file_name), index=False)

    save_glcm_as_image()
    save_glcm_as_array()
    save_glcm_features_as_csv()


# def batch_processing(
#     input_directory: str, filename_pattern: str, output_directory: str
# ):
#     pass
