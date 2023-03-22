"""# a set of functions for extracting GLCM features based on analysis of co-ocurring
intensity values in pairs of pixels

## GLCM is short for gray scale co-occurrence matrix

## read_image(...) to read image using PIL library.
This function expects the full absolute path to an input image file as input
parameters. In practice, the input image will be an image patch after tiling
of a whole slide image. This function returns a numpy array of the image.

## construct_glcm(...) to construct the GLCM using skimage library.
This function expects an image and a mask, both in the format of a numpy array, 
and settings for GLCM construction as input parameters. The settings include a list of
pixel-pair distances, a list of pixel-pair angles, number of gray levels, whether
GLCM is symmetric, and whether the elements in GLCM are normalised. This function
returns GLCM, namely, P[i,j,d,theta] with respect to different combinations of
distances and angles, in the format of a numpy array, and the masked image array.

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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


def read_image(path_to_img: str) -> np.ndarray:
    """read gray scale image patch into a numpy array
    This function reads a gray scale image using PIL library and returns a numpy array

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
    image: np.ndarray,
    mask: np.ndarray,
    distances: List[int],
    angles: List[float],
    levels: int = 256,
    symmetric: bool = True,
    normed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """construct GLCM for the input image
    This function inputs a gray scale image in a format of a numpy array and constructs the GLCM.
    The GLCM is constructed for each combination of unique distance and angle values provided.
    The GLCM, P[i,j,d,theta], with respect to different distance and angle values and the masked
    image are returned in the format of numpy arrays.
    Note that in implementation of a mask, only pixels outside the mask are assigned to have zero
    intensities. The first row and the first column of the GLCM, which reflect any analyses involving
    pixels with zero intensities, are excluded from further calculations (i.e., not returned from this
    function).

    Parameters
    ----------
    image : np.ndarray
        An input gray scale image in the format of a numpy array
    image_mask : np.ndarray
        An binary image in the format of a numpy array
    distances : List[int]
        A list of distance values, each used as an input parameter for a given realisation of GLCM
        construction.
    angles : List[float]
        A list of angle values in radians, each used as an input parameter for a given realisation of
        of GLCM construction.
    levels : int, optional
        The number of gray levels, by default 256
    symmetric : bool, optional
        A boolean variable indicating if GLCM is symmetric, by default True
    normed : bool, optional
        A boolean variable indicating if GLCM is normalised, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays including
        * The GLCM with respect to different levels of distances and angles,
        i.e., P[i,j,d,theta].
        * The masked image as a numpy array.
    """

    masked_image = image.copy()
    masked_image[masked_image == 0] = 1

    masked_image = np.multiply(masked_image, mask).astype(np.uint8)

    glcm = graycomatrix(
        image=masked_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )

    return glcm[1:, 1:, :, :], masked_image


def extract_glcm_features(
    matrix: np.ndarray, features: Tuple[str]
) -> Dict[str, np.ndarray]:
    """extract quantitative features based on the input GLCM
    This function inputs the GLCM in the format of a numpy array, namely, P[i,j,d,theta],
    and extracts a set of quantitative features as instructed by the user.
    Note: Scikit Image library provides implementation of an imcomplete set of GLCM features,
    which are currently implemented in this function. In the future, this function will also
    contain customised implementation of the other GLCM features.

    Parameters
    ----------
    matrix : np.ndarray
        The GLCM with respect to different levels of distances and angles, i.e., P[i,j,d,theta],
        in the format of a numpy array
    features : Tuple[str]
        A tuple containing names of quantitative features to extract based on GLCM

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of [feature name : feature value] with outputs saved with respect to different
        conditions of distance and angle values
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
        glcm_features_output[feature] = graycoprops(matrix, feature)

    return glcm_features_output


def save_glcm_features(
    matrix: np.ndarray,
    glcm_features_output: Dict[str, np.ndarray],
    distances: List[int],
    angles: List[float],
    output_directory: str,
    analysis_type: str = "masked",
) -> None:
    """save GLCM and quantitative features
    This function inputs the constructed GLCM and extracted quantitative features and outputs these
    results into numpy arrays and data frames, respectively.

    Parameters
    ----------
    matrix : np.ndarray
        The GLCM with respect to different conditions of distance and angle values, i.e., P[i,j,d,theta],
        in the format of a numpy array
    glcm_features_output : Dict[str, np.ndarray]
        A dictionary of [feature name : feature value] with feature value stored with respect to different
        conditions of distance and angle values
    distances : List[int]
        A list of distance values, each used as an input parameter for a given realisation of GLCM
        construction.
    angles : List[float]
        A list of angle values in radians, each used as an input parameter for a given realisation of
        of GLCM construction.
    output_directory: str
        Directory in which the feature outputs are saved
    """

    def save_glcm_as_image() -> None:
        """save GLCM as image
        This function outputs GLCM as images using matplotlib library, with respect
        to different distances and angles
        """
        for i, distance in enumerate(distances):
            for j, angle in enumerate(angles):
                figure_name = f"glcm_distance_{distance}_angle_{int(angle/np.pi*180)}_{analysis_type}.pdf"
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
        array_name = f"glcm_array_{analysis_type}.npy"
        with open(os.path.join(output_directory, array_name), "wb") as fout:
            np.save(fout, matrix)

    def save_glcm_features_as_csv() -> None:
        """save GLCM quantitative features
        This function outputs GLCM quantitative features into a .csv file.
        """
        file_name = f"glcm_features_{analysis_type}.csv"
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

    # save_glcm_as_image()
    # save_glcm_as_array()
    save_glcm_features_as_csv()
