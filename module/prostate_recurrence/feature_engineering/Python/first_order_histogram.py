"""# a set of functions for extracting quantitative features based on the
histogram of pixel intensities.

## read_image(...) to read image using PIL library.
This function expects the full absolute path to an input image file as input
parameters. In practice, the input image will be an image patch after tiling
of a whole slide image. This function returns a numpy array of the image.

## construct_histogram(...) to construct the histogram.
This function expects an image in the format of a numpy array and settings as
input parameters. Settings include the number of gray levels and whether the
histogram is normalised. This function uses numpy library to create a histogram,
with bin size of one. This function returns the histogram in the format of a
numpy array.

## extract_histogram_features(...) to extract quantitative features from the
histogram.
This function expects an image in the format of a numpy array, the histogram of
the image's pixel intensities, and a tuple containing names of quantitative features 
as input parameters. Numpy and scipy libraries are used to extract features. 
This function returns a dictionary of [feature name: feature value].

## save_histogram_features(...) to save histogram and quantitative features.
This function expects the histogram, a dictionary of the extracted features, and
the directory for saving outputs into as input parameters. Histogram is saved into
an image using matplotlib library and into a numpy array. Histogram based quantitative
features are saved in a .csv file.

"""

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kurtosis, skew


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


def construct_histogram(
    image: np.ndarray, mask: np.ndarray, levels: int = 256, normed: bool = True
) -> np.ndarray:
    """construct a histogram of pixel intensities.
    This function inputs the gray scale image and its corresponding binary
    tissue mask, both in the format of a numpy array, and configuration
    configuration parameters including number of binns and whether histogram
    should be normalised. This function outputs a histogram of pixel intensities
    and the masked image, both in the format of a numpy array.
    The histogram is created using np.histogram().

    Parameters
    ----------
    image : np.ndarray
        An input gray scale image in the format of a numpy array
    mask : np.ndarray
        An binary image in the format of a numpy array
    levels : int
        The number of bins for creating the histogram
    normed : bool
        A boolean variable indicating if the histogram is normalised

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays, including:
        * A histogram of pixel intesities in the format of a numpy array,
        with elements indicating the frequency or probability of pixel
        intensities at different gray levels, i.e., P[i].
        * The masked image in the format of a numpy array.
    """

    masked_image = np.multiply(image, mask).astype(np.uint8)

    histogram, _ = np.histogram(
        masked_image[masked_image > 0], bins=np.arange(-0.5, levels + 0.5, 1)
    )
    if normed:
        histogram = histogram / histogram.sum()

    return histogram, masked_image


def extract_histogram_features(
    image: np.ndarray,
    histogram: np.ndarray,
    exclude_background_pixels: bool = True,
    features: Tuple[str] = (),
) -> Dict[str, float]:
    """extract quantitative features based on the input histogram
    This function inputs the histogram of pixel intensities in the format of
    a numpy array, namely, P[i], and extracts a set of quantitative features
    as instructed by the user.
    Note that the input image should be a masked image generated in the
    construct_histogram() function if a tissue mask is considered.
    Quantitative features are generated using functions in numpy and scipy
    libraries.

    Parameters
    ----------
    image : np.ndarray
        An input gray scale image in the format of numpy array.
    histogram : np.ndarray
        The histogram of pixel intensities, with elements indicating
        the frequency or probability at different gray levels, i.e., P[i].
    exclude_background_pixels : bool
        A boolean variable indicating whether to exclude pixels with zero
        intensity.
    features : Tuple[str]
        A tuple containing the names of quantitative features that are to be
        extracted based on the histogram of pixel intensities.

    Returns
    -------
    Dict[str, float]
        A dictionary of [feature name : feature value].
    """

    if exclude_background_pixels:
        image = image[image > 0]
        histogram = histogram[1:]

    image_flattened = image.flatten()
    histogram_features_all = {
        "median": np.median(image_flattened),
        "mean": np.mean(image_flattened),
        "variance": np.var(image_flattened),
        "skewness": skew(image_flattened),
        "kurtosis": kurtosis(image_flattened, fisher=False),
    }
    if histogram.sum() != 1:
        histogram /= histogram.sum()
    histogram_features_all.update(
        {
            "energy": np.power(histogram, 2).sum(),
            "entropy": -np.sum(
                histogram[histogram > 0] * np.log2(histogram[histogram > 0])
            ),
        }
    )

    histogram_features_output = {
        feature: histogram_features_all[feature]
        for feature in features
        if feature in histogram_features_all.keys()
    }

    return histogram_features_output


def save_histogram_features(
    histogram: np.ndarray,
    histogram_features_output: Dict[str, float],
    output_directory: str,
) -> None:
    """save histogram and quantitative features
    This function inputs the histogram of pixel intensities, the quantitative
    features, and the directory to write outputs into.

    Parameters
    ----------
    histogram : np.ndarray
        The histogram of pixel intensities in the form of numpy array,
        with elements indicating the frequency or probability of pixel
        intensities at certain gray levels, i.e., P[i].
    histogram_features_output : Dict[str, float]
        A dictionary of [feature name : feature value].
    output_directory : str
        Directory in which the feature outputs are saved
    """

    def save_histogram_as_image() -> None:
        """save histogram as image
        This function outputs histogram as image using matplotlib library
        """

        figure_name = "intensity_histogram.pdf"
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
        ax.bar(np.arange(histogram.size), height=histogram)
        ax.set_xticks([0, 64, 128, 192, 255])
        plt.savefig(
            os.path.join(output_directory, figure_name),
            dpi=300,
            transparent=True,
        )
        plt.close()

    def save_histogram_as_array() -> None:
        """save histogram as numpy array
        This function outputs histogram as a numpy array
        """
        array_name = "intensity_histogram_array.npy"
        with open(os.path.join(output_directory, array_name), "wb") as fout:
            np.save(fout, histogram)

    def save_histogram_features_as_csv() -> None:
        """save histogram quantitative features
        This function outputs histogram quantitative features into a .csv file.
        """
        file_name = "intensity_features.csv"
        columns = ["feature", "value"]
        data_rows = []
        for feature_name, feature_value in histogram_features_output.items():
            feature_name_this_analysis = f"intensity_{feature_name}"
            feature_value_this_analysis = feature_value
            data_rows.append((feature_name_this_analysis, feature_value_this_analysis))
        data_frame = pd.DataFrame(columns=columns, data=data_rows)
        data_frame.to_csv(os.path.join(output_directory, file_name), index=False)

    # save_histogram_as_image()
    # save_histogram_as_array()
    save_histogram_features_as_csv()
