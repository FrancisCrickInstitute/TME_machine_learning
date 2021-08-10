"""validate image tiles according to the collagen content

## read_psr_image(...) to read psr image tile using pillow library.
This function expects the full absolute path to a psr image saved as .tif as the
input parameter and returns the image in the format of a numpy array.

## validate_psr_image(...) to validate psr image tile.
This function expects a numpy array of the psr image tile and validation settings
as input parameters. Settings include the minimum pixel intensity for the detection
of collagen content and the minimum number of pixels with at least the minimum intensity.
These settings may need to be tuned according to the types of tumours. This function
returns a tuple containing an integer indicating whether the input image tile is valid
based on the settings and a float reflecting the fraction of pixels with the minimum
intensity.

## write_validation_summary(...) to write a summary of psr image validation.
This function expects a data frame recording the summary and the output directory for
saving the output file into as input parameters. The data frame recording the summary
is expected to have a header of ['path_to_image_tile', 'row', 'column', 'valid',
'fraction_with_min_intensity'].

"""


import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


def read_psr_image(path_to_img: str) -> np.ndarray:
    """read .tif psr image into a numpy array

    Parameters
    ----------
    path_to_img : str
        path to the psr image

    Returns
    -------
    np.array
        psr image in the format of a numpy array
    """

    image_arr = np.array(Image.open(path_to_img))

    return image_arr


def validate_psr_image(
    image: np.ndarray,
    min_intensity: int = 200,
    min_fraction: float = 0.005,
    # method: str = "fraction_with_min_intensity",
) -> Tuple[int, float]:
    """validate psr image tile
    This function reads an input psr image tile in the format of a numpy
    array and returns an integer indicating the validity of the image and
    a float reflecting the minimum number of pixels with at least the minimum
    intensity.

    Parameters
    ----------
    image : np.ndarray
        An input image in the format of a numpy array. The dimension should
        be N x M.
    min_intensity : int, optional
        The minimum intensity a pixel has to reach so as to be counted as a
        valid pixel, by default 200
    min_fraction : float, optional
        The minimum fraction of pixels the image has to have so as to be
        considered a valid image tile with enough psr content, by default 0.1

    Returns
    -------
    List[int, float]
        A tuple containing an integer indicating whether the image is valid (1 for
        valid while 0 for invalid) and a float reflecting the fraction of pixels
        counted as valid pixels.
    """

    assert image.ndim == 2 and image.max() <= 255

    (size_x, size_y) = image.shape

    fraction_with_min_intensity = np.sum(image >= min_intensity) / size_x / size_y

    if fraction_with_min_intensity >= min_fraction:
        return (1, fraction_with_min_intensity)
    else:
        return (0, fraction_with_min_intensity)


def write_validation_summary(summary: pd.DataFrame, output_directory: str) -> None:
    """write the validation summary as a .csv file

    Parameters
    ----------
    summary : pd.DataFrame
        A summary dataframe containing information about the validation of image tiles,
        with a header of ['path_to_image_tile', 'row', 'column', 'valid',
        'fraction_with_min_intensity']
    output_directory : str
        Directory to save the summary file into.
    """

    filename_summary = "summary_valid_image_tiles.csv"
    summary.to_csv(os.path.join(output_directory, filename_summary), index=False)
