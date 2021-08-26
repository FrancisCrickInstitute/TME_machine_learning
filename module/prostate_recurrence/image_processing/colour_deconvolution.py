"""# colour deconvolution

## create_cmap(...) to create a color map using matplotlib library.
This function expects a valid string for a user-defined color as the input
parameter. LinearSegmentedColormap module from matplotlib.colors is called
to create a colour map between white (color value of 0) and the user-
defined color (color value of 255, assuming 8 bit). This function returns the
created color map.

## deconvolve_image(...) to perform colour deconvolution on the input image
using histomicstk library.
This function expects an input image in a format of a numpy array with a
dimension of N x M x 3. A stain map is defined within the function for the
intended colour deconvolution for extracting picroserius red staining of
collagen. Note that an issue remains that not all Python versions
are compatible with czifile. Python 3.7.x was used to succesfully perform
colour deconvolution. This function returns the outputs from colour convolution
in a format of a numpy array with a dimension of N x M x 3.

## save_deconvolved_images(...) to save deconvolved images using pillow and
matplotlib libraries.
This function expects the deconvolved image in the format of a numpy array,
the path to the raw image, stains used for deconvolution, and the colour
map for picrosirius red as input parameters. Grey scale images of each of
the channels are saved using Image module in pillow library; a separate
PSR image is saved using a color map mimicking picroserius red staining of
collagen. This function doesn't return a value

"""

import os
from typing import List, Tuple

import histomicstk as htk  # pip install histomicstk
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_cmap(
    color: str = "#D63B82",  # color for PSR
) -> matplotlib.colors.LinearSegmentedColormap:
    """create a linear segmented colour map
    This function creates and returns a linear segmented
    colour map from white (colour value of 0) to a
    user-defined colour (colour value of 255)

    Parameters
    ----------
    color : str, optional
        A string indicating the colour used for constructing
        the linear segmented colour map, by default "#D63B82".
        The default value of color reflects the hue seen in
        typical picosirus red staining.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A linear segmeneted colour map from white to the user-
        defined colour.
    """
    cvals = [0, 255]
    colors = ["white", color]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    return cmap


def deconvolve_image(image_arr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """deconvolve the input image
    This function reads an input image in the format of a numpy
    array and deconvolves it using the stain color map
    defined within this function
    Note: the library histomicstk doesn't seem to be loaded
    properly in Python 3.8 environments. Use Python 3.7 for using
    this function.

    Parameters
    ----------
    image_arr : np.ndarray
        An input image in the format of a numpy array. The dimension
        of the image should be N x M x 3.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A numpy array as the output from colour deconvolution. The
        dimension of the array is N x M x 3.
        A list of names for the stains
    """
    stain_color_map = {
        "psr": [0.174309, 0.8309804, 0.5282877],
        "nucleus1": [0.37535256, 0.61940926, 0.6852346],
        "nucleus2": [0.22552978, 0.6076955, 0.7614739],
    }
    stains = ["psr", "nucleus1", "nucleus2"]
    W = np.array([stain_color_map[st] for st in stains]).T

    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
        image_arr, W
    )
    imDeconvolvedInverted = np.subtract(255, imDeconvolved.Stains)

    return imDeconvolvedInverted, stains


def save_deconvolved_images(
    image_deconvolved: np.ndarray,
    image_path: str,
    output_directory: str,
    stains: List[str],
    cmap_psr: matplotlib.colors.LinearSegmentedColormap,
) -> None:
    """save deconvolved images

    Parameters
    ----------
    image_deconvolved : np.ndarray
        deconvolved image in the format of a numpy array. The dimension
        of the image should be N x M x 3.
    image_path : str
        path to the raw image
    output_directory : str
        directory to save deconvolved images into
    stains : List[str]
        a list of names for the stains
    cmap_psr : matplotlib.colors.LinearSegmentedColormap
        A linear segmeneted colour map from white to the user-
        defined colour
    """

    for channel, stain in enumerate(stains):
        output_subdirectory = os.path.join(output_directory, stain)
        os.makedirs(output_subdirectory, exist_ok=True)
        save_path = os.path.join(
            output_subdirectory,
            os.path.basename(image_path).split(".")[0] + f"_{stain}.tif",
        )

        stained_image = Image.fromarray(image_deconvolved[:, :, channel], mode="L")

        stained_image.save(save_path)

        if stain == "psr":
            output_subdirectory_inverted_grayscale = os.path.join(
                output_subdirectory, "inverted_grayscale"
            )
            save_path_inverted_grayscale = os.path.join(
                output_subdirectory_inverted_grayscale,
                os.path.basename(image_path).split(".")[0] + f"_{stain}.tif",
            )
            os.makedirs(output_subdirectory_inverted_grayscale, exist_ok=True)
            os.chmod(output_subdirectory_inverted_grayscale, mode=0o777)
            os.rename(save_path, save_path_inverted_grayscale)

            fig = plt.figure(figsize=(1.707, 1.707), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(stained_image, cmap=cmap_psr)
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    output_subdirectory,
                    os.path.basename(save_path_inverted_grayscale).split(".")[0]
                    + "_coloured.tif",
                ),
                dpi=300,
            )
            plt.close()
