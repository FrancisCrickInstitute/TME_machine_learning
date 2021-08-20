"""# stitching

## get_row_col(...) to get row and column id for an image tile.
This function expects the file name of the image tile as input and returns a tuple
of integers reflecting the row and column ids of the image tile.

## reconstruct_whole_slide(...) to stitch image tiles into a whole slide image.
This function expects a list of paths to image tiles, the path to save the whole
slide image to, the total number of rows of image tiles, the total number of columns
of image tiles, and the dimensions of each image tile as input parameters. The image
tile can be in either grey scale or rgb. This function returns the stiched whole-
slide image in the format of a PIL Image object.

## visualise_overlay(...) to generate a heatmap of feature of interest.
This function expects the raw PSR image, a list of paths to image tiles, a data frame
recording the tile-level features, a particular feature to map, the path to save the
whole slide image to, the total number of rows of image tiles, the total number of columns
of image tiles, and the size of each image tile as input parameters. This function returns
the whole-slide heatmap in hte format of a numpy array.

"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_row_col(image_name: str) -> Tuple[int, int]:
    """get the location (i.e., row and column numbers) of an image tile
    in the whole slide image
    This function returns the row and column numbers of an image tile.
    The row and column numbers are in the image name following "*tile_",
    so the image name is parsed accordingly for extraction of these numbers.

    Parameters
    ----------
    image_name : str
        name of an image tile

    Returns
    -------
    Tuple[int, int]
        the row and column number of an image tile
    """

    image_name_parts = image_name.split("_")
    position_str_tile = image_name_parts.index("tile")

    row = int(image_name_parts[position_str_tile + 1])

    col_str = image_name_parts[position_str_tile + 2]
    if "." in col_str:
        col = int(col_str.split(".")[0])
    else:
        col = int(col_str)
    return (row, col)


def reconstruct_whole_slide(
    file_paths: List[str],
    position_in_path_has_tile_row_col: int,
    save_path: str,
    nrow: int,
    ncol: int,
    dim: Tuple,
) -> Image:
    """stitch image tiles to reconstruct the whole slide image
    This function locates individual image tiles according to the row and
    columns numbers and stitches them together to reconstruct the whole
    slice image.

    Parameters
    ----------
    file_paths : List[str]
        a list of paths to image tiles
    position_in_path_has_tile_row_col : int
        position in path has tile row and column locations
    save_path : str
        the path to save stitched whole slide image
    nrow : int
        number of rows of image tiles in total for the whole slide image
    ncol : int
        number of columns of image tiles in total for the whole slide image
    dim : Tuple
        dimension of image tiles. can only be 2- or 3-dimensional, which
        reflects greyscale or rgb, respectively


    Returns
    -------
    Image
        the stitched whole-slide image in the format of a PIL Image object

    """

    assert len(dim) in [2, 3] and dim[0] == dim[1]
    size = dim[0]

    if len(dim) == 3:
        wsi_image = np.empty([nrow * size, ncol * size, 3], dtype=np.uint8)
    elif len(dim) == 2:
        wsi_image = np.empty([nrow * size, ncol * size], dtype=np.uint8)
    wsi_image.fill(255)

    for file_path in tqdm(file_paths):
        image_tile_name = file_path.split("/")[position_in_path_has_tile_row_col]
        row, col = get_row_col(image_tile_name)
        image = Image.open(file_path)
        image_arr = np.array(image)

        if np.array(image).shape[0] != size or np.array(image).shape[1] != size:
            print("this is not a proper heatmap ... ")
            return None

        if len(dim) == 3:
            wsi_image[
                row * size : (row + 1) * size, col * size : (col + 1) * size, :
            ] = image_arr[:, :, :3]
        elif len(dim) == 2:
            wsi_image[
                row * size : (row + 1) * size, col * size : (col + 1) * size
            ] = image_arr[:, :]

    if len(dim) == 3:
        im = Image.fromarray(wsi_image.astype(np.uint8))
    elif len(dim) == 2:
        im = Image.fromarray(wsi_image.astype(np.uint8), mode="L")
    im.save(save_path)

    return im


def visualise_overlay(
    raw_image: Image,
    file_paths: List[str],
    position_in_path_has_tile_row_col: int,
    features_all: pd.DataFrame,
    feature_to_map: str,
    save_path: str,
    nrow: int,
    ncol: int,
    size: int = 1024,
) -> np.ndarray:
    """heatmap and overlay single-value quantitative features on whole slide
    This function generates a standalone whole-slide heatmap colour-coding the
    tile-level single-value quantitative features and overlays this heatmap on
    top of the raw PSR whole-slide image.

    Parameters
    ----------
    raw_image : Image
        whole slide image of PSR staining.
    file_paths : List[str]
        a list of paths to .csv files recording tile-level features
    position_in_path_has_tile_row_col : int
        position in path has tile row and column locations
    features_all : pd.DataFrame
        a data frame recording tile-level features
    feature_to_map : str
        the feature to heatmap and overlay
    save_path : str
        the path to save stitched whole slide heatmap of feature
    nrow : int
        number of rows of image tiles in total for the whole slide image
    ncol : int
        number of columns of image tiles in total for the whole slide image
    size : int, optional
        size of an image tile in pixels, by default 1024

    Returns
    -------
    np.ndarray
        the whole-slide heatmap of the feature to map in a format of numpy
        array
    """
    mask = np.empty([nrow * size, ncol * size])
    mask.fill(np.nan)

    feature_value_min = features_all.loc[
        features_all.feature == feature_to_map
    ].value.min()
    feature_value_max = features_all.loc[
        features_all.feature == feature_to_map
    ].value.max()

    for file_path in tqdm(file_paths):
        image_name = file_path.split("/")[position_in_path_has_tile_row_col]
        row, col = get_row_col(image_name)

        feature_value = features_all.loc[
            (features_all.tile == file_path.split("/")[-2])
            & (features_all.feature == feature_to_map),
            "value",
        ].values[0]

        scaled_feature_value = (feature_value - feature_value_min) / (
            feature_value_max - feature_value_min + 1e-5
        )

        mask[
            row * size : (row + 1) * size,
            col * size : (col + 1) * size,
        ] = scaled_feature_value

    figsize = (int(3 * ncol / float(nrow)), 3)
    # make plot -- feature standalone
    if True:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(mask, cmap=plt.cm.Blues)
        plt.xticks([])
        plt.yticks([])

        if True:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

    # make plot -- feature overlay on raw image
    if True:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(raw_image, zorder=1)
        ax.imshow(mask, cmap=plt.cm.Blues, alpha=0.5, zorder=2)
        plt.xticks([])
        plt.yticks([])

        if True:
            plt.savefig(save_path.split(".")[0] + "_overlay.jpg", dpi=300)
        plt.show()
        plt.close()

    return mask
