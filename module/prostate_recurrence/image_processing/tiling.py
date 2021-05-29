"""# a set of functions for tiling of whole slide image

## read_image(...) to read whole slide image using czifile library.
This function expects the full absolute path to a .czi image image and a valid
tiling method as input parameters. Currently, the only tiling method implemented
is using czifile library. Note that an issue remains that not all Python versions
are compatible with czifile. Python 3.7.x was used to succesfully generate image tiles.
This function returns a numpy array of the whole slide image.

## create_tiles(...) to create image tiles based on the whole slide image.
This function expects a numpy array of the whole slide image and a user-defined
tile size as input parameters. By default, the tile size is set to be 512 pixels.
This function returns a dictionary of image tiles, each stored as a numpy array, and
the number of rows and columns of the tiled whole slide image.

## save_tiles(...) to save image tiles as .tif files.
This function expects a dictionary of image tiles, image id, image type, the number
of rows and columns of the tiled whole slide image, path to save the output images, and
the tile size as input parameters. The name of output image files will contain information
about image id, image type ("HE" or "PSR"), spatial location of the image tile within the
whole slide image, tile size used for tiling.
Note that a simple way that checks whether all pixels are close to white or to black
is used to exclude background image tiles from being saved.
This function saves output images using Image module from the PIL library.

"""

import os
from typing import Dict, Tuple

import numpy as np
from czifile import CziFile  # pip install czifile
from PIL import Image


def read_image(path_to_img: str, method: str = "czifile") -> np.array:
    """read .czi image into a numpy array
    note: need implementation of alterative methods
    for reading .czi image
    czifile method reads by default the image with
    highest resolution in the series
    allowed methods include "czifile"

    Parameters
    ----------
    path_to_img : str
        The full path to the image file
    method : str, optional
        Method used for reading the image, by default "czifile"

    Returns
    -------
    np.array
        The whole slide image as Numpy array
    """

    allowed_methods = ["czifile"]

    if method not in allowed_methods:
        print(f"Please use one of the allowed methods : {allowed_methods}")
        return None
    elif method == "czifile":
        with CziFile(path_to_img) as czi:
            return czi.asarray()


def create_tiles(
    img: np.array, size: int = 512
) -> Tuple[Dict[int, np.array], int, int]:
    """create image tiles with user-defined size
    note: need to add extra functionalities such as option of
    tile overlapping, offsetting, etc

    Parameters
    ----------
    img : np.array
        An input whole slide image as Numpy array
    size : int, optional
        Tile size in pixels, by default 512

    Returns
    -------
    Tuple[Dict[int, np.array], int, int]
        A Tuple of outputs, including a Dictionary of
        [tile id : tile as Numpy array], the number of
        rows and columns of tiles.
        Tile id is related to the row and column number
        of a tile, and therefore can be used to reconstruct
        the whole slide image.
    """

    height = img.shape[0]
    width = img.shape[1]

    nrow = height // size
    ncol = width // size

    dict_img_tiles = {}

    for irow in range(nrow):
        for icol in range(ncol):
            tile_id = irow * ncol + icol

            img_tile = img[
                irow * size : (irow + 1) * size,
                icol * size : (icol + 1) * size,
                :,
            ]

            dict_img_tiles[tile_id] = img_tile

    return (dict_img_tiles, nrow, ncol)


def save_tiles(
    dict_img_tiles: Dict[int, np.array],
    img_id: str,
    img_type: str,
    nrow: int,
    ncol: int,
    save_path: str,
    size: int = 512,
):
    """save image tiles into user-defined directory
    note: need to implement options to filter out all the background
    image tiles, possibly by loading a user-defined background-foreground
    mask that segments tissue.
    A temporary solution is used by filering out images with most pixels
    in colour close to black or white

    Parameters
    ----------
    dict_img_tiles : Dict[int, np.array]
        a Dictionary of
        [tile id : tile as Numpy array]
    img_id : str
        image identifier, e.g., CHIIP id
    img_type : str
        image type, e.g., "PSR" or "HE"
    nrow : int
        the number of rows of image tiles
    ncol : int
        the number of columns of image tiles
    save_path : str
        path to save the tiles into
    size : int, optional
        Tile size in pixels, by default 512
    """

    for irow in range(nrow):
        if irow % 5 == 0:
            print(f"... saving {ncol} tiles in row {irow + 1} ...")
        for icol in range(ncol):
            tile_id = irow * ncol + icol
            img_tile = dict_img_tiles[tile_id]

            if irow == 10 and icol == 10:
                print(np.sum(img_tile > 200))

            # if x% of values are near 255 (white space) or near 0, continue
            if (
                np.sum(img_tile > 200) / size ** 2 / 3
                + np.sum(img_tile < 30) / size ** 2 / 3
            ) > 0.95:
                continue

            # save image tile
            im = Image.fromarray(img_tile)
            im.save(
                os.path.join(
                    save_path,
                    "{}_{}_image_tile_{}_{}.tif".format(
                        img_type, img_id, str(irow).zfill(3), str(icol).zfill(3)
                    ),
                )
            )
