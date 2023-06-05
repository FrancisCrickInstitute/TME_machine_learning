"""# a set of functions for tiling of whole slide image

## read_image(...) to read whole slide image using czifile library.
This function expects the full absolute path to a .czi image image and a valid
tiling method as input parameters. Implemented reading methods include 'czifile'
and 'aicsimageio'.

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
from glob import glob
from typing import Dict, Tuple


import numpy as np
from aicsimageio import AICSImage  # pip install AICSImage[czi]
from czifile import CziFile  # pip install czifile
from natsort import natsorted
from PIL import Image
from skimage import color


def read_image(path_to_img: str, method: str = "aicsimageio") -> Dict[str, np.ndarray]:
    """read .czi image into scene specific numpy arrays
    by default, the method "aicsimageio" is recommended, which allows
    reading of images by scene iteratively.
    "czifile" method reads by default the image with highest resolution
    in the series, of the first scene and doesn't work well for images
    with multiple scan regions as in the CHHiP cohort of prostate data.

    Parameters
    ----------
    path_to_img : str
        The full path to the image file
    method : str, optional
        Method used for reading the image, by default "czifile"

    Returns
    -------
    Dict[str, np.array]
        A dictionary of key:value pair reflecting
        scene name : image as Numpy array
    """

    allowed_methods = ["czifile", "aicsimageio"]

    if method not in allowed_methods:
        print(f"Please use one of the allowed methods : {allowed_methods}")
    elif method == "czifile":
        with CziFile(path_to_img) as czi:
            return {"ScanRegion0": czi.asarray()[0, 0]}
    elif method == "aicsimageio":
        # read the highest resolution
        img = AICSImage(path_to_img)
        dict_imgs: Dict[str, np.ndarray] = {}
        for iid, scene in enumerate(img.scenes):
            img.set_scene(scene)
            img_data = img.get_image_data(
                "YXS", T=0, C=0, Z=0
            )  # returns 3D YXS numpy array
            dict_imgs[scene] = img_data
        return dict_imgs

    return {}


def read_image_mask(
    directory_to_image_mask: str,
    image_mask_name: str,
) -> Dict[str, np.ndarray]:
    image_mask_arrays = {}
    subdirectory_to_image_masks = natsorted(
        glob(os.path.join(directory_to_image_mask, "*"))
    )
    for subdirectory_to_image_mask in subdirectory_to_image_masks:
        scene = os.path.basename(subdirectory_to_image_mask)
        path_to_image_mask = os.path.join(subdirectory_to_image_mask, image_mask_name)
        image_mask_array = Image.open(path_to_image_mask)
        image_mask_array = np.array(image_mask_array)

        # convert to grayscale
        if image_mask_array.ndim == 3:
            if image_mask_array.shape[-1] == 4:
                image_mask_array = np.uint8(
                    color.rgb2gray(color.rgba2rgb(image_mask_array)) * 255
                )
            elif image_mask_array.shape[-1] == 3:
                image_mask_array = np.uint8(color.rgb2gray(image_mask_array) * 255)

        image_mask_arrays[scene] = image_mask_array
    return image_mask_arrays


def create_tiles(
    img: np.ndarray, size: int = 512
) -> Tuple[Dict[int, np.ndarray], int, int]:
    """create image tiles with user-defined size
    note: need to add extra functionalities such as option of
    tile overlapping, offsetting, etc

    Parameters
    ----------
    img : np.ndarray
        An input whole slide image as Numpy array
    size : int, optional
        Tile size in pixels, by default 512

    Returns
    -------
    Tuple[Dict[int, np.ndarray], int, int]
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

            if img.ndim == 3:
                img_tile = img[
                    irow * size : (irow + 1) * size,
                    icol * size : (icol + 1) * size,
                    :,
                ]
            elif img.ndim == 2:
                img_tile = img[
                    irow * size : (irow + 1) * size,
                    icol * size : (icol + 1) * size,
                ]

            dict_img_tiles[tile_id] = img_tile

    return (dict_img_tiles, nrow, ncol)


def save_tiles(
    dict_img_tiles: Dict[int, np.ndarray],
    nrow: int,
    ncol: int,
    save_path: str,
    img_id: str = "",
    img_type: str = "",
    size: int = 512,
):
    """save image tiles into user-defined directory

    Parameters
    ----------
    dict_img_tiles : Dict[int, np.array]
        a Dictionary of
        [tile id : tile as Numpy array]
    nrow : int
        the number of rows of image tiles
    ncol : int
        the number of columns of image tiles
    save_path : str
        path to save the tiles into
    img_id : str, optional
        image identifier, e.g., CHIIP id, by default ''
    img_type : str, optional
        image type, e.g., "PSR" or "HE", by default ''
    size : int, optional
        Tile size in pixels, by default 512
    """

    zero_padding = 5
    for irow in range(nrow):
        if irow % 5 == 0:
            print(f"... saving {ncol} tiles in row {irow + 1} ...")
        for icol in range(ncol):
            tile_id = irow * ncol + icol
            img_tile = dict_img_tiles[tile_id]
            im = Image.fromarray(img_tile)

            # resize if needed
            if img_tile.ndim == 2 and img_tile.shape[0] != size:
                im = im.resize((size, size))

            # save image tile
            if img_id and img_type:
                im.save(
                    os.path.join(
                        save_path,
                        "{}_{}_image_tile_{}_{}.tif".format(
                            img_type,
                            img_id,
                            str(irow).zfill(zero_padding),
                            str(icol).zfill(zero_padding),
                        ),
                    )
                )
            elif img_id and not img_type:
                im.save(
                    os.path.join(
                        save_path,
                        "{}_image_tile_{}_{}.tif".format(
                            img_id,
                            str(irow).zfill(zero_padding),
                            str(icol).zfill(zero_padding),
                        ),
                    )
                )
            elif not img_id and img_type:
                im.save(
                    os.path.join(
                        save_path,
                        "{}_image_tile_{}_{}.tif".format(
                            img_type,
                            str(irow).zfill(zero_padding),
                            str(icol).zfill(zero_padding),
                        ),
                    )
                )
            else:
                im.save(
                    os.path.join(
                        save_path,
                        "image_tile_{}_{}.tif".format(
                            str(irow).zfill(zero_padding), str(icol).zfill(zero_padding)
                        ),
                    )
                )
