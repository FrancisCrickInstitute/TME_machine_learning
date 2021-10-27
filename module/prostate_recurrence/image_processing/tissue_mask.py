"""[summary]

"""


import os
import numpy as np
from aicsimageio import AICSImage
from skimage import color, filters, morphology, transform
from typing import Dict, Tuple
from PIL import Image


def read_image(path_to_img: str, method: str = "aicsimageio") -> Dict[str, np.ndarray]:
    """read .czi image into a numpy array
    "aicsimageio" method reads the image with highest resolution in
    the series; when there are multiple series (i.e., scanned regions),
    all

    Parameters
    ----------
    path_to_img : str
        The full path to the image file
    method : str, optional
        Method used for reading the image, by default "aicsimageio"

    Returns
    -------
    np.array
        The whole slide image as Numpy array
    """

    allowed_methods = ["aicsimageio"]

    if method not in allowed_methods:
        print(f"Please use one of the allowed methods : {allowed_methods}")
    elif method == "aicsimageio":
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


def create_tissue_mask(
    dict_imgs: Dict[str, np.ndarray],
    gaussian_blur_sigma: float = 2.0,
    resize_factor: int = 4,
    erosion_n: int = 5,
    dilation_n: int = 50,
) -> Dict[str, np.ndarray]:

    dict_masks: Dict[str, np.ndarray] = {}

    for scene, image_arr in dict_imgs.items():
        gray_image = color.rgb2gray(image_arr)
        blurred_image = filters.gaussian(gray_image, sigma=gaussian_blur_sigma)
        original_size = (gray_image.shape[0], gray_image.shape[1])
        small_size = (
            original_size[0] // resize_factor,
            original_size[1] // resize_factor,
        )
        blurred_image_small = transform.resize(blurred_image, small_size)
        val = filters.threshold_otsu(blurred_image_small)
        mask = blurred_image_small < val
        mask_eroded = morphology.erosion(mask, morphology.square(erosion_n))
        mask_dilated = morphology.dilation(mask_eroded, morphology.square(dilation_n))
        mask_dilated_original_size = transform.resize(mask_dilated, original_size)

        dict_masks[scene] = mask_dilated_original_size

    return dict_masks


def create_mask_tiles(
    img: np.ndarray, size: int = 512
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
                irow * size : (irow + 1) * size, icol * size : (icol + 1) * size
            ]

            dict_img_tiles[tile_id] = img_tile

    return (dict_img_tiles, nrow, ncol)


def save_mask_tiles(
    dict_img_tiles: Dict[int, np.ndarray],
    nrow: int,
    ncol: int,
    save_path: str,
    size: int = 512,
):
    zero_padding = 5
    for irow in range(nrow):
        if irow % 5 == 0:
            print(f"... saving {ncol} tiles in row {irow + 1} ...")
        for icol in range(ncol):
            tile_id = irow * ncol + icol
            img_tile = dict_img_tiles[tile_id]
            im = Image.fromarray(img_tile, mode="L")

            im.save(
                os.path.join(
                    save_path,
                    "image_tile_{}_{}_mask.tif".format(
                        str(irow).zfill(zero_padding), str(icol).zfill(zero_padding)
                    ),
                )
            )
