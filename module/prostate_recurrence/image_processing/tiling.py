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
import pandas as pd
from typing import Dict, Tuple

import numpy as np
from czifile import CziFile  # pip install czifile
from PIL import Image
from aicsimageio import AICSImage  # pip install AICSImage[czi]
import javabridge, bioformats  # pip install javabridge, bioformats

javabridge.start_vm(class_path=bioformats.JARS)


def read_image(
    path_to_img: str, method: str = "czifile", highest_resolution: str = "20x"
) -> Dict[str, np.ndarray]:
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

    allowed_methods = ["czifile", "bioformats", "aicsimageio"]

    if method not in allowed_methods:
        print(f"Please use one of the allowed methods : {allowed_methods}")
        return {}
    elif method == "czifile":
        with CziFile(path_to_img) as czi:
            return {"ScanRegion0": czi.asarray()}
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
    elif method == "bioformats":
        omexml = bioformats.get_omexml_metadata(path_to_img)
        o = bioformats.OMEXML(omexml)
        # get image dimensions (and series)
        image_dims = []
        for i in range(o.image_count):
            image_dim = (
                o.image(i).Pixels.get_SizeX(),
                o.image(i).Pixels.get_SizeY(),
            )
            # print(image_dim)
            image_dims.append(image_dim)
        df_image_dims = pd.DataFrame(
            columns=["scene_dim_x", "scene_dim_y"], data=image_dims
        )
        all_Xs = df_image_dims.scene_dim_x.values
        all_Ys = df_image_dims.scene_dim_y.values
        res = int(highest_resolution.split("x")[0])
        resolutions = [highest_resolution]
        for j, Xi, Xj, Xk, Yi, Yj, Yk in zip(
            np.arange(all_Xs[1:].size),
            all_Xs[:-1],
            all_Xs[1:],
            all_Xs[2:],
            all_Ys[:-1],
            all_Ys[1:],
            all_Ys[2:],
        ):
            if Xi // Xj == 2:
                res /= 2
                res_str = f"{res}x"
            else:
                if Xj // Xk == 2:
                    res = int(highest_resolution.split("x")[0])
                    res_str = highest_resolution
                else:
                    break
            resolutions.append(res_str)
        # keep only rows reflecting data
        df_image_dims_keep = df_image_dims.copy().iloc[: len(resolutions)]
        df_image_dims_keep["Res"] = resolutions
        df_image_dims_keep_largest_tiff = df_image_dims_keep.loc[
            df_image_dims_keep.Res == resolutions[0]
        ]

        # read images
        dict_imgs = {}
        cnt = 0
        for image_id in df_image_dims_keep_largest_tiff.index:
            reader_this = bioformats.load_image(
                path=path_to_img, series=image_id, rescale=True
            )
            dict_imgs[f"ScanRegion{cnt}"] = (
                reader_this / reader_this.max() * 255
            ).astype(np.uint8)
            cnt += 1
        return dict_imgs


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
    nrow: int,
    ncol: int,
    save_path: str,
    img_id: str = "",
    img_type: str = "",
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

            # if irow == 10 and icol == 10:
            #    print(np.sum(img_tile > 200))

            # if x% of values are near 255 (white space) or near 0, continue
            # if (
            #    np.sum(img_tile > 200) / size ** 2 / 3
            #    + np.sum(img_tile < 30) / size ** 2 / 3
            # ) > 0.95:
            #    continue

            # save image tile
            im = Image.fromarray(img_tile)
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
