"""# stitching

## reconstruct_whole_slide(...) to stitch image tiles into a whole slide image.
This function expects a list of paths to image tiles, the path to save the whole
slide image to, the total number of rows of image tiles, the total number of columns
of image tiles, and the dimensions of each image tile as input parameters. The image
tile can be in either grey scale or rgb.

"""

from typing import List, Tuple

import numpy as np
from PIL import Image


def reconstruct_whole_slide(
    file_paths: List[str], save_path: str, nrow: int, ncol: int, dim: Tuple
) -> None:
    """stitch image tiles to reconstruct the whole slide image
    This function locates individual image tiles according to the row and
    columns numbers and stitches them together to reconstruct the whole
    slice image.

    Parameters
    ----------
    file_paths : List[str]
        a list of paths to image tiles
    save_path : str
        the path to save stitched whole slide image
    nrow : int
        number of rows of image tiles in total for the whole slide image
    ncol : int
        number of columns of image tiles in total for the whole slide image
    dim : Tuple
        dimension of image tiles. can only be 2- or 3-dimensional, which
        reflects greyscale or rgb, respectively
    """

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

    assert len(dim) in [2, 3] and dim[0] == dim[1]
    size = dim[0]

    if len(dim) == 3:
        wsi_image = np.empty([nrow * size, ncol * size, 3], dtype=np.uint8)
    elif len(dim) == 2:
        wsi_image = np.empty([nrow * size, ncol * size], dtype=np.uint8)
    wsi_image.fill(255)

    for file_path in file_paths:
        image_name = file_path.split("/")[-1]
        row, col = get_row_col(image_name)
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
