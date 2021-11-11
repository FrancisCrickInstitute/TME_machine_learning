"""[summary]

"""


import os
import numpy as np
import pandas as pd
from aicsimageio import AICSImage   # pip install AICSImage[czi]
import javabridge, bioformats    # pip install javabridge, bioformats
from skimage import color, filters, morphology, transform
from typing import Dict, Tuple
from PIL import Image, ImageOps

javabridge.start_vm(class_path=bioformats.JARS)


def read_image_information(
    path_to_img: str,
    method: str = "aicsimageio",
    resolution: str = "20x",
) -> pd.DataFrame:
    slide_id = os.path.basename(path_to_img).split('_')[0]
    if method == "aicsimageio":
        img = AICSImage(path_to_img)
        img_info_cols = [
            'slide_id', 'scene_name', 'scene_dim_x', 'scene_dim_y', 'resolution'
        ]
        img_info_rows = []

        for iid, scene in enumerate(img.scenes):
            img.set_scene(scene)
            dim_x, dim_y = img.dims['X'][0], img.dims['Y'][0]
            img_info_rows.append(
                (slide_id, scene, dim_x, dim_y, resolution)
            )

        img_info = pd.DataFrame(
            columns=img_info_cols,
            data=img_info_rows
        )

        return img_info

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

            image_dims.append(
                image_dim
            )
        df_image_dims = pd.DataFrame(
            columns=['scene_dim_x', 'scene_dim_y'],
            data=image_dims
        )

        all_Xs = df_image_dims.scene_dim_x.values;
        all_Ys = df_image_dims.scene_dim_y.values
        res = int(resolution.split('x')[0]);
        resolutions = [f'{res}x']
        for j, Xi, Xj, Xk, Yi, Yj, Yk in zip(
            np.arange(all_Xs[1:].size),
            all_Xs[:-1], all_Xs[1:], all_Xs[2:],
            all_Ys[:-1], all_Ys[1:], all_Ys[2:]
        ):
            if Xi // Xj == 2:
                res /= 2;
                res_str = f'{res}x'
            else:
                if Xj // Xk == 2:
                    res = int(resolution.split('x')[0]);
                    res_str = f'{res}x'
                else:
                    break
            resolutions.append(res_str)# keep only rows reflecting data
        df_image_dims_keep = df_image_dims.copy().iloc[:len(resolutions)]
        df_image_dims_keep['resolution'] = resolutions

        df_image_information = df_image_dims_keep.loc[
            df_image_dims_keep.resolution == resolution
        ]
        num_scan_regions = df_image_information.shape[0]
        scan_regions = [f'ScanRegion{idx}' for idx in range(num_scan_regions)]
        df_image_information['slide_id'] = [slide_id for _ in range(num_scan_regions)]
        df_image_information['scene_name'] = scan_regions
        print(f'... {num_scan_regions} scenes in total.')

        return df_image_information

    return pd.DataFrame()



def read_image(
    path_to_img: str,
    method: str = "bioformats",
    highest_resolution: int = 20,
    low_resolution_keep: int = 5
) -> Dict[str, Dict[str, np.ndarray]]:
    """read .czi image into a numpy array
    "aicsimageio" method reads the image only with the highest resolution
    in the series;
    "bioformats" method reads the image with the highest resolution and one
    low resolution in the series.
    when there are multiple series (i.e., scanned regions), image from each
    series is read.

    Parameters
    ----------
    path_to_img : str
        The full path to the image file
    method : str, optional
        Method used for reading the image, by default "aicsimageio"
    highest_resolution: int, optional
        Highest resolution of the image series, by default 20
    low_resolution_keep: int, optional
        Low resolution of the image to be kept, by default 5

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A dictionary of the whole slide image as Numpy array with keys
        reflecting different resolutions and different series of images
    """

    allowed_methods = ["aicsimageio", "bioformats"]

    if method not in allowed_methods:
        print(f"Please use one of the allowed methods : {allowed_methods}")
    elif method == "aicsimageio":
        img = AICSImage(path_to_img)
        dict_imgs_with_res = {}
        dict_imgs: Dict[str, np.ndarray] = {}
        for iid, scene in enumerate(img.scenes):
            img.set_scene(scene)
            img_data = img.get_image_data(
                "YXS", T=0, C=0, Z=0
            )  # returns 3D YXS numpy array
            dict_imgs[scene] = img_data
        dict_imgs_with_res[highest_resolution] = dict_imgs
        return dict_imgs_with_res
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
            print(image_dim)

            image_dims.append(
                image_dim
            )
        df_image_dims = pd.DataFrame(
            columns = ['X','Y'],
            data=image_dims
        )
        all_Xs = df_image_dims.X.values; all_Ys = df_image_dims.Y.values
        res = highest_resolution; resolutions = [f'{res}x']
        for j, Xi, Xj, Xk, Yi, Yj, Yk in zip(
            np.arange(all_Xs[1:].size),
            all_Xs[:-1], all_Xs[1:], all_Xs[2:],
            all_Ys[:-1], all_Ys[1:], all_Ys[2:]
        ):

            if Xi // Xj == 2:
                res /= 2;
                res_str = f'{res}x'
            else:
                if Xj // Xk == 2:
                    res = highest_resolution;
                    res_str = f'{res}x'
                else:
                    break
            resolutions.append(res_str)
        # keep only rows reflecting data
        df_image_dims_keep = df_image_dims.copy().iloc[:len(resolutions)]
        df_image_dims_keep['Res'] = resolutions
        keep_res = f'{low_resolution_keep}x'
        if keep_res not in df_image_dims_keep.Res.unique():
            keep_res = f'{low_resolution_keep}.0x'
        df_image_dims_keep_small_tiff = df_image_dims_keep.loc[
            df_image_dims_keep.Res == keep_res
        ]
        df_image_dims_keep_large_tiff = df_image_dims_keep.loc[
            df_image_dims_keep.Res == resolutions[0]
        ]
        # read images
        dict_imgs_with_res = {}
        for res, df in zip(
            [
                keep_res,
                resolutions[0],
            ],
            [
                df_image_dims_keep_small_tiff,
                df_image_dims_keep_large_tiff
            ]
        ):
            dict_imgs_with_res[res] = {}
            cnt = 0
            for image_id in df.index:
                reader_this = bioformats.load_image(
                    path=path_to_img,
                    series=image_id,
                    rescale=True
                )
                dict_imgs_with_res[res][
                    f'ScanRegion{cnt}'
                ] = reader_this
                cnt += 1
        return dict_imgs_with_res

    return {}


def resize(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    return transform.resize(image, target_size)


def create_tissue_mask(
    dict_imgs: Dict[str, np.ndarray],
    gaussian_blur_sigma: float = 2.0,
    resize_factor: int = 4,
    erosion_n: int = 5,
    dilation_n: int = 5,
) -> Dict[str, np.ndarray]:

    def erosion(
        image: np.ndarray
    ):
        if erosion_n:
            eroded = morphology.erosion(image, morphology.square(erosion_n))
        else:
            eroded = image
        return eroded

    def dilation(
        image: np.ndarray
    ):
        if dilation_n:
            dilated = morphology.dilation(image, morphology.square(dilation_n))
        else:
            dilated = image
        return dilated

    def fill_holes(
        image: np.ndarray
    ):
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.max()
        to_fill = image
        filled = morphology.reconstruction(seed, to_fill, method='erosion')
        return filled

    dict_masks: Dict[str, np.ndarray] = {}

    for scene, image_arr in dict_imgs.items():
        image_arr[np.where(image_arr==0)] = image_arr.max() # remove the black regions at the edge
        gray_image = color.rgb2gray(image_arr)
        blurred_image = filters.gaussian(gray_image, sigma=gaussian_blur_sigma)
        if resize_factor == 1:
            blurred_image_small = blurred_image
        else:
            original_size = (gray_image.shape[0], gray_image.shape[1])
            small_size = (
                original_size[0] // resize_factor,
                original_size[1] // resize_factor,
            )
            blurred_image_small = transform.resize(blurred_image, small_size)

        val = filters.threshold_isodata(blurred_image_small)
        mask = blurred_image_small < val

        # hole filling => erosion => dilation
        mask_filled = fill_holes(mask)
        mask_eroded = erosion(mask_filled)
        mask_dilated = dilation(mask_eroded)

        # erosion => hole filling => dilation
        # mask_eroded = erosion(mask)
        # mask_filled = fill_holes(mask_eroded)
        # mask_dilated = dilation(mask_filled)

        mask_dilated_original_size = transform.resize(mask_dilated, original_size)

        dict_masks[scene] = mask_dilated_original_size

    return dict_masks


def create_mask_tiles(
    img: np.ndarray,
    size: int = 512
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


def save_low_res_whole_slide_image(
    dict_imgs: Dict[str, np.ndarray],
    dict_masks: Dict[str, np.ndarray],
    save_path: str,
    resolution: str = '5x'
):
    for img_name in dict_imgs.keys():
        img = (dict_imgs[img_name] * 255).astype(np.uint8)
        mask = dict_masks[img_name]

        im = Image.fromarray(img)
        ma = ImageOps.grayscale(Image.fromarray(mask*255))
        ma = ma.convert(mode="L")

        save_path_subdir = os.path.join(
            save_path,
            f"{img_name}"
        )
        os.makedirs(save_path_subdir, exist_ok=True)

        im.save(
            os.path.join(
                save_path_subdir,
                f"raw_PSR_{resolution}.tif"
            )
        )
        ma.save(
            os.path.join(
                save_path_subdir,
                f"mask_{resolution}.tif"
            )
        )