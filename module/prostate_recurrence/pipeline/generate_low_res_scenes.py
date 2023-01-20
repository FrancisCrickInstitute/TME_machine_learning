import argparse
import os
import sys
from glob import glob
from typing import Dict, Tuple

import javabridge, bioformats
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image, ImageOps

javabridge.start_vm(class_path=bioformats.JARS)

parser = argparse.ArgumentParser(prog="tme-ml-raw-data")
parser.add_argument(
    "--raw_data_path",
    dest="raw_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path containing raw data.",
)
parser.add_argument(
    "--processed_data_path",
    dest="processed_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path saving processed data.",
)
parser.add_argument(
    "--raw_data_type",
    dest="raw_data_type",
    action="store",
    type=str,
    default="PSR",
    help="options are PSR, HandE, or Both. currently only PSR is implemented.",
)
parser.add_argument(
    "--raw_data_res",
    dest="raw_data_res",
    action="store",
    type=str,
    default="20x",
    help="options are 20x, 10x. By default it's 20x",
)
parser.add_argument(
    "--mask_res",
    dest="mask_res",
    action="store",
    type=str,
    default="2.5x",
    help="By default it's 2.5x",
)


args = parser.parse_args()
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
MASK_RES = args.mask_res
PROCESSED_DATA_PATH = args.processed_data_path


def read_image(
    path_to_img: str,
    method: str = "bioformats",
    highest_resolution: str = "20x",
    resolution: str = "5x",
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
        # read the highest resolution
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

        keep_res = resolution
        if keep_res not in df_image_dims_keep.Res.unique():
            resolution_int = int(resolution.split("x")[0])
            keep_res = f"{resolution_int}.0x"
        df_image_dims_keep_small_tiff = df_image_dims_keep.loc[
            df_image_dims_keep.Res == keep_res
        ]
        # read images
        dict_imgs_with_res = {}
        for res, df in zip(
            [
                keep_res,
                # resolutions[0],
            ],
            [
                df_image_dims_keep_small_tiff,
                # df_image_dims_keep_large_tiff
            ],
        ):
            dict_imgs_with_res[res] = {}
            cnt = 0
            for image_id in df.index:
                reader_this = bioformats.load_image(
                    path=path_to_img, series=image_id, rescale=True
                )
                dict_imgs_with_res[res][f"ScanRegion{cnt}"] = reader_this
                cnt += 1
        return dict_imgs_with_res

    return {}


def save_low_res_whole_slide_image(
    dict_imgs: Dict[str, np.ndarray],
    save_path: str,
    resolution: str = "5x",
):
    for img_name in dict_imgs.keys():
        img = (dict_imgs[img_name] * 255).astype(np.uint8)

        im = Image.fromarray(img)

        save_path_subdir = os.path.join(save_path, f"{img_name}")
        os.makedirs(save_path_subdir, exist_ok=True)

        im.save(os.path.join(save_path_subdir, f"raw_PSR_{resolution}.tif"))


if __name__ == "__main__":
    MASK_RES_0 = MASK_RES
    if MASK_RES == "5x":
        MASK_RES_0 = "5.0x"

    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    for path in all_paths_to_data:
        print(f"> processing path : {path}")
        slide_id = "_".join(
            [os.path.basename(path).split("_")[0], os.path.basename(path).split("_")[1]]
        )
        dict_imgs_with_res = read_image(
            path_to_img=path,
            method="bioformats",
            highest_resolution=RAW_DATA_RES,
            resolution=MASK_RES,
        )

        output_directory_low_res_scenes = os.path.join(
            PROCESSED_DATA_PATH, slide_id, "low_res_scenes", f"resolution_{MASK_RES}"
        )
        os.makedirs(output_directory_low_res_scenes, exist_ok=True)

        print("... saving low resolution whole slide")
        mask_summary = save_low_res_whole_slide_image(
            dict_imgs=dict_imgs_with_res[MASK_RES_0],
            save_path=output_directory_low_res_scenes,
            resolution=MASK_RES_0,
        )
