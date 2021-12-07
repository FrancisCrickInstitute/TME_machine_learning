from operator import sub
import os
from glob import glob
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from skimage import color


def read_dataframe_contours(
    path_to_contours: str, path_to_centres: str
) -> Tuple[pd.DataFrame]:
    contours = pd.read_csv(path_to_contours)
    centres = pd.read_csv(path_to_centres)
    return (contours, centres)


def read_scene_arrays(
    directory_to_scenes: str, scene_image_name: str = "raw_PSR_2.5x.tif"
) -> Dict[str, np.ndarray]:
    scene_arrays = {}
    subdirectory_to_scenes = natsorted(glob(os.path.join(directory_to_scenes, "*")))
    for subdirectory_to_scene in subdirectory_to_scenes:
        scene = os.path.basename(subdirectory_to_scene)
        path_to_scene_image = os.path.join(subdirectory_to_scene, scene_image_name)
        scene_image = Image.open(path_to_scene_image)
        scene_array = np.array(scene_image)
        scene_arrays[scene] = scene_array
    return scene_arrays


def read_scene_mask_arrays(
    directory_to_scene_masks: str,
    scene_mask_image_name: str = "contour_derived_mask_2.5x.tif",
):
    scene_mask_arrays = {}
    subdirectory_to_scene_masks = natsorted(
        glob(os.path.join(directory_to_scene_masks, "*"))
    )
    for subdirectory_to_scene_mask in subdirectory_to_scene_masks:
        scene = os.path.basename(subdirectory_to_scene_mask)
        path_to_scene_mask_image = os.path.join(
            subdirectory_to_scene_mask, scene_mask_image_name
        )
        scene_mask_image = Image.open(path_to_scene_mask_image)
        scene_mask_array = np.array(scene_mask_image)
        scene_mask_arrays[scene] = scene_mask_array
    return scene_mask_arrays


def process_contours(contours: pd.DataFrame, centres: pd.DataFrame) -> pd.DataFrame:

    contours_processed = pd.DataFrame()
    for scene in centres.Scene.values:
        contours_offset = contours.copy()
        contours_offset_scene = contours_offset.copy().loc[
            contours_offset.Scene == scene
        ]
        contours_offset_scene["X"] = (
            contours_offset_scene["X"]
            - centres.loc[centres.Scene == scene, "X"].values[0]
        )
        contours_offset_scene["Y"] = (
            contours_offset_scene["Y"]
            - centres.loc[centres.Scene == scene, "Y"].values[0]
        )
        contours_offset_scene["Scene"] = scene

        contours_processed = contours_processed.append(contours_offset_scene)

    return contours_processed


def create_psr_mask(
    scene_mask_arrays: Dict[str, np.ndarray],
    scene_arrays: Dict[str, np.ndarray],
):
    psr_mask_arrays = {}

    for scene in sorted(scene_arrays.keys()):

        scene_array = scene_arrays[scene]
        scene_mask_array = scene_mask_arrays[scene]
        gray_image = color.rgb2gray(scene_array)
        gray_scene_mask = color.rgb2gray(color.rgba2rgb(scene_mask_array))

        # print(scene, gray_image.shape, gray_scene_mask.shape) # the dimensions may not match. need debugging!

        # ... find out border gray levels => PSR mask
        nrow, ncol = gray_image.shape
        border_graylevels = []
        for gl in (
            gray_image[[0, nrow - 1], :ncol].flatten().tolist()
            + gray_image[1 : nrow - 1, [0, ncol - 1]].flatten().tolist()
        ):
            border_graylevels.append(gl)
        arr_border_graylevels = np.array(border_graylevels)
        half_max_gl = np.max(arr_border_graylevels) * 0.5
        mask_half_max = gray_image >= half_max_gl
        gray_image_copy = gray_image.copy()
        gray_image_copy[mask_half_max] = 0

        # ... clean the PSR mask using gray scene mask
        # this is turned off for now, as the gray_scene_mask needs improving
        # gray_scene_mask_for_psr = gray_scene_mask > 0
        # gray_image_copy_copy = gray_image_copy.copy()
        # gray_image_copy_copy[gray_scene_mask_for_psr] = 0
        gray_image_copy_copy = gray_image_copy.copy()

        # ... minimum filter
        nonzero_pixels = gray_image_copy_copy > 0
        thresholded_scene = gray_image_copy_copy.copy()
        thresholded_scene[nonzero_pixels] = 255

        psr_mask_arrays[scene] = thresholded_scene

    return psr_mask_arrays


def save_tissue_mask(
    contours_processed: pd.DataFrame,
    scene_arrays: Dict[str, np.ndarray],
    directory_to_save_images: str,
    resolution_micron_per_pixel: float = 0.22,
    downsize_factor: float = 0.125,
    downsized_resolution: str = "2.5x",
):
    for scene in sorted(scene_arrays.keys()):

        subdirectory_to_save_images = os.path.join(directory_to_save_images, scene)
        os.makedirs(subdirectory_to_save_images, exist_ok=True)
        os.chmod(subdirectory_to_save_images, mode=0o777)

        scene_array = scene_arrays[scene]
        contours_offset_scene = contours_processed.loc[
            contours_processed.Scene == scene
        ]

        # plot overlay with contour
        fig = plt.figure(
            figsize=(scene_array.shape[1] / 300, scene_array.shape[0] / 300),
            dpi=300,
        )
        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(scene_array)
        ax.scatter(
            scene_array.shape[1] * 0.5
            + contours_offset_scene["X"].values
            / resolution_micron_per_pixel
            * downsize_factor,
            scene_array.shape[0] * 0.5
            + contours_offset_scene["Y"].values
            / resolution_micron_per_pixel
            * downsize_factor,
            edgecolor="none",
            s=4,
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(
                subdirectory_to_save_images,
                f"contour_overlaid_with_raw_{downsized_resolution}.tif",
            ),
            dpi=300,
        )
        plt.close()

        # plot overlay with mask (transparent)
        for mask_overlay_with_raw in [True, False]:
            fig = plt.figure(
                figsize=(scene_array.shape[1] / 300, scene_array.shape[0] / 300),
                dpi=300,
            )
            ax = fig.add_axes([0, 0, 1, 1])

            if mask_overlay_with_raw:
                ax.fill(
                    scene_array.shape[1] * 0.5
                    + contours_offset_scene["X"].values
                    / resolution_micron_per_pixel
                    * downsize_factor,
                    scene_array.shape[0] * 0.5
                    + contours_offset_scene["Y"].values
                    / resolution_micron_per_pixel
                    * downsize_factor,
                    fc="k",
                    ec="none",
                    alpha=0.2,
                    zorder=2,
                )
                ax.imshow(scene_array, zorder=1)
                path_to_save_figure = os.path.join(
                    subdirectory_to_save_images,
                    f"contour_derived_mask_overlaid_with_raw_{downsized_resolution}.tif",
                )
            else:
                ax.fill(
                    scene_array.shape[1] * 0.5
                    + contours_offset_scene["X"].values
                    / resolution_micron_per_pixel
                    * downsize_factor,
                    scene_array.shape[0] * 0.5
                    + contours_offset_scene["Y"].values
                    / resolution_micron_per_pixel
                    * downsize_factor,
                    fc="k",
                    ec="none",
                )
                path_to_save_figure = os.path.join(
                    subdirectory_to_save_images,
                    f"contour_derived_mask_{downsized_resolution}.tif",
                )

            ax.set_xlim(0, scene_array.shape[1] - 1)
            ax.set_ylim(scene_array.shape[0] - 1, 0)
            ax.set_aspect("equal")
            plt.axis("off")

            plt.savefig(path_to_save_figure, dpi=300)
            plt.close()


def save_psr_mask(
    scene_arrays: Dict[str, np.ndarray],
    psr_mask_arrays: Dict[str, np.ndarray],
    directory_to_save_images: str,
    downsized_resolution: str = "2.5x",
):
    for scene in sorted(scene_arrays.keys()):

        scene_array = scene_arrays[scene]

        subdirectory_to_save_images = os.path.join(directory_to_save_images, scene)
        os.makedirs(subdirectory_to_save_images, exist_ok=True)
        os.chmod(subdirectory_to_save_images, mode=0o777)

        # save psr mask
        thresholded_scene = psr_mask_arrays[scene]
        fig = plt.figure(
            figsize=(scene_array.shape[1] / 300, scene_array.shape[0] / 300),
            dpi=300,
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(thresholded_scene, cmap=plt.cm.Greys)
        plt.axis("off")
        plt.savefig(
            os.path.join(
                subdirectory_to_save_images,
                f"mask_PSR_{downsized_resolution}.tif",
            ),
            dpi=300,
        )
        plt.close()
