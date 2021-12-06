import os
from glob import glob
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image


def read_scene_arrays(
    directory_to_scenes: str, scene_image_name: str = "raw_PSR_2.5x.tif"
) -> Dict[str, np.ndarray]:
    scene_arrays = {}
    subdirectory_to_scenes = natsorted(glob(directory_to_scenes + "*"))
    for subdirectory_to_scene in subdirectory_to_scenes:
        scene = os.path.basename(subdirectory_to_scene)
        path_to_scene_image = os.path.join(subdirectory_to_scene, scene_image_name)
        scene_image = Image.open(path_to_scene_image)
        scene_array = np.array(scene_image)
        scene_arrays[scene] = scene_array
    return scene_arrays


def read_dataframe_contours(
    path_to_contours: str, path_to_centres: str
) -> Tuple[pd.DataFrame]:
    contours = pd.read_csv(path_to_contours)
    centres = pd.read_csv(path_to_centres)
    return (contours, centres)


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


def save_tissue_mask(
    contours_processed: pd.DataFrame,
    scene_arrays: Dict[str, np.ndarray],
    directory_to_save_images: str,
    resolution_micron_per_pixel: float = 0.22,
    downsize_factor: float = 0.125,
):
    for scene in sorted(scene_arrays.keys()):
        scene_array = scene_arrays[scene]
        contours_offset_scene = contours_processed.loc[
            contours_processed.Scene == scene
        ]

        # plot overlay with contour
        fig, axes = plt.subplots(
            ncols=1,
            nrows=1,
            figsize=(6, 6 / scene_array.shape[0] * scene_array.shape[1]),
            dpi=300,
        )
        axes.imshow(scene_array)
        axes.scatter(
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
        plt.savefig(
            os.path.join(
                directory_to_save_images, "contour_overlaid_with_raw_2.5x.tif"
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
                    directory_to_save_images,
                    "contour_derived_mask_overlaid_with_raw_2.5x.tif",
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
                    directory_to_save_images, "contour_derived_mask_2.5x.tif"
                )

            ax.set_xlim(0, scene_array.shape[1] - 1)
            ax.set_ylim(scene_array.shape[0] - 1, 0)
            ax.set_aspect("equal")
            plt.axis("off")

            plt.savefig(path_to_save_figure, dpi=300)
            plt.close()
