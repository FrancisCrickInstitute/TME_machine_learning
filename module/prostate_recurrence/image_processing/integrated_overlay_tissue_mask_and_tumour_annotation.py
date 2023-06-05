"""This script contains functions for overlaying tissue mask with tumour annotation

## read_image_scene_information(...) to obtain the information about image scenes,
including the relative positions of individual scenes relative to the whole slide.

## read_tumour_annotation(...) to read tiles of tumour contour line overlay with raw
PSR+H and tiles of binary tumour mask.

## process_tumour_annotation(...) calling stitch_da_tiles(...) to stitch tiles of binary
tumour mask to whole slide binary tumour mask in Numpy array.

Inputs
- tiles containing nonzero tumour areas (ICR tiling system)
- tiles containing nonzero tissue areas (Crick tiling system)

Outputs
- overlay of tumour annotation on top of tissue mask, at the whole slide level
- same as above, saved as numpy array
- a summary data frame of percentage of tumour area in individual tiles

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from glob import glob
from natsort import natsorted
from typing import Dict, Tuple
from aicsimageio import AICSImage


def read_image_scene_information(
    path_to_image: str,
    resolution_micron_per_pixel=0.22,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """obtain information about scenes in the image

    Function "AICSImage" from the aicsimageio library is used to obtain information
    about image scenes. Positions of individual scenes relative to the whole slide
    are obtained via image metadata, i.e., image.metadata

    Parameters
    ----------
    path_to_image : str
        path to the image in a .czi format.
    resolution_micron_per_pixel : float, optional
        image resolution in micron per pixel, by default 0.22

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        A dataframe containing information about image scenes.
        A dictionary of positions of the four corners of the whole slide.
    """
    image = AICSImage(path_to_image)

    # from reading scenes
    scene_information = {}
    scene_information_dataframe_cols = ["scene", "width", "height"]
    scene_information_dataframe_rows = []
    for scene in image.scenes:
        image.set_scene(scene)
        image_width = image.dims.X
        image_height = image.dims.Y
        scene_information[scene] = {"width": image_width, "height": image_height}
        scene_information_dataframe_rows.append([scene, image_width, image_height])
    scene_information_dataframe = pd.DataFrame(
        data=scene_information_dataframe_rows, columns=scene_information_dataframe_cols
    )
    #     print(scene_information_dataframe.head(5))

    # from reading metadata
    metadata = image.metadata
    scene_center_positions = []
    scene_names = []
    slide_information = {}
    for elem in metadata.iter():
        if elem.tag == "CenterPosition" and elem.text not in scene_center_positions:
            scene_center_positions.append(elem.text)
        elif elem.tag == "Scene" and elem.attrib["Name"] not in scene_names:
            scene_names.append(elem.attrib["Name"])
        elif elem.tag in ["SizeX", "SizeY"]:
            slide_information[elem.tag] = elem.text
    # print(slide_information)

    slide_topleft_x_in_um = 1e5
    slide_topleft_y_in_um = 1e5
    slide_bottomright_x_in_um = -1e5
    slide_bottomright_y_in_um = -1e5

    scene_information_dataframe_2_cols = [
        "scene",
        "xcentre_in_um",
        "ycentre_in_um",
        "scene_topleft_x_in_um",
        "scene_topleft_y_in_um",
        "scene_bottomright_x_in_um",
        "scene_bottomright_y_in_um",
    ]
    scene_information_dataframe_2_rows = []

    for scene, scene_center in zip(scene_names, scene_center_positions):
        xcstr, ycstr = scene_center.split(",")
        xc_in_um, yc_in_um = int(float(xcstr)), int(float(ycstr))

        #     xc_in_allowed_scan_area = xc - allowed_scan_area_xstart
        #     yc_in_allowed_scan_area = yc - allowed_scan_area_ystart

        scene_width = scene_information[scene]["width"]
        scene_height = scene_information[scene]["height"]
        scene_width_in_um = scene_width * resolution_micron_per_pixel
        scene_height_in_um = scene_height * resolution_micron_per_pixel

        scene_topleft_x_in_um = xc_in_um - scene_width_in_um * 0.5
        scene_topleft_y_in_um = yc_in_um - scene_height_in_um * 0.5
        scene_bottomright_x_in_um = xc_in_um + scene_width_in_um * 0.5
        scene_bottomright_y_in_um = yc_in_um + scene_height_in_um * 0.5

        if scene_topleft_x_in_um < slide_topleft_x_in_um:
            slide_topleft_x_in_um = scene_topleft_x_in_um
        if scene_topleft_y_in_um < slide_topleft_y_in_um:
            slide_topleft_y_in_um = scene_topleft_y_in_um
        if scene_bottomright_x_in_um > slide_bottomright_x_in_um:
            slide_bottomright_x_in_um = scene_bottomright_x_in_um
        if scene_bottomright_y_in_um > slide_bottomright_y_in_um:
            slide_bottomright_y_in_um = scene_bottomright_y_in_um

        #     xstart = xc_in_allowed_scan_area - scene_width *0.5
        #     ystart = yc_in_allowed_scan_area - scene_height *0.5

        scene_information_dataframe_2_rows.append(
            [
                scene,
                xc_in_um,
                yc_in_um,
                scene_topleft_x_in_um,
                scene_topleft_y_in_um,
                scene_bottomright_x_in_um,
                scene_bottomright_y_in_um,
            ]
        )
    scene_information_dataframe_2 = pd.DataFrame(
        data=scene_information_dataframe_2_rows,
        columns=scene_information_dataframe_2_cols,
    )

    scene_information_dataframe_complete = pd.merge(
        left=scene_information_dataframe,
        right=scene_information_dataframe_2,
        on="scene",
    )
    # print(scene_information_dataframe_complete)

    # slide information
    slide_information.update(
        {
            "slide_topleft_x_in_um": slide_topleft_x_in_um,
            "slide_topleft_y_in_um": slide_topleft_y_in_um,
            "slide_bottomright_x_in_um": slide_bottomright_x_in_um,
            "slide_bottomright_y_in_um": slide_bottomright_y_in_um,
        }
    )
    # print(slide_information)

    return scene_information_dataframe_complete, slide_information


def read_tumour_annotation(
    input_directories: Dict[str, str],
    image_tile_name_pattern: str = "Da*",
    tile_size: int = 2000,
    downscale_factor: float = 0.125,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """read tiles of tumour annotation

    Parameters
    ----------
    input_directories : Dict[str, str]
        Directories to tumour overlay tiles (contour lines overlaid with raw PSR+H image)
        and binary tumour mask tiles. The latter is used for tissue/tumour overlay.
    image_tile_name_pattern : str, optional
        Naming patterns for image tiles using the ICR tiling system, by default "Da*"
    tile_size : int, optional
        Width of each image tile in pixels, by default 2000
    downscale_factor : float, optional
        Downscaling factor when reading in image tiles, by default 0.125

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        A dictionary of tumour overlay tiles and a dictionary of binary tumour mask
        tiles. Keys are the tile names according to the ICR tiling system.
    """
    tile_size_downscaled = int(tile_size * downscale_factor)

    # overlay of tumour contour on raw PSR+H
    if "input_directory_annotation_overlay" in input_directories.keys():
        input_directory_annotation_overlay = input_directories[
            "input_directory_annotation_overlay"
        ]
        paths_to_annotation_overlay_image_tiles = natsorted(
            glob(
                os.path.join(
                    input_directory_annotation_overlay, image_tile_name_pattern
                )
            )
        )
        annotation_overlay_image_arrays = [
            np.array(
                Image.open(path).resize((tile_size_downscaled, tile_size_downscaled))
            )
            for path in paths_to_annotation_overlay_image_tiles
        ]
        dict_annotation_overlay_image_arrays = {
            os.path.splitext(os.path.basename(path))[0]: image_array
            for path, image_array in zip(
                paths_to_annotation_overlay_image_tiles, annotation_overlay_image_arrays
            )
        }
        da_names = dict_annotation_overlay_image_arrays.keys()
        print(da_names)
    else:
        dict_annotation_overlay_image_arrays = {}

    # binary tumour mask
    if "input_directory_annotation_binary" in input_directories.keys():
        input_directory_annotation_binary = input_directories[
            "input_directory_annotation_binary"
        ]
        paths_to_annotation_binary_image_tiles = natsorted(
            glob(
                os.path.join(input_directory_annotation_binary, image_tile_name_pattern)
            )
        )
        annotation_binary_image_arrays = [
            np.array(
                ImageOps.grayscale(Image.open(path)).resize(
                    (tile_size_downscaled, tile_size_downscaled)
                )
            )
            for path in paths_to_annotation_binary_image_tiles
        ]
        dict_annotation_binary_image_arrays = {
            os.path.splitext(os.path.basename(path))[0]: image_array
            for path, image_array in zip(
                paths_to_annotation_binary_image_tiles, annotation_binary_image_arrays
            )
        }
        da_names = dict_annotation_binary_image_arrays.keys()
        print(da_names)
    else:
        dict_annotation_binary_image_arrays = {}

    return dict_annotation_overlay_image_arrays, dict_annotation_binary_image_arrays


def stitch_da_tiles(
    slide_information: Dict[str, float],
    dict_image_arrays: Dict[str, np.ndarray],
    image_dim: int = 2,
    tile_size: int = 2000,
    downscale_factor: float = 0.125,
) -> np.ndarray:
    """stitch tiles of tumour annotation

    Input image tiles, with a name starting with "Da", were created using the ICR tiling system.

    Parameters
    ----------
    slide_information : Dict[str, float]
        A dictionary of positions of the four corners of the whole slide.
    dict_image_arrays : Dict[str, np.ndarray]
        A dictionary of binary tumour mask tiles. Keys are the tile names according
        to the ICR tiling system.
    image_dim : int, optional
        Dimensionality of the image. 2 for greyscale image and 3 for RGB image, by default 2
    tile_size : int, optional
        Width of each image tile in pixels, by default 2000
    downscale_factor : float, optional
        Downscaling factor applied when reading in image tiles, by default 0.125

    Returns
    -------
    np.ndarray
        Stitched whole slide binary tumour mask in Numpy array.
    """
    slide_width = int(float(slide_information["SizeX"]) * downscale_factor)
    slide_height = int(float(slide_information["SizeY"]) * downscale_factor)

    tile_size_downscaled = int(tile_size * downscale_factor)
    ncols_outer = int(slide_width // tile_size_downscaled + 1)
    nrows_outer = int(slide_height // tile_size_downscaled + 1)

    if image_dim == 2:
        stitched_downscaled = np.zeros(
            (nrows_outer * tile_size_downscaled, ncols_outer * tile_size_downscaled)
        )
    if image_dim == 3:
        stitched_downscaled = np.zeros(
            (nrows_outer * tile_size_downscaled, ncols_outer * tile_size_downscaled, 3)
        )

    # print(stitched_downscaled.shape)

    for da_name, image_array in dict_image_arrays.items():
        image_tile_id = int(da_name.split("Da")[1])
        image_tile_row = image_tile_id // ncols_outer
        image_tile_col = image_tile_id % ncols_outer
        if image_dim == 2:
            stitched_downscaled[
                image_tile_row
                * tile_size_downscaled : (image_tile_row + 1)
                * tile_size_downscaled,
                image_tile_col
                * tile_size_downscaled : (image_tile_col + 1)
                * tile_size_downscaled,
            ] = image_array.copy()
        if image_dim == 3:
            stitched_downscaled[
                image_tile_row
                * tile_size_downscaled : (image_tile_row + 1)
                * tile_size_downscaled,
                image_tile_col
                * tile_size_downscaled : (image_tile_col + 1)
                * tile_size_downscaled,
                :,
            ] = image_array.copy()

    return stitched_downscaled


def process_tumour_annotation(
    slide_information: Dict[str, float],
    dict_annotation_binary_image_arrays: Dict[str, np.ndarray],
    tile_size: int = 2000,
    downscale_factor: float = 0.125,
) -> np.ndarray:
    """processing the stitching of binary tumour masks

    Parameters
    ----------
    slide_information : Dict[str, float]
        A dictionary of positions of the four corners of the whole slide.
    dict_annotation_binary_image_arrays : Dict[str, np.ndarray]
        A dictionary of binary tumour mask tiles. Keys are the tile names according
        to the ICR tiling system.
    tile_size : int, optional
        Width of each image tile in pixels, by default 2000
    downscale_factor : float, optional
        Downscaling factor applied when reading in image tiles, by default 0.125

    Returns
    -------
    np.ndarray
        Stitched whole slide binary tumour mask in Numpy array.
    """
    stitch_downscaled_annotation_binary = stitch_da_tiles(
        slide_information=slide_information,
        dict_image_arrays=dict_annotation_binary_image_arrays,
        tile_size=tile_size,
        downscale_factor=downscale_factor,
    )
    plt.imshow(stitch_downscaled_annotation_binary, cmap=plt.cm.Greys)

    return stitch_downscaled_annotation_binary


def read_tissue_mask(
    camp_input_directory_tissue_mask_tiles, tile_size=2000, downscale_factor=0.125
):
    tile_size_downscaled = int(tile_size * downscale_factor)

    camp_input_directory_tissue_mask_tiles_scene_subdirs = natsorted(
        glob(os.path.join(camp_input_directory_tissue_mask_tiles, "ScanRegion*"))
    )
    scene_names = [
        os.path.basename(subdir)
        for subdir in camp_input_directory_tissue_mask_tiles_scene_subdirs
    ]

    # print("loading tissue mask image tiles")
    tissue_mask_image_tile_name_pattern = "image_tile*"
    dict_tissue_mask_tiles_scene_level = {}

    for scene, subdir in zip(
        scene_names, camp_input_directory_tissue_mask_tiles_scene_subdirs
    ):
        # print(scene)
        paths_to_tissue_mask_image_tiles = natsorted(
            glob(os.path.join(subdir, tissue_mask_image_tile_name_pattern))
        )

        image_tile_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in paths_to_tissue_mask_image_tiles
        ]

        tissue_mask_image_arrays = {
            image_tile_name: np.array(
                Image.open(path).resize((tile_size_downscaled, tile_size_downscaled))
            )
            for image_tile_name, path in zip(
                image_tile_names, paths_to_tissue_mask_image_tiles
            )
        }
        dict_tissue_mask_tiles_scene_level[scene] = tissue_mask_image_arrays

    return dict_tissue_mask_tiles_scene_level


def stitch_image_tiles(
    slide_information,
    scene_information,
    dict_image_arrays_scene_level,
    image_dim=2,
    tile_size=2000,
    downscale_factor=0.125,
    resolution_micron_per_pixel=0.22,
):
    tile_size_downscaled = int(tile_size * downscale_factor)

    (
        slide_bottomright_x_in_um,
        slide_bottomright_y_in_um,
        slide_topleft_x_in_um,
        slide_topleft_y_in_um,
    ) = (
        slide_information["slide_bottomright_x_in_um"],
        slide_information["slide_bottomright_y_in_um"],
        slide_information["slide_topleft_x_in_um"],
        slide_information["slide_topleft_y_in_um"],
    )

    slide_width_in_um = slide_bottomright_x_in_um - slide_topleft_x_in_um
    slide_height_in_um = slide_bottomright_y_in_um - slide_topleft_y_in_um
    slide_width = int(
        slide_width_in_um / resolution_micron_per_pixel * downscale_factor
    )
    slide_height = int(
        slide_height_in_um / resolution_micron_per_pixel * downscale_factor
    )

    ncols_outer = int(slide_width // tile_size_downscaled + 1)
    nrows_outer = int(slide_height // tile_size_downscaled + 1)

    if image_dim == 2:
        stitched_downscaled = np.zeros(
            (nrows_outer * tile_size_downscaled, ncols_outer * tile_size_downscaled)
        )

    # print(stitched_downscaled.shape)

    for scene in dict_image_arrays_scene_level.keys():
        # print(scene)

        scene_topleft_x_in_um = scene_information.loc[
            scene_information.scene == scene
        ].scene_topleft_x_in_um.values[0]
        scene_topleft_y_in_um = scene_information.loc[
            scene_information.scene == scene
        ].scene_topleft_y_in_um.values[0]

        scene_topleft_x_in_pixel_downscaled = int(
            (scene_topleft_x_in_um - slide_topleft_x_in_um)
            / resolution_micron_per_pixel
            * downscale_factor
        )
        scene_topleft_y_in_pixel_downscaled = int(
            (scene_topleft_y_in_um - slide_topleft_y_in_um)
            / resolution_micron_per_pixel
            * downscale_factor
        )

        for image_tile_name, image_array in dict_image_arrays_scene_level[
            scene
        ].items():
            if (image_array == 0).all():
                continue

            image_tile_row = int(image_tile_name.split("_")[2])
            image_tile_col = int(image_tile_name.split("_")[3])

            if image_dim == 2:
                stitched_downscaled[
                    image_tile_row * tile_size_downscaled
                    + scene_topleft_y_in_pixel_downscaled : (image_tile_row + 1)
                    * tile_size_downscaled
                    + scene_topleft_y_in_pixel_downscaled,
                    image_tile_col * tile_size_downscaled
                    + scene_topleft_x_in_pixel_downscaled : (image_tile_col + 1)
                    * tile_size_downscaled
                    + scene_topleft_x_in_pixel_downscaled,
                ] = image_array.copy()

    return stitched_downscaled


def process_tissue_mask(
    slide_information,
    scene_information,
    dict_image_arrays_scene_level,
    tile_size=2000,
    downscale_factor=0.125,
    resolution_micron_per_pixel=0.22,
):
    stitch_downscaled_tissue_mask = stitch_image_tiles(
        slide_information=slide_information,
        scene_information=scene_information,
        dict_image_arrays_scene_level=dict_image_arrays_scene_level,
        tile_size=tile_size,
        downscale_factor=downscale_factor,
        resolution_micron_per_pixel=resolution_micron_per_pixel,
    )

    plt.imshow(stitch_downscaled_tissue_mask, cmap=plt.cm.Greys)

    return stitch_downscaled_tissue_mask


def overlay_tumour_annotation_with_tissue_mask(
    stitch_downscaled_annotation_binary,
    stitch_downscaled_tissue_mask,
    main_output_directory,
    slide,
):
    shapex = min(
        stitch_downscaled_tissue_mask.shape[0],
        stitch_downscaled_annotation_binary.shape[0],
    )
    shapey = min(
        stitch_downscaled_tissue_mask.shape[1],
        stitch_downscaled_annotation_binary.shape[1],
    )

    combined_mask = (stitch_downscaled_tissue_mask[:shapex, :shapey] > 0) * (
        stitch_downscaled_annotation_binary[:shapex, :shapey] > 0
    )

    stitch_downscaled_annotation_overlay_with_tissue_mask = np.divide(
        stitch_downscaled_tissue_mask[:shapex, :shapey].copy(), 2
    )
    stitch_downscaled_annotation_overlay_with_tissue_mask = np.add(
        stitch_downscaled_annotation_overlay_with_tissue_mask[:shapex, :shapey],
        combined_mask * 255 / 2,
    ).astype(np.uint8)

    # save overlay image

    plt.imshow(stitch_downscaled_annotation_overlay_with_tissue_mask, cmap=plt.cm.Greys)

    output_directory = os.path.join(main_output_directory, slide)
    os.makedirs(output_directory, exist_ok=True)

    plt.savefig(
        os.path.join(
            output_directory,
            f"{slide}_stitched_binary_tumour_annotation_with_tissue_mask.pdf",
        ),
        dpi=600,
    )

    plt.show()

    # save numpy arrays
    np.save(
        os.path.join(
            output_directory,
            f"{slide}_stitched_binary_tumour_annotation_with_tissue_mask.npy",
        ),
        stitch_downscaled_annotation_overlay_with_tissue_mask,
    )

    return stitch_downscaled_annotation_overlay_with_tissue_mask


def get_subtile_tumour_information(
    slide, scene, irow, icol, image_tile_name, image_tile, summary_subtile_rows
):
    subtile_size = image_tile.shape[0] // 2
    for subtile_row in range(2):
        for subtile_col in range(2):
            image_subtile_name = (
                f"subtile_{str(subtile_row).zfill(5)}_{str(subtile_col).zfill(5)}"
            )

            image_subtile = image_tile[
                subtile_row * subtile_size : (subtile_row + 1) * subtile_size,
                subtile_col * subtile_size : (subtile_col + 1) * subtile_size,
            ]

            tissue_percentage_of_tile_area = np.sum(image_subtile > 0) / np.sum(
                image_subtile >= 0
            )
            tumour_percentage_of_tissue_area = (
                np.sum(image_subtile == 255) / np.sum(image_subtile > 0)
                if np.sum(image_subtile > 0)
                else 0
            )

            summary_subtile_rows.append(
                (
                    slide,
                    scene,
                    irow,
                    icol,
                    image_tile_name,
                    image_subtile_name,
                    tissue_percentage_of_tile_area,
                    tumour_percentage_of_tissue_area,
                )
            )

    return summary_subtile_rows


def tiles_of_overlay_tumour_annotation_with_tissue_mask(
    slide_information,
    scene_information_dataframe_complete,
    stitch_downscaled_annotation_overlay_with_tissue_mask,
    main_output_directory,
    slide,
    tile_size=2000,
    downscale_factor=0.125,
    resolution_micron_per_pixel=0.22,
):
    image_dim = 2

    tile_size_downscaled = int(tile_size * downscale_factor)

    (
        slide_bottomright_x_in_um,
        slide_bottomright_y_in_um,
        slide_topleft_x_in_um,
        slide_topleft_y_in_um,
    ) = (
        slide_information["slide_bottomright_x_in_um"],
        slide_information["slide_bottomright_y_in_um"],
        slide_information["slide_topleft_x_in_um"],
        slide_information["slide_topleft_y_in_um"],
    )

    slide_width_in_um = slide_bottomright_x_in_um - slide_topleft_x_in_um
    slide_height_in_um = slide_bottomright_y_in_um - slide_topleft_y_in_um
    slide_width = int(
        slide_width_in_um / resolution_micron_per_pixel * downscale_factor
    )
    slide_height = int(
        slide_height_in_um / resolution_micron_per_pixel * downscale_factor
    )

    ncols_outer = int(slide_width // tile_size_downscaled + 1)
    nrows_outer = int(slide_height // tile_size_downscaled + 1)

    if image_dim == 2:
        stitched_downscaled = np.zeros(
            (nrows_outer * tile_size_downscaled, ncols_outer * tile_size_downscaled)
        )

    # print(stitched_downscaled.shape)

    summary_cols = [
        "slide",
        "scene",
        "row",
        "col",
        "image_tile_name",
        "tissue_fraction_of_tile_area",
        "tumour_fraction_of_tissue_area",
    ]
    summary_rows = []

    summary_subtile_cols = [
        "slide",
        "scene",
        "row",
        "col",
        "image_tile_name",
        "image_subtile_name",
        "tissue_fraction_of_tile_area",
        "tumour_fraction_of_tissue_area",
    ]
    summary_subtile_rows = []

    for scene in scene_information_dataframe_complete.scene.unique():
        #     for scene in ['ScanRegion0']:
        # print(scene)

        scene_topleft_x_in_um = scene_information_dataframe_complete.loc[
            scene_information_dataframe_complete.scene == scene
        ].scene_topleft_x_in_um.values[0]
        scene_topleft_y_in_um = scene_information_dataframe_complete.loc[
            scene_information_dataframe_complete.scene == scene
        ].scene_topleft_y_in_um.values[0]

        scene_topleft_x_in_pixel_downscaled = int(
            (scene_topleft_x_in_um - slide_topleft_x_in_um)
            / resolution_micron_per_pixel
            * downscale_factor
        )
        scene_topleft_y_in_pixel_downscaled = int(
            (scene_topleft_y_in_um - slide_topleft_y_in_um)
            / resolution_micron_per_pixel
            * downscale_factor
        )

        scene_width = scene_information_dataframe_complete.loc[
            scene_information_dataframe_complete.scene == scene
        ].width.values[0]
        scene_height = scene_information_dataframe_complete.loc[
            scene_information_dataframe_complete.scene == scene
        ].height.values[0]
        ncol = scene_width // tile_size
        nrow = scene_height // tile_size

        for irow in range(nrow):
            for icol in range(ncol):
                # tile_id = irow * ncol + icol

                image_tile_name = "image_tile_{}_{}.tif".format(
                    str(irow).zfill(5), str(icol).zfill(5)
                )
                # print(image_tile_name)

                image_tile = stitch_downscaled_annotation_overlay_with_tissue_mask[
                    scene_topleft_y_in_pixel_downscaled
                    + irow * tile_size_downscaled : scene_topleft_y_in_pixel_downscaled
                    + (irow + 1) * tile_size_downscaled,
                    scene_topleft_x_in_pixel_downscaled
                    + icol * tile_size_downscaled : scene_topleft_x_in_pixel_downscaled
                    + (icol + 1) * tile_size_downscaled,
                ]

                ## information
                tissue_percentage_of_tile_area = np.sum(image_tile > 0) / np.sum(
                    image_tile >= 0
                )
                tumour_percentage_of_tissue_area = (
                    np.sum(image_tile == 255) / np.sum(image_tile > 0)
                    if np.sum(image_tile > 0)
                    else 0
                )

                ## information at subtile level
                summary_subtile_rows = get_subtile_tumour_information(
                    slide,
                    scene,
                    irow,
                    icol,
                    image_tile_name,
                    image_tile,
                    summary_subtile_rows,
                )

                # print(
                #     f"tissue_percentage_of_tile_area = {tissue_percentage_of_tile_area:6.2%}; tumour_percentage_of_tissue_area = {tumour_percentage_of_tissue_area:6.2%}\n"
                # )

                summary_rows.append(
                    (
                        slide,
                        scene,
                        irow,
                        icol,
                        image_tile_name,
                        tissue_percentage_of_tile_area,
                        tumour_percentage_of_tissue_area,
                    )
                )

    summary = pd.DataFrame(columns=summary_cols, data=summary_rows)
    summary_subtile = pd.DataFrame(
        columns=summary_subtile_cols, data=summary_subtile_rows
    )

    output_directory = os.path.join(main_output_directory, slide)
    os.makedirs(output_directory, exist_ok=True)

    summary.to_csv(
        os.path.join(output_directory, f"{slide}_summary_of_tumour_percentage.csv"),
        index=False,
    )
    summary_subtile.to_csv(
        os.path.join(
            output_directory, f"{slide}_summary_of_tumour_percentage_in_subtiles.csv"
        ),
        index=False,
    )

    return summary, summary_subtile


def read_deconv_psr(
    directory_to_deconv_psr_tiles, tile_size=2000, downscale_factor=0.125
):
    tile_size_downscaled = int(tile_size * downscale_factor)

    directory_to_deconv_psr_tiles_scene_subdirs = natsorted(
        glob(os.path.join(directory_to_deconv_psr_tiles, "ScanRegion*"))
    )
    scene_names = [
        os.path.basename(subdir)
        for subdir in directory_to_deconv_psr_tiles_scene_subdirs
    ]

    # print("loading tissue mask image tiles")
    subsubdir_deconv_psr = "/psr/inverted_grayscale_tissue_masked/"
    deconv_psr_image_tile_name_pattern = "image_tile*"
    dict_deconv_psr_tiles_scene_level = {}

    for scene, subdir in zip(scene_names, directory_to_deconv_psr_tiles_scene_subdirs):
        print(scene, subdir)
        paths_to_deconv_psr_image_tiles = natsorted(
            glob(
                os.path.join(
                    subdir + subsubdir_deconv_psr, deconv_psr_image_tile_name_pattern
                )
            )
        )
        #         print(paths_to_deconv_psr_image_tiles)

        image_tile_names = [
            os.path.splitext(os.path.basename(path))[0]
            for path in paths_to_deconv_psr_image_tiles
        ]

        deconv_psr_image_arrays = {
            image_tile_name: np.array(
                Image.open(path).resize((tile_size_downscaled, tile_size_downscaled))
            )
            for image_tile_name, path in zip(
                image_tile_names, paths_to_deconv_psr_image_tiles
            )
        }
        dict_deconv_psr_tiles_scene_level[scene] = deconv_psr_image_arrays

    return dict_deconv_psr_tiles_scene_level


def read_cell_annotation_wsi(
    path_to_cell_annotation_wsi,
):
    cell_annotation_wsi = pd.read_csv(path_to_cell_annotation_wsi)
    return cell_annotation_wsi


def overlay_deconv_psr_with_cell_annotation(
    stitch_downscaled_deconv_psr,
    cell_annotation_wsi,
    main_output_directory,
    slide,
    downscale_factor_ecm_to_cells=0.25,
):
    colormap = {
        "leukocytes": "blue",
        "cancer": "green",
        "normal": "yellow",
        "blood": "red",
        "fibroblast": "magenta",
        "others": "white",
        "necrosis": "brown",
    }

    fig, axes = plt.subplots()

    # plot deconvolved psr
    axes.imshow(stitch_downscaled_deconv_psr, cmap=plt.cm.Greys, zorder=1)

    # plot cell annotation
    cell_types = cell_annotation_wsi["class"].unique()
    for ctype in cell_types:
        axes.scatter(
            x=cell_annotation_wsi.loc[cell_annotation_wsi["class"] == ctype].x
            / downscale_factor_ecm_to_cells,
            y=cell_annotation_wsi.loc[cell_annotation_wsi["class"] == ctype].y
            / downscale_factor_ecm_to_cells,
            c=colormap[ctype],
            edgecolor="none",
            s=0.1,
            zorder=2,
        )
    #     axes.set_ylim(axes.get_ylim()[::-1])
    axes.legend(cell_types, loc="upper right", fontsize="xx-small", markerscale=4)

    output_directory = os.path.join(main_output_directory, slide)
    os.makedirs(output_directory, exist_ok=True)

    plt.savefig(
        os.path.join(
            output_directory,
            f"{slide}_stitched_deconvolved_psr_with_cell_annotation.pdf",
        ),
        dpi=600,
    )
