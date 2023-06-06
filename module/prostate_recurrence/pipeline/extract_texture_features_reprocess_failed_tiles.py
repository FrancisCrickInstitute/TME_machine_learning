"""# Script for batch-processing texture feature extraction in Prostate dataset

Customised arguments are to be defined using argparse library by the user when
calling this script.

This script is only applicable to previously failed processing jobs in the
Prostate dataset. The PATH_TO_FAILED_TILES_IN_BATCHES is a user-defined
argument for locating previously failed tiles, generated in MATLAB.

## get_paths_to_psr_and_tissue_mask(...) to obtain the path to the tissue masked
deconvolved PSR image tile and the path to the corresponding tissue mask tile.
The function expects arguments indicating the slide id, scene name, and image tile
name. Note that the paths are hard-coded according to the current folder organisation
of the Prostate dataset and therefore are subject to changes in application to other
datasets.

## extract_texture_features_this_image(...), calling process_intensity_features(...), 
process_glcm_features(...), and process_perception_features(...), to perform texture
feature extraction of a single PSR tile. This function expects arguments reflecting 
the path to the tissue masked deconvolved PSR tile and its corresponding tissue mask
tile. Options to process only a subset of texture feature domains can be set using
flag_intensity_features, flag_glcm_features, and flag_perception_features.

"""

import argparse
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(prog="tme-ml-pipeline-texture-features")
parser.add_argument(
    "--module_path",
    dest="module_path",
    action="store",
    type=str,
    default="../../../module/",
    help="module path to pipeline functions.",
)
parser.add_argument(
    "--logfile_path",
    dest="logfile_path",
    action="store",
    type=str,
    default="./log_test.txt",
    help="log file to record progress",
)
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
    help="By default, 20x.",
)
parser.add_argument(
    "--tile_size",
    dest="tile_size",
    action="store",
    type=int,
    default=2000,
    help="provide the number of pixels for tile size.",
)
parser.add_argument(
    "--subdivision",
    dest="subdivision",
    action="store",
    type=int,
    default=2,
    help="provide the level of subdivision.",
)
parser.add_argument(
    "--path_to_failed_tiles_in_batches",
    dest="path_to_failed_tiles_in_batches",
    action="store",
    type=str,
    default="",
    help="provide the path to failed tiles in batches",
)
parser.add_argument(
    "--batch_id",
    dest="batch_id",
    action="store",
    type=int,
    default=0,
    help="provide the batch id to process.",
)


args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
PROCESSED_DATA_PATH = args.processed_data_path
TILE_SIZE = args.tile_size
SUBDIVISION = args.subdivision

PATH_TO_FAILED_TILES_IN_BATCHES = args.path_to_failed_tiles_in_batches

BATCH_ID = args.batch_id

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.feature_engineering.Python import (
    first_order_histogram,
    glcm,
    perception,
)


def get_paths_to_psr_and_tissue_mask(
    slide_id: str, scene_id: str, tile: str
) -> Tuple[str, str]:
    """obtain the paths to tissue masked deconvolved PSR tile and
    tissue mask tile

    Parameters
    ----------
    slide_id : str
        Slide id.
    scene_id : str
        Scene name.
    tile : str
        Image tile name.

    Returns
    -------
    Tuple[str, str]
        The path to the tissue masked deconvolved PSR tile and
        the path to the tissue mask tile.
    """
    deconvolved_psr_image_tile_name = f"image_tile_{tile}_psr"
    tissue_mask_image_tile_name = f"image_tile_{tile}"

    path_to_deconvolved_psr_tile = os.path.join(
        PROCESSED_DATA_PATH,
        slide_id,
        f"tile_size_{TILE_SIZE}/whole_slide/PSR",
        "deconvolutions",
        scene_id,
        "psr/inverted_grayscale_tissue_masked",
        f"{deconvolved_psr_image_tile_name}.tif",
    )

    path_to_tissue_mask_tile = os.path.join(
        PROCESSED_DATA_PATH,
        slide_id,
        f"tile_size_{TILE_SIZE}/whole_slide/PSR",
        "tissue_masks/tissue_mask_v2",
        scene_id,
        f"{tissue_mask_image_tile_name}.tif",
    )

    return (path_to_deconvolved_psr_tile, path_to_tissue_mask_tile)


def process_intensity_features(
    image_array: np.ndarray, tissue_mask_array: np.ndarray, output_subdir: str
) -> None:
    """extract intensity features from this image tile

    Parameters
    ----------
    image_array : np.ndarray
        The tissue masked deconvolved PSR image array.
    tissue_mask_array : np.ndarray
        The corresponding tissue mask image array.
    output_subdir : str
        Directory to save feature extraction outputs.
    """
    histogram, masked_image_array = first_order_histogram.construct_histogram(
        image=image_array, mask=tissue_mask_array
    )

    histogram_features_output = first_order_histogram.extract_histogram_features(
        image=masked_image_array,
        histogram=histogram,
        features=(
            "median",
            "mean",
            "variance",
            "skewness",
            "kurtosis",
            "energy",
            "entropy",
        ),
    )

    first_order_histogram.save_histogram_features(
        histogram=histogram,
        histogram_features_output=histogram_features_output,
        output_directory=output_subdir,
    )


def process_glcm_features(
    image_array: np.ndarray, tissue_mask_array: np.ndarray, output_subdir: str
) -> None:
    """extract grey level co-occurence matrix features from this image tile

    Parameters
    ----------
    image_array : np.ndarray
        The tissue masked deconvolved PSR image array.
    tissue_mask_array : np.ndarray
        The corresponding tissue mask image array.
    output_subdir : str
        Directory to save feature extraction outputs.
    """
    distances = [1, 2, 5, 11, 22, 45, 90, 182, 364]
    angles = [0, np.pi / 4.0, np.pi / 2.0, np.pi * 3 / 4.0]
    symmetric = True
    normed = True

    analysis_type = "masked"

    matrix_glcm, masked_image_array = glcm.construct_glcm(
        image=image_array,
        mask=tissue_mask_array,
        distances=distances,
        angles=angles,
        symmetric=symmetric,
        normed=normed,
    )

    glcm_features_output = glcm.extract_glcm_features(
        matrix=matrix_glcm,
        features=(
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ),
    )

    glcm.save_glcm_features(
        matrix=matrix_glcm,
        glcm_features_output=glcm_features_output,
        distances=distances,
        angles=angles,
        output_directory=output_subdir,
        analysis_type=analysis_type,
    )


def process_perception_features(
    image_array: np.ndarray, tissue_mask_array: np.ndarray, output_subdir: str
) -> None:
    """extract perception features from this image tile

    Parameters
    ----------
    image_array : np.ndarray
        The tissue masked deconvolved PSR image array.
    tissue_mask_array : np.ndarray
        The corresponding tissue mask image array.
    output_subdir : str
        Directory to save feature extraction outputs.
    """
    (coarseness_arrays, S, coarseness) = perception.calculate_coarseness(
        image=image_array, mask=tissue_mask_array
    )
    perception_contrast_features_output = perception.calculate_contrast(
        image=image_array, mask=tissue_mask_array
    )
    contrast = perception_contrast_features_output["Contrast"]
    perception_std = perception_contrast_features_output["_std"]
    perception_kur = perception_contrast_features_output["_kurtosis"]

    # temporary code for saving outputs - need to save some heat maps as well
    perception_features_output = pd.DataFrame(
        columns=["feature", "value"],
        data=[
            ("perception_coarseness", coarseness),
            ("perception_contrast", contrast),
            ("perception_std", perception_std),
            ("perception_kurtosis", perception_kur),
        ],
    )

    perception_features_output.to_csv(
        os.path.join(
            output_subdir,
            "perception_features.csv",
        ),
        index=False,
    )


def extract_texture_features_this_image(
    path_to_valid_image_tile: str,
    path_to_corresponding_tissue_mask_tile: str,
    flag_intensity_features: bool,
    flag_glcm_features: bool,
    flag_perception_features: bool,
) -> None:
    """perform texture feature extraction of a single image tile

    Parameters
    ----------
    path_to_valid_image_tile : str
        The path to the tissue masked deconvolved PSR image tile
    path_to_corresponding_tissue_mask_tile : str
        The path to the corresponding tissue mask tile
    flag_intensity_features : bool
        A boolean variable indicating whether to extract intensity features.
    flag_glcm_features : bool
        A boolean variable indicating whether to extract grey level co-occurence
        matrix features.
    flag_perception_features : bool
        A boolean variable indicating whether to extract perception features.
    """

    dirname_valid_image_tile = os.path.dirname(path_to_valid_image_tile)
    basename_valid_image_tile = os.path.basename(path_to_valid_image_tile)
    output_directory_processed_texture_features_this_image_tile = os.path.join(
        dirname_valid_image_tile.replace("pre_processed_data", "feature_engineering"),
        "tile_level_features_texture_v2",
        basename_valid_image_tile.split(".")[0],
    )
    os.makedirs(
        output_directory_processed_texture_features_this_image_tile,
        exist_ok=True,
    )
    os.chmod(
        output_directory_processed_texture_features_this_image_tile,
        mode=0o777,
    )

    # read image
    image_array = glcm.read_image(path_to_img=path_to_valid_image_tile)
    tissue_mask_array = glcm.read_image(
        path_to_img=path_to_corresponding_tissue_mask_tile
    )

    # texture - indensity
    if flag_intensity_features:
        process_intensity_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )

    # texture - glcm
    if flag_glcm_features:
        process_glcm_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )

    # texture - perception
    if flag_perception_features:
        process_perception_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )


if __name__ == "__main__":

    # failed intensity
    path_intensity = os.path.join(
        PATH_TO_FAILED_TILES_IN_BATCHES,
        f"failed_intensity_batch_{str(BATCH_ID).zfill(4)}.csv",
    )
    failed_intensity = pd.DataFrame()
    if os.path.exists(path_intensity):
        failed_intensity = pd.read_csv(path_intensity)

    # failed glcm
    path_glcm = os.path.join(
        PATH_TO_FAILED_TILES_IN_BATCHES,
        f"failed_glcm_batch_{str(BATCH_ID).zfill(4)}.csv",
    )
    failed_glcm = pd.DataFrame()
    if os.path.exists(path_glcm):
        failed_glcm = pd.read_csv(path_glcm)

    # failed perception
    path_perception = os.path.join(
        PATH_TO_FAILED_TILES_IN_BATCHES,
        f"failed_perception_batch_{str(BATCH_ID).zfill(4)}.csv",
    )
    failed_perception = pd.DataFrame()
    if os.path.exists(path_perception):
        failed_perception = pd.read_csv(path_perception)

    logstr = (
        f"===== EXTRACTION OF TEXTURE FEATURES (by batch : id = {BATCH_ID}) =====\n"
    )
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    logstr += f"batch id = {BATCH_ID}; started at {date_time}\n"
    logstr += "\n"
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(logstr)
    logfile.close()
    logstr = ""

    for df, flags in zip(
        [failed_intensity, failed_glcm, failed_perception],
        [
            {"intensity": True, "glcm": False, "perception": False},
            {"intensity": False, "glcm": True, "perception": False},
            {"intensity": False, "glcm": False, "perception": True},
        ],
    ):
        if df.shape[0] == 0:
            continue

        for slide_id, scene, tile in df[["slide_id", "scene", "tile"]].values:
            (
                path_to_deconvolved_psr_tile,
                path_to_tissue_mask_tile,
            ) = get_paths_to_psr_and_tissue_mask(
                slide_id=slide_id, scene_id=scene, tile=tile
            )
            extract_texture_features_this_image(
                path_to_valid_image_tile=path_to_deconvolved_psr_tile,
                path_to_corresponding_tissue_mask_tile=path_to_tissue_mask_tile,
                flag_intensity_features=flags["intensity"],
                flag_glcm_features=flags["glcm"],
                flag_perception_features=flags["perception"],
            )

    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    logstr += f"batch id = {BATCH_ID} - {failed_intensity.shape[0]} data paths processed; finished at {date_time}\n"
    logstr += "\n"
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(logstr)
    logfile.close()
    logstr = ""
