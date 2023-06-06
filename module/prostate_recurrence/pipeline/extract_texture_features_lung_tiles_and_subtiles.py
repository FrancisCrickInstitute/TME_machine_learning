"""# Script for batch-processing texture feature extraction in Lung dataset

Customised arguments are to be defined using argparse library by the user when
calling this script.

Extraction of texture features is performed both at the tile level and at the
subtile level.

## get_path_to_image_tile(...) to obtain the path to the tissue masked deconvolved
PSR image tile given provided slide id, scene name, and image tile name.

## get_path_to_corresponding_tissue_mask_tile(...) to obtain the path to the tissue
mask tile correponding to the path to the tissue masked deconvolved PSR image tile.

## extract_texture_features_this_image(...), calling process_intensity_features(...),
process_glcm_features(...), and process_perception_features(...), to perform texture
feature extraction of a single PSR tile. This function expects arguments reflecting
the path to the tissue masked deconvolved PSR tile and its corresponding tissue mask
tile. Modules in first_order_histogram.py, glcm.py, and perception.py are called to
extract intensity features, grey level co-occurence matrix features, and perception
features, respectively. Options to process only a subset of texture feature domains
can be set using FLAG_INTENSITY_FEATURES, FLAG_GLCM_FEATURES, and
FLAG_PERCEPTION_FEATURES.

"""

import argparse
import os
import sys
from datetime import datetime

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
    "--flag_intensity_features",
    dest="flag_intensity_features",
    action="store",
    type=int,
    default=0,
    help="indicate whether to extract intensity features. set it to 0 if not.",
)
parser.add_argument(
    "--flag_glcm_features",
    dest="flag_glcm_features",
    action="store",
    type=int,
    default=0,
    help="indicate whether to extract GLCM features. set it to 0 if not.",
)
parser.add_argument(
    "--flag_perception_features",
    dest="flag_perception_features",
    action="store",
    type=int,
    default=0,
    help="indicate whether to extract perception features. set it to 0 if not.",
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
    "--flag_by_batch_job",
    dest="flag_by_batch_job",
    action="store",
    type=int,
    default=0,
    help="indicate whether to perform analysis by batch job. set it to 0 if not.",
)
parser.add_argument(
    "--path_to_job_batch_information",
    dest="path_to_job_batch_information",
    action="store",
    type=str,
    default="",
    help="provide the path to job batch information file.",
)

args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
PROCESSED_DATA_PATH = args.processed_data_path
TILE_SIZE = args.tile_size
SUBDIVISION = args.subdivision

FLAG_INTENSITY_FEATURES = args.flag_intensity_features
FLAG_GLCM_FEATURES = args.flag_glcm_features
FLAG_PERCEPTION_FEATURES = args.flag_perception_features

FLAG_BY_BATCH_JOB = args.flag_by_batch_job
PATH_TO_JOB_BATCH_INFORMATION = args.path_to_job_batch_information

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.feature_engineering.Python import (
    first_order_histogram,
    glcm,
    perception,
)


def get_path_to_image_tile(slide_id: str, scene: str, tile: str) -> str:
    """obtain the path to tissue masked deconvolved PSR tile

    Parameters
    ----------
    slide_id : str
        Slide id.
    scene : str
        Scene name.
    tile : str
        Image tile name.

    Returns
    -------
    str
        The path to the tissue masked deconvolved PSR tile
    """
    path_to_valid_image_tile = os.path.join(
        PROCESSED_DATA_PATH,
        slide_id,
        "tile_size_2000/whole_slide/PSR/deconvolutions/",
        scene,
        "psr/inverted_grayscale_tissue_masked/",
        f"image_tile_{tile}_psr.tif",
    )
    return path_to_valid_image_tile


def get_path_to_corresponding_tissue_mask_tile(path_to_valid_image_tile: str) -> str:
    """obtain the path to the corresponding tissue mask tile of the input
    tissue masked deconvolved image tile

    Parameters
    ----------
    path_to_valid_image_tile : str
        The path to the tissue masked deconvolved PSR tile

    Returns
    -------
    str
        The path to the corresponding tissue mask tile
    """
    path_splited = path_to_valid_image_tile.split("/")
    slide_keyword_index = path_splited.index("PSR_20X") + 1
    scene_keyword_index = path_splited.index("deconvolutions") + 1
    slide_id = path_splited[slide_keyword_index]
    scene_id = path_splited[scene_keyword_index]
    image_tile_name = os.path.basename(path_to_valid_image_tile).split("_psr.tif")[0]

    path_to_corresponding_tissue_mask_tile = os.path.join(
        "/".join(path_splited[:slide_keyword_index]),
        slide_id,
        f"tile_size_{TILE_SIZE}/whole_slide/PSR",
        "tissue_masks/tissue_mask_v2",
        scene_id,
        f"{image_tile_name}.tif",
    )

    return path_to_corresponding_tissue_mask_tile


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
    path_to_valid_image_tile: str, path_to_corresponding_tissue_mask_tile: str
) -> None:
    """perform texture feature extraction of a single image tile

    This function focuses on processing extraction of texture features both at the tile
    level and at the subtile level.
    User-defined SUBDIVISION controls the size of subtiles; by default, a subtile is half
    the width of a tile.

    If only a subset of texture feature domains need processing, boolean variables
        FLAG_INTENSITY_FEATURES, FLAG_GLCM_FEATURES, FLAG_PERCEPTION_FEATURES can be set
        accordingly to turn on only relevant parts of processing.

    Parameters
    ----------
    path_to_valid_image_tile : str
        The path to the tissue masked deconvolved PSR image tile
    path_to_corresponding_tissue_mask_tile : str
        The path to the corresponding tissue mask tile
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

    # [1] process this tile
    # texture - indensity
    if FLAG_INTENSITY_FEATURES:
        process_intensity_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )

    # texture - glcm
    if FLAG_GLCM_FEATURES:
        process_glcm_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )

    # texture - perception
    if FLAG_PERCEPTION_FEATURES:
        process_perception_features(
            image_array=image_array,
            tissue_mask_array=tissue_mask_array,
            output_subdir=output_directory_processed_texture_features_this_image_tile,
        )

    # [2] process subtiles
    subtile_size = TILE_SIZE // SUBDIVISION
    output_directory_subtiles = os.path.join(
        output_directory_processed_texture_features_this_image_tile,
        f"features_texture_subtile_size_{subtile_size}",
    )
    os.makedirs(output_directory_subtiles, exist_ok=True)
    os.chmod(output_directory_subtiles, mode=0o777)

    for row_subtile in range(SUBDIVISION):
        for col_subtile in range(SUBDIVISION):
            subtile_image_array = image_array[
                row_subtile * subtile_size : (row_subtile + 1) * subtile_size,
                col_subtile * subtile_size : (col_subtile + 1) * subtile_size,
            ]
            subtile_tissue_mask_array = tissue_mask_array[
                row_subtile * subtile_size : (row_subtile + 1) * subtile_size,
                col_subtile * subtile_size : (col_subtile + 1) * subtile_size,
            ]
            # check if tissue mask contains all zeros
            if (subtile_tissue_mask_array == 0).all():
                continue
            else:
                # create subfolder for the subtile
                row_subtile_str = str(row_subtile).zfill(5)
                col_subtile_str = str(col_subtile).zfill(5)
                output_directory_this_subtile = os.path.join(
                    output_directory_subtiles,
                    f"subtile_{row_subtile_str}_{col_subtile_str}",
                )
                os.makedirs(output_directory_this_subtile, exist_ok=True)
                os.chmod(output_directory_this_subtile, mode=0o777)

                # texture - indensity
                if FLAG_INTENSITY_FEATURES:
                    process_intensity_features(
                        image_array=subtile_image_array,
                        tissue_mask_array=subtile_tissue_mask_array,
                        output_subdir=output_directory_this_subtile,
                    )

                # texture - glcm
                if FLAG_GLCM_FEATURES:
                    process_glcm_features(
                        image_array=subtile_image_array,
                        tissue_mask_array=subtile_tissue_mask_array,
                        output_subdir=output_directory_this_subtile,
                    )

                # texture - perception
                if FLAG_PERCEPTION_FEATURES:
                    process_perception_features(
                        image_array=subtile_image_array,
                        tissue_mask_array=subtile_tissue_mask_array,
                        output_subdir=output_directory_this_subtile,
                    )


if __name__ == "__main__":
    job_batch_information = pd.read_csv(
        PATH_TO_JOB_BATCH_INFORMATION
    )  # ["slide_id", "scene", "tile", "tile_index"]

    logstr = f"===== EXTRACTION OF TEXTURE FEATURES (job batch: {os.path.basename(PATH_TO_JOB_BATCH_INFORMATION)}) =====\n"
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    logstr += f"started at {date_time}\n"
    logstr += "\n"
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(logstr)
    logfile.close()
    logstr = ""

    for slide_id, scene, tile in job_batch_information[
        ["slide_id", "scene", "tile"]
    ].values:
        path_to_valid_image_tile = get_path_to_image_tile(
            slide_id=slide_id, scene=scene, tile=tile
        )

        path_to_corresponding_tissue_mask_tile = (
            get_path_to_corresponding_tissue_mask_tile(
                path_to_valid_image_tile=path_to_valid_image_tile
            )
        )
        extract_texture_features_this_image(
            path_to_valid_image_tile=path_to_valid_image_tile,
            path_to_corresponding_tissue_mask_tile=path_to_corresponding_tissue_mask_tile,
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += (
            f"... image tile {path_to_valid_image_tile}; finished at {date_time}\n"
        )
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""

    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    logstr += f"{job_batch_information.shape[0]} data paths processed; finished at {date_time}\n"
    logstr += "\n"
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(logstr)
    logfile.close()
    logstr = ""
