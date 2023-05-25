"""
    This script processes Lung TDA ROIs
    No tissue mask tiles are used for analysis
    Subtile analysis is not performed
"""


import argparse
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from natsort import natsorted

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
    "--image_data_path",
    dest="image_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path saving image data.",
)
parser.add_argument(
    "--image_filename_start",
    dest="image_filename_start",
    action="store",
    type=str,
    default="LTX001",
    help="provide the start of image filenames.",
)
parser.add_argument(
    "--output_directory",
    dest="output_directory",
    action="store",
    type=str,
    default="",
    help="provide the directory to save outputs into.",
)

args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
IMAGE_DATA_PATH = args.image_data_path
IMAGE_FILENAME_START = args.image_filename_start
OUTPUT_DIRECTORY = args.output_directory

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.feature_engineering.Python import glcm


def extract_glcm_features_this_image(path_to_valid_image_tile):
    basename_valid_image_tile = os.path.basename(path_to_valid_image_tile)
    output_directory_processed_texture_features_this_image_tile = os.path.join(
        OUTPUT_DIRECTORY,
        "tile_level_features_texture",
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
    tissue_mask_array = np.ones_like(image_array)

    # texture - glcm
    if True:
        distances = [1, 2, 5, 11, 22, 45, 90, 182, 364]
        angles = [0, np.pi / 4.0, np.pi / 2.0, np.pi * 3 / 4.0]
        symmetric = True
        normed = True

        min_intensity_thresholds = [8, 16, 32, 64, 106, 128, 212]

        for analysis_type, tissue_mask in zip(["masked"], [tissue_mask_array]):

            matrix_glcm, masked_image_array = glcm.construct_glcm(
                image=image_array,
                mask=tissue_mask,
                distances=distances,
                angles=angles,
                symmetric=symmetric,
                normed=normed,
            )

            for min_intensity_threshold in min_intensity_thresholds:
                output_subdir = os.path.join(
                    output_directory_processed_texture_features_this_image_tile,
                    f"min_intensity_threshold_{min_intensity_threshold}",
                )
                os.makedirs(output_subdir, exist_ok=True)
                os.chmod(output_subdir, mode=0o777)

                start = min_intensity_threshold - 1
                matrix_glcm_small = matrix_glcm[start:, start:, :, :]

                glcm_features_output = glcm.extract_glcm_features(
                    matrix=matrix_glcm_small,
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


if __name__ == "__main__":

    paths_to_valid_image_tiles = natsorted(
        glob(
            os.path.join(IMAGE_DATA_PATH, f"**/{IMAGE_FILENAME_START}*.tif"),
            recursive=True,
        )
    )
    print(paths_to_valid_image_tiles)

    logstr = "===== EXTRACTION OF TEXTURE FEATURES (lung tda rois) =====\n"

    for k, path_to_valid_image_tile in enumerate(paths_to_valid_image_tiles):
        extract_glcm_features_this_image(
            path_to_valid_image_tile=path_to_valid_image_tile
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += (
            f"{k+1} / {len(paths_to_valid_image_tiles)} data paths\n"
            f"... image tile {path_to_valid_image_tile}; finished at {date_time}\n"
        )
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""

    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    logstr += f"all {len(paths_to_valid_image_tiles)} data paths processed; finished at {date_time}\n"
    logstr += "\n"
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(logstr)
    logfile.close()
    logstr = ""