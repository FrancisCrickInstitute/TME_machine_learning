"""# a set of functions for running the complete tme-ml pipeline
[to do] User has options to run some modules of the pipeline.

## a full tme-ml pipeline (as of 2021.08) is described as below
(1) [image_processing] read raw image(s) from user defined input data path
(2) [image_processing] perform tiling and save image tiles into output path
(3) [image_processing] perform colour deconvolution and save psr tiles in to output path
(4) [feature_engineering/MATLAB] perform feature engineering to extract ECM architectural features
(5) [feature_engineering/Python] perform feature engineering to extract ECM textural features
(6) [data_exploration/MATLAB] perform PCA and unsupervised clustering
(7) [data_exploration/Python] perform stitching to construct whole slide feature heatmaps
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

module_path = os.path.abspath(os.path.join("../../../module/"))
if module_path not in sys.path:
    sys.path.append(module_path)

from prostate_recurrence.image_processing import (
    colour_deconvolution,
    tiling,
    validate_psr_image,
)

from prostate_recurrence.data_exploration import stitching

parser = argparse.ArgumentParser(prog="tme-ml-pipeline")
parser.add_argument(
    "--flag_run_tiling",
    dest="flag_run_tiling",
    action="store",
    type=int,
    default=1,
    help="indicate whether tiling module needs running. set it to 0 if not.",
)
parser.add_argument(
    "--flag_run_colour_deconvolution",
    dest="flag_run_colour_deconvolution",
    action="store",
    type=int,
    default=1,
    help="indicate whether colour deconvolution module needs running. set it to 0 if not.",
)
parser.add_argument(
    "--flag_run_validate_psr_image",
    dest="flag_run_validate_psr_image",
    action="store",
    type=int,
    default=1,
    help="indicate whether validate psr image module needs running. set it to 0 if not.",
)
parser.add_argument(
    "--flag_run_stitching",
    dest="flag_run_stitching",
    action="store",
    type=int,
    default=1,
    help="indicate whether stitching module needs running. set it to 0 if not.",
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    action="store",
    type=int,
    default=10,
    help="provide the size of batch for batch processing.",
)
parser.add_argument(
    "--batch_id",
    dest="batch_id",
    action="store",
    type=int,
    default=0,
    help="provide the current batch id for batch processing.",
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
    "--features_path",
    dest="features_path",
    action="store",
    type=str,
    default="",
    help="provide the path saving features.",
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
    "--tile_size",
    dest="tile_size",
    action="store",
    type=int,
    default=1024,
    help="provide the number of pixels for tile size.",
)
args = parser.parse_args()
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
PROCESSED_DATA_PATH = args.processed_data_path
FEATURES_PATH = args.features_path
TILE_SIZE = args.tile_size
BATCH_SIZE = args.batch_size
BATCH_ID = args.batch_id
FLAG_RUN_TILING = args.flag_run_tiling
FLAG_RUN_COLOUR_DECONVOLUTION = args.flag_run_colour_deconvolution
FLAG_RUN_VALIDATE_PSR_IMAGE = args.flag_run_validate_psr_image
FLAG_RUN_STITCHING = args.flag_run_stitching

assert RAW_DATA_PATH and PROCESSED_DATA_PATH


def run_tiling():
    print("===== TILING =====")
    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(glob(os.path.join(RAW_DATA_PATH, "*.czi")))
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])
        output_directory_processed_raw_tiling = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            RAW_DATA_TYPE,
            f"tile_size_{TILE_SIZE}",
            "pre_processing",
            "raw_tiling",
        )
        os.makedirs(output_directory_processed_raw_tiling, exist_ok=True)

        img = tiling.read_image(data_path)
        (dict_img_tiles, nrow, ncol) = tiling.create_tiles(img[0, 0], size=TILE_SIZE)
        tiling.save_tiles(
            dict_img_tiles=dict_img_tiles,
            nrow=nrow,
            ncol=ncol,
            img_id="",
            img_type="",
            save_path=output_directory_processed_raw_tiling,
        )


def run_colour_deconvolution():
    print("===== COLOUR DECONVOLUTION =====")
    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(glob(os.path.join(RAW_DATA_PATH, "*.czi")))
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])
        output_directory_processed_raw_tiling = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            RAW_DATA_TYPE,
            f"tile_size_{TILE_SIZE}",
            "pre_processing",
            "raw_tiling",
        )
        assert os.path.exists(output_directory_processed_raw_tiling)

        output_directory_processed_deconvolutions = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            RAW_DATA_TYPE,
            f"tile_size_{TILE_SIZE}",
            "pre_processing",
            "deconvolutions",
        )
        os.makedirs(output_directory_processed_deconvolutions, exist_ok=True)

        image_tile_paths = glob(
            os.path.join(output_directory_processed_raw_tiling, "*.tif")
        )
        for image_tile_path in image_tile_paths:
            img_arr = np.array(Image.open(image_tile_path))
            image_deconvolved, stains = colour_deconvolution.deconvolve_image(img_arr)
            cmap_psr = colour_deconvolution.create_cmap()
            colour_deconvolution.save_deconvolved_images(
                image_deconvolved=image_deconvolved,
                image_path=image_tile_path,
                output_directory=output_directory_processed_deconvolutions,
                stains=stains,
                cmap_psr=cmap_psr,
            )


def run_validate_psr_image():
    print("===== VALIDATE PSR IMAGE =====")
    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(glob(os.path.join(RAW_DATA_PATH, "*.czi")))
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])
        output_directory_processed_deconvolutions_psr = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions",
            "psr",
            "inverted_grayscale",
        )
        assert os.path.exists(output_directory_processed_deconvolutions_psr)

        output_directory_summary = os.path.join(
            output_directory_processed_deconvolutions_psr, "valid_psr_images_summary/"
        )
        os.makedirs(output_directory_summary, exist_ok=True)
        os.chmod(output_directory_summary, mode=0o777)

        psr_image_tile_paths = glob(
            os.path.join(output_directory_processed_deconvolutions_psr, "*psr.tif")
        )
        summary_columns = [
            "path_to_image_tile",
            "row",
            "column",
            "valid",
            "fraction_with_min_intensity",
        ]
        summary_rows = []
        for image_path in tqdm(psr_image_tile_paths):
            image_arr = validate_psr_image.read_psr_image(image_path)
            (
                valid,
                fraction_with_min_intensity,
            ) = validate_psr_image.validate_psr_image(image_arr)
            summary_rows.append(
                (
                    image_path,
                    int(image_path.split("/")[-1].split("_")[2]),
                    int(image_path.split("/")[-1].split("_")[3]),
                    valid,
                    fraction_with_min_intensity,
                )
            )
        df_summary = pd.DataFrame(data=summary_rows, columns=summary_columns)
        validate_psr_image.write_validation_summary(
            summary=df_summary, output_directory=output_directory_summary
        )


# def run_feature_engineering():
#     pass


def run_stitching():
    print("===== STITCHING =====")
    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(glob(os.path.join(RAW_DATA_PATH, "*.czi")))
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])

        output_directory_processed_raw_tiling = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "raw_tiling",
        )
        output_directory_processed_tile_level_features = os.path.join(
            FEATURES_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions/psr/inverted_grayscale",
            "tile_level_features",
        )
        assert os.path.exists(output_directory_processed_raw_tiling) and os.path.exists(
            output_directory_processed_tile_level_features
        )

        output_directory_processed_stitching = os.path.join(
            FEATURES_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions/psr/inverted_grayscale",
            "slide_level_features/stitching",
        )
        os.makedirs(output_directory_processed_stitching, exist_ok=True)
        os.chmod(output_directory_processed_stitching, mode=0o777)

        # feature image
        which_image_tiles = os.listdir(output_directory_processed_tile_level_features)
        which_tif_files = glob(
            os.path.join(
                output_directory_processed_tile_level_features,
                which_image_tiles[0],
                "*tif",
            )
        )
        row_col_pairs = np.array(
            [
                [
                    int(image_tile_folder.split("_")[2]),
                    int(image_tile_folder.split("_")[3]),
                ]
                for image_tile_folder in which_image_tiles
            ]
        )
        nrow = ncol = row_col_pairs.max() + 1

        file_patterns = [which_tif.split("/")[-1] for which_tif in which_tif_files]
        file_patterns_skip = [
            file_pattern for file_pattern in file_patterns if "_cm" in file_pattern
        ]
        for file_pattern in file_patterns:
            if file_pattern in file_patterns_skip:
                print(f"skipping file : {file_pattern}")
                continue

            print(f"> processing feature = {file_pattern}")
            file_paths = glob(
                os.path.join(
                    output_directory_processed_tile_level_features,
                    f"image_tile_*_psr/{file_pattern}",
                )
            )

            save_path = os.path.join(
                output_directory_processed_stitching,
                f"stitched_image_{file_pattern}.jpg",
            )

            print(f"... saving output to {save_path}")
            stitching.reconstruct_whole_slide(
                file_paths=file_paths,
                position_in_path_has_tile_row_col=-2,
                save_path=save_path,
                nrow=nrow,
                ncol=ncol,
                dim=(1024, 1024, 3),
            )

        # feature heatmap
        file_pattern_overlay = "features_out.csv"

        file_paths_features = glob(
            os.path.join(
                output_directory_processed_tile_level_features,
                f"image_tile_*_psr/{file_pattern_overlay}",
            )
        )
        row_col_strings = [
            "_".join(
                [
                    file_path_feature.split("/")[-2].split("_")[2],
                    file_path_feature.split("/")[-2].split("_")[3],
                ]
            )
            for file_path_feature in file_paths_features
        ]
        file_paths_all = glob(
            os.path.join(output_directory_processed_raw_tiling, "image_tile*.tif")
        )

        file_paths = [
            file_path
            for file_path in file_paths_all
            if "_".join(
                [
                    file_path.split("/")[-1].split("_")[2],
                    file_path.split("/")[-1].split("_")[3].split(".")[0],
                ]
            )
            in row_col_strings
        ]

        save_path = os.path.join(
            output_directory_processed_stitching, "stitched_image_raw_PSR.jpg"
        )
        raw_image = stitching.reconstruct_whole_slide(
            file_paths=file_paths,
            position_in_path_has_tile_row_col=-1,
            save_path=save_path,
            nrow=nrow,
            ncol=ncol,
            dim=(1024, 1024, 3),
        )

        features_all = pd.DataFrame()
        for file_path_feature in file_paths_features:
            subdir = file_path_feature.split("/")[-2]
            features = pd.read_csv(
                file_path_feature, header=None, names=["feature", "value"]
            )
            features["tile"] = subdir
            features_all = features_all.append(features)

        for feature_to_map in features_all.feature.unique():
            print(f"> mapping feature : {feature_to_map}")
            save_path = os.path.join(
                output_directory_processed_stitching,
                f"stitched_heatmap_{feature_to_map}.jpg",
            )
            mask = stitching.visualise_overlay(
                raw_image=raw_image,
                file_paths=file_paths_features,
                position_in_path_has_tile_row_col=-2,
                features_all=features_all,
                feature_to_map=feature_to_map,
                save_path=save_path,
                nrow=nrow,
                ncol=ncol,
                size=1024,
            )


if FLAG_RUN_TILING:
    run_tiling()
if FLAG_RUN_COLOUR_DECONVOLUTION:
    run_colour_deconvolution()
if FLAG_RUN_VALIDATE_PSR_IMAGE:
    run_validate_psr_image()
if FLAG_RUN_STITCHING:
    run_stitching()
