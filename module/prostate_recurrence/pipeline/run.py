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
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="tme-ml-pipeline")
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
    default=1024,
    help="provide the number of pixels for tile size.",
)
args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
PROCESSED_DATA_PATH = args.processed_data_path
FEATURES_PATH = args.features_path
TILE_SIZE = args.tile_size
BATCH_SIZE = args.batch_size
BATCH_ID = args.batch_id
FLAG_RUN_TILING = args.flag_run_tiling
FLAG_RUN_COLOUR_DECONVOLUTION = args.flag_run_colour_deconvolution
FLAG_RUN_VALIDATE_PSR_IMAGE = args.flag_run_validate_psr_image
FLAG_RUN_STITCHING = args.flag_run_stitching
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.data_exploration import stitching
from prostate_recurrence.image_processing import (
    colour_deconvolution,
    tiling,
    validate_psr_image,
)

assert RAW_DATA_PATH and PROCESSED_DATA_PATH


def run_tiling():
    """perform tiling of raw image data in .czi format

    This function calls module in tiling.py to perform tiling. The default reading method
    is "aicsimageio". Paths to all image data in the folder RAW_DATA_PATH are obtained.

    Tip: when testing this code for a small number of images, the user can run
        "for data_path in data_paths[:1]: ..." and check if the outputs are generated and
        saved properly.

    For each whole slide,
        a dictionary of images are returned from function call
        "tiling.read_image(...)" where key:value pairs refer to
        scene name : image as a Numpy array.

    For each image scene,
        a dictionary of image tiles are returned from function call
        "tiling.create_tiles(...)" where key:value pairs refer to
        tile id : image as a Numpy array.
        additionally, the number of rows and the number of cols in this image scene
        are also returned.
        finally, image tiles are saved into the designated folders.

    """
    print("===== TILING =====")

    logstr = "===== TILING =====\n"

    # reading_method = "bioformats"
    reading_method = "aicsimageio"

    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(
            glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
        )
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {data_path} at {date_time}\n"

        print(f"> processing: {data_path}")
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])
        output_directory_processed_raw_tiling = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "raw_tiling",
        )
        os.makedirs(output_directory_processed_raw_tiling, exist_ok=True)

        dict_imgs = tiling.read_image(data_path, method=reading_method)
        print(f"scenes of this slide: {dict_imgs}")

        for scan_region, img in dict_imgs.items():
            print(f"> processing scene: {scan_region}")
            output_directory_processed_raw_tiling_scan_region = os.path.join(
                output_directory_processed_raw_tiling, scan_region
            )
            os.makedirs(
                output_directory_processed_raw_tiling_scan_region, exist_ok=True
            )
            (dict_img_tiles, nrow, ncol) = tiling.create_tiles(
                img,
                size=TILE_SIZE,
            )
            tiling.save_tiles(
                dict_img_tiles=dict_img_tiles,
                nrow=nrow,
                ncol=ncol,
                img_id="",
                img_type="",
                save_path=output_directory_processed_raw_tiling_scan_region,
            )

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"... scene : {scan_region} saved at {date_time}\n"

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(data_paths.index(data_path)+1)} data paths processed (total: {len(data_paths)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""


def run_colour_deconvolution():
    """perform colour deconvolution of raw image tiles

    This function calls module in colour_deconvolution.py to perform colour deconvolution.
    Paths to all image data in the folder RAW_DATA_PATH are obtained.

    For each whole slide,
        paths to all of its image scenes in the processed data directory are obtained.

    For each image scene,
        paths to all of its image tiles in .tif format are obtained.

    For each image tile,
        data is read as Numpy array by calling PIL.Image.open(...). Deconvolved images in
        Numpy array are returned from function call "colour_deconvolution.deconvolve_image(...)".
        A colour map from white to picrosirius red is needed to generate coloured visualisation of
        deconolved psr images and created by function call "colour_deconvolution.create_cmap(...)".
        Deconolved images are saved into (inverted) greyscale images by function call
        "colour_deconvolution.save_deconvolved_images(...)".

    """
    print("===== COLOUR DECONVOLUTION =====")
    logstr = "===== COLOUR DECONVOLUTION =====\n"

    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(
            glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
        )
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {data_path} at {date_time}\n"
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])
        output_directory_processed_raw_tiling = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "raw_tiling",
        )
        assert os.path.exists(output_directory_processed_raw_tiling)

        output_directory_processed_deconvolutions = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions",
        )
        os.makedirs(output_directory_processed_deconvolutions, exist_ok=True)

        output_directory_processed_raw_tiling_scan_region_paths = natsorted(
            glob(os.path.join(output_directory_processed_raw_tiling, "ScanRegion*"))
        )

        for (
            output_directory_processed_raw_tiling_scan_region_path
        ) in output_directory_processed_raw_tiling_scan_region_paths:
            scan_region = os.path.basename(
                output_directory_processed_raw_tiling_scan_region_path
            )
            print(f"> processing scene: {scan_region}")

            output_directory_processed_deconvolutions_scan_region_path = os.path.join(
                output_directory_processed_deconvolutions, scan_region
            )
            os.makedirs(
                output_directory_processed_deconvolutions_scan_region_path,
                exist_ok=True,
            )
            os.chmod(
                output_directory_processed_deconvolutions_scan_region_path, mode=0o777
            )

            image_tile_paths = natsorted(
                glob(
                    os.path.join(
                        output_directory_processed_raw_tiling_scan_region_path,
                        "image_tile*.tif",
                    )
                )
            )
            for image_tile_path in image_tile_paths:
                img_arr = np.array(Image.open(image_tile_path))
                image_deconvolved, stains = colour_deconvolution.deconvolve_image(
                    img_arr
                )
                cmap_psr = colour_deconvolution.create_cmap()
                colour_deconvolution.save_deconvolved_images(
                    image_deconvolved=image_deconvolved,
                    image_path=image_tile_path,
                    output_directory=output_directory_processed_deconvolutions_scan_region_path,
                    stains=stains,
                    cmap_psr=cmap_psr,
                )

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"... scene : {scan_region} saved at {date_time}\n"

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(data_paths.index(data_path)+1)} data paths processed (total: {len(data_paths)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""


def run_validate_psr_image():
    """perform validation of image tiles (to be deprecated)

    This function calls module in validate_psr_image.py to perform validation of
    image tiles. Image tiles with too few pixels with a minimum of intensity level
    are labelled as invalid during this process.

    An alternative and better implementation with a similar intention is available
    in MATLAB. Therefore, this function is NOT recommended for use.

    """
    print("===== VALIDATE PSR IMAGE =====")
    logstr = "===== VALIDATE PSR IMAGE =====\n"
    do_batch_processing = True
    if ".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = natsorted(
            glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
        )
        data_paths = data_paths[
            BATCH_SIZE * BATCH_ID : min(BATCH_SIZE * (BATCH_ID + 1), len(data_paths))
        ]
        print(f"> batch processing ON < \n data paths are \n {data_paths}")
    else:
        data_paths = [RAW_DATA_PATH]

    for data_path in data_paths:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {data_path} at {date_time}\n"
        data_id = "_".join(data_path.split("/")[-1].split("_")[:2])

        output_directory_processed_deconvolutions = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions",
        )
        assert os.path.exists(output_directory_processed_deconvolutions)

        output_directory_processed_deconvolutions_scan_region_paths = natsorted(
            glob(os.path.join(output_directory_processed_deconvolutions, "ScanRegion*"))
        )

        for (
            output_directory_processed_deconvolutions_scan_region_path
        ) in output_directory_processed_deconvolutions_scan_region_paths:
            scan_region = os.path.basename(
                output_directory_processed_deconvolutions_scan_region_path
            )
            print(f"> processing scene: {scan_region}")

            output_directory_processed_deconvolutions_scan_region_psr = os.path.join(
                output_directory_processed_deconvolutions_scan_region_path,
                "psr",
                "inverted_grayscale",
            )
            assert os.path.exists(
                output_directory_processed_deconvolutions_scan_region_psr
            )

            output_directory_summary = os.path.join(
                output_directory_processed_deconvolutions_scan_region_psr,
                "valid_psr_images_summary/",
            )
            os.makedirs(output_directory_summary, exist_ok=True)
            os.chmod(output_directory_summary, mode=0o777)

            psr_image_tile_paths = natsorted(
                glob(
                    os.path.join(
                        output_directory_processed_deconvolutions_scan_region_psr,
                        "*psr.tif",
                    )
                )
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

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"... scene : {scan_region} saved at {date_time}\n"

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(data_paths.index(data_path)+1)} data paths processed (total: {len(data_paths)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""


def run_stitching():
    """perform re-stitching of image tiles (to be deprecated)

    This functionality is re-implemented elsewhere. Do NOT use.

    """
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
            _ = stitching.visualise_overlay(
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
if False or FLAG_RUN_VALIDATE_PSR_IMAGE:  # function to be deprecated
    run_validate_psr_image()
if False or FLAG_RUN_STITCHING:  # function to be deprecated
    run_stitching()
