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
    default=1024,
    help="provide the number of pixels for tile size.",
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


def extract_texture_features_this_image(path_to_valid_image_tile):

    dirname_valid_image_tile = os.path.dirname(path_to_valid_image_tile)
    basename_valid_image_tile = os.path.basename(path_to_valid_image_tile)
    output_directory_processed_texture_features_this_image_tile = os.path.join(
        dirname_valid_image_tile.replace("pre_processed_data", "feature_engineering"),
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

    # texture - indensity
    if FLAG_INTENSITY_FEATURES:
        nlevels = 16
        histogram = first_order_histogram.construct_histogram(
            image=image_array, levels=nlevels
        )

        histogram_features_output = first_order_histogram.extract_histogram_features(
            image=image_array,
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

        # temporary code for saving outputs - need to save some histograms as well
        columns = ["feature", "value"]
        data_rows = []
        for (
            feature_name,
            feature_value,
        ) in histogram_features_output.items():
            feature_name_this_analysis = f"intensity_{feature_name}"
            feature_value_this_analysis = feature_value
            data_rows.append(
                (
                    feature_name_this_analysis,
                    feature_value_this_analysis,
                )
            )
        data_frame = pd.DataFrame(columns=columns, data=data_rows)
        data_frame.to_csv(
            os.path.join(
                output_directory_processed_texture_features_this_image_tile,
                "intensity_features.csv",
            ),
            index=False,
        )

    # texture - glcm
    if FLAG_GLCM_FEATURES:
        distances = [1, 10, 100]
        angles = [0, np.pi / 4.0, np.pi / 2.0, np.pi * 3 / 4.0]
        symmetric = True
        normed = True

        matrix_glcm = glcm.construct_glcm(
            image=image_array,
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
            output_directory=output_directory_processed_texture_features_this_image_tile,
        )

    # texture - perception
    if FLAG_PERCEPTION_FEATURES:

        (S, coarseness) = perception.calculate_coarseness(image_array)

        contrast = perception.calculate_contrast(image_array)

        # temporary code for saving outputs - need to save some heat maps as well
        perception_features_output = pd.DataFrame(
            columns=["feature", "value"],
            data=[
                ("perception_coarseness", coarseness),
                ("perception_contrast", contrast),
            ],
        )

        perception_features_output.to_csv(
            os.path.join(
                output_directory_processed_texture_features_this_image_tile,
                "perception_features.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    if not FLAG_BY_BATCH_JOB:
        all_paths_to_data = natsorted(
            glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
        )

        logstr = "===== EXTRACTION OF TEXTURE FEATURES =====\n"

        for path in all_paths_to_data:
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"> processing: {path} at {date_time}\n"
            print(f"> processing path : {path}")
            data_id = "_".join(os.path.basename(path).split("_")[:2])

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
                glob(
                    os.path.join(
                        output_directory_processed_deconvolutions, "ScanRegion*"
                    )
                )
            )

            for (
                output_directory_processed_deconvolutions_scan_region_path
            ) in output_directory_processed_deconvolutions_scan_region_paths:
                scan_region = os.path.basename(
                    output_directory_processed_deconvolutions_scan_region_path
                )
                print(f"> processing scene: {scan_region}")

                output_directory_processed_deconvolutions_scan_region_psr = (
                    os.path.join(
                        output_directory_processed_deconvolutions_scan_region_path,
                        "psr",
                        "inverted_grayscale",
                    )
                )
                assert os.path.exists(
                    output_directory_processed_deconvolutions_scan_region_psr
                )

                output_directory_summary = os.path.join(
                    output_directory_processed_deconvolutions_scan_region_psr,
                    "valid_psr_images_summary/",
                )
                filename_summary = "summary_valid_image_tiles.csv"
                path_to_summary_valid_image_tiles = os.path.join(
                    output_directory_summary, filename_summary
                )
                if os.path.exists(output_directory_summary):
                    summary_valid_image_tiles = pd.read_csv(
                        path_to_summary_valid_image_tiles
                    )
                    paths_to_valid_image_tiles = summary_valid_image_tiles.loc[
                        summary_valid_image_tiles.valid == 1
                    ].path_to_image_tile.values.tolist()

                    for path_to_valid_image_tile in paths_to_valid_image_tiles:
                        extract_texture_features_this_image(
                            path_to_valid_image_tile=path_to_valid_image_tile
                        )

                else:
                    print(
                        "no summmary valid image tiles file found for:"
                        + f"slide {data_id} - scene {scan_region}"
                    )

                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
                logstr += f"... scene : {scan_region} saved at {date_time}\n"

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"{int(all_paths_to_data.index(path)+1)} data paths processed (total: {len(all_paths_to_data)}); finished at {date_time}\n"
            logstr += "\n"
            logfile = open(LOGFILE_PATH, "a")
            logfile.write(logstr)
            logfile.close()
            logstr = ""

    else:
        job_batch_information = pd.read_csv(PATH_TO_JOB_BATCH_INFORMATION)
        batch_id = job_batch_information.batch_id.values[0]
        paths_to_valid_image_tiles = job_batch_information.path_to_image.values
        for path_to_valid_image_tile in paths_to_valid_image_tiles:
            extract_texture_features_this_image(
                path_to_valid_image_tile=path_to_valid_image_tile
            )

        logstr = (
            f"===== EXTRACTION OF TEXTURE FEATURES (by batch : id = {batch_id}) =====\n"
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"batch id = {batch_id} - {job_batch_information.shape[0]} data paths processed; finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""
