"""# a set of functions for checking the completion of tme-ml pipeline jobs
"""

import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser(prog="tme-ml-pipeline-checker")
parser.add_argument(
    "--flag_check_tiling",
    dest="flag_check_tiling",
    action="store",
    type=int,
    default=1,
    help="indicate whether tiling module needs checking. set it to 0 if not.",
)
parser.add_argument(
    "--flag_check_colour_deconvolution",
    dest="flag_check_colour_deconvolution",
    action="store",
    type=int,
    default=1,
    help="indicate whether colour deconvolution module needs checking. set it to 0 if not.",
)
parser.add_argument(
    "--flag_check_validate_psr_image",
    dest="flag_check_validate_psr_image",
    action="store",
    type=int,
    default=1,
    help="indicate whether validate psr image module needs checking. set it to 0 if not.",
)
parser.add_argument(
    "--flag_check_feature_engineering",
    dest="flag_check_feature_engineering",
    action="store",
    type=int,
    default=1,
    help="indicate whether feature engineering module needs checking. set it to 0 if not.",
)
parser.add_argument(
    "--flag_check_stitching",
    dest="flag_check_stitching",
    action="store",
    type=int,
    default=1,
    help="indicate whether stitching module needs checking. set it to 0 if not.",
)
parser.add_argument(
    "--job_batch_file_path",
    dest="job_batch_file_path",
    action="store",
    type=str,
    default="",
    help="provide the path containing a file listing jobs.",
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
JOB_BATCH_FILE_PATH = args.job_batch_file_path
TILE_SIZE = args.tile_size
FLAG_CHECK_TILING = args.flag_check_tiling
FLAG_CHECK_COLOUR_DECONVOLUTION = args.flag_check_colour_deconvolution
FLAG_CHECK_VALIDATE_PSR_IMAGE = args.flag_check_validate_psr_image
FLAG_CHECK_FEATURE_ENGINEERING = args.flag_check_feature_engineering
FLAG_CHECK_STITCHING = args.flag_check_stitching

assert (
    RAW_DATA_PATH
    and PROCESSED_DATA_PATH
    and FEATURES_PATH
    and os.path.exists(RAW_DATA_PATH)
    and os.path.exists(PROCESSED_DATA_PATH)
    and os.path.exists(FEATURES_PATH)
)


def check_tiling():
    pass


def check_colour_deconvolution():
    pass


def check_validate_psr_image():
    pass


def check_feature_engineering():
    """This function checks the completion of feature engineering"""
    print("===== CHECK : FEATURE ENGINEERING =====")

    job_batch_file = pd.read_csv(JOB_BATCH_FILE_PATH, header=None, names=["slide"])

    check_summary_columns = [
        "slide",
        "image_tile",
        "image_tile_index",
        "feature_engineering_performed",
        "feature_engineering_completed",
    ]
    check_summary_rows = []

    for slide_name in job_batch_file.slide.values:
        # get a list of valid image tiles
        summary_file_valid_image_tiles = os.path.join(
            PROCESSED_DATA_PATH,
            slide_name,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            "PSR",
            "deconvolutions",
            "psr",
            "inverted_grayscale",
            "valid_psr_images_summary",
            "job_batch_valid_image_tiles.csv",
        )
        output_directory_feature_engineering_tiles = os.path.join(
            FEATURES_PATH,
            slide_name,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            "PSR",
            "deconvolutions",
            "psr",
            "inverted_grayscale",
            "tile_level_features",
        )
        if not os.path.exists(summary_file_valid_image_tiles):
            check_summary_rows.append(
                (slide_name, "none_validated_for_psr", "n/a", "n/a", "n/a")
            )
        else:
            if not os.path.exists(output_directory_feature_engineering_tiles):
                check_summary_rows.append(
                    (
                        slide_name,
                        "none_performed_for_feature_engineering",
                        "n/a",
                        "n/a",
                        "n/a",
                    )
                )
            else:
                job_batch_valid_image_tiles = pd.read_csv(
                    summary_file_valid_image_tiles
                )
                for _, row in job_batch_valid_image_tiles.iterrows():
                    image_tile, image_tile_index = (
                        row["path_to_image_tile"],
                        row["index"],
                    )
                    image_tile_name = os.path.basename(image_tile).split(".")[0]
                    # check if there are outputs for this image tile
                    flag_feature_engineering_performed: str = "no"
                    flag_feature_engineering_completed: str = "no"

                    if os.path.exists(
                        os.path.join(
                            output_directory_feature_engineering_tiles, image_tile_name
                        )
                    ):
                        flag_feature_engineering_performed = "yes"
                        if os.path.exists(
                            os.path.join(
                                output_directory_feature_engineering_tiles,
                                image_tile_name,
                                "features_out.mat",
                            )
                        ):
                            flag_feature_engineering_completed = "yes"

                    check_summary_rows.append(
                        (
                            slide_name,
                            image_tile,
                            image_tile_index,
                            flag_feature_engineering_performed,
                            flag_feature_engineering_completed,
                        )
                    )

    check_summary = pd.DataFrame(data=check_summary_rows, columns=check_summary_columns)

    path_to_write_check_summary = os.path.join(
        os.path.dirname(JOB_BATCH_FILE_PATH),
        f"{os.path.basename(JOB_BATCH_FILE_PATH).split('.')[0]}_check_summary_feature_engineering.csv",
    )

    check_summary.to_csv(path_to_write_check_summary, index=False)


def check_stitching():
    pass


if FLAG_CHECK_TILING:
    check_tiling()
if FLAG_CHECK_COLOUR_DECONVOLUTION:
    check_colour_deconvolution()
if FLAG_CHECK_VALIDATE_PSR_IMAGE:
    check_validate_psr_image()
if FLAG_CHECK_FEATURE_ENGINEERING:
    check_feature_engineering()
if FLAG_CHECK_STITCHING:
    check_stitching()
