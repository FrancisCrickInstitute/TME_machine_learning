import argparse
import os
from datetime import datetime
from glob import glob

import pandas as pd
from natsort import natsorted

parser = argparse.ArgumentParser(prog="tme-ml-pipeline-texture-features")
parser.add_argument(
    "--raw_data_path",
    dest="raw_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path containing raw data.",
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
    "--output_directory",
    dest="output_directory",
    action="store",
    type=str,
    default="",
    help="provide the directory to save outputs.",
)

args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
PROCESSED_DATA_PATH = args.processed_data_path
OUTPUT_DIRECTORY = args.output_directory
TILE_SIZE = args.tile_size

FLAG_INTENSITY_FEATURES = args.flag_intensity_features
FLAG_GLCM_FEATURES = args.flag_glcm_features
FLAG_PERCEPTION_FEATURES = args.flag_perception_features


def save_dataframes(
    combined_glcm_features_masked: pd.DataFrame,
    combined_glcm_features_notmasked: pd.DataFrame,
):
    for mask_condition, combined_glcm_features in zip(
        ["masked", "notmasked"],
        [combined_glcm_features_masked, combined_glcm_features_notmasked],
    ):
        path_to_save_combined_glcm_features = os.path.join(
            OUTPUT_DIRECTORY, f"combined_glcm_features_{mask_condition}.csv"
        )
        combined_glcm_features.to_csv(path_to_save_combined_glcm_features, index=False)


if __name__ == "__main__":

    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )

    assert FLAG_GLCM_FEATURES or FLAG_INTENSITY_FEATURES or FLAG_PERCEPTION_FEATURES

    logstr = "===== EXTRACTION OF TEXTURE FEATURES =====\n"

    if FLAG_GLCM_FEATURES:
        combined_glcm_features_masked = pd.DataFrame()
        combined_glcm_features_notmasked = pd.DataFrame()

    for path in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path} at {date_time}\n"
        print(f"> processing path : {path}")
        data_id = "_".join(os.path.basename(path).split("_")[:2])

        output_directory_processed_features = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions",
        )

        if not os.path.exists(output_directory_processed_features):
            continue

        output_directory_processed_features_scan_region_paths = natsorted(
            glob(os.path.join(output_directory_processed_features, "ScanRegion*"))
        )

        for (
            output_directory_processed_features_scan_region_path
        ) in output_directory_processed_features_scan_region_paths:
            scan_region = os.path.basename(
                output_directory_processed_features_scan_region_path
            )
            print(f"> processing scene: {scan_region}")

            output_directory_processed_features_scan_region_texture = os.path.join(
                output_directory_processed_features_scan_region_path,
                "psr",
                "inverted_grayscale_tissue_masked",
                "tile_level_features_texture_v2",
            )

            if not os.path.exists(
                output_directory_processed_features_scan_region_texture
            ):
                continue

            output_directory_processed_features_scan_region_texture_image_tiles = (
                natsorted(
                    glob(
                        os.path.join(
                            output_directory_processed_features_scan_region_texture,
                            "image_tile*",
                        )
                    )
                )
            )

            for (
                output_directory_processed_features_scan_region_texture_image_tile
            ) in output_directory_processed_features_scan_region_texture_image_tiles:
                image_tile = os.path.basename(
                    output_directory_processed_features_scan_region_texture_image_tile
                )
                print(f"> processing image tile: {image_tile}")

                if FLAG_GLCM_FEATURES:
                    for mask_condition in ["masked", "notmasked"]:
                        glcm_features_filename = f"glcm_features_{mask_condition}.csv"
                        path_to_glcm_features = os.path.join(
                            output_directory_processed_features_scan_region_texture_image_tile,
                            glcm_features_filename,
                        )
                        if not os.path.exists(path_to_glcm_features):
                            continue

                        glcm_features_rotated = pd.read_csv(path_to_glcm_features)

                        if glcm_features_rotated.empty:
                            continue

                        glcm_features_rotated.set_index("feature")
                        glcm_features = glcm_features_rotated.T
                        glcm_features["slide_id"] = data_id
                        glcm_features["scene_id"] = scan_region
                        glcm_features["image_tile"] = image_tile
                        glcm_features[
                            "path_to_image_tile"
                        ] = output_directory_processed_features_scan_region_texture_image_tile

                        if mask_condition == "masked":
                            combined_glcm_features_masked = (
                                combined_glcm_features_masked.append(glcm_features)
                            )
                        elif mask_condition == "notmasked":
                            combined_glcm_features_notmasked = (
                                combined_glcm_features_notmasked.append(glcm_features)
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

        if FLAG_GLCM_FEATURES and (int(all_paths_to_data.index(path) + 1) % 20 == 0):
            save_dataframes(
                combined_glcm_features_masked, combined_glcm_features_notmasked
            )

    if FLAG_GLCM_FEATURES:
        save_dataframes(combined_glcm_features_masked, combined_glcm_features_notmasked)
