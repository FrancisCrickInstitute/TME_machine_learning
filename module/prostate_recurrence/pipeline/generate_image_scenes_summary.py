import argparse
import os
import sys
from glob import glob

import pandas as pd
from natsort import natsorted


parser = argparse.ArgumentParser(prog="tme-ml-raw-data-summary")
parser.add_argument(
    "--module_path",
    dest="module_path",
    action="store",
    type=str,
    default="../../../module/",
    help="module path to pipeline functions.",
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
TILE_SIZE = args.tile_size
RAW_DATA_TYPE = args.raw_data_type
# module_path = os.path.abspath(os.path.join("../../../module/"))
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

# from prostate_recurrence.image_processing import tissue_mask
from prostate_recurrence.image_processing import scene_information

if __name__ == "__main__":
    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    all_image_information = pd.DataFrame()
    for path in all_paths_to_data:
        print(f"> processing path : {path}")
        # img_info = tissue_mask.read_image_information(
        #     path_to_img=path, method="bioformats", resolution="20x"
        # )
        img_info = scene_information.extract_scene_information(
            path_to_image=path, tile_size=TILE_SIZE
        )
        all_image_information = all_image_information.append(img_info)

    directory_to_save_summary = os.path.join(RAW_DATA_PATH, "summary_of_image_scenes")
    os.makedirs(directory_to_save_summary, exist_ok=True)
    os.chmod(directory_to_save_summary, mode=0o777)
    all_image_information.to_csv(
        os.path.join(directory_to_save_summary, "all_image_information_v2.csv"),
        index=False,
    )