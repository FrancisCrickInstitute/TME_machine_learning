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
args = parser.parse_args()
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
# module_path = os.path.abspath(os.path.join("../../../module/"))
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import tissue_mask

if __name__ == "__main__":
    all_paths_to_data = natsorted(glob(
        os.path.join(
            RAW_DATA_PATH,
            f"*{RAW_DATA_TYPE}.czi"
        )
    ))
    all_image_information = pd.DataFrame()
    for path in all_paths_to_data:
        print(f"> processing path : {path}")
        img_info = tissue_mask.read_image_information(
            path_to_img=path,
            method='bioformats',
            resolution="20x"
        )
        all_image_information = all_image_information.append(img_info)

    all_image_information.to_csv(
        os.path.join(
            RAW_DATA_PATH,
            "all_image_information.csv"
        ),
        index=False
    )