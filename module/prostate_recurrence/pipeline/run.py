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

import os
import sys
module_path = os.path.abspath(os.path.join('../../../module/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from glob import glob
from prostate_recurrence.image_processing import tiling
import argparse

parser = argparse.ArgumentParser(prog="tme-ml-pipeline")
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
TILE_SIZE = args.tile_size


assert RAW_DATA_PATH and PROCESSED_DATA_PATH


def run_tiling():
    do_batch_processing = True
    if f".czi" in RAW_DATA_PATH:
        do_batch_processing = False

    if do_batch_processing:
        data_paths = glob(f"*{RAW_DATA_TYPE}.czi")
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

    pass


run_tiling()

