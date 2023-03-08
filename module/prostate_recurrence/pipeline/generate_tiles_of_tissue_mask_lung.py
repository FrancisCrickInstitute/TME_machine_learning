"""# a set of functions for generating tiles of tissue masks

## input images are downsized tissue masks generated in MATLAB

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

parser = argparse.ArgumentParser(prog="tme-ml-raw-data-tissue-psr-mask")
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
    "--processed_data_path",
    dest="processed_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path saving processed data.",
)
parser.add_argument(
    "--path_to_tissue_masks",
    dest="path_to_tissue_masks",
    action="store",
    type=str,
    default="",
    help="provide the path to the folder containing tissue masks.",
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
    help="options are 20x, 10x. By default it's 20x",
)
parser.add_argument(
    "--mask_res",
    dest="mask_res",
    action="store",
    type=str,
    default="2.5x",
    help="By default it's 2.5x",
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
PROCESSED_DATA_PATH = args.processed_data_path
PATH_TO_TISSUE_MASKS = args.path_to_tissue_masks
RAW_DATA_RES = args.raw_data_res
MASK_RES = args.mask_res
TILE_SIZE = args.tile_size
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import tiling

if __name__ == "__main__":

    all_paths_to_data = natsorted(
        glob(os.path.join(PATH_TO_TISSUE_MASKS, "*binarised_image.tif"))
    )
    logstr = "===== TILING OF TISSUE MASKS =====\n"

    downsize_factor = float(MASK_RES.split("x")[0]) / float(RAW_DATA_RES.split("x")[0])
    downsized_tile_size = int(TILE_SIZE * downsize_factor)

    for path_to_tissue_mask in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path_to_tissue_mask} at {date_time}\n"
        print(f"> processing path : {path_to_tissue_mask}")
        slide_id = os.path.basename(path_to_tissue_mask).split("_")[0]
        scene = os.path.basename(path_to_tissue_mask).split("_")[1]

        # check if slide_id is in summary of good tissue masks

        output_directory_processed_tissue_and_psr_mask_tiles = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "tissue_masks",
        )
        os.makedirs(output_directory_processed_tissue_and_psr_mask_tiles, exist_ok=True)
        os.chmod(output_directory_processed_tissue_and_psr_mask_tiles, mode=0o777)

        directory_to_save_mask_tiles = os.path.join(
            output_directory_processed_tissue_and_psr_mask_tiles,
            "tissue_mask_v2",
        )
        os.makedirs(directory_to_save_mask_tiles, exist_ok=True)
        os.chmod(directory_to_save_mask_tiles, mode=0o777)

        # processing scene
        subdirectory_to_save_mask_tiles = os.path.join(
            directory_to_save_mask_tiles, scene
        )
        os.makedirs(subdirectory_to_save_mask_tiles, exist_ok=True)
        os.chmod(subdirectory_to_save_mask_tiles, mode=0o777)

        tissue_mask = Image.open(path_to_tissue_mask)
        mask_array = np.array(tissue_mask)

        # tiling with downsized resolution
        dict_img_tiles_mask, nrow, ncol = tiling.create_tiles(
            img=mask_array, size=downsized_tile_size
        )

        # save tiles with original resolution
        tiling.save_tiles(
            dict_img_tiles=dict_img_tiles_mask,
            ncol=ncol,
            nrow=nrow,
            save_path=subdirectory_to_save_mask_tiles,
            size=TILE_SIZE,
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_paths_to_data.index(path_to_tissue_mask)+1)} data paths processed (total: {len(all_paths_to_data)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""
