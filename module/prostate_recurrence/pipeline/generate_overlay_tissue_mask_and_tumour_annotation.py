""" This script calls functions for overlaying tissue mask with tumour annotation 
"""

import argparse
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
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
parser.add_argument(
    "--directory_to_tumour_annotation",
    dest="directory_to_tumour_annotation",
    action="store",
    type=str,
    help="Need to be provided",
)
parser.add_argument(
    "--downscale_factor",
    dest="downscale_factor",
    action="store",
    type=float,
    default=0.125,
    help="Downscale factor from 20x for saving overlay image & array. By default it's 0.125",
)
parser.add_argument(
    "--resolution_micron_per_pixel",
    dest="resolution_micron_per_pixel",
    action="store",
    type=float,
    default=0.22,
    help="Resolution in micron per pixel. By default it's 0.22",
)
parser.add_argument(
    "--output_directory",
    dest="output_directory",
    action="store",
    type=str,
    help="Need to be provided",
)


args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
PROCESSED_DATA_PATH = args.processed_data_path
RAW_DATA_RES = args.raw_data_res
DIR_TUMOUR_ANNOTATION = args.directory_to_tumour_annotation

MASK_RES = args.mask_res
TILE_SIZE = args.tile_size
DOWNSCALE_FACTOR = args.downscale_factor
RES_MICRON_PER_PIXEL = args.resolution_micron_per_pixel

OUTPUT_DIR = args.output_directory

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import (
    integrated_overlay_tissue_mask_and_tumour_annotation,
)

if __name__ == "__main__":
    all_slides_with_tumour_annotation = [
        "_".join(os.path.basename(slide_czi).split("_")[:2])
        for slide_czi in natsorted(
            glob(os.path.join(DIR_TUMOUR_ANNOTATION, "freehandlabels_png/*czi"))
        )
    ]
    print(all_slides_with_tumour_annotation)

    logstr = "===== TISSUE MASK + TUMOUR ANNOTATION =====\n"

    for slide in all_slides_with_tumour_annotation:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {slide} at {date_time}\n"

        # raw
        path_to_raw_czi = os.path.join(RAW_DATA_PATH, f"{slide}_PSR.czi")
        # tissue mask
        input_directory_tissue_mask_tiles = os.path.join(
            PROCESSED_DATA_PATH,
            slide,
            "tile_size_2000/whole_slide/PSR/tissue_masks/tissue_mask_v2",
        )
        # tumour annotation
        input_directory_annotation_overlay = os.path.join(
            DIR_TUMOUR_ANNOTATION, f"AnnotatedTiles/{slide}_PSR.czi/"
        )
        input_directory_annotation_binary = os.path.join(
            DIR_TUMOUR_ANNOTATION, f"freehandlabels_png/{slide}_PSR.czi/"
        )
        input_directories = {
            # "input_directory_annotation_overlay": input_directory_annotation_overlay, # this is not yet available for data on CAMP
            "input_directory_annotation_binary": input_directory_annotation_binary,
        }
        image_tile_name_pattern = "Da*"

        # get image information
        (
            scene_information_dataframe_complete,
            slide_information,
        ) = integrated_overlay_tissue_mask_and_tumour_annotation.read_image_scene_information(
            path_to_image=path_to_raw_czi,
            resolution_micron_per_pixel=RES_MICRON_PER_PIXEL,
        )

        # process tumour annotation
        (
            dict_annotation_overlay_image_arrays,
            dict_annotation_binary_image_arrays,
        ) = integrated_overlay_tissue_mask_and_tumour_annotation.read_tumour_annotation(
            input_directories=input_directories,
            tile_size=TILE_SIZE,
            downscale_factor=DOWNSCALE_FACTOR,
        )
        stitch_downscaled_annotation_binary = integrated_overlay_tissue_mask_and_tumour_annotation.process_tumour_annotation(
            slide_information=slide_information,
            dict_annotation_binary_image_arrays=dict_annotation_binary_image_arrays,
            tile_size=TILE_SIZE,
            downscale_factor=DOWNSCALE_FACTOR,
        )

        # process tissue mask
        dict_tissue_mask_tiles_scene_level = integrated_overlay_tissue_mask_and_tumour_annotation.read_tissue_mask(
            camp_input_directory_tissue_mask_tiles=input_directory_tissue_mask_tiles,
            tile_size=TILE_SIZE,
            downscale_factor=DOWNSCALE_FACTOR,
        )
        stitch_downscaled_tissue_mask = (
            integrated_overlay_tissue_mask_and_tumour_annotation.process_tissue_mask(
                slide_information=slide_information,
                scene_information=scene_information_dataframe_complete,
                dict_image_arrays_scene_level=dict_tissue_mask_tiles_scene_level,
                tile_size=TILE_SIZE,
                downscale_factor=DOWNSCALE_FACTOR,
                resolution_micron_per_pixel=RES_MICRON_PER_PIXEL,
            )
        )

        # overlay tumour annotation with tissue mask
        stitch_downscaled_annotation_overlay_with_tissue_mask = integrated_overlay_tissue_mask_and_tumour_annotation.overlay_tumour_annotation_with_tissue_mask(
            stitch_downscaled_annotation_binary=stitch_downscaled_annotation_binary,
            stitch_downscaled_tissue_mask=stitch_downscaled_tissue_mask,
            main_output_directory=OUTPUT_DIR,
            slide=slide,
        )

        # summarise tiles of the overlay
        summary, summary_subtile = integrated_overlay_tissue_mask_and_tumour_annotation.tiles_of_overlay_tumour_annotation_with_tissue_mask(
            slide_information=slide_information,
            scene_information_dataframe_complete=scene_information_dataframe_complete,
            stitch_downscaled_annotation_overlay_with_tissue_mask=stitch_downscaled_annotation_overlay_with_tissue_mask,
            main_output_directory=OUTPUT_DIR,
            slide=slide,
            tile_size=TILE_SIZE,
            downscale_factor=DOWNSCALE_FACTOR,
            resolution_micron_per_pixel=RES_MICRON_PER_PIXEL,
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_slides_with_tumour_annotation.index(slide)+1)} slides processed (total: {len(all_slides_with_tumour_annotation)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""
