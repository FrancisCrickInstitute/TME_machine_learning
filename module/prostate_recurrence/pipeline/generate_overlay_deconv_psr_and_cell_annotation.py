""" This script calls functions for overlaying deconvolved psr with cell annotation 
"""

import argparse
import os
import sys
from datetime import datetime
from glob import glob

from natsort import natsorted

parser = argparse.ArgumentParser(prog="tme-ml-overlay")
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
    "--directory_to_cell_annotation",
    dest="directory_to_cell_annotation",
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
    "--downscale_factor_ecm_to_cells",
    dest="downscale_factor_ecm_to_cells",
    action="store",
    type=float,
    default=0.25,
    help="Downscale factor ecm to cell resolution. By default it's 0.25",
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
DIR_CELL_ANNOTATION = args.directory_to_cell_annotation

MASK_RES = args.mask_res
TILE_SIZE = args.tile_size
DOWNSCALE_FACTOR = args.downscale_factor
DOWNSCALE_FACTOR_ECM_TO_CELLS = args.downscale_factor_ecm_to_cells
RES_MICRON_PER_PIXEL = args.resolution_micron_per_pixel

OUTPUT_DIR = args.output_directory

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import (
    integrated_overlay_tissue_mask_and_tumour_annotation,
)

if __name__ == "__main__":
    all_slides_with_cell_annotation = [
        "_".join(os.path.basename(slide_csv).split("_")[:2])
        for slide_csv in natsorted(glob(os.path.join(DIR_CELL_ANNOTATION, "*csv")))
    ]
    print(all_slides_with_cell_annotation)

    logstr = "===== DECONV PSR + CELL ANNOTATION =====\n"

    for slide in all_slides_with_cell_annotation:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {slide} at {date_time}\n"

        # raw
        path_to_raw_czi = os.path.join(RAW_DATA_PATH, f"{slide}_PSR.czi")
        # deconvolved PSR
        directory_to_deconv_psr_tiles = os.path.join(
            PROCESSED_DATA_PATH,
            slide,
            "tile_size_2000/whole_slide/PSR/deconvolutions",
        )
        # cell annotation
        path_to_cell_annotation = f"/Volumes/lab-sahaie/working/Hanyun/tx_PSR/PSR-H-PROSTATE/csv_WSI/CellPos/{slide}_PSR_cellPos.csv"

        # get image information
        (
            scene_information_dataframe_complete,
            slide_information,
        ) = integrated_overlay_tissue_mask_and_tumour_annotation.read_image_scene_information(
            path_to_image=path_to_raw_czi,
            resolution_micron_per_pixel=RES_MICRON_PER_PIXEL,
        )

        # process cell annotation
        cell_annotation_wsi = integrated_overlay_tissue_mask_and_tumour_annotation.read_cell_annotation_wsi(
            path_to_cell_annotation_wsi=path_to_cell_annotation
        )

        # process deconv psr
        dict_deconv_psr_tiles_scene_level = (
            integrated_overlay_tissue_mask_and_tumour_annotation.read_deconv_psr(
                directory_to_deconv_psr_tiles=directory_to_deconv_psr_tiles,
                tile_size=TILE_SIZE,
                downscale_factor=DOWNSCALE_FACTOR,
            )
        )
        stitch_downscaled_deconv_psr = (
            integrated_overlay_tissue_mask_and_tumour_annotation.process_tissue_mask(
                slide_information=slide_information,
                scene_information=scene_information_dataframe_complete,
                dict_image_arrays_scene_level=dict_deconv_psr_tiles_scene_level,
                tile_size=TILE_SIZE,
                downscale_factor=DOWNSCALE_FACTOR,
                resolution_micron_per_pixel=RES_MICRON_PER_PIXEL,
            )
        )

        # overlay deconv psr with cell annotation
        integrated_overlay_tissue_mask_and_tumour_annotation.overlay_deconv_psr_with_cell_annotation(
            stitch_downscaled_deconv_psr=stitch_downscaled_deconv_psr,
            cell_annotation_wsi=cell_annotation_wsi,
            main_output_directory=OUTPUT_DIR,
            slide=slide,
            downscale_factor_ecm_to_cells=DOWNSCALE_FACTOR_ECM_TO_CELLS,
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_slides_with_cell_annotation.index(slide)+1)} slides processed (total: {len(all_slides_with_cell_annotation)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""