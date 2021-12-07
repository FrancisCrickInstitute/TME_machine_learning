import argparse
import os
import sys
from datetime import datetime
from glob import glob

from natsort import natsorted

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


args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
PROCESSED_DATA_PATH = args.processed_data_path
RAW_DATA_RES = args.raw_data_res
MASK_RES = args.mask_res
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import tissue_mask_from_tissue_contour

if __name__ == "__main__":

    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    logstr = "===== TISSUE & PSR MASKS =====\n"

    for path in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path} at {date_time}\n"
        print(f"> processing path : {path}")
        slide_id = "_".join(
            [os.path.basename(path).split("_")[0], os.path.basename(path).split("_")[1]]
        )

        output_directory_processed_extracted_contours = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            "tissue_contours_from_metadata",
        )
        assert os.path.exists(output_directory_processed_extracted_contours)

        output_directory_processed_mask_whole_slide = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            "tissue_mask_whole_slide",
            f"resolution_{MASK_RES}",
        )
        assert os.path.exists(output_directory_processed_mask_whole_slide)

        output_directory_processed_tissue_and_psr_mask_whole_slide = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            "tissue_and_psr_mask_whole_slide_based_on_contours",
            f"resolution_{MASK_RES}",
        )
        os.makedirs(
            output_directory_processed_tissue_and_psr_mask_whole_slide, exist_ok=True
        )
        os.chmod(output_directory_processed_tissue_and_psr_mask_whole_slide, mode=0o777)

        # [1] create tissue mask based on metadata-derived tissue contour
        path_to_contours = os.path.join(
            output_directory_processed_extracted_contours,
            "tissue_contours_extracted_from_metadata_contours.csv",
        )
        path_to_centres = os.path.join(
            output_directory_processed_extracted_contours,
            "tissue_contours_extracted_from_metadata_contour_center_positions.csv",
        )
        scene_arrays = tissue_mask_from_tissue_contour.read_scene_arrays(
            directory_to_scenes=output_directory_processed_mask_whole_slide
        )
        (contours, centres) = tissue_mask_from_tissue_contour.read_dataframe_contours(
            path_to_centres=path_to_centres, path_to_contours=path_to_contours
        )

        contours_processed = tissue_mask_from_tissue_contour.process_contours(
            contours=contours,
            centres=centres,
        )

        downsize_factor = float(MASK_RES.split("x")[0]) / float(
            RAW_DATA_RES.split("x")[0]
        )
        tissue_mask_from_tissue_contour.save_tissue_mask(
            contours_processed=contours_processed,
            scene_arrays=scene_arrays,
            directory_to_save_images=output_directory_processed_tissue_and_psr_mask_whole_slide,
            downsize_factor=downsize_factor,
            downsized_resolution=MASK_RES,
        )

        # [2] create psr mask, with refinement by tissue mask and contour
        scene_mask_arrays = tissue_mask_from_tissue_contour.read_scene_mask_arrays(
            directory_to_scene_masks=output_directory_processed_tissue_and_psr_mask_whole_slide
        )
        psr_mask_arrays = tissue_mask_from_tissue_contour.create_psr_mask(
            scene_mask_arrays=scene_mask_arrays, scene_arrays=scene_arrays
        )
        tissue_mask_from_tissue_contour.save_psr_mask(
            scene_arrays=scene_arrays,
            psr_mask_arrays=psr_mask_arrays,
            directory_to_save_images=output_directory_processed_tissue_and_psr_mask_whole_slide,
            downsized_resolution=MASK_RES,
        )
