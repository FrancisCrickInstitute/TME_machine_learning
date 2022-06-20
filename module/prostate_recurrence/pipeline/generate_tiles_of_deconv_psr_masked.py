import argparse
import os
import sys
from datetime import datetime
from glob import glob
from PIL import Image
import numpy as np

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
RAW_DATA_RES = args.raw_data_res
MASK_RES = args.mask_res
TILE_SIZE = args.tile_size
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)


if __name__ == "__main__":

    all_paths_to_data_unfiltered = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    logstr = "===== MASKING DECONVOLVED PSR =====\n"

    # filter out paths that don't have deconvolutions & tissue mask ready
    all_paths_to_data = []
    for path in all_paths_to_data_unfiltered:
        slide_id = "_".join(
            [os.path.basename(path).split("_")[0], os.path.basename(path).split("_")[1]]
        )
        if os.path.exists(
            os.path.join(
                PROCESSED_DATA_PATH,
                slide_id,
                f"tile_size_{TILE_SIZE}",
                "whole_slide",
                RAW_DATA_TYPE,
                "tissue_masks",
            )
        ) and os.path.exists(
            os.path.join(
                PROCESSED_DATA_PATH,
                slide_id,
                f"tile_size_{TILE_SIZE}",
                "whole_slide",
                RAW_DATA_TYPE,
                "deconvolutions",
            )
        ):
            all_paths_to_data.append(path)

    for path in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path} at {date_time}\n"
        print(f"> processing path : {path}")
        slide_id = "_".join(
            [os.path.basename(path).split("_")[0], os.path.basename(path).split("_")[1]]
        )

        output_directory_processed_tissue_mask_tiles = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "tissue_masks",
            "tissue_mask_v2",
        )
        if not os.path.exists(output_directory_processed_tissue_mask_tiles):
            logstr += "... tissue mask tiles n/a; skipped ..."
            continue

        output_directory_processed_tissue_mask_tiles_scan_region_paths = natsorted(
            glob(
                os.path.join(
                    output_directory_processed_tissue_mask_tiles, "ScanRegion*"
                )
            )
        )

        for (
            output_directory_processed_tissue_mask_tiles_scan_region_path
        ) in output_directory_processed_tissue_mask_tiles_scan_region_paths:
            scan_region = os.path.basename(
                output_directory_processed_tissue_mask_tiles_scan_region_path
            )
            print(f"> processing scene: {scan_region}")

            # confirm if the deconvolved psr exists
            output_directory_processed_deconvolved_psr_scan_region_path = os.path.join(
                PROCESSED_DATA_PATH,
                slide_id,
                f"tile_size_{TILE_SIZE}",
                "whole_slide",
                RAW_DATA_TYPE,
                "deconvolutions",
                f"{scan_region}",
                "psr",
                "inverted_grayscale",
            )
            assert os.path.exists(
                output_directory_processed_deconvolved_psr_scan_region_path
            )

            # get all image paths for this scan region
            output_directory_processed_tissue_mask_tiles_scan_region_image_paths = natsorted(
                glob(
                    os.path.join(
                        output_directory_processed_tissue_mask_tiles_scan_region_path,
                        "image_tile*.tif",
                    )
                )
            )
            for (
                output_directory_processed_tissue_mask_tiles_scan_region_image_path
            ) in output_directory_processed_tissue_mask_tiles_scan_region_image_paths:
                tissue_mask_tile_name = os.path.splitext(
                    os.path.basename(
                        output_directory_processed_tissue_mask_tiles_scan_region_image_path
                    )
                )[0]
                deconvolved_psr_tile_name_with_extension = (
                    tissue_mask_tile_name + "_psr.tif"
                )
                output_directory_processed_deconvolved_psr_scan_region_image_path = (
                    os.path.join(
                        output_directory_processed_deconvolved_psr_scan_region_path,
                        deconvolved_psr_tile_name_with_extension,
                    )
                )
                if os.path.exists(
                    output_directory_processed_deconvolved_psr_scan_region_image_path
                ):
                    tissue_mask_tile = Image.open(
                        output_directory_processed_tissue_mask_tiles_scan_region_image_path
                    )
                    deconvolved_psr_tile = Image.open(
                        output_directory_processed_deconvolved_psr_scan_region_image_path
                    )

                    tissue_mask_array = np.array(tissue_mask_tile)
                    deconvolved_psr_array = np.array(deconvolved_psr_tile)

                    # multiple the above two
                    deconvolved_psr_tissue_masked_array = np.multiply(
                        deconvolved_psr_array,
                        tissue_mask_array / 255,
                    ).astype(np.uint8)

                    deconvolved_psr_tissue_masked = Image.fromarray(
                        deconvolved_psr_tissue_masked_array
                    )
                    output_directory_processed_deconvolved_psr_tissue_masked_scan_region_image_path = output_directory_processed_deconvolved_psr_scan_region_image_path.replace(
                        "inverted_grayscale", "inverted_grayscale_tissue_masked"
                    )
                    os.makedirs(
                        os.path.dirname(
                            output_directory_processed_deconvolved_psr_tissue_masked_scan_region_image_path
                        ),
                        exist_ok=True,
                    )
                    os.chmod(
                        os.path.dirname(
                            output_directory_processed_deconvolved_psr_tissue_masked_scan_region_image_path
                        ),
                        mode=0o777,
                    )

                    deconvolved_psr_tissue_masked.save(
                        output_directory_processed_deconvolved_psr_tissue_masked_scan_region_image_path
                    )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_paths_to_data.index(path)+1)} data paths processed (total: {len(all_paths_to_data)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""
