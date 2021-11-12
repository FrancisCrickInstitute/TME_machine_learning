import argparse
import os
import sys
from glob import glob

import pandas as pd
from natsort import natsorted


parser = argparse.ArgumentParser(prog="tme-ml-raw-data-tissue-mask")
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

args = parser.parse_args()
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
PROCESSED_DATA_PATH = args.processed_data_path
# module_path = os.path.abspath(os.path.join("../../../module/"))
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import tissue_mask

if __name__ == "__main__":
    #MASK_RES = '5x'; MASK_RES_0 = '5.0x'
    MASK_RES = '2.5x'; MASK_RES_0 = '2.5x'
    all_paths_to_data = natsorted(glob(
        os.path.join(
            RAW_DATA_PATH,
            f"*{RAW_DATA_TYPE}.czi"
        )
    ))
    all_mask_information = pd.DataFrame()
    for path in all_paths_to_data:
        print(f"> processing path : {path}")
        slide_id = '_'.join(
            [
                os.path.basename(path).split('_')[0],
                os.path.basename(path).split('_')[1]
            ]
        )
        dict_imgs_with_res = tissue_mask.read_image(
            path_to_img=path,
            method='bioformats',
            highest_resolution='10x',
            resolution=MASK_RES
        )
        dict_masks = tissue_mask.create_tissue_mask(
            dict_imgs=dict_imgs_with_res[MASK_RES_0],
            gaussian_blur_sigma=2.,
            resize_factor=4,
            erosion_n=10,
            dilation_n=10
        )

        output_directory_processed_mask_whole_slide = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            "tissue_mask_whole_slide",
            f"resolution_{MASK_RES}"
        )
        os.makedirs(output_directory_processed_mask_whole_slide, exist_ok=True)

        print('... saving low resolution whole slide')
        mask_summary = tissue_mask.save_low_res_whole_slide_image(
            dict_imgs=dict_imgs_with_res[MASK_RES_0],
            dict_masks=dict_masks,
            slide_id=slide_id,
            save_path=output_directory_processed_mask_whole_slide,
            resolution=MASK_RES_0
        )

        all_mask_information = all_mask_information.append(mask_summary)

        # also save tiles of masks - output_directory_processed_mask_tiling

    directory_to_save_summary = os.path.join(
        RAW_DATA_PATH,
        "summary_of_image_scenes"
    )
    os.makedirs(directory_to_save_summary, exist_ok=True)
    all_mask_information.to_csv(
        os.path.join(
            directory_to_save_summary,
            "all_tissue_mask_information.csv"
        ),
        index=False
    )