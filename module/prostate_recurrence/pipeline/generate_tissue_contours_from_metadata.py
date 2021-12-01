import argparse
import os
import sys
from datetime import datetime
from glob import glob

import pandas as pd
from natsort import natsorted

parser = argparse.ArgumentParser(prog="tme-ml-raw-data-tissue-contour")
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

args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
PROCESSED_DATA_PATH = args.processed_data_path
MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.image_processing import tissue_contour_from_metadata

if __name__ == "__main__":

    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    all_contour_size_information = pd.DataFrame()
    logstr = "===== TISSUE CONTOUR EXTRACTION =====\n"
    for path in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path} at {date_time}\n"
        print(f"> processing path : {path}")
        slide_id = "_".join(
            [os.path.basename(path).split("_")[0], os.path.basename(path).split("_")[1]]
        )
        contours_extracted = tissue_contour_from_metadata.extract_tissue_contour(
            path_to_img=path
        )

        output_directory_processed_extracted_contours = os.path.join(
            PROCESSED_DATA_PATH,
            slide_id,
            "tissue_contours_from_metadata",
        )
        os.makedirs(output_directory_processed_extracted_contours, exist_ok=True)

        print("... saving scatterplot of tissue contours")
        path_to_save_plot = os.path.join(
            output_directory_processed_extracted_contours,
            "tissue_contours_extracted_from_metadata.pdf",
        )
        tissue_contour_from_metadata.scatterplot_contours(
            contours_extracted=contours_extracted, path_to_save_plot=path_to_save_plot
        )

        print("... saving dataframes of tissue contours")
        path_to_save_dataframe = os.path.join(
            output_directory_processed_extracted_contours,
            "tissue_contours_extracted_from_metadata",
        )
        (
            df_contour_center_positions,
            df_contour_sizes,
            df_contours,
        ) = tissue_contour_from_metadata.dataframe_contours(
            contours_extracted=contours_extracted,
            path_to_save_dataframe=path_to_save_dataframe,
        )

        all_contour_size_information = all_contour_size_information.append(
            df_contour_sizes
        )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_paths_to_data.index(path)+1)} data paths processed (total: {len(all_paths_to_data)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""

    directory_to_save_summary = os.path.join(
        RAW_DATA_PATH, "summary_of_tissue_contours_from_metadata"
    )
    os.makedirs(directory_to_save_summary, exist_ok=True)
    all_contour_size_information.to_csv(
        os.path.join(
            directory_to_save_summary, "all_tissue_contour_size_information.csv"
        ),
        index=False,
    )
