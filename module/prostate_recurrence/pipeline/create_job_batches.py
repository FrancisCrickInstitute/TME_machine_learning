import argparse
import os
import sys
from datetime import datetime
from glob import glob
from typing import List, Generator

import pandas as pd
from natsort import natsorted

parser = argparse.ArgumentParser(prog="tme-ml-pipeline-job-batches")
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
    help="By default, 20x.",
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
    "--job_batch_size",
    dest="job_batch_size",
    action="store",
    type=int,
    default=8,
    help="provide the number of jobs per batch.",
)
parser.add_argument(
    "--job_batch_output_dir",
    dest="job_batch_output_dir",
    action="store",
    type=int,
    default=8,
    help="provide the directory for saving job batch files.",
)


args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
RAW_DATA_PATH = args.raw_data_path
RAW_DATA_TYPE = args.raw_data_type
RAW_DATA_RES = args.raw_data_res
PROCESSED_DATA_PATH = args.processed_data_path
TILE_SIZE = args.tile_size
JOB_BATCH_SIZE = args.job_batch_size
JOB_BATCH_OUTPUT_DIR = args.job_batch_output_dir

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)


def generate_job_batch(all_jobs: List[str], job_batch_size: int) -> Generator:
    n_job_batches = len(all_jobs) // JOB_BATCH_SIZE + 1
    for job_batch_i in range(n_job_batches):
        yield all_jobs[
            job_batch_i
            * job_batch_size : min(
                (job_batch_i + 1) * job_batch_size, len(all_jobs) - 1
            )
        ]


def write_job_batch(
    job_batch_information: pd.DataFrame, path_to_save_job_batch_information: str
) -> None:
    job_batch_information.to_csv(path_to_save_job_batch_information, index=False)


if __name__ == "__main__":
    all_paths_to_data = natsorted(
        glob(os.path.join(RAW_DATA_PATH, f"*{RAW_DATA_TYPE}.czi"))
    )
    logstr = "===== CREATION OF JOB BATCHES =====\n"

    all_paths_to_valid_image_tiles = []

    for path in all_paths_to_data:
        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"> processing: {path} at {date_time}\n"
        print(f"> processing path : {path}")
        data_id = "_".join(os.path.basename(path).split("_")[:2])

        output_directory_processed_deconvolutions = os.path.join(
            PROCESSED_DATA_PATH,
            data_id,
            f"tile_size_{TILE_SIZE}",
            "whole_slide",
            RAW_DATA_TYPE,
            "deconvolutions",
        )
        assert os.path.exists(output_directory_processed_deconvolutions)

        output_directory_processed_deconvolutions_scan_region_paths = natsorted(
            glob(os.path.join(output_directory_processed_deconvolutions, "ScanRegion*"))
        )

        for (
            output_directory_processed_deconvolutions_scan_region_path
        ) in output_directory_processed_deconvolutions_scan_region_paths:
            scan_region = os.path.basename(
                output_directory_processed_deconvolutions_scan_region_path
            )
            print(f"> processing scene: {scan_region}")

            output_directory_processed_deconvolutions_scan_region_psr = os.path.join(
                output_directory_processed_deconvolutions_scan_region_path,
                "psr",
                "inverted_grayscale",
            )
            assert os.path.exists(
                output_directory_processed_deconvolutions_scan_region_psr
            )

            output_directory_summary = os.path.join(
                output_directory_processed_deconvolutions_scan_region_psr,
                "valid_psr_images_summary/",
            )
            filename_summary = "summary_valid_image_tiles.csv"
            path_to_summary_valid_image_tiles = os.path.join(
                output_directory_summary, filename_summary
            )
            if os.path.exists(output_directory_summary):
                summary_valid_image_tiles = pd.read_csv(
                    path_to_summary_valid_image_tiles
                )
                paths_to_valid_image_tiles = summary_valid_image_tiles.loc[
                    summary_valid_image_tiles.valid == 1
                ].path_to_image_tile.values.tolist()

                all_paths_to_valid_image_tiles.extend(paths_to_valid_image_tiles)

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
            logstr += f"... scene : {scan_region} saved at {date_time}\n"

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += f"{int(all_paths_to_data.index(path)+1)} data paths processed (total: {len(all_paths_to_data)}); finished at {date_time}\n"
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""

    # set up output dir
    job_batch_output_subdir = os.path.join(
        JOB_BATCH_OUTPUT_DIR,
        f"raw_res_{RAW_DATA_RES}_tile_size_{TILE_SIZE}_batch_size_{JOB_BATCH_SIZE}_valid_image_tiles",
    )
    os.makedirs(
        job_batch_output_subdir,
        exist_ok=True,
    )
    os.chmod(
        job_batch_output_subdir,
        mode=0o777,
    )

    # save all paths to valid image tiles
    jobs_all_information = pd.DataFrame(
        {
            "path_to_image_tile": all_paths_to_valid_image_tiles,
            "batch_id": ["all" for _ in range(len(all_paths_to_valid_image_tiles))],
        }
    )
    write_job_batch(
        job_batch_information=jobs_all_information,
        path_to_save_job_batch_information=os.path.join(
            job_batch_output_subdir, "all_paths_to_valid_image_tiles.csv"
        ),
    )

    # save paths in a job batch
    job_batch_generator = generate_job_batch(
        all_jobs=all_paths_to_valid_image_tiles, job_batch_size=JOB_BATCH_SIZE
    )
    batch_id = 0
    while True:
        try:
            job_batch = next(job_batch_generator)
            batch_id_str = f"{str(batch_id).zfill(4)}"
            job_batch_information = pd.DataFrame(
                {
                    "path_to_image_tile": job_batch,
                    "batch_id": [batch_id_str for _ in range(len(job_batch))],
                }
            )
            write_job_batch(
                job_batch_information=job_batch_information,
                path_to_save_job_batch_information=os.path.join(
                    job_batch_output_subdir,
                    f"paths_to_valid_image_tiles_batch_size_{JOB_BATCH_SIZE}_batch_id_{batch_id_str}.csv",
                ),
            )
            batch_id += 1
        except StopIteration:
            break

    logfile = open(LOGFILE_PATH, "a")
    logfile.write("finshed writing all job batch information!")
    logfile.close()
    logstr = ""
