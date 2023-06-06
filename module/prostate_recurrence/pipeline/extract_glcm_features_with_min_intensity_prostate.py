"""# Script for batch-processing GLCM feature extraction in Prostate dataset

Customised arguments are to be defined using argparse library by the user when
calling this script.

Extraction of GLCM features is performed only at the tile level. For each tile,
only pixels with a minimum intensity level are included for GLCM feature analysis.

## get_path_to_valid_image_tile(...) to obtain the path to the corresponding tissue
masked deconvolved PSR image tile given the input argument indicating a folder
containing previously generated texture feature outputs.

## get_path_to_corresponding_tissue_mask_tile(...) to obtain the corresponding
path to the tissue mask tile of the provided tissue masked deconvolved PSR tile.

## extract_glcm_features_this_image(...) to perform GLCM feature extraction
of a single PSR tile. This function expects arguments reflecting the path to the
tissue masked deconvolved PSR tile, the path to its corresponding tissue mask tile,
and the path to the folder containing texture feature outputs.
Modules in glcm.py are called to extract grey level co-occurence matrix features.

"""

import argparse
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
from natsort import natsorted

parser = argparse.ArgumentParser(prog="tme-ml-pipeline-glcm-features")
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
    "--processed_data_path",
    dest="processed_data_path",
    action="store",
    type=str,
    default="",
    help="provide the path saving processed data.",
)


args = parser.parse_args()
LOGFILE_PATH = args.logfile_path
PROCESSED_DATA_PATH = args.processed_data_path

MODULE_PATH = args.module_path
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from prostate_recurrence.feature_engineering.Python import glcm


def get_path_to_valid_image_tile(path_to_image_tile_texture_feature_folder: str) -> str:
    """obtain the path to the tissue masked deconvolved PSR image tile

    Parameters
    ----------
    path_to_image_tile_texture_feature_folder : str
        The path to the folder containing previously processed texture feature outputs.

    Returns
    -------
    str
        The path to the tissue masked deconvolved PSR image tile
    """
    dirname_valid_image_tile = os.path.dirname(
        os.path.dirname(
            path_to_image_tile_texture_feature_folder.replace(
                "feature_engineering", "pre_processed_data"
            )
        )
    )
    basename_valid_image_tile = (
        os.path.basename(path_to_image_tile_texture_feature_folder) + ".tif"
    )
    path_to_valid_image_tile = os.path.join(
        dirname_valid_image_tile, basename_valid_image_tile
    )

    return path_to_valid_image_tile


def get_path_to_corresponding_tissue_mask_tile(path_to_valid_image_tile: str) -> str:
    """obtain the path to the corresponding tissue mask tile

    Parameters
    ----------
    path_to_valid_image_tile : str
        The path to the tissue masked deconvolved PSR image tile

    Returns
    -------
    str
        The path to the corresponding tissue mask tile
    """
    path_splited = path_to_valid_image_tile.split("/")
    slide_keyword_index = path_splited.index("slide_20X") + 1
    scene_keyword_index = path_splited.index("deconvolutions") + 1
    slide_id = path_splited[slide_keyword_index]
    scene_id = path_splited[scene_keyword_index]
    image_tile_name = os.path.basename(path_to_valid_image_tile).split("_psr.tif")[0]

    path_to_corresponding_tissue_mask_tile = os.path.join(
        "/".join(path_splited[:slide_keyword_index]),
        slide_id,
        "tile_size_2000/whole_slide/PSR",
        "tissue_masks/tissue_mask_v2",
        scene_id,
        f"{image_tile_name}.tif",
    )

    return path_to_corresponding_tissue_mask_tile


def extract_glcm_features_this_image(
    path_to_valid_image_tile: str,
    path_to_corresponding_tissue_mask_tile: str,
    path_to_image_tile_texture_feature_folder: str,
) -> None:
    """perform glcm feature extraction of a single image tile

    Parameters
    ----------
    path_to_valid_image_tile : str
        The path to the tissue masked deconvolved PSR image tile
    path_to_image_tile_texture_feature_folder : str
        The path to the folder containing previously processed texture feature outputs.
    """
    output_directory_processed_texture_features_this_image_tile = (
        path_to_image_tile_texture_feature_folder
    )

    # read image
    image_array = glcm.read_image(path_to_img=path_to_valid_image_tile)
    tissue_mask_array = glcm.read_image(
        path_to_img=path_to_corresponding_tissue_mask_tile
    )

    # texture - glcm
    if True:
        distances = [1, 2, 5, 11, 22, 45, 90, 182, 364]
        angles = [0, np.pi / 4.0, np.pi / 2.0, np.pi * 3 / 4.0]
        symmetric = True
        normed = True

        min_intensity_thresholds = [8, 16, 32, 64, 106, 128, 212]

        for analysis_type, tissue_mask in zip(["masked"], [tissue_mask_array]):

            matrix_glcm, masked_image_array = glcm.construct_glcm(
                image=image_array,
                mask=tissue_mask,
                distances=distances,
                angles=angles,
                symmetric=symmetric,
                normed=normed,
            )

            for min_intensity_threshold in min_intensity_thresholds:
                output_subdir = os.path.join(
                    output_directory_processed_texture_features_this_image_tile,
                    f"min_intensity_threshold_{min_intensity_threshold}",
                )
                os.makedirs(output_subdir, exist_ok=True)
                os.chmod(output_subdir, mode=0o777)

                start = min_intensity_threshold - 1
                matrix_glcm_small = matrix_glcm[start:, start:, :, :]

                glcm_features_output = glcm.extract_glcm_features(
                    matrix=matrix_glcm_small,
                    features=(
                        "contrast",
                        "dissimilarity",
                        "homogeneity",
                        "energy",
                        "correlation",
                        "ASM",
                    ),
                )

                glcm.save_glcm_features(
                    matrix=matrix_glcm,
                    glcm_features_output=glcm_features_output,
                    distances=distances,
                    angles=angles,
                    output_directory=output_subdir,
                    analysis_type=analysis_type,
                )


if __name__ == "__main__":

    paths_to_slides_folders = list(
        natsorted(glob(os.path.join(PROCESSED_DATA_PATH, "C*")))
    )
    print(
        "total number of image slides: ",
        len(paths_to_slides_folders),
    )
    print("... first few paths\n", paths_to_slides_folders[:3])

    logstr = "===== EXTRACTION OF GLCM FEATURES =====\n"
    for k, path_to_slide_folder in enumerate(paths_to_slides_folders):

        paths_to_scenes_folders = natsorted(
            glob(
                os.path.join(
                    path_to_slide_folder,
                    "tile_size_2000/whole_slide/PSR/deconvolutions/",
                    "ScanRegion*",
                )
            )
        )

        if not paths_to_scenes_folders:
            continue

        for path_to_scene_folder in paths_to_scenes_folders:

            paths_to_image_tile_texture_feature_folders = natsorted(
                glob(
                    os.path.join(
                        path_to_scene_folder,
                        "psr/inverted_grayscale_tissue_masked/tile_level_features_texture_v2/",
                        "image_tile*",
                    )
                )
            )

            if not paths_to_image_tile_texture_feature_folders:
                continue

            for (
                path_to_image_tile_texture_feature_folder
            ) in paths_to_image_tile_texture_feature_folders:
                path_to_valid_image_tile = get_path_to_valid_image_tile(
                    path_to_image_tile_texture_feature_folder=path_to_image_tile_texture_feature_folder
                )
                path_to_corresponding_tissue_mask_tile = (
                    get_path_to_corresponding_tissue_mask_tile(
                        path_to_valid_image_tile=path_to_valid_image_tile
                    )
                )

                if not (
                    os.path.exists(path_to_valid_image_tile)
                    and os.path.exists(path_to_corresponding_tissue_mask_tile)
                ):
                    continue

                extract_glcm_features_this_image(
                    path_to_valid_image_tile=path_to_valid_image_tile,
                    path_to_corresponding_tissue_mask_tile=path_to_corresponding_tissue_mask_tile,
                    path_to_image_tile_texture_feature_folder=path_to_image_tile_texture_feature_folder,
                )

        now = datetime.now()
        date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
        logstr += (
            f"{k+1} / {len(paths_to_slides_folders)} slides\n"
            f"... slide {path_to_slide_folder}; finished at {date_time}\n"
        )
        logstr += "\n"
        logfile = open(LOGFILE_PATH, "a")
        logfile.write(logstr)
        logfile.close()
        logstr = ""
