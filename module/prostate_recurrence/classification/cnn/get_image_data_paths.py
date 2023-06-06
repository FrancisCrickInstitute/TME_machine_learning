import itertools
import os

# import sys
from glob import glob

# from typing import Dict, List

# import numpy as np
import pandas as pd
from natsort import natsorted

from typing import Dict


def get_data_paths_prostate_based_on_feature_df(
    feature_df: pd.DataFrame,
    processed_data_directory: str
) -> Dict[int, str]:

    list_of_paths_to_image_tiles = []
    for slide_id, scene, tile in feature_df[
        ['slide_id', 'scene', 'tile']
    ].values:
        path_to_image_tile = os.path.join(
            processed_data_directory,
            slide_id,
            "tile_size_2000/whole_slide/PSR/deconvolutions",
            scene,
            "psr/inverted_grayscale_tissue_masked",
            f"image_tile_{tile}_psr.tif"
        )
        list_of_paths_to_image_tiles.append(path_to_image_tile)

    # convert data frame into dictionary
    dict_image_paths = {
        ID: image_path
        for ID, image_path in enumerate(
            list_of_paths_to_image_tiles
        )
    }
    print(f"... consisting of {len(dict_image_paths)} valid image tiles in total")

    return dict_image_paths


def get_data_paths_prostate(
    main_data_directory: str, slide_id_pattern: str, tile_size: int = 2000
) -> Dict[int, str]:

    paths_to_slides = natsorted(
        glob(os.path.join(main_data_directory, slide_id_pattern))
    )
    slide_ids = [os.path.basename(path) for path in paths_to_slides]

    print(f"found {len(slide_ids)} slides with name pattern {slide_id_pattern}.")

    # get paths to scenes
    paths_to_scenes = list(
        itertools.chain.from_iterable(
            natsorted(
                glob(
                    os.path.join(
                        path_to_slide,
                        f"tile_size_{tile_size}/whole_slide/PSR/deconvolutions",
                        "ScanRegion*",
                    )
                )
            )
            for path_to_slide in paths_to_slides
        )
    )
    print(f"... consisting of {len(paths_to_scenes)} scenes in total")

    # find out valid jobs in each slide / scene
    combined_summary_valid_tiles = pd.DataFrame()
    for path_to_scene in paths_to_scenes:

        path_to_summary_valid_tiles = os.path.join(
            path_to_scene,
            "psr/inverted_grayscale/valid_psr_images_summary",
            "summary_valid_image_tiles.csv",
        )

        summary_valid_tiles = pd.read_csv(path_to_summary_valid_tiles)

        combined_summary_valid_tiles = pd.concat(
            [
                combined_summary_valid_tiles,
                summary_valid_tiles.loc[summary_valid_tiles.valid == 1].copy(),
            ],
            ignore_index=True,
        )

    # convert data frame into dictionary
    dict_image_paths = {
        ID: image_path
        for ID, image_path in enumerate(
            combined_summary_valid_tiles.path_to_image_tile.values.tolist()
        )
    }
    print(f"... consisting of {len(dict_image_paths)} valid image tiles in total")

    return dict_image_paths


def get_data_paths_lung(
    main_data_directory: str, slide_id_pattern: str, tile_size: int = 2000
) -> Dict[int, str]:

    paths_to_slides = natsorted(
        glob(os.path.join(main_data_directory, slide_id_pattern))
    )
    slide_ids = [os.path.basename(path) for path in paths_to_slides]

    print(f"found {len(slide_ids)} slides with name pattern {slide_id_pattern}.")

    # get paths to scenes
    paths_to_scenes = list(
        itertools.chain.from_iterable(
            natsorted(
                glob(
                    os.path.join(
                        path_to_slide,
                        f"tile_size_{tile_size}/whole_slide/PSR/deconvolutions",
                        "ScanRegion*",
                    )
                )
            )
            for path_to_slide in paths_to_slides
        )
    )
    print(f"... consisting of {len(paths_to_scenes)} scenes in total")

    # find out paths to image tiles
    dataset_summary_cols = ["path_to_scene", "n_image_tiles"]
    dataset_summary_rows = []

    for path_to_scene in paths_to_scenes:
        print(f"> processing scene : {path_to_scene}")

        paths_to_image_tiles = natsorted(
            glob(
                os.path.join(
                    path_to_scene,
                    "psr/inverted_grayscale/",
                    "image_tile*",
                )
            )
        )

        print(f"... found {len(list(paths_to_image_tiles))} image tiles.")
        dataset_summary_rows.append([path_to_scene, len(list(paths_to_image_tiles))])

    dataset_summary = pd.DataFrame(
        data=dataset_summary_rows, columns=dataset_summary_cols
    )

    return dataset_summary
