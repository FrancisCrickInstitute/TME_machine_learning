"""#A set of functions to get the paths to PSR image tiles

## get_data_paths_prostate_based_on_feature_df(...) to obtain paths to PSR
image tiles in Prostate dataset according to the input feature dataframe, which
contains information about (a subset of) image tiles, including the slide id,
scene name, and image tile name. This method has advantage/convenience as the
input feature dataframe can come from data split for other classification methods,
thereby permitting concordant data splits across methods.

## get_data_paths_prostate(...) to obtain paths to PSR image tiles in Prostate
dataset by iterating over all slide ids, scenes, and image tiles. This function
needs to be updated or to be deprecated.

## get_data_paths_lung(...) to obtain paths to PSR image tiles in Lung
dataset by iterating over all slide ids, scenes, and image tiles. This function
needs to be updated or to be deprecated.

"""

import itertools
import os
from glob import glob
from typing import Dict

import pandas as pd
from natsort import natsorted


def get_data_paths_prostate_based_on_feature_df(
    feature_df: pd.DataFrame, processed_data_directory: str
) -> Dict[int, str]:
    """obtain paths to PSR image tiles in Prostate dataset

    This function finds out paths to PSR image tiles by iterating over the
    provided feature dataframe, which contains information about (a subset of)
    image tiles, including the slide id, scene name, and image tile name.

    Spliting of dataset is expected to take place when the feature dataframe
    is constructed.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature dataframe containing information about slide id, scene name,
        and image tile name.
    processed_data_directory : str
        Directory that contains the processed data, where the tissue masked
        deconvolved image tiles can be located.

    Returns
    -------
    Dict[int, str]
        A dictionary of key:value pairs reflecting the id of a tissue masked
        deconvolved image tile and the path to the image tile.
    """

    list_of_paths_to_image_tiles = []
    for slide_id, scene, tile in feature_df[["slide_id", "scene", "tile"]].values:
        path_to_image_tile = os.path.join(
            processed_data_directory,
            slide_id,
            "tile_size_2000/whole_slide/PSR/deconvolutions",
            scene,
            "psr/inverted_grayscale_tissue_masked",
            f"image_tile_{tile}_psr.tif",
        )
        list_of_paths_to_image_tiles.append(path_to_image_tile)

    # convert data frame into dictionary
    dict_image_paths = {
        ID: image_path for ID, image_path in enumerate(list_of_paths_to_image_tiles)
    }
    print(f"... consisting of {len(dict_image_paths)} valid image tiles in total")

    return dict_image_paths


def get_data_paths_prostate(
    main_data_directory: str, slide_id_pattern: str, tile_size: int = 2000
) -> Dict[int, str]:
    """obtain paths to PSR image tiles in Prostate dataset

    This function needs to be updated OR to be deprecated.

    This function finds out paths to PSR image tiles by iterating over all
    whole slides, scenes, and image tiles.

    Note that the train/test split of dataset needs to be taken care of
    separately from this function.

    Parameters
    ----------
    main_data_directory : str
        Directory that contains raw image data in .czi format.
    slide_id_pattern : str
        Naming pattern of slide ids.
    tile_size : int, optional
        The width of an image tile, by default 2000

    Returns
    -------
    Dict[int, str]
        A dictionary of key:value pairs reflecting the id of a tissue masked
        deconvolved image tile and the path to the image tile.
    """

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
    """obtain paths to PSR image tiles in Lung dataset

    This function needs to be updated OR to be deprecated.

    This function finds out paths to PSR image tiles by iterating over all
    whole slides, scenes, and image tiles.

    Note that the train/test split of dataset needs to be taken care of
    separately from this function.

    Parameters
    ----------
    main_data_directory : str
        Directory that contains raw image data in .czi format.
    slide_id_pattern : str
        Naming pattern of slide ids.
    tile_size : int, optional
        The width of an image tile, by default 2000

    Returns
    -------
    Dict[int, str]
        A dictionary of key:value pairs reflecting the id of a tissue masked
        deconvolved image tile and the path to the image tile.
    """

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

    # the code appears to be incomplete. returned variable is strange, too.
    # similar functionalities to
    # get_data_paths_prostate(...) above need to be implemented.

    return dataset_summary
