import pandas as pd
from aicsimageio import AICSImage


def extract_scene_information(
    path_to_image: str, tile_size: int = 2000
) -> pd.DataFrame:
    scene_information_rows = []
    scene_information_cols = [
        "path_to_image",
        "scene",
        "image_height",
        "image_width",
        "ncols_inner",
        "ncols_outer",
        "nrows_inner",
        "nrows_outer",
    ]

    image = AICSImage(path_to_image)
    for scene in image.scenes:
        image.set_scene(scene)
        image_width = image.dims.X
        image_height = image.dims.Y
        ncols_inner = image_width // tile_size
        ncols_outer = ncols_inner + 1
        nrows_inner = image_height // tile_size
        nrows_outer = nrows_inner + 1

        scene_information_rows.append(
            [
                path_to_image,
                scene,
                image_height,
                image_width,
                ncols_inner,
                ncols_outer,
                nrows_inner,
                nrows_outer,
            ]
        )

    scene_information = pd.DataFrame(
        data=scene_information_rows, columns=scene_information_cols
    )

    return scene_information
