from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsimageio import AICSImage  # pip install AICSImage[czi]
from shapely.geometry import Polygon


def extract_tissue_contour(
    path_to_img: str,
) -> Dict:
    img = AICSImage(path_to_img)
    metad = img.metadata

    text_contours = []
    text_contour_sizes = []
    text_contour_center_positions = []

    for elem in metad.iter():
        if (
            elem.tag == "CenterPosition"
            and elem.text not in text_contour_center_positions
        ):
            text_contour_center_positions.append(elem.text)
        if elem.tag == "ContourSize" and elem.text not in text_contour_sizes:
            text_contour_sizes.append(elem.text)
        if elem.tag == "Points" and elem.text not in text_contours:
            text_contours.append(elem.text)

    contours_extracted = {}
    for idx, text_contour_size in enumerate(text_contour_sizes):
        contour_size = [float(size) for size in text_contour_size.split(",")]
        # print(contour_size)

        text_center_position = text_contour_center_positions[idx]
        contour_center_position = [
            float(pos) for pos in text_center_position.split(",")
        ]
        # print(contour_center_position)

        text_contour = text_contours[idx]
        contour = [
            [float(coord_str.split(",")[0]), float(coord_str.split(",")[1])]
            for coord_str in text_contour.split(" ")
        ]

        contours_extracted[idx] = {
            "contour_center_position": contour_center_position,
            "contour_size": contour_size,
            "contour": contour,
        }

    return contours_extracted


def dataframe_contours(
    contours_extracted: Dict, path_to_save_dataframe: str = ""
) -> pd.DataFrame:

    df_contour_center_positions = pd.DataFrame()
    df_contour_sizes = pd.DataFrame()
    df_contours = pd.DataFrame()

    for idx in sorted(contours_extracted.keys()):
        contour_center_position = np.array(
            contours_extracted[idx]["contour_center_position"]
        )
        contour_size = np.array(contours_extracted[idx]["contour_size"])
        contour = np.array(contours_extracted[idx]["contour"])

        df_contour_center_position = pd.DataFrame(
            data=[contour_center_position], columns=["X", "Y"]
        )
        df_contour_center_position["Scene"] = f"ScanRegion{idx}"

        df_contour_size = pd.DataFrame(data=[contour_size], columns=["Xsize", "Ysize"])
        df_contour_size["Scene"] = f"ScanRegion{idx}"

        df_contour = pd.DataFrame(data=contour, columns=["X", "Y"])
        df_contour["Scene"] = f"ScanRegion{idx}"

        pgon = Polygon(df_contour[["X", "Y"]].values)
        area, perimeter = pgon.area, pgon.length
        df_contour_size["Area"] = area
        df_contour_size["Perimeter"] = perimeter

        df_contours = df_contours.append(df_contour)
        df_contour_sizes = df_contour_sizes.append(df_contour_size)
        df_contour_center_positions = df_contour_center_positions.append(
            df_contour_center_position
        )

    if path_to_save_dataframe:
        df_contour_sizes.to_csv(
            path_to_save_dataframe + "_contour_sizes.csv", index=False
        )
        df_contour_center_positions.to_csv(
            path_to_save_dataframe + "_contour_center_positions.csv", index=False
        )
        df_contours.to_csv(path_to_save_dataframe + "_contours.csv", index=False)

    return df_contour_center_positions, df_contour_sizes, df_contours


def scatterplot_contours(contours_extracted: Dict, path_to_save_plot: str = ""):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 4), dpi=300)
    for idx in sorted(contours_extracted.keys()):
        contour = np.array(contours_extracted[idx]["contour"])

        axes.scatter(
            contour[:, 0],
            contour[:, 1],
            s=2,
            edgecolor="none",
            label=f"ScanRegion{idx}",
        )

    axes.set_ylim(
        [
            axes.get_ylim()[1],
            axes.get_ylim()[0],
        ]
    )
    axes.set_aspect("equal")
    plt.legend(loc="best")

    if path_to_save_plot:
        plt.savefig(path_to_save_plot, dpi=300)
    else:
        plt.show()
