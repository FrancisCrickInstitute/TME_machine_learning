import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import kurtosis
from typing import Tuple, List, Dict


def calculate_coarseness(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """calculate coarseness

    Parameters
    ----------
    image : np.ndarray
        An input gray scale image as numpy array

    Returns
    -------
    Tuple[np.ndarray, float]
        A numpy array of coarseness values at different positions within
        the input image and the mean coarseness value over the image.
    """

    def calculate_A(k: int) -> np.ndarray:
        """calculate the mean pixel intensity over a square neighbourhood
        of a size of 2^k, over the image.

        Parameters
        ----------
        k : int
            Log2 the size of a square neighbourhood.

        Returns
        -------
        np.ndarray
            A numpy array recording the mean pixel intensity over a square
            neighbourhood of a size of 2^k.
        """
        A = np.full_like(image, fill_value=np.nan)
        nh_size_half = int(2 ** (k - 1))

        low = nh_size_half
        high = A.shape[1] - nh_size_half

        for row in np.arange(low, high):
            for col in np.arange(low, high):
                A[row, col] = (
                    np.sum(
                        image[
                            row - nh_size_half : row + nh_size_half,
                            col - nh_size_half : col + nh_size_half,
                        ]
                    )
                    / nh_size_half
                    / nh_size_half
                    / 4
                )

        return A

    def calculate_E(k: int, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """calculate the difference between adjacent square neighbourhoods,
        in the mean pixel intensity over a square neighbourhood of a size of 2^k,
        in both horizontal and vertical directions.

        Parameters
        ----------
        k : int
            Log2 the size of a square neighbourhood.
        A : np.ndarray
            A numpy array recording the mean pixel intensity over a square
            neighbourhood of a size of 2^k.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of numpy arrays recording the difference between adjacent
            square neighbourhoods, in the mean pixel intensity over a square
            neighbourhood of a size of 2^k, in both horizontal and vertical
            directions, respectively.
        """
        E_h = np.full_like(image, fill_value=np.nan)
        E_v = np.full_like(image, fill_value=np.nan)
        nh_size_half = int(2 ** (k - 1))

        low = nh_size_half * 2
        high = E_h.shape[1] - nh_size_half * 2

        for row in np.arange(low, high):
            for col in np.arange(low, high):
                E_h[row, col] = np.abs(
                    A[row, col + nh_size_half] - A[row, col - nh_size_half]
                )
                E_v[row, col] = np.abs(
                    A[row + nh_size_half, col] - A[row - nh_size_half, col]
                )

        return (E_h, E_v)

    assert image.ndim == 2 and image.shape[0] == image.shape[1]

    K_MAX = int(np.floor(np.log2(image.shape[0])))

    E_h_all_k = np.zeros(((K_MAX, image.shape[0], image.shape[1])))
    E_v_all_k = np.zeros(((K_MAX, image.shape[0], image.shape[1])))
    for k in np.arange(1, K_MAX + 1):
        print(f"> calculation for k = {k}, window size = {int(2**k)}")
        A_k = calculate_A(k)
        (E_h_all_k[k - 1], E_v_all_k[k - 1]) = calculate_E(k, A_k)

    S = np.full_like(image, fill_value=np.nan)
    for row in np.arange(image.shape[0]):
        for col in np.arange(image.shape[1]):
            if np.isnan(E_h_all_k[:, row, col]).any():
                continue
            k_h_opt, val_h = (
                E_h_all_k[:, row, col].argmax() + 1,
                E_h_all_k[:, row, col].max(),
            )
            k_v_opt, val_v = (
                E_v_all_k[:, row, col].argmax() + 1,
                E_v_all_k[:, row, col].max(),
            )

            k_opt = k_h_opt if val_h > val_v else k_v_opt

            S[row, col] = 2 ** k_opt

    Coarseness = np.nanmean(S)

    return (S, Coarseness)


def calculate_contrast(image: np.ndarray):
    assert image.ndim == 2 and image.shape[0] == image.shape[1]

    image_flattened = image.flatten()
    image_flattened_notnan = image_flattened[~np.isnan(image_flattened)]

    kur = kurtosis(image_flattened_notnan)
    std = np.std(image_flattened_notnan)

    Contrast = std / np.power(kur, 1 / 4)

    print("std:", std, "kurtosis:", kur, "Contrast: ", Contrast)

    return Contrast