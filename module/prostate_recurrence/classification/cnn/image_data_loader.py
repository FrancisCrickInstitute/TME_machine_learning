"""#MyDataGenerator Class

"""


import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, List, Tuple


class MyDataGenerator(tf.keras.utils.Sequence):
    """Generate a Subclass of Sequence class

    Returns
    -------
    tf.keras.utils.Sequence
        A Sequence object that can be used to get a minibatch of image
        data in Numpy array format and their labels.
    """

    def __init__(
        self,
        dict_image_paths: Dict[int, str],
        labels: List[int],
        batch_size: int = 32,
        dim: Tuple[int, int] = (2000, 2000),
        n_channels: int = 3,
        n_classes: int = 2,
        shuffle: bool = True,
        return_image_paths: bool = False,
    ):
        """Initialisation of the object's attributes

        Parameters
        ----------
        dict_image_paths : Dict[int, str]
            Dictionary of key:value pairs reflecting the ID of an image tile
            and the path to the image tile.
        labels : List[int]
            List of labels, with indices corresponding to the IDs of image
            tiles.
        batch_size : int, optional
            The number of image tiles per minibatch, by default 32
        dim : Tuple[int, int], optional
            Dimensions of an image tile, by default (2000, 2000)
        n_channels : int, optional
            Number of colour channels, by default 3
        n_classes : int, optional
            Number of classes encoded by the label, by default 2
        shuffle : bool, optional
            A boolean variable indicating whether to shuffle the data, by
            default True
        return_image_paths : bool, optional
            A boolean variable indicating whether to return the paths corresponding
            to the minibatch of image data, by default False
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.dict_image_paths = dict_image_paths
        self.list_IDs = list(dict_image_paths.keys())
        self.list_image_paths = list(dict_image_paths.values())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.return_image_paths = return_image_paths
        self.on_epoch_end()

    def __len__(self) -> int:
        """get the length of the Sequence object

        The length of the Sequence object reflects the number of minibatches.

        Returns
        -------
        int
            The length of the Sequence object
        """
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index: int) -> Tuple:
        """get an item from the Sequence object

        Parameters
        ----------
        index : int
            The index in the Sequence

        Returns
        -------
        Tuple
            A minibatch of image tiles in a Numpy array format with corresponding
            labels, by default, if the boolean variable "self.return_image_paths"
            is False. Otherwise, image tiles, labels, and paths are returned.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, image_paths = self.__data_generation(list_IDs_temp)

        if self.return_image_paths:
            return X, y, image_paths
        else:
            return X, y

    def on_epoch_end(self) -> None:
        """Shuffle data at the end of an epoch

        Data shuffling is performed if the boolean variable "self.shuffle" is True.

        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp: List[int]) -> Tuple:
        """read image data for the minibatch

        Parameters
        ----------
        list_IDs_temp : List[int]
            List of IDs of image tiles, for which corresponding image data will be read
            as Numpy arrays.

        Returns
        -------
        Tuple
            A minibatch of image tiles in Numpy array, their labels, and the paths to the
            image tiles, by default, if labels are provided.
            Otherwise, image arrays themselves are returned as labels, assuming that the
            model is an Encoder-Decoder configuration.
        """
        "Generates data containing batch_size samples"
        # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        image_paths = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = np.array(Image.open(self.dict_image_paths[ID]))
            # Normalise
            img = img / 255.0
            if img.ndim == 2:
                img = img[..., np.newaxis]
            X[
                i,
            ] = img
            image_paths.append(self.dict_image_paths[ID])

            if self.labels:
                # Store class
                y[i] = self.labels[ID]

        if self.labels:
            return (
                X,
                tf.keras.utils.to_categorical(y, num_classes=self.n_classes),
                image_paths,
            )

        else:
            return X, X, image_paths
