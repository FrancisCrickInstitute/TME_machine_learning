import numpy as np
import tensorflow as tf
from PIL import Image


class MyDataGenerator(tf.keras.utils.Sequence):
    "Generates data for memory-heavy dataset"

    def __init__(
        self,
        dict_image_paths,
        labels,
        batch_size=32,
        dim=(2000, 2000),
        n_channels=3,
        n_classes=2,
        shuffle=True,
        return_image_paths=True
    ):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.dict_image_paths = dict_image_paths
        self.list_IDs = list(dict_image_paths.keys())
        self.list_image_paths = list(dict_image_paths.values())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.return_image_paths=return_image_paths
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
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

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
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
            img = img / 255.
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
