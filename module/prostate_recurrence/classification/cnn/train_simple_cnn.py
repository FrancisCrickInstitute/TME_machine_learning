"""# Script to train a simple cnn model

This script is work in progress.

This script calls modules in get_image_data_paths.py and image_data_loader.py to
construct a MyDataGenerator (subclass of Sequence) object for effeciently loading
image tiles from the folder tree in the Prostate dataset.

Refactoring of the code into a seperate simple_cnn_model.py is to be implemented.

"""


import argparse
import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser(prog="tme-ml-cnn")
parser.add_argument(
    "--module_path",
    dest="module_path",
    action="store",
    type=str,
    default="../../../module/",
    help="module path to pipeline functions.",
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
    help="provide the path containing processed data.",
)
parser.add_argument(
    "--path_to_feature_df",
    dest="path_to_feature_df",
    action="store",
    type=str,
    default="",
    help="provide the path to matrix feature dataframe.",
)

args = parser.parse_args()
RAW_DATA_PATH = args.raw_data_path
PROCESSED_DATA_PATH = args.processed_data_path
MODULE_PATH = args.module_path
PATH_TO_FEATURE_DF = args.path_to_feature_df
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import get_image_data_paths
import image_data_loader

if __name__ == "__main__":
    # ===== read feature dataframe =====
    feature_df = pd.read_csv(PATH_TO_FEATURE_DF)
    # for testing purpose, focus on tiles with high tissue proportion and high tumour proportion
    feature_df_filtered = feature_df.loc[
        (feature_df.tissue_proportion >= 0.5) & (feature_df.tumour_proportion >= 0.5)
    ].copy()

    # ===== train/test split =====
    # for testing purpose, read in cross_validation_set_1
    dirname_path_to_feature_df = os.path.dirname(PATH_TO_FEATURE_DF)
    path_to_cv_set_1 = os.path.join(
        dirname_path_to_feature_df, "cross_validation_set_1.csv"
    )
    cv_set_1 = pd.read_csv(path_to_cv_set_1)
    training_set_patient_ids = cv_set_1.training_set_patient_id.unique()
    test_set_patient_ids = [
        pid for pid in cv_set_1.test_set_patient_id.unique() if type(pid) == str
    ]
    print(len(training_set_patient_ids), len(test_set_patient_ids))

    # split the feature dataframe into train/test
    feature_df_train = feature_df_filtered.loc[
        feature_df_filtered.patient_id.isin(training_set_patient_ids)
    ].copy()
    labels_train = feature_df_train.case.tolist()
    print(len(labels_train))
    feature_df_test = feature_df_filtered.loc[
        feature_df_filtered.patient_id.isin(test_set_patient_ids)
    ].copy()
    labels_test = feature_df_test.case.tolist()
    print(len(labels_test))

    # ===== fetch relevant image paths for train & test =====
    dict_image_paths_train = (
        get_image_data_paths.get_data_paths_prostate_based_on_feature_df(
            feature_df=feature_df_train, processed_data_directory=PROCESSED_DATA_PATH
        )
    )
    dict_image_paths_test = (
        get_image_data_paths.get_data_paths_prostate_based_on_feature_df(
            feature_df=feature_df_test, processed_data_directory=PROCESSED_DATA_PATH
        )
    )
    print(len(dict_image_paths_train), len(dict_image_paths_test))

    # ===== construct MyDataGenerator objects for train & test =====
    TILE_SIZE = 2000
    BATCH_SIZE = 4  # 32 causes oom error
    params = {
        "dim": (TILE_SIZE, TILE_SIZE),
        "batch_size": BATCH_SIZE,
        "n_classes": 2,
        "n_channels": 1,
        "shuffle": True,
        "return_image_paths": False,
    }
    my_data_generator_train = image_data_loader.MyDataGenerator(
        dict_image_paths=dict_image_paths_train, labels=labels_train, **params
    )
    my_data_generator_test = image_data_loader.MyDataGenerator(
        dict_image_paths=dict_image_paths_test, labels=labels_test, **params
    )

    # ===== train a simple CNN model =====
    # work in progress!
    # this needs to be re-factored into a simple_cnn_model.py module
    if True:
        # create a simple CNN model
        model = models.Sequential()

        model.add(
            layer=layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                input_shape=(2000, 2000, 1),
            )
        )
        model.add(layer=layers.MaxPool2D(pool_size=(4, 4)))  # 500 x 500 x 32
        model.add(
            layer=layers.Conv2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            )
        )
        model.add(layer=layers.MaxPool2D(pool_size=(4, 4)))  # 125 x 125 x 64
        model.add(
            layer=layers.Conv2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            )
        )
        model.add(layer=layers.MaxPool2D(pool_size=(5, 5)))  # 25 x 25 x 64
        model.add(
            layer=layers.Conv2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            )
        )

        model.add(layer=layers.Flatten())
        model.add(layer=layers.Dense(units=32, activation="relu"))
        model.add(layer=layers.Dense(units=1, activation="sigmoid"))
        print(model.summary())

        # compile the model
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
            metrics="accuracy",
        )

        # fit the model, with data generators as input
        history = model.fit(
            x=my_data_generator_train,
            validation_data=my_data_generator_test,
            epochs=50,
            workers=4,
            #     use_multiprocessing=True
        )

        # save the training summary
        summary = pd.DataFrame(history.history)
        print(summary)
        summary.to_csv("./summary.csv")
