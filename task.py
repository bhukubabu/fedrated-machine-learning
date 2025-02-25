"""flower_tensorflow: A Flower / TensorFlow app for federated learning"""

import os
#import ray
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from itertools import chain
from tensorflow import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers

# Make TensorFlow log less verbose
#os.environ["RAY_DEDUP_LOGS"]="0"


def load_model(learning_rate: float = 0.002):
    model = keras.Sequential(
        [
            keras.Input(shape=(96,96,3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu",padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            #layers.Dense(256,activation='relu'),
            #layers.Dense(128,activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset

def load_data(partition_id, num_partition):
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partition) #num_partition specifies the number of client
        fds = FederatedDataset(
            dataset="FastJobs/Visual_Emotional_Analysis",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train") #partition_id specifies a unique id for selected client
    partition.set_format("numpy")

    # Divide data on each node
    partition = partition.train_test_split(test_size=0.25)
    partition=partition.shuffle(seed=42)
    x_train, y_train = partition["train"]["image"]/255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["image"]/255.0, partition["test"]["label"]

    x_train=x_train.astype("float32")/255.0
    x_test=x_test.astype("float32")/255.0

    return x_train, y_train, x_test, y_test
