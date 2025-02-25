"""flower_tensorflow: A Flower / TensorFlow app."""

import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower_tensorflow.task import load_data, load_model


def activate_gpu():
    devices=tf.config.list_physical_devices('GPU')
    if devices:
        print(f"Utilizing {len(devices)} GPU's")
        try:
            for gpu in devices:
                tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError as e:
            print(f"Error in setting memory growth: {e}")
    else:
        print("No GPUs found. Using CPU.")


# Defining the Flower Client that is derrived from the abstract base class NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self, learning_rate, data, epochs, batch_size,verbose):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        print("Received parameter shapes:", [p.shape for p in parameters])
        print("Expected model weights shapes:", [w.shape for w in self.model.get_weights()])
        self.model.set_weights(parameters)
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else ('/CPU:0')):
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """This function will evaluate the model on the data this client has."""
        print(parameters)
        self.model.set_weights(parameters)
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else ('/CPU:0')):
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"test accuracy : {accuracy} and loss : {loss}")
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"] 
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose",True)
    learning_rate = context.run_config["learning-rate"]

    activate_gpu()
    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
