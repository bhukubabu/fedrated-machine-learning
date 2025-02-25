"""flower_tensorflow: A Flower / TensorFlow app for federated learning"""

import json
import flwr as fl
from typing import List, Tuple
import tensorflow as tf
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flower_tensorflow.task import load_model


k=0
############ Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    ########################## Multiply accuracy of each client by number of examples used
    global k
    print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    with open("flower_tensorflow/result.json",'r+') as f:
        data=json.load(f)
        k+=1
        data["final_result"].append(
            {
                "round": k,
                "accuracy" : sum(accuracies) / sum(examples)
            }
        )
        f.seek(0)
        json.dump(data,f)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context:Context):
    """Construct components that set the ServerApp behaviour."""
    
    parameters = ndarrays_to_parameters(load_model().get_weights())
    print(f"Server is running on GPU: {tf.config.list_physical_devices('GPU')}")
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        inplace=True
    )
 
    num_rounds = 3
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

#config_, strategy=server_fn()

#fl.server.start_server(
 #   server_address='0.0.0.0:8080',
  #  server=ServerAppComponents(strategy=strategy, config=config_),
   # config=config_,
    #strategy=strategy,
#)
# Create ServerApp
app = ServerApp(server_fn=server_fn)


