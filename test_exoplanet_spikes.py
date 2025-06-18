import torch
import pandas as pd
import numpy as np
import os
from models import get_model
from core import count_spikes, get_model_stats, calculate_memory_traffic
from config import MODEL_CONFIG

def load_test_data(csv_path, batch_size=None):
    """Load and batch test data from CSV file."""
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
        
    df = pd.read_csv(csv_path)
    
    #label (1=no planet, 2=planet) so convert to 0/1 by subtracting 1
    labels = df.iloc[:, 0].values - 1
    features = df.iloc[:, 1:].values.astype(np.float32) #pytroch uses float32 by default
    
    X = torch.tensor(features)
    y = torch.tensor(labels)
    
    batches = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        batches.append((batch_X, batch_y))
    
    return batches



if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "data", "exoTest.csv")
    data_batches = load_test_data(csv_path)
    model = get_model("exoplanet_snn", return_spikes=True)

    print("\n1. Get Model Stats")

    total_neurons, total_weights, layer_info = get_model_stats(model)
    print(f"Total neurons: {total_neurons}")
    print(f"Total weights: {total_weights}")
    print(f"Layer info: {layer_info}")

    print("\n2. Count Spikes:")

    total_spikes, total_samples, spike_counts = count_spikes(model, data_batches)
    print(f"Total spikes: {total_spikes}")
    print(f"Total samples: {total_samples}")
    print(f"Average spikes per sample: {total_spikes / total_samples:.2f}")
    
    print("\n3. Calculate Memory Traffic:")
    results = calculate_memory_traffic(model, data_batches, verbose=True, hardware_config = "loihi2")
    # print(f"Calculation results {results}")