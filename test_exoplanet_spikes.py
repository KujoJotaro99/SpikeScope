import torch
import pandas as pd
import numpy as np
import os
from models import get_model
from core import count_spikes

def load_exoplanet_test_data(csv_path, batch_size=64):
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
    batch_size = 64

    batches = load_exoplanet_test_data(csv_path, batch_size)

    model = get_model("exoplanet_snn", return_spikes=True)

    total_spikes, total_samples, spike_counts = count_spikes(model, batches)
    print(f"Total spikes: {total_spikes}")
    print(f"Total samples: {total_samples}")
    print(f"Average spikes per sample: {total_spikes / total_samples:.2f}")

    for layer, spikes in spike_counts.items():
        avg = spikes / total_samples
        print(f"{layer}: {avg:.2f} spikes/sample")