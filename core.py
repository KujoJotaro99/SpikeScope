import torch
import json
import matplotlib.pyplot as plt
import os

def count_spikes(model, data_loader, num_steps, device=None):
    """
    Count the total number of spikes emitted by the model across the dataset.

    Args:
        model (nn.Module): Spiking neural network.
        data_loader (DataLoader): PyTorch DataLoader.
        num_steps (int): Number of time steps per sample.
        device (torch.device): CPU or CUDA device.

    Returns:
        total_spikes (int): Total number of spikes over all neurons and timesteps.
        total_samples (int): Number of samples processed.
        spike_counts (dict): Spike tensors per layer.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #move model to available device and puts in evaluation mode
    model.to(device)
    model.eval()

    total_spikes = 0
    total_samples = 0
    spike_counts = {}

    #running in inference mode
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = list(model(data, num_steps))
            batch_spike_total = 0

            for i, spikes in enumerate(output):
                layer_key = f"layer_{i}"
                if layer_key not in spike_counts:
                    spike_counts[layer_key] = spikes.sum()
                else:
                    spike_counts[layer_key] += spikes.sum()
                batch_spike_total += spikes.sum().item()

            total_spikes += batch_spike_total
            total_samples += data.size(0)

            return total_spikes, total_samples, spike_counts

    return

