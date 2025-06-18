import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
from config import MODEL_PATHS

beta = 0.5
spike_grad = surrogate.fast_sigmoid(slope=25)
batch_size = 64

class ExoplanetSNN(nn.Module):
    def __init__(self, return_spikes=False):
        super().__init__()

        # Initialize layers (3 linear layers and 3 leaky layers)
        self.fc1 = nn.Linear(3197, 128) # takes an input of 3197 and outputs 128
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 64) # takes an input of 64 and outputs 68
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(32, 2) # takes in 32 inputs and outputs our two outputs (planet with/without an exoplanet)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.softmax = nn.Softmax(dim=1) # softmax applied with a dimension of 1
        self.return_spikes = return_spikes

    def forward(self, x):
        actual_batch_size = x.size(0)

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = F.max_pool1d(self.fc1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.fc2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2.view(actual_batch_size, -1))

        # return cur3
        if self.return_spikes:
            # Return both output and spikes for counting
            return cur3, [spk1, spk2]
        else:
            return cur3

def load_exoplanet_model(weights_path=None, device=None, **kwargs):
    """
    Load the exoplanet model specifically, you can use your own weights. Default is the one we acquired from the Google Colab tutorial for snnTorch by Jason Eshraghian.
    """
    if weights_path is None:
        weights_path = MODEL_PATHS.get("exoplanet_snn", None)
    model = ExoplanetSNN(**kwargs)
    if weights_path:
        _load_weights(model, weights_path, device)
    return model

def _load_weights(model, weights_path, device=None):
    """
    Load weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

def get_model(name, weights_path=None, device=None, **kwargs):
    """
    Get the model from registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](weights_path=weights_path, device=device, **kwargs)

MODEL_REGISTRY = {
    "exoplanet_snn": load_exoplanet_model
}
