import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
from config import MODEL_PATHS

class ExoplanetSNN(nn.Module):
    def __init__(self, beta=0.5, spike_grad=None, input_size=3197, batch_size=64):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.batch_size = batch_size

        self.fc1 = nn.Linear(input_size, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(64, 64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3 = nn.Linear(32, 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=1):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = F.max_pool1d(self.fc1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.fc2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        out = self.fc3(spk2.view(self.batch_size, -1))
        return out


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
