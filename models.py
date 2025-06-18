import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
from config import MODEL_PATHS, MODEL_CONFIG

class ExoplanetSNN(nn.Module):
    def __init__(self, return_spikes=False):
        super().__init__()

        beta = MODEL_CONFIG['beta']
        slope = MODEL_CONFIG['spike_grad_slope']
        spike_grad = surrogate.fast_sigmoid(slope=slope)

        self.fc1 = nn.Linear(3197, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(32, 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.softmax = nn.Softmax(dim=1)
        self.return_spikes = return_spikes

    def forward(self, x):
        actual_batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = F.max_pool1d(self.fc1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.fc2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2.view(actual_batch_size, -1))

        if self.return_spikes:
            return cur3, [spk1, spk2]
        else:
            return cur3

def load_model_weights(model, weights_path, device=None):
    """Load model weights from file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def get_model(model_name, weights_path=None, device=None, **kwargs):
    """
    Get a model by name with optional custom weights.
    """
    if model_name == "exoplanet_snn":
        model = ExoplanetSNN(**kwargs)
        
        if weights_path is None:
            weights_path = MODEL_PATHS.get("exoplanet_snn")
            
        model = load_model_weights(model, weights_path, device)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: ['exoplanet_snn']")
