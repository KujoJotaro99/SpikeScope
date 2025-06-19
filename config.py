import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATHS = {
    "exoplanet_snn": os.path.join(BASE_DIR, "weights", "exoplanet_snn.pth")
}

MODEL_CONFIG = {
    "beta": 0.5,
    "spike_grad_slope": 25,
    "batch_size": 64
}

HARDWARE_CONFIGS = {
    'loihi1': {
        'name': 'Intel Loihi 1',
        'cache_size_kb': 256,
        'memory_size_mb': 32,
        'weight_precision': 8,
        'spike_precision': 1,
        'static_power_per_core': 28e-3,
        'energy_per_spike': 26.6e-12,
    },
    
    'loihi2': {
        'name': 'Intel Loihi 2',
        'cache_size_kb': 512,
        'memory_size_mb': 128,
        'weight_precision': 8,
        'spike_precision': 1,
        'static_power_per_core': 15e-3,
        'energy_per_spike': 15e-12, 
    },
    
    'conventional': {
        'name': 'Conventional Processor',
        'cache_size_kb': 1024,
        'memory_size_mb': 8192,
        'weight_precision': 32,
        'spike_precision': 32,
        'static_power_per_core': 10,
        'energy_per_spike': 3e-12, 
    }
}

PROFILER_DEFAULTS = {
    'cache_size_kb': 1,
    'memory_size_mb': 4,
    'weight_bits': 8,
    'activation_bits': 1,
    'weight_sparsity': 1.0,
    'num_steps': 100
}