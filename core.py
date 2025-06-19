import torch
from config import HARDWARE_CONFIGS, PROFILER_DEFAULTS

def compute_weight_sparsity(model):
    """
    Computes per-linear-layer density p_w and a global p_w.
    """
    layer_stats = {}
    total_nonzero = 0
    total_params  = 0

    #only go through dense weight matrices
    for name, module in model.named_modules():
        #linear and conv layers have weight calculation
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            #get raw weight tensor and flatten
            w = module.weight.data.abs().view(-1)
            nonzero = int((w > 1e-3).sum().item())
            count = w.numel()
            p_w = nonzero / count if count else 0.0
            layer_stats[name] = {
                'nonzero': nonzero,
                'total':   count,
                'p_w':     p_w,
            }
            total_nonzero += nonzero
            total_params  += count

    global_p_w = total_nonzero / total_params if total_params else 0.0
    return layer_stats, global_p_w

def count_spikes(model, data_loader, device=None):
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
            output = model(data)

            if isinstance(output, tuple):
                output, spike_list = output
                output = spike_list
            else:
                #when return_spikes=false
                output = [output]

            batch_spike_total = 0.0

            for i, spikes in enumerate(output):
                layer_key = f"layer_{i}"
                layer_spikes = spikes.sum().item() 
                if layer_key not in spike_counts:
                    spike_counts[layer_key] = layer_spikes
                else:
                    spike_counts[layer_key] += layer_spikes
                batch_spike_total += layer_spikes

            total_spikes += batch_spike_total
            total_samples += data.size(0)

    return total_spikes, total_samples, spike_counts

def get_model_stats(model):
    """Extract information about the model architecture."""
    total_neurons = 0
    total_weights = 0
    layer_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            input_size = module.in_features
            output_size = module.out_features
            weights = input_size * output_size
            
            total_neurons += output_size
            total_weights += weights
            
            layer_info[name] = {
                'input_size': input_size,
                'output_size': output_size,
                'weights': weights
            }
    
    return total_neurons, total_weights, layer_info

def calculate_memory_traffic(model, data_loader, hardware_config='loihi1', num_steps=None, device=None, verbose=False):
    """
    Calculate memory traffic for SNN inference.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if num_steps is None:
        num_steps = PROFILER_DEFAULTS['num_steps']
    
    if isinstance(hardware_config, str):
        hw_config = HARDWARE_CONFIGS.get(hardware_config, HARDWARE_CONFIGS['conventional'])
    else:
        hw_config = hardware_config

    layer_stats, global_p_w = compute_weight_sparsity(model)
    
    total_neurons, total_weights, layer_info = get_model_stats(model)
    
    total_spikes, total_samples, spike_counts = count_spikes(model, data_loader, device)
    
    #firing rate per neuron per timestep
    if total_samples > 0 and total_neurons > 0:
        avg_spikes_per_sample = total_spikes / total_samples
        firing_rate = avg_spikes_per_sample / (total_neurons * num_steps)
    else:
        firing_rate = 0
    
    #memory traffic calculations
    b_w = hw_config['weight_precision']
    b_a = hw_config['spike_precision']
    
    #static weight traffic in bits
    weight_traffic = total_weights * b_w * global_p_w
    
    #dynamic activation traffic in bits
    activation_traffic = b_a * firing_rate * total_neurons * num_steps
    
    #total traffic per inference
    total_traffic = weight_traffic + activation_traffic
    
    #cache movement
    cache_size_bits = hw_config['cache_size_kb'] * 1024 * 8
    cache_overflow = max(activation_traffic - cache_size_bits, 0) #goes to dram
    
    #energy estimation, figure from loihi paper
    static_energy = hw_config['static_power_per_core'] * (num_steps * 1e-3)
    dynamic_energy = avg_spikes_per_sample * hw_config['energy_per_spike']
    total_energy = static_energy + dynamic_energy
    
    results = {
        'hardware_config': hw_config['name'],
        'weight_sparsity': {
            'per_layer': layer_stats,
            'global_p_w': global_p_w,
        },
        'memory_traffic_bits': {
            'weight_traffic': weight_traffic,
            'activation_traffic': activation_traffic,
            'total_traffic': total_traffic,
            'cache_overflow': cache_overflow
        },
        'energy_estimate_J': {
            'static_energy': static_energy,
            'dynamic_energy': dynamic_energy,
            'total_energy': total_energy
        }
    }
    
    if verbose:
        print(f"Hardware profile: {hw_config['name']}")
        print(f"Model: {total_neurons:,} neurons, {total_weights:,} weights")
        print(f"Firing rate: {firing_rate:.6f} spikes/neuron/timestep")
        print(f"Weight traffic: {weight_traffic/1e6:.2f} Mbits")
        print(f"Activation traffic: {activation_traffic} bits")
        print(f"Total traffic: {total_traffic/1e6:.2f} Mbits")
        print(f"Cache overflow: {cache_overflow/1e6:.2f} Mbits")
        print(f"Total energy: {total_energy*1e6:.2f} µJ")
    
    return results

