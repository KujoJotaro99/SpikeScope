import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from math import log2

# Define the spiking-aware function to estimate data movement (in bits) for a forward pass.
def data_movement_snn(
        blocks,
        layers,
        input_dim,
        neurons,
        weight_bw,
        weight_sparsity,
        activation_bw,
        firing_rate,
        cache_size,
        memory_size
    ):
    # 0 denotes register
    # 1 denotes cache
    # 2 denotes memory
    # 3 denotes persistent storage
    # Dictionary entries are formatted so that it's (src, dest)
    mem_mvmt = defaultdict(int) # Default value of 0

    # Weights
    total_weight_bits = blocks * layers * neurons * input_dim * weight_bw * weight_sparsity

    # Initial load from storage -> memory (once, but we include it here for completeness)
    mem_mvmt[(3, 2)] += total_weight_bits
    mem_mvmt[(2, 1)] += total_weight_bits
    mem_mvmt[(1, 0)] += total_weight_bits

    # Sparse activations
    total_act_bits = blocks * layers * neurons * activation_bw * firing_rate

    # All activations are written from register to cache
    mem_mvmt[(0, 1)] += total_act_bits
    # Memory <-> cache for activations
    cache_overflow = total_act_bits - min(cache_size, total_act_bits)
    mem_mvmt[(1, 2)] += cache_overflow
    # Ideally no activations go to storage due to insufficient memory
    memory_overflow = cache_overflow - min(memory_size, cache_overflow)
    mem_mvmt[(2, 3)] += memory_overflow

    return {
        (0, 1): mem_mvmt[(0, 1)] + mem_mvmt[(1, 0)],
        (1, 2): mem_mvmt[(1, 2)] + mem_mvmt[(2, 1)],
        (2, 3): mem_mvmt[(2, 3)] + mem_mvmt[(3, 2)],
    }

if __name__ == "__main__":
    # Default values
    blocks          = 1024
    layers          = 16
    input_dim       = 256 * 2**20
    neurons         = 256 * 2**20
    weight_bw       = 32
    weight_sparsity = 0.5
    activation_bw   = 1
    cache_size      = 8 * 2**30 * 8 # bits
    memory_size     = 64 * 2**30 * 8
    firing_rate     = 0.1

    # Sweep firing rates
    firing_rates = np.linspace(0.1, 0.9, 201)
    records = []
    for rate in firing_rates:
        rec = data_movement_snn(
            blocks, layers, input_dim, neurons, weight_bw, weight_sparsity,
            activation_bw, rate, cache_size, memory_size
        )
        #rec["firing_rate"] = rate*100  # in percent
        records.append(rec)
    plot_x = firing_rates * 100

    # Sweep weight sparsity
    # weight_sparsity = np.linspace(0.1, 0.9, 201)
    # records = []
    # for sparsity in weight_sparsity:
    #     rec = data_movement_snn(
    #         blocks, layers, input_dim, neurons, weight_bw, sparsity,
    #         activation_bw, firing_rate, cache_size, memory_size
    #     )
    #     records.append(rec)
    # plot_x = weight_sparsity * 100

    # Sweep weight bitwidth
    # records = []
    # wbitwidths = list(range(1, 65))
    # for wbw in wbitwidths:
    #     rec = data_movement_snn(
    #         blocks, layers, input_dim, neurons, wbw, weight_sparsity,
    #         activation_bw, firing_rate, cache_size, memory_size
    #     )
    #     records.append(rec)
    # plot_x = wbitwidths

    # Sweep neurons
    # records = []
    # neurons = list(range(64, 512, 8))
    # for n in neurons:
    #     rec = data_movement_snn(
    #         blocks, layers, input_dim, n, weight_bw, weight_sparsity,
    #         activation_bw, firing_rate, cache_size, memory_size
    #     )
    #     records.append(rec)
    # plot_x = neurons

    # Convert to arrays for plotting
    regs  = [r[(0, 1)] for r in records]
    cache = [r[(1, 2)] for r in records]
    mem   = [r[(2, 3)] for r in records]

    # Plot
    plt.rc('font',**{'family':'serif','serif':['Times New Roman'],'size':24,'weight':'50'})
    # plt.rc('font',**{'family':'serif','serif':['Times New Roman'],'size':18})
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(bin_formatter))
    plt.plot(plot_x, np.array(regs)/(2**70), label="R↔C", linestyle="dotted")#, color="black")
    plt.plot(plot_x, np.array(cache)/(2**70), label="C↔M", linestyle="dashed")#, color="black")
    plt.plot(plot_x, np.array(mem)/(2**70), label="M↔S", linestyle="dashdot")#, color="black")
    plt.xlabel("Firing Rate (percent)")
    # plt.xlabel("Weight sparsity (percent)", weight="50")
    # plt.xlabel("Number of neurons", weight="50")
    plt.ylabel("Data Movement (Zbit)", weight="50")

    # plt.title("Data Movement Breakdown vs. Sparsity")
    plt.legend()
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=8*2**70)
    plt.ticklabel_format(useMathText=True, useOffset=True)
    plt.tight_layout()
    plt.show()
