import numpy as np
from collections import Counter
from scipy.stats import entropy

def compute_amax_entropy(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    if calib_bin_edges is None and calib_hist is None:
        return None

    def _normalize_distr(distr):
        summ = np.sum(distr)
        if summ != 0:
            distr = distr / summ

    bins = calib_hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    nbins = 1 << (num_bits - 1 + int(unsigned)) # 对称量化 nbins=128

    starting = start_bin
    stop = len(bins) # 4028

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0) 
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1
        digitized_space[bins[:i] == 0] = -1 # 直方图值为0 对应 digitized_space 值取-1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space) # Counter：统计可迭代序列中每个元素出现次数
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        _normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)
        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_amax = calib_bin_edges[last_argmin * stride + starting]
    
    return calib_amax