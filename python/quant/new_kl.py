import numpy as np
from collections import Counter
from scipy.stats import entropy

def _compute_amax_entropy(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    # 返回使收集的直方图的KL散度最小化的amax
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    # If calibrator hasn't collected any data, return none
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

    # we are quantizing to 128 values + sign if num_bits=8
    nbins = 1 << (num_bits - 1 + int(unsigned)) # 对称量化 nbins=128

    starting = start_bin
    stop = len(bins) # 4028

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    #首次遍历 i=128
    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0) 
        # 这里是先进行量化，再计算数据分布Q,耗时比较大
        # 把bin[0],...,bin[i-1]量化为128个bin
        space = np.linspace(0, i, num=nbins + 1)
        # numpy.digitize(array_x, bins, right=False)：返回array中每一个值在bins中所属的位置
        # 记录量化前的i在量化后的位置
        digitized_space = np.digitize(range(i), space) - 1
        digitized_space[bins[:i] == 0] = -1 # 直方图值为0 对应 digitized_space 值取-1

        # 计算量化后的数据分布Q
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                # 将直方图柱子不为0依次累加
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space) # Counter：统计可迭代序列中每个元素出现次数
        for key, val in counter.items():
            if key != -1:
                # 计算分布Q:new_density_counts
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        # 剔除直方图值为0的
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        # 归一化
        _normalize_distr(new_density)

        # 取前i个bin
        reference_density = np.array(bins[:len(digitized_space)])
        # 选择从第i个bin截断，i后面的bin加到i-1 上，得到一个基本没有损失的直方图P_clip
        # reference_density 代表原始float数据截断后的分布情况
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)
        # 计算KL散度，散度越小代表分布越相似
        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    # calib_amax代表截断float的最大值，在此处截断，量化效果最好
    calib_amax = calib_bin_edges[last_argmin * stride + starting]
    
    return calib_amax

# 定义量化函数和反量化函数,需要基于上面的_compute_amax_entropy函数


if __name__ == '__main__':
    pass