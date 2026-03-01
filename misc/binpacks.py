import numpy as np
import prtpy


def analyze_output(bins):
    N = sum([len(b) for b in bins])
    M = len(bins)
    sums = [sum(b) for b in bins]
    # print(sums)
    m = np.mean(sums)
    s = np.std(sums)
    ma = np.max(sums)
    mi = np.min(sums)
    med = np.median(sums)
    print(f"{N=}, {M=}, {m=:0.3f}, {s=:0.10f}, {ma=:0.3f}, {mi=:0.3f}, {med=:0.3f}")


def multifit_lpt(nums, M):
    sums = [0] * M
    bins = []
    for i in range(M):
        bins.append([])
    nums_ = sorted(nums, reverse=True)
    for n in nums_:
        lowest_i = np.argmin(sums)
        sums[lowest_i] += n
        bins[lowest_i].append(n)
    for i in range(M):
        assert sums[i] == sum(bins[i])
    return bins


def test_multifit_lpt_1():
    N = 16384
    M = 16
    np.random.seed(0x31)
    nums = np.log2(np.arange(N // 2, N))

    bins = multifit_lpt(nums, M)
    analyze_output(bins)

    bins_prtpy = prtpy.partition(
        algorithm=prtpy.partitioning.kk,
        numbins=M,
        items=nums,
        # time_limit=3,
        # iterations=1000,
    )

    analyze_output(bins_prtpy)


if __name__ == "__main__":
    test_multifit_lpt_1()
