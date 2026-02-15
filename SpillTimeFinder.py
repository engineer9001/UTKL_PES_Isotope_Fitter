import numpy as np
from tqdm import tqdm

def SpillTime(event_time_array, window_size=0.1):
    print("Running optimized spill time finder.......")

    # 步骤1: 确保数据是排好序的 NumPy 数组
    # to_numpy() 处理了从Pandas Series传入的情况，sort()确保了算法的前提条件
    event_time_array = np.sort(event_time_array.to_numpy())
    n_events = len(event_time_array)
    if n_events == 0:
        print("Warning: Event time array is empty.")
        return 0.0

    hysteresis_high = 0.8
    hysteresis_low = 0.6
    
    # --- 第一遍扫描 (O(N) 复杂度): 寻找全局最大事件率 ---
    print("Pass 1: Finding global max event rate...")
    max_rate = 0
    left_ptr = 0
    for right_ptr in tqdm(range(n_events)):
        # 当窗口过大时，收缩左边界
        while event_time_array[right_ptr] - event_time_array[left_ptr] > window_size:
            left_ptr += 1
        # 当前窗口内的事件数就是指针的差值
        current_rate = right_ptr - left_ptr + 1
        if current_rate > max_rate:
            max_rate = current_rate

    if max_rate == 0:
        print("Warning: No events found within any window.")
        return 0.0

    # --- 第二遍扫描 (O(N) 复杂度): 使用迟滞比较定位最终溢出 ---
    print("Pass 2: Locating final spill using hysteresis...")
    final_spill_max_rate = 0
    final_spill_end_time = None
    
    # 我们需要从后向前扫描，所以反转数组
    reversed_time_array = event_time_array[::-1]
    
    left_ptr = 0
    for right_ptr in tqdm(range(n_events)):
        # 窗口宽度计算方式变为 (end - start)
        while reversed_time_array[left_ptr] - reversed_time_array[right_ptr] > window_size:
            left_ptr += 1
        
        current_rate = right_ptr - left_ptr + 1
        
        if current_rate >= final_spill_max_rate:
            final_spill_max_rate = current_rate
            # 窗口的结束时间是左指针对应的时间点 (因为数组是反转的)
            final_spill_end_time = reversed_time_array[left_ptr]

        # 迟滞判断条件 (从后向前扫描，逻辑不变)
        if final_spill_max_rate > hysteresis_high * max_rate and current_rate < hysteresis_low * max_rate:
            break

    print(f'Max event rate in a {window_size*1000} millisecond window found to be {max_rate/window_size:.1f} Hz')
    print(f"Final spill's max event rate in a {window_size*1000} millisecond window found to be {final_spill_max_rate/window_size:.1f} Hz")
    print(f"Final spill end time found to be at {final_spill_end_time} seconds")
    
    return final_spill_end_time


def SpillStartAndBackground(event_time_array, window_size=0.1):
    """Forward hysteresis scan to locate the first spill onset and background.

    Returns tuple (spill_start_time, background_rate, background_rate_uncertainty).
    """
    print("Running forward scan for spill start and background...")

    t = np.sort(event_time_array.to_numpy())
    n = len(t)
    if n == 0:
        print("Warning: Event time array is empty.")
        return 0.0, 0.0, 0.0

    hysteresis_high = 0.9
    hysteresis_low = 0.7

    # Pass 1: global max rate for normalization
    max_rate = 0
    l = 0
    for r in range(n):
        while t[r] - t[l] > window_size:
            l += 1
        cur = r - l + 1
        if cur > max_rate:
            max_rate = cur
    if max_rate == 0:
        return 0.0, 0.0, 0.0

    # Pass 2: forward scan with hysteresis condition
    l = 0
    start_max_rate = 0
    candidate_time = t[0]
    for r in range(n):
        while t[r] - t[l] > window_size:
            l += 1
        cur = r - l + 1
        if cur >= start_max_rate:
            start_max_rate = cur
            candidate_time = t[l]

        if start_max_rate > hysteresis_high * max_rate and cur < hysteresis_low * max_rate:
            break
    spill_start_time = candidate_time
    spill_start_time = np.floor(spill_start_time)

    # Background from data strictly before the start time
    pre_mask = t < spill_start_time
    if not np.any(pre_mask):
        return spill_start_time, 0.0, 0.0

    t_pre = t[pre_mask]
    duration = t_pre[-1] - t_pre[0]
    if duration <= 0:
        return spill_start_time, 0.0, 0.0

    count_pre = len(t_pre)
    background_rate = count_pre / duration
    background_rate_unc = np.sqrt(count_pre) / duration
    print(
        f"Estimated background rate: {background_rate:.2f} ± {background_rate_unc:.2f} Hz "
        f"from {duration:.2f}s pre-spill region"
    )
    return spill_start_time, background_rate, background_rate_unc

if __name__ == "__main__":   #Prototyping
    import pandas as pd # type: ignore

    file_name = "/Users/yuanwu/Library/CloudStorage/Box-Box/PET/MDA1011/Run1_DerenzoPhant4mm_DerenzoColi4mm_PMMASpacer_119.9MeV_200MU_10cyc_15min_HWTrigOn_coinc.dat"  #A file you want to look at to try out the spill time finder

    df = pd.read_csv(file_name, delimiter="\t", usecols=(2,3,4,7,8,9))
    df.columns = ["TimeL", "ChargeL", "ChannelIDL", "TimeR", "ChargeR", "ChannelIDR"]

    df["TimeL"] = df["TimeL"] / 1000000000000   #1E12   converting picoseconds to seconds presumably
    #print(df)
    SpillTime(df["TimeL"])
