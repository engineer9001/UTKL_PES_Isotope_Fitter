import numpy as np

def SpillTime(event_time_array, window_size=0.1):     #funciton that looks at list of event times and computer the maximum event rate over a sliding window returning the time window with the highest rate
    print("Running spill time finder.......") #Takes a bit to run
    event_time_array = np.array(event_time_array)

    max_rate = -999999.0
    min_rate = 999999999.0
    max_event_rate_time = None
    
    for i in range(len(event_time_array)):
        window_start = event_time_array[-i]
        window_end = window_start + window_size         #window size is the size in seconds; Guessing we should look on the order of ~100 microseconds
        num_events = np.sum((event_time_array >= window_start) & (event_time_array < window_end))
        
        if num_events < min_rate:
            min_rate = num_events

        if num_events > max_rate:
            max_rate = num_events
            max_event_rate_time = window_end    #Want to start the fitting at the end of the provided window

    final_spill_max_rate = -999999.0
    final_spill_end_time = None

    for i in range(len(event_time_array)):
        window_start = event_time_array[-i]
        window_end = window_start + window_size         #window size is the size in seconds; Guessing we should look on the order of ~100 microseconds
        num_events = np.sum((event_time_array >= window_start) & (event_time_array < window_end))

        if num_events > final_spill_max_rate:
            final_spill_max_rate = num_events
            final_spill_end_time = window_end

        if final_spill_max_rate > 0.9 * max_rate and num_events < 0.7 * max_rate:   #Scanning from the end of the file, if your workign max is over 90% the global max rate, and the working rate is < 70% the global max, stop scanning and you should only be looking at the time of the final spill. These percentages may need to be tuned
            break
        
    #print(window_size)
    print(f'Max event rate in a {window_size*1000} millisecond window found to be {max_rate/window_size} Hz')
    print(f"Final spill's max event rate in a {window_size*1000} millisecond window found to be {final_spill_max_rate/window_size} Hz")
    #print(f'Min event rate in a {window_size*1000} millisecond window found to be {min_rate/window_size} Hz')
    print(f"Final spill end time found to be at {final_spill_end_time} seconds")
    return final_spill_end_time


if __name__ == "__main__":   #Prototyping
    import pandas as pd # type: ignore

    file_name = "Flash_Therapy/PET_3-5-23/MDA-Cu-10min-210mm-NoCu_coinc.dat"  #A file you want to look at to try out the spill time finder

    df = pd.read_csv(file_name, delimiter="\t", usecols=(2,3,4,7,8,9))
    df.columns = ["TimeL", "ChargeL", "ChannelIDL", "TimeR", "ChargeR", "ChannelIDR"]

    df["TimeL"] = df["TimeL"] / 1000000000000   #1E12   converting picoseconds to seconds presumably
    #print(df)
    SpillTime(df["TimeL"])
