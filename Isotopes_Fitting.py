#Code initially received from Kyle Klein
#Modified by Dalton Myers <dgmyers@utexas.edu> January 2025

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # type: ignore
#from scipy.optimize import curve_fit
#import scipy as sp
from iminuit import cost, Minuit # type: ignore
import iminuit as im # type: ignore
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys, os
import string
import inspect
import argparse
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from time import perf_counter
from tqdm import tqdm

from FLASH_Fitting_Command_Line_Args import parse_arguments
from channel_indices import indices
#from fit_functions import * #As much as I hate using this syntax, this shouldn't be a problem as long as we always put new functions in fit_functions.py
from fit_params import Isotopes_Lifetimes_Dict
from SpillTimeFinderFast import SpillTime, SpillStartAndBackground 

ln2 = np.log(2)

# plt.style.use('ChannelPairStyles.mplstyle')

args = parse_arguments()

def generateTextY(num,max,subplots=False):
    if max > 50 and subplots==False:
        y = [max - 0.04*(i+1)*max for i in range(0,num+1)]
    elif max <= 50 and subplots == False:
        y = [max - 0.06*(i+1)*max for i in range(0,num+1)]
    elif subplots == True:
        y = [0.96*max - 0.08*(i+1)*max for i in range(0,num+1)]
    return y

def halferror(x,xerr):
    return ln2*x * xerr/x

def toGeo(x):
    #converts PETSys ID to geometric ID
    y = 8*indices.get(x)[0] + indices.get(x)[1]
    return y

def Constant(x,A):
    return A+(10**-7)*x

geo_channels = []
for i in range(128):
    geo_channels.append([i,toGeo(i)])
geo_channels = np.asarray(geo_channels)

def toGeoChannelID(AbsChannelID):
    # Convert PETSys absolute channel IDs to geomteric IDs
    slaveID = AbsChannelID // 4096
    chipID = (AbsChannelID - slaveID*4096) // 64
    channelID = AbsChannelID % 64

    PCB_ChanID = 64*(chipID % 2) + channelID
    AbsPCB_ChanID = geo_channels[geo_channels[:,0] == PCB_ChanID][0][1]

    #General formula can be found in above function "to AbsChannelID"
    GeoChannelID = 10**4 * slaveID + 10**2 * chipID + AbsPCB_ChanID % 64
    return GeoChannelID

def chiSq(o,e):
    return (o-e)**2/e

def halferror(x,xerr):
    return ln2*x * xerr/x

def fill_valleys(signal, kernel_size=21, threshold=0.6):
    """
    Fill valleys (artificial dips) in a 1D signal based on deviation from local median.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    signal = np.asarray(signal)
    local_median = medfilt(signal, kernel_size=kernel_size)
    is_valley = signal < (local_median * threshold)
    x = np.arange(len(signal))
    valid_mask = ~is_valley
    interp_fn = interp1d(x[valid_mask], signal[valid_mask], kind='linear', fill_value="extrapolate")
    filled_signal = signal.copy()
    filled_signal[is_valley] = interp_fn(x[is_valley])
    return filled_signal

def create_decaying_background_func(amplitudes_str, isotopes, time_offset):
    """
    Creates a function that models the decaying background from a previous run.
    """
    try:
        amplitudes = [float(a) for a in amplitudes_str]
    except (ValueError, TypeError):
        print("Error: Could not convert previous run amplitudes to numbers.")
        sys.exit(1)

    def decaying_background(t):
        """ The actual background function. t is the time in the current run. """
        total_bkg = 0.0
        for amp, iso in zip(amplitudes, isotopes):
            if iso in Isotopes_Lifetimes_Dict:
                tau = Isotopes_Lifetimes_Dict[iso]
                total_bkg += amp * np.exp(-(t + time_offset) / tau)
        return total_bkg
    
    return decaying_background

# ## Main script logic starts here

file_name = args.data_file
dir = args.in_dir
datafile_path = dir+file_name
imgs_dir = args.output_dir + "/"   #Add trailing slash so that it doens't matter if use provides it or not

if not os.path.exists(datafile_path):
    datafile_path = os.path.dirname(__file__) + datafile_path
    if not os.path.exists(datafile_path):
        raise FileNotFoundError(f"Error: The file '{datafile_path}' does not exist.")

if not os.path.exists(imgs_dir) and imgs_dir != "":
    print("Output directory does not exist; making it now")
    os.makedirs(imgs_dir)

timings: dict[str, float] = {}

#For showing ricardo the data run 1 was MidV1, run 2 was MidV2, and run3 was MaxV1
timings['data_loading'] = -perf_counter()
with tqdm(total=1, desc="Loading data", unit="file") as pbar:
    df = pd.read_csv(datafile_path, delimiter="\t", usecols=(args.use_columns))
    pbar.update()
df.columns = args.column_names
timings['data_loading'] += perf_counter()

df["GeoChannelIDL"] = df["ChannelIDL"].apply(toGeoChannelID)
df["GeoChannelIDR"] = df["ChannelIDR"].apply(toGeoChannelID)
df["TimeL"] = df["TimeL"] / 1000000000000   #1E12   converting picoseconds to seconds presumably
df["TimeR"] = df["TimeR"] / 1000000000000   #1E12 
#Adjusting the time so the start time of the first spill is at 0:

spill_time_end = args.spill_time_end
if args.spill_time_end < 0.0:  #Have to assume that negative values of this are always invalid!
    spill_time_end = SpillTime(df["TimeL"], args.spill_time_finder_window)
    if not args.dont_write and args.file !=None and spill_time_end !=0:
        config_filepath = args.file[0]
        with open(config_filepath, 'a') as file:
            file.write(f"\n--spill_time_end {spill_time_end} #Added by spill time finder algorithm")  #don't recalculate the spill time start every time
            print("Spill time start written to config file")

# Determine fixed background rate before shifting the timeline
fixed_background_rate = args.fixed_background_rate
fixed_background_rate_unc = args.fixed_background_rate_uncertainty
if fixed_background_rate < 0.0:
    timings['background_estimation'] = -perf_counter()
    spill_start_time, fixed_background_rate, fixed_background_rate_unc = SpillStartAndBackground(
        df["TimeL"], args.spill_time_finder_window
    )
    timings['background_estimation'] += perf_counter()
    if fixed_background_rate > 0.0:
        if fixed_background_rate_unc >= 0.0:
            print(
                f"Using fixed background rate {fixed_background_rate:.2f} ± {fixed_background_rate_unc:.2f} Hz "
                f"(pre-spill data before {spill_start_time:.6f} s)"
            )
        else:
            print(
                f"Using fixed background rate {fixed_background_rate:.2f} Hz "
                f"(pre-spill data before {spill_start_time:.6f} s)"
            )
        if not args.dont_write and args.file is not None:
            config_filepath = args.file[0]
            with open(config_filepath, 'a') as file:
                file.write(f"\n--fixed_background_rate {fixed_background_rate} #Added by background estimator")
                if fixed_background_rate_unc >= 0.0:
                    file.write(
                        f"\n--fixed_background_rate_uncertainty {fixed_background_rate_unc} "
                        "#Added by background estimator"
                    )
                print("Fixed background rate written to config file")
    else:
        fixed_background_rate_unc = -1.0
        print("Warning: Could not determine background rate; falling back to fitting constant background")
elif fixed_background_rate > 0.0:
    if fixed_background_rate_unc >= 0.0:
        print(
            f"Using fixed background rate from config: {fixed_background_rate:.2f} ± {fixed_background_rate_unc:.2f} Hz"
        )
    else:
        print(f"Using fixed background rate from config: {fixed_background_rate:.2f} Hz")

df["TimeL"] = df["TimeL"] - spill_time_end #44.66

#Total Process for fitting the data to 4 exponentials
binwidth = args.bin_width
num_bins = args.num_bins
# --- REMOVED: Initial guess calculation for constant background is no longer needed. ---
# A simple default guess will be used instead.
constant_bkg_guess = 1.0

# Set figure size (width, height)
fig_width = 20
fig_height = 9
fig, (ax, ax_residuals) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(fig_width, fig_height), sharex=True)
plt.subplots_adjust(hspace=0)

# Create the histogram (counts), then convert to rates for fitting/plotting
# Match the binning strategy used in my_FLASH_Fitting.py but avoid forcing
# the range to run_length. We compute rates = counts / binwidth so that the
# plotted data truly reflects PET event rates.
t_series = df["TimeL"][df["TimeL"] >= 1]
if len(t_series) == 0:
    raise ValueError("No events remain after spill-time subtraction >= 1s")
t = t_series - t_series.min()
timings['histogram'] = -perf_counter()
new_num_bins = int((t.max() - t.min()) / binwidth) if len(t) > 0 else num_bins
counts, bins = np.histogram(t, bins=new_num_bins)
if args.valleys:
    counts = fill_valleys(counts)
counts = np.asarray(counts)
bins = np.asarray(bins)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
values = counts / binwidth
plot_values = np.linspace(bin_centers.min(), bin_centers.max(), num=25*len(bin_centers))

bar_widths = bins[1:] - bins[:-1]
ax.bar(bin_centers, values, width=bar_widths, align='center', alpha=0.8, label='PET Data')
timings['histogram'] += perf_counter()

# Create the decaying background model if applicable
decaying_bkg_model = None
if args.previous_run_amplitudes:
    print("--- Subsequent Run Mode: Including decaying background ---")
    # Convert string amplitudes to floats for the function
    prev_amps_float = [float(amp) for amp in args.previous_run_amplitudes]
    decaying_bkg_model = create_decaying_background_func(
        prev_amps_float,
        args.previous_run_isotopes,
        args.time_since_previous_run
    )

# Create fit functions
fit_functions = []
for func_name in args.fit_isotopes:
    if func_name not in Isotopes_Lifetimes_Dict.keys():
        print(f"Function {func_name} not found in fit_params.py")
        sys.exit(1)
    fit_functions.append(func_name)

# This function now creates a model with new decays + decaying bkg + constant bkg
def create_sum_function(fit_functions, fixed_decaying_bkg=None, fixed_const_bkg=None):
    include_const_param = fixed_const_bkg is None
    param_names = list(string.ascii_uppercase[:len(fit_functions)]) + (["ConstBkg"] if include_const_param else [])

    def sum_function(t, *params):
        amplitudes = params[:len(fit_functions)]
        const_bkg = params[len(fit_functions)] if include_const_param else (fixed_const_bkg or 0.0)

        new_decays = sum(amp*np.exp(-t/Isotopes_Lifetimes_Dict[func]) for func, amp in zip(fit_functions, amplitudes))
        decaying_bkg = fixed_decaying_bkg(t) if fixed_decaying_bkg else 0.0
        return new_decays + decaying_bkg + const_bkg

    parameters = [
        inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ] + [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in param_names
    ]
    sum_function.__signature__ = inspect.Signature(parameters)
    sum_function.__name__ = "sum_of_fits"

    return sum_function

# Create the final composite fit function
fixed_const_value = None
fixed_const_unc = None
if fixed_background_rate > 0.0:
    fixed_const_value = fixed_background_rate
    if fixed_background_rate_unc >= 0.0:
        fixed_const_unc = fixed_background_rate_unc

fit_function = create_sum_function(fit_functions, decaying_bkg_model, fixed_const_bkg=fixed_const_value)

if args.fit_function != "":
    fit_function = globals()[args.fit_function]

# Fit the model using Poisson errors (rates derived from counts)
count_uncertainties = np.sqrt(np.clip(counts, 1e-9, None))
rate_uncertainties = count_uncertainties / binwidth
c = cost.LeastSquares(bin_centers, values, rate_uncertainties, fit_function)

# Building the initial parameters for Minuit
param_names = list(string.ascii_uppercase[:len(args.fit_isotopes)])
initial_params_dict = {}
# Initial guess for new isotopes
initial_amp_guess = max(values) / len(args.fit_isotopes) if len(values) > 0 else 1.0
for i, name in enumerate(param_names):
    initial_params_dict[name] = initial_amp_guess
# Add the initial guess for the constant background only when it is not fixed
if fixed_const_value is None:
    initial_params_dict["ConstBkg"] = constant_bkg_guess

fitter = Minuit(c, **initial_params_dict)

fitter.limits = [(0, None)] * len(fitter.values)

timings['migrad'] = -perf_counter()
fitter.migrad()
timings['migrad'] += perf_counter()

timings['hesse'] = -perf_counter()
fitter.hesse()
timings['hesse'] += perf_counter()

# Calculate reduced chi-squared
redChiSq = fitter.fval / (len(values) - fitter.nfit)

# Plotting section to visualize all components
# Plot each individual isotope function from the current run
for i, isotope in enumerate(args.fit_isotopes):
    param_name = string.ascii_uppercase[i]
    # NOTE: The individual component does NOT include background for clarity
    fit_line = fitter.values[param_name] * np.exp(-plot_values/Isotopes_Lifetimes_Dict[isotope])
    ax.plot(plot_values, fit_line, linewidth=4, linestyle='dotted', label=f"{isotope} (new)")

# Plot the background components
if fixed_const_value is None:
    const_bkg_val = fitter.values["ConstBkg"]
    const_bkg_line = np.full_like(plot_values, const_bkg_val)
    ax.plot(plot_values, const_bkg_line, linewidth=2, linestyle=':', color='gray', label='Constant Bkg (fit)')
else:
    const_bkg_val = fixed_const_value
    const_bkg_line = np.full_like(plot_values, const_bkg_val)
    ax.plot(plot_values, const_bkg_line, linewidth=2, linestyle=':', color='gray', label='Constant Bkg (fixed)')

if decaying_bkg_model:
    decaying_bkg_line = decaying_bkg_model(plot_values)
    ax.plot(plot_values, decaying_bkg_line, linewidth=2, linestyle='--', color='black', label='Decaying Bkg (fixed)')

# Plot the sum function with more visibility
sum_fit_line = fit_function(plot_values, *fitter.values)
ax.plot(plot_values, sum_fit_line, linewidth=4, color="r", alpha=0.7, label="Sum")

plt.tick_params(axis='both', which='major', labelsize=20)

fit_values_at_bin_centers = fit_function(bin_centers, *fitter.values)
residuals = values - fit_values_at_bin_centers
residual_errors = rate_uncertainties

# Residuals plot
ax_residuals.axhline(0, color='black', linewidth=1, linestyle='dashed')
ax_residuals.errorbar(bin_centers, residuals, yerr=residual_errors, fmt='.', color='black', capsize=4, elinewidth=1.5)
ax_residuals.set_ylabel("Residuals\n(Data - Fit)", fontsize=16)

ax.tick_params(labelbottom=False)
ax_residuals.set_xlabel('Time from Spill End [s]', fontsize=25)
ax.set_ylabel(r'PET Event Rate [s$^{-1}$]', fontsize=25)
ax.yaxis.set_tick_params(labelsize=20)
ax_residuals.yaxis.set_tick_params(labelsize=16)
ax.legend(ncol=2, fontsize=25)

# Text annotation to include constant background result
y_start = 0.95
y_step = 0.12
last_y_pos = y_start
for i, isotope in enumerate(args.fit_isotopes):
    y_pos = y_start - i * y_step
    last_y_pos = y_pos - 0.06
    param_name = string.ascii_uppercase[i]
    isotope_number_start = next((idx for idx, char in enumerate(isotope) if char.isdigit()), len(isotope))

    ax.text(0.05, y_pos, f"{param_name} = {fitter.values[param_name]:.1f} ± {fitter.errors[param_name]:.1f}", 
            fontsize=20, transform=ax.transAxes)
    if isotope in Isotopes_Lifetimes_Dict.keys():
        half_life = Isotopes_Lifetimes_Dict[isotope] * ln2
        ax.text(0.05, y_pos - 0.06,
                fr"     T$_{{1/2}}$ (assumed) = {half_life:.3f} s ($^{{{isotope[isotope_number_start:]}}}${isotope[:isotope_number_start]})", 
                fontsize=20, transform=ax.transAxes)

# Add constant background result (fit or fixed)
if fixed_const_value is None:
    const_bkg_val = fitter.values["ConstBkg"]
    const_bkg_err = fitter.errors["ConstBkg"]
    ax.text(0.05, last_y_pos - y_step, f"Const Bkg = {const_bkg_val:.1f} ± {const_bkg_err:.1f}",
            fontsize=20, transform=ax.transAxes)
else:
    if fixed_const_unc is not None:
        ax.text(0.05, last_y_pos - y_step, f"Const Bkg (fixed) = {const_bkg_val:.1f} ± {fixed_const_unc:.1f}",
                fontsize=20, transform=ax.transAxes)
    else:
        ax.text(0.05, last_y_pos - y_step, f"Const Bkg (fixed) = {const_bkg_val:.1f}",
                fontsize=20, transform=ax.transAxes)
last_y_pos -= y_step

reduced_chi2_y = last_y_pos - 0.08
title_y = reduced_chi2_y - 0.08
ax.text(0.05, reduced_chi2_y, fr"reduced $\chi^{{2}}$ = {redChiSq:.3f}", fontsize=20, transform=ax.transAxes)
ax.text(0.05, title_y, args.run_config.replace("_", " "), fontsize=32, transform=ax.transAxes)

ax.set_xlim(0, max(bins) * 1.01)
if args.run_end_time > 0.0:
    ax.set_xlim(0, args.run_end_time)

ax.set_ylim(0, max(values)*1.1*args.y_margin_adjust)

# Save the plot with dynamic fit name
fit_name = "_".join(args.fit_isotopes)
if args.fit_function != "":
    fit_name = args.fit_function
config_filename = args.run_config.replace(" ", "_")

plt.savefig(f'{imgs_dir}PES_Activity_Fit_{config_filename}_{fit_name}.png')

print(f"\nPlot saved to {imgs_dir}PES_Activity_Fit_{config_filename}_{fit_name}.png")

# Console summary of fit results
print("\nFit parameter results (total rates):")
previous_amp_map: dict[str, float] = {}
previous_sigma_map: dict[str, float] = {}
if args.previous_run_isotopes and args.previous_run_amplitudes:
    try:
        previous_amp_map = {iso: float(val) for iso, val in zip(args.previous_run_isotopes, args.previous_run_amplitudes)}
        if args.previous_run_amplitude_uncertainties:
            previous_sigma_map = {iso: float(sig) for iso, sig in zip(args.previous_run_isotopes, args.previous_run_amplitude_uncertainties)}
    except ValueError:
        print("Warning: could not parse previous run amplitudes/uncertainties; ignoring them.")
        previous_amp_map = {}
        previous_sigma_map = {}

decay_time = max(args.time_since_previous_run, 0.0)
prev_decayed_map: dict[str, float] = {}
prev_decayed_sigma_map: dict[str, float] = {}
for iso, prev_val in previous_amp_map.items():
    tau = Isotopes_Lifetimes_Dict.get(iso)
    if tau is None or tau <= 0:
        print(f"Warning: missing lifetime for {iso}; previous contribution ignored for subtraction.")
        continue
    decay_factor = np.exp(-decay_time / tau)
    prev_decayed_map[iso] = prev_val * decay_factor
    prev_decayed_sigma_map[iso] = previous_sigma_map.get(iso, 0.0) * decay_factor

total_amp_map: dict[str, float] = {}
total_sigma_map: dict[str, float] = {}
new_amp_map: dict[str, float] = {}
new_sigma_map: dict[str, float] = {}

for i, isotope in enumerate(args.fit_isotopes):
    param_name = string.ascii_uppercase[i]
    total_val = fitter.values[param_name]
    total_err = fitter.errors[param_name]
    prev_decayed = prev_decayed_map.get(isotope, 0.0)
    prev_decayed_sigma = prev_decayed_sigma_map.get(isotope, 0.0)

    new_val = total_val - prev_decayed
    new_err = (total_err ** 2 + prev_decayed_sigma ** 2) ** 0.5

    total_amp_map[isotope] = total_val
    total_sigma_map[isotope] = total_err
    new_amp_map[isotope] = new_val
    new_sigma_map[isotope] = new_err

    prev_text = "none"
    if isotope in previous_amp_map:
        raw_prev = previous_amp_map[isotope]
        prev_text = f"prev raw {raw_prev:.3f}"
        if raw_prev != prev_decayed:
            prev_text += f" -> decayed {prev_decayed:.3f}"
        prev_sigma_raw = previous_sigma_map.get(isotope, 0.0)
        if prev_sigma_raw > 0:
            prev_text += f" (σ {prev_sigma_raw:.3f} -> {prev_decayed_sigma:.3f})"
    warning_note = " [warning: negative new contribution]" if new_val < 0 else ""
    print(
        f"  {param_name} ({isotope}): total {total_val:.3f} ± {total_err:.3f} s^-1 | "
        f"{prev_text} -> new {new_val:.3f} ± {new_err:.3f}{warning_note}"
    )

if fixed_const_value is None:
    print(f"  ConstBkg: {fitter.values['ConstBkg']:.3f} ± {fitter.errors['ConstBkg']:.3f} s^-1")
else:
    if fixed_const_unc is not None:
        print(f"  ConstBkg (fixed): {fixed_const_value:.3f} ± {fixed_const_unc:.3f} s^-1")
    else:
        print(f"  ConstBkg (fixed): {fixed_const_value:.3f} s^-1")

print(f"  Reduced chi^2: {redChiSq:.3f}")

# Prepare amplitudes to carry forward (total rates at spill end)
combined_isotopes = sorted(set(total_amp_map.keys()) | set(previous_amp_map.keys()))
print("\nTotal amplitudes to carry forward (s^-1):")
forward_amplitudes: list[str] = []
forward_uncertainties: list[str] = []
for isotope in combined_isotopes:
    if isotope in total_amp_map:
        carry_val = total_amp_map[isotope]
        carry_sigma = total_sigma_map.get(isotope, 0.0)
    else:
        carry_val = prev_decayed_map.get(isotope, previous_amp_map.get(isotope, 0.0))
        carry_sigma = prev_decayed_sigma_map.get(isotope, previous_sigma_map.get(isotope, 0.0))
    carry_text = f"{carry_val:.3f}"
    if carry_sigma > 0:
        carry_text += f" ± {carry_sigma:.3f}"
        sigma_export = f"{carry_sigma:.6g}"
    else:
        sigma_export = "0.0"
    print(f"  {isotope}: {carry_text}")
    forward_amplitudes.append(f"{carry_val:.6g}")
    forward_uncertainties.append(sigma_export)

print("\nArguments for next run background transfer:")
print(f"  --previous_run_isotopes {','.join(combined_isotopes)}")
print(f"  --previous_run_amplitudes {','.join(forward_amplitudes)}")
print(f"  --previous_run_amplitude_uncertainties {','.join(forward_uncertainties)}")

print("\nNew production during this run (s^-1):")
for isotope in args.fit_isotopes:
    new_val = new_amp_map.get(isotope, 0.0)
    new_sigma = new_sigma_map.get(isotope, 0.0)
    new_text = f"{new_val:.3f}"
    if new_sigma > 0:
        new_text += f" ± {new_sigma:.3f}"
    print(f"  {isotope}: {new_text}")

# Timing summary
if timings:
    print("\nStep timings (s):")
    for step, duration in timings.items():
        print(f"  {step}: {duration:.3f}")
    longest_step = max(timings, key=timings.get)
    print(f"Longest step: {longest_step} ({timings[longest_step]:.3f} s)")
