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

from FLASH_Fitting_Command_Line_Args import parse_arguments
from channel_indices import indices
#from fit_functions import * #As much as I hate using this syntax, this shouldn't be a problem as long as we always put new functions in fit_functions.py
from fit_params import Isotopes_Lifetimes_Dict
from SpillTimeFinder import SpillTime

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


# ## fit to background
# 
# Pass in file name/dir


file_name = args.data_file
dir = args.in_dir
imgs_dir = args.output_dir + "/"   #Add trailing slash so that it doens't matter if use provides it or not

if not os.path.exists(imgs_dir) and imgs_dir != "":
    os.makedirs(imgs_dir)

#For showing ricardo the data run 1 was MidV1, run 2 was MidV2, and run3 was MaxV1
df = pd.read_csv(dir+file_name, delimiter="\t", usecols=(2,3,4,7,8,9))
df.columns = ["TimeL", "ChargeL", "ChannelIDL", "TimeR", "ChargeR", "ChannelIDR"]

df["GeoChannelIDL"] = df["ChannelIDL"].apply(toGeoChannelID)
df["GeoChannelIDR"] = df["ChannelIDR"].apply(toGeoChannelID)
df["TimeL"] = df["TimeL"] / 1000000000000   #1E12   converting picoseconds to seconds presumably
df["TimeR"] = df["TimeR"] / 1000000000000   #1E12 
#Adjusting the time so the start time of the first spill is at 0:


spill_time_end = args.spill_time_end
if args.spill_time_end < 0.0:  #Have to assume that negative values of this are always invalid!
    spill_time_end = SpillTime(df["TimeL"], args.spill_time_finder_window)
    if not args.dont_write and args.file !=None:
        config_filepath = args.file[0]
        with open(config_filepath, 'a') as file:
            file.write(f"\n--spill_time_end {spill_time_end} #Added by spill time finder algorithm")  #don't recalculate the spill time start every time
            print("Spill time start written to config file")


df["TimeL"] = df["TimeL"] - spill_time_end #44.66

#Total Process for fitting the data to 4 exponentials
num_bins = args.num_bins

binwidth = args.run_length/num_bins

if args.create_first_plot:
    ### TODO: Fix this code
    #fitting the background to a constant and then subtracting it from the data for scaling purposes
    values,bins,params = plt.hist(df["TimeL"], bins=num_bins, fill=False, ec="C0")
    values = np.array(values)
    bins = np.array(bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    values_const = values[(bin_centers <= -1)]
    bin_centers_const = bin_centers[(bin_centers <= -1)]
    c = cost.LeastSquares(bin_centers_const, values_const,np.sqrt(values_const), Constant)
    fitter = Minuit(c,A=100)
    fitter.limits = [(0,None)]
    fitter.migrad()
    fitter.hesse()
    y_shift = fitter.values["A"]
    print("Constant Background Fit")
    print(fitter.values)
    print(fitter.errors)
    print(fitter.fval /(len(bins) - 2) )
    # bounds = list(ax.get_xlim())
    # leftbound = bounds[0]
    # rightbound = bounds[0]
    plt.figure(figsize=(20, 9))
    plt.xlim(-30,200)
    plt.ylim(0,4000)
    x_f = np.linspace(-5,0,len(values_const))
    plt.plot(x_f, Constant(x_f, *fitter.values))
    plt.savefig(f'{imgs_dir}FirstImage.png')     #I have no clue what this plot is supposed to be
    plt.close()
    #print("Minuit isn't robust enough for you to set a value, put it in the big fitting function")

# Set figure size (width, height)
fig_width = 20
fig_height = 9
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Create the histogram
t = df["TimeL"][df["TimeL"] >= 1] - min(df["TimeL"][df["TimeL"] >= 1])
new_num_bins = int((max(t) - min(t)) / binwidth)
values, bins, parameters = ax.hist(t, bins=new_num_bins, label='PET Data', alpha=0.8)
values = np.array(values)
bins = np.array(bins)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Find the maximum value of the histogram
max_bin_value = np.max(values)

# Set the y-axis to ensure the histogram is not cut off, while also allowing space for the fit lines
y_margin = max_bin_value * 0.05 # Increase margin to make room for the fit lines
ax.set_ylim(0, (max_bin_value + y_margin))

text_positions = {     #This is a dictionary of text positions calculated here so they won't change after adjuting the y-axis
    "reduced_chi2": max_bin_value * 0.5,
    "title": max_bin_value * 0.40,
}
isotope_text_positions = []
text_y_offset_factor = 0.10

for i, isotope in enumerate(args.fit_isotopes):
    y_offset = max_bin_value * (1 - i * text_y_offset_factor)
    isotope_text_positions.append(y_offset)

# Now, the histogram should fill most of the plot while leaving space above it
ax.set_ylim(0, (max_bin_value + y_margin) * args.y_margin_adjust)

#print(args.fit_isotopes)
#print(args.initial_fit_params)
# Create fit functions (same as before)
fit_functions = []
for func_name in args.fit_isotopes:
    if func_name not in Isotopes_Lifetimes_Dict.keys():
        print(f"Function {func_name} not found in fit_functions.py")
        sys.exit(1)
    fit_functions.append(func_name)

#print(fit_functions)
# Define the sum of functions for fitting (same as before)   This is done using black magic that I understood at one point
def create_sum_function(fit_functions):
    param_names = string.ascii_uppercase[:len(fit_functions)]

    def sum_function(t, *amplitudes):
        return sum(amp*np.exp(-t/Isotopes_Lifetimes_Dict[func]) for func, amp in zip(fit_functions, amplitudes))

    parameters = [
        inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ] + [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in param_names
    ]
    sum_function.__signature__ = inspect.Signature(parameters)
    sum_function.__name__ = "sum_of_fits"
    
    return sum_function

# Create the sum function for fitting
fit_function = create_sum_function(fit_functions)

if args.fit_function != "":
    fit_function = globals()[args.fit_function]

# Fit the model (same as before)
c = cost.LeastSquares(bin_centers, values, np.sqrt(values), fit_function)

param_names = string.ascii_uppercase[:len(args.fit_isotopes)]


initial_fit_params = [string.ascii_uppercase[i] + "=" + str(max(values)/len(args.fit_isotopes)) for i in range(len(args.fit_isotopes))]  # Initial guess for each isotope is evenly distributed unless otherwise specified
if args.initial_fit_params  != [""]:
    initial_fit_params = args.initial_fit_params

Fit_param_kwargs = ",".join(initial_fit_params)
fitter = eval(f"Minuit(c, {Fit_param_kwargs})")

fitter.limits = [(0, None)] * len(fitter.values)
fitter.migrad()
fitter.hesse()

# Calculate reduced chi-squared (same as before)
redChiSq = fitter.fval / (len(values) - len(initial_fit_params))
#print(len(initial_fit_params))

# Plot each individual isotope function dynamically (ensure they're visible)
for i, isotope in enumerate(args.fit_isotopes):
    if isotope in Isotopes_Lifetimes_Dict.keys():
        fit_line = fitter.values[i]*np.exp(-bin_centers/Isotopes_Lifetimes_Dict[isotope])  # Use bin_centers for fit calculation
        ax.plot(bin_centers, fit_line, linewidth=4, linestyle='dotted' if i % 2 == 0 else 'dashdot', label=isotope)


# Plot the sum function with more visibility (adjust line width and color if necessary)
sum_fit_line = fit_function(bin_centers, *fitter.values)
ax.plot(bin_centers, sum_fit_line, linewidth=4, color="r", alpha=0.7, label="Sum")

# Set legend
ax.legend(ncol=2, fontsize=25)

# Set axis labels
ax.set_ylabel(r'PET Event Rate [s$^{-1}$]', fontsize=25)
ax.set_xlabel('Time from Spill End [s]', fontsize=25)
plt.rcParams['figure.figsize'] = [fig_width, fig_height]

# Adjust text positioning (doubling the line spacing for the fit result text)
text_y_offset_factor = 0.10  # Double the spacing from before (previously was 0.05)
y_start = 0.95  # Initial position for first isotope label
y_step = 0.11   # Increased spacing between isotope labels

for i, isotope in enumerate(args.fit_isotopes):
    y_pos = y_start - i * y_step  # Double the spacing
    last_y_pos = y_pos - 0.06

    isotope_number_start = 0 #Silly nonsense that captures every possilbe isotope name
    for char in isotope:
        if char.isalpha():
            isotope_number_start += 1
        else:
            break

    # Fit function label
    ax.text(0.05, y_pos, f"{chr(65 + i)} = {fitter.values[i]:.1f} ± {fitter.errors[i]:.1f}", 
            fontsize=20, transform=ax.transAxes)

    # T 1/2 

    if isotope in Isotopes_Lifetimes_Dict.keys():
        half_life = Isotopes_Lifetimes_Dict[isotope] * ln2
        ax.text(0.05, y_pos - 0.06,  # Offset to prevent overlap
                fr"     T$_{{1/2}}$ (assumed) = {half_life:.3f} s ($^{{{isotope[isotope_number_start:]}}}${isotope[:isotope_number_start]})", 
                fontsize=20, transform=ax.transAxes)

reduced_chi2_y = last_y_pos - 0.08
title_y = reduced_chi2_y - 0.08
# Reduced chi-squared text
ax.text(0.05, reduced_chi2_y, fr"reduced $\chi^{{2}}$ = {redChiSq:.3f}", fontsize=20, transform=ax.transAxes)

# Title annotation
ax.text(0.05, title_y, args.run_config.replace("_", " "), fontsize=35, transform=ax.transAxes)

# Set x-axis limits

ax.set_xlim(0, max(bins) * 1.01) #go 1% past the last bin
if args.run_end_time > 0.0:
    ax.set_xlim(0, args.run_end_time)

plt.tick_params(axis='both', which='major', labelsize=20)

# Save the plot with dynamic fit name
fit_name = "_".join(args.fit_isotopes)
if args.fit_function != "":
    fit_name = args.fit_function

config_filename = args.run_config.replace(" ", "_")


plt.savefig(f'{imgs_dir}PES_Activity_Fit_{config_filename}_{fit_name}.png')
plt.savefig(f'{imgs_dir}PES_Activity_Fit_{config_filename}_{fit_name}.pdf')