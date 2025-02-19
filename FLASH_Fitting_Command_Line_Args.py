import sys
import argparse

def parse_arguments():

    # Start this parser up just to extract the -f or --file argument if it's
    # there, and push all of those args into the command line for the real
    # parser to deal with.


    parser = argparse.ArgumentParser(description="Fit the FLASH data and plot fit results.")

    parser.add_argument("-f", "--file", type=str, action="append",
                        help="""Text file containing any arguments to this utility. NOTE that this is an either/or with other command line arguments. If additional arguments are provided, THEY WILL BE IGNORED. 
                        Arguments in the file should be separated by newlines or spaces. Lines starting with '#' will be ignored. 
                        Example: -f my_args.cfg""", metavar="FILE")

    fit_args = parser.add_argument_group("Fit options", "Configure how the FLASH fitting is set up and run.")

    fit_args.add_argument("--spill_time_end", type=float, default=-99999999.0,
                          help="End of the spill time within the run (sec); defaults to finding manually with SpillTimeFinder")
    
    fit_args.add_argument("--spill_time_finder_window", type=float, default=0.1,
                          help="Time window to consider when trying to find the spill start time; defaults to 100 microseconds")
    
    fit_args.add_argument("-rl", "--run_length", type=float, default=600.0,
                         help="Length of the run in seconds; default 600s")
    
    fit_args.add_argument("-n", "--num_bins", type=int, default=100,
                         help="Number of bins to use in NAME OF PLOT; default 100")
    
    fit_args.add_argument("-ff", "--fit_function", type=str, default="",
                            help="Function to fit the data to. Prefer to specify isotopes, and only use this if you need to avoid that; default to empty string")
    
    fit_args.add_argument("-fi", "--fit_isotopes", type=parse_comma_separated_list, default=["Cu58","Cu62","Cu59"],
                            help="Positron emitting isotopes to fit the data to; default Cu58 Cu62 Cu59")
    
    fit_args.add_argument("-rcfg", "--run_config", type=str, default="Ni (21 cm)",
                          help="Configuration of the run; default Ni (21 cm)")
    
    fit_args.add_argument("--initial_fit_params", type=parse_comma_separated_list, default=[""],
                            help="Initial guess values to be passed to minuit fitter. NOTE: the parameter order must match what is expected by the corresponding fit funciton in fit_functions.py; default ''")

    fit_args.add_argument("--y_margin_adjust", type=float, default=1.0,
                            help="Play with this to adjust the y-axis margin to make room for the fit lines; default 1.0")
    
    fit_args.add_argument("--run_end_time", type=float, default=-1.0,
                            help="Time after run start to consider in fit; defaults to looking at all data after the spill time start")

    fit_args.add_argument("--create_first_plot", type=bool, default=False,
                            help="Decides whether to create the first plot (to be elaborated on). Defaults to False until fixed.")

    software_args = parser.add_argument_group("File Options", "Specify the filepath and the data file options")

    software_args.add_argument("--in_dir", type=str, default="Flash_Therapy/PET_3-5-23/",
                               help="Filepath to find the input data files; defaults to Flash_Therapy/PET_3-5-23/.=")

    software_args.add_argument("-d", "--data_file", type=str, default="MDA-Ni-10min-210mm-NoCu_coinc.dat",
                               help="Data file name to read in. Reset --in_dir if using a nonstandard location; default MDA-Ni-10min-210mm-NoCu_coinc.dat")

    software_args.add_argument("-o", "--output_dir", type=str, default="./",
                               help="Output directory to host images. By default, outputs images into the same directory as the fitting code.")

    software_args.add_argument("-dw", "--dont_write", type=bool, default=False,
                               help="Flag to indicate whether to write the found spill start time to the config file; default is False")

    args, remaining_argv = parser.parse_known_args()


    if args.file:
        filename = args.file      #workaround to get the filename accessible later
        config_args = parse_config_file(args.file[0])
        args = parser.parse_args(config_args + remaining_argv)  # Reparse with config options

        args.file=filename          #reappend the filename into the code
    if len(args.initial_fit_params) != len(args.fit_isotopes) and args.initial_fit_params != [""]:
        print("Number of initial fit parameters guesses must match the number of isotopes to fit!")
        sys.exit(1)
    return args

#End of parse_arguments()

def parse_config_file(filename):
    args = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and full-line comments
                line = line.split("#", 1)[0].strip()  # Remove inline comments
                if line:  # Ensure there's still content after stripping the comment
                    parts = line.split(maxsplit=1)
                    args.extend(parts)  # Properly tokenize the line
    return args
# END of parse_config_file()



def parse_comma_separated_list(value):
    return [item.strip() for item in value.split(",")]
#END of parse_comma_separated_list()
#-----------------------------------------------------------------------------------------